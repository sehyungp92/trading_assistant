"""Worker — consumes events from the queue, routes through the brain, executes actions.

The worker is the bridge between the event queue and the rest of the system.
It pulls pending events, asks the OrchestratorBrain what to do, and dispatches.
"""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

from orchestrator.conversation_tracker import ConversationTracker
from orchestrator.db.queue import EventQueue
from orchestrator.event_stream import EventStream
from orchestrator.orchestrator_brain import OrchestratorBrain, Action, ActionType
from orchestrator.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        queue: EventQueue,
        registry: TaskRegistry,
        brain: OrchestratorBrain,
        event_stream: EventStream | None = None,
        conversation_tracker: ConversationTracker | None = None,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._brain = brain
        self._event_stream: EventStream | None = event_stream
        self._conversation_tracker: ConversationTracker | None = conversation_tracker

        # Pluggable handlers — set these to hook into the action pipeline
        self.on_alert: Callable[[Action], Awaitable[None]] | None = None
        self.on_heartbeat: Callable[[Action], Awaitable[None]] | None = None
        self.on_triage: Callable[[Action], Awaitable[None]] | None = None
        self.on_daily_analysis: Callable[[Action], Awaitable[None]] | None = None
        self.on_weekly_analysis: Callable[[Action], Awaitable[None]] | None = None
        self.on_wfo: Callable[[Action], Awaitable[None]] | None = None
        self.on_notification: Callable[[Action], Awaitable[None]] | None = None
        self.on_feedback: Callable[[Action], Awaitable[None]] | None = None

        self.daily_queue_counts: dict[str, int] = {}
        self.weekly_queue_counts: dict[str, int] = {}

    def _emit(self, event_type: str, data: dict | None = None) -> None:
        """Broadcast an event to the SSE stream (if attached)."""
        if self._event_stream:
            self._event_stream.broadcast(event_type, data)

    async def process_batch(self, limit: int = 10) -> int:
        """Process up to `limit` pending events. Returns count processed."""
        events = await self._queue.peek(limit=limit)
        if not events:
            return 0

        processed = 0
        for event in events:
            event_id = event.get("event_id", "")
            chain_id = event.get("chain_id", "")

            # Begin or extend conversation chain for loop protection
            if self._conversation_tracker:
                if chain_id:
                    ok = self._conversation_tracker.extend_chain(chain_id, event_id)
                    if not ok:
                        logger.warning(
                            "Loop detected for chain %s on event %s — skipping",
                            chain_id, event_id,
                        )
                        await self._queue.ack(event_id)
                        processed += 1
                        continue
                else:
                    chain = self._conversation_tracker.begin_chain(event_id)
                    chain_id = chain.chain_id

            try:
                self._emit("event_processing_start", {"event_id": event_id})
                actions = self._brain.decide(event)
                for action in actions:
                    action.chain_id = chain_id
                    await self._dispatch(action)
                await self._queue.ack(event_id)
                processed += 1
                self._emit("event_processing_complete", {"event_id": event_id})
            except Exception as exc:
                logger.exception("Failed to process event %s", event_id)
                self._emit("event_processing_error", {"event_id": event_id, "error": str(exc)})
                is_dead = await self._queue.nack(event_id, str(exc))
                if is_dead:
                    logger.error(
                        "Event %s moved to dead-letter after exhausting retries", event_id,
                    )

        return processed

    def _record_queued_event(self, bot_id: str, action_type: ActionType) -> None:
        """Track event counts for QUEUE_FOR_DAILY/WEEKLY actions."""
        if action_type == ActionType.QUEUE_FOR_DAILY:
            self.daily_queue_counts[bot_id] = self.daily_queue_counts.get(bot_id, 0) + 1
        elif action_type == ActionType.QUEUE_FOR_WEEKLY:
            self.weekly_queue_counts[bot_id] = self.weekly_queue_counts.get(bot_id, 0) + 1

    def get_and_reset_daily_counts(self) -> dict[str, int]:
        """Get accumulated daily event counts and reset. Called by daily analysis trigger."""
        counts = dict(self.daily_queue_counts)
        self.daily_queue_counts = {}
        return counts

    def get_and_reset_weekly_counts(self) -> dict[str, int]:
        """Get accumulated weekly event counts and reset. Called by weekly analysis trigger."""
        counts = dict(self.weekly_queue_counts)
        self.weekly_queue_counts = {}
        return counts

    async def _dispatch(self, action: Action) -> None:
        """Route an action to the appropriate handler."""
        if action.type == ActionType.ALERT_IMMEDIATE:
            if self.on_alert:
                await self.on_alert(action)
            else:
                logger.warning("ALERT (no handler): %s — %s", action.bot_id, action.details)

        elif action.type == ActionType.SPAWN_TRIAGE:
            if self.on_triage:
                await self.on_triage(action)
            else:
                logger.info("TRIAGE needed: %s — %s", action.bot_id, action.details)

        elif action.type == ActionType.UPDATE_HEARTBEAT:
            if self.on_heartbeat:
                await self.on_heartbeat(action)
            else:
                logger.debug("Heartbeat: %s", action.bot_id)

        elif action.type == ActionType.SPAWN_DAILY_ANALYSIS:
            if self.on_daily_analysis:
                await self.on_daily_analysis(action)
            else:
                logger.info("Daily analysis triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.SPAWN_WEEKLY_SUMMARY:
            if self.on_weekly_analysis:
                await self.on_weekly_analysis(action)
            else:
                logger.info("Weekly analysis triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.SPAWN_WFO:
            if self.on_wfo:
                await self.on_wfo(action)
            else:
                logger.info("WFO triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.SEND_NOTIFICATION:
            if self.on_notification:
                await self.on_notification(action)
            else:
                logger.info("Notification triggered but no handler set: %s", action.event_id)

        elif action.type == ActionType.PROCESS_FEEDBACK:
            if self.on_feedback:
                await self.on_feedback(action)
            else:
                logger.info("Feedback received but no handler set: %s", action.event_id)

        elif action.type == ActionType.QUEUE_FOR_DAILY:
            self._record_queued_event(action.bot_id, action.type)
            logger.debug("Queued for daily: %s", action.event_id)

        elif action.type == ActionType.QUEUE_FOR_WEEKLY:
            self._record_queued_event(action.bot_id, action.type)
            logger.debug("Queued for weekly: %s", action.event_id)

        elif action.type == ActionType.LOG_UNKNOWN:
            logger.warning("Unknown event type from %s: %s", action.bot_id, action.event_id)
