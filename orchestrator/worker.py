"""Worker — consumes events from the queue, routes through the brain, executes actions.

The worker is the bridge between the event queue and the rest of the system.
It pulls pending events, asks the OrchestratorBrain what to do, and dispatches.
"""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

from orchestrator.db.queue import EventQueue
from orchestrator.orchestrator_brain import OrchestratorBrain, Action, ActionType
from orchestrator.task_registry import TaskRegistry

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        queue: EventQueue,
        registry: TaskRegistry,
        brain: OrchestratorBrain,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._brain = brain

        # Pluggable handlers — set these to hook into the action pipeline
        self.on_alert: Callable[[Action], Awaitable[None]] | None = None
        self.on_heartbeat: Callable[[Action], Awaitable[None]] | None = None
        self.on_triage: Callable[[Action], Awaitable[None]] | None = None
        self.on_daily_analysis: Callable[[Action], Awaitable[None]] | None = None
        self.on_weekly_analysis: Callable[[Action], Awaitable[None]] | None = None

    async def process_batch(self, limit: int = 10) -> int:
        """Process up to `limit` pending events. Returns count processed."""
        events = await self._queue.peek(limit=limit)
        if not events:
            return 0

        processed = 0
        for event in events:
            try:
                actions = self._brain.decide(event)
                for action in actions:
                    await self._dispatch(action)
                await self._queue.ack(event["event_id"])
                processed += 1
            except Exception:
                logger.exception("Failed to process event %s", event.get("event_id"))

        return processed

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

        elif action.type == ActionType.QUEUE_FOR_DAILY:
            logger.debug("Queued for daily: %s", action.event_id)

        elif action.type == ActionType.LOG_UNKNOWN:
            logger.warning("Unknown event type from %s: %s", action.bot_id, action.event_id)
