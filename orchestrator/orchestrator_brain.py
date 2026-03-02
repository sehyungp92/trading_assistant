"""Orchestrator brain — deterministic event routing.

Maps incoming events to actions. No LLM calls.
The brain decides WHAT should happen; workers execute HOW.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    QUEUE_FOR_DAILY = "queue_for_daily"
    ALERT_IMMEDIATE = "alert_immediate"
    SPAWN_TRIAGE = "spawn_triage"
    SPAWN_DAILY_ANALYSIS = "spawn_daily_analysis"
    SPAWN_WEEKLY_SUMMARY = "spawn_weekly_summary"
    SPAWN_WFO = "spawn_wfo"
    QUEUE_FOR_WEEKLY = "queue_for_weekly"
    UPDATE_HEARTBEAT = "update_heartbeat"
    LOG_UNKNOWN = "log_unknown"


@dataclass
class Action:
    type: ActionType
    event_id: str
    bot_id: str
    details: dict | None = None


class OrchestratorBrain:
    """Deterministic decision engine for incoming events."""

    def decide(self, event: dict) -> list[Action]:
        """Given a raw event dict, return a list of actions to take."""
        event_type = event.get("event_type", "")
        event_id = event.get("event_id", "")
        bot_id = event.get("bot_id", "")

        handler = self._handlers.get(event_type)
        if handler is not None:
            # _handlers stores unbound functions; pass self explicitly
            return handler(self, event_id, bot_id, event)
        return self._handle_unknown(event_id, bot_id, event)

    def _handle_trade(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_missed_opportunity(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_error(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        payload = json.loads(event.get("payload", "{}"))
        severity = payload.get("severity", "MEDIUM").upper()

        if severity == "CRITICAL":
            return [
                Action(type=ActionType.ALERT_IMMEDIATE, event_id=event_id, bot_id=bot_id, details=payload),
            ]
        elif severity == "HIGH":
            return [
                Action(type=ActionType.SPAWN_TRIAGE, event_id=event_id, bot_id=bot_id, details=payload),
            ]
        elif severity == "LOW":
            return [Action(type=ActionType.QUEUE_FOR_WEEKLY, event_id=event_id, bot_id=bot_id)]
        else:  # MEDIUM or unrecognized
            return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_heartbeat(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.UPDATE_HEARTBEAT, event_id=event_id, bot_id=bot_id)]

    def _handle_daily_analysis_trigger(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id=event_id, bot_id=bot_id)]

    def _handle_weekly_summary_trigger(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.SPAWN_WEEKLY_SUMMARY, event_id=event_id, bot_id=bot_id)]

    def _handle_wfo_trigger(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.SPAWN_WFO, event_id=event_id, bot_id=bot_id)]

    def _handle_unknown(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.LOG_UNKNOWN, event_id=event_id, bot_id=bot_id)]

    _handlers: dict = {
        "trade": _handle_trade,
        "missed_opportunity": _handle_missed_opportunity,
        "error": _handle_error,
        "heartbeat": _handle_heartbeat,
        "daily_analysis_trigger": _handle_daily_analysis_trigger,
        "weekly_summary_trigger": _handle_weekly_summary_trigger,
        "wfo_trigger": _handle_wfo_trigger,
    }
