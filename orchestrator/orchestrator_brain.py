"""Orchestrator brain — deterministic event routing.

Maps incoming events to actions. No LLM calls.
The brain decides WHAT should happen; workers execute HOW.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
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
    SEND_NOTIFICATION = "send_notification"
    UPDATE_HEARTBEAT = "update_heartbeat"
    PROCESS_FEEDBACK = "process_feedback"
    LOG_UNKNOWN = "log_unknown"


@dataclass
class Action:
    type: ActionType
    event_id: str
    bot_id: str
    details: dict | None = None
    chain_id: str = ""


class ErrorRateTracker:
    """Sliding window counter for error frequency tracking per bot+error_type."""

    def __init__(self, window_seconds: int = 3600, storm_threshold: int = 3) -> None:
        self._window = window_seconds
        self._storm_threshold = storm_threshold
        self._events: dict[str, list[float]] = defaultdict(list)
        self._triage_spawned: dict[str, bool] = defaultdict(bool)

    def _prune(self, key: str, cutoff: float) -> None:
        """Prune expired timestamps and remove empty keys."""
        self._events[key] = [t for t in self._events[key] if t > cutoff]
        if not self._events[key]:
            del self._events[key]
            self._triage_spawned.pop(key, None)

    def record_and_check(self, bot_id: str, error_type: str) -> tuple[int, bool, bool]:
        """Record an error and return (count_in_window, is_suppressed, is_storm)."""
        key = f"{bot_id}:{error_type}"
        now = time.monotonic()
        cutoff = now - self._window

        self._prune(key, cutoff)
        self._events[key].append(now)

        count = len(self._events[key])
        already_triaging = self._triage_spawned.get(key, False)
        is_storm = count >= self._storm_threshold

        if is_storm:
            self._triage_spawned[key] = True
            return count, False, True

        if already_triaging:
            return count, True, False

        self._triage_spawned[key] = True
        return count, False, False

    def total_count(self) -> int:
        """Return the total error count across all keys within the current window."""
        now = time.monotonic()
        cutoff = now - self._window
        total = 0
        for key in list(self._events):
            self._prune(key, cutoff)
            total += len(self._events.get(key, []))
        return total


class OrchestratorBrain:
    """Deterministic decision engine for incoming events."""

    def __init__(self) -> None:
        self._error_tracker = ErrorRateTracker()
        self.last_daily_analysis: str | None = None
        self.last_weekly_analysis: str | None = None

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
            error_type = payload.get("error_type", "unknown")
            count, suppressed, is_storm = self._error_tracker.record_and_check(bot_id, error_type)

            if suppressed:
                return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

            details = dict(payload)
            if is_storm:
                details["urgency"] = "error_storm"
                details["error_count"] = count

            return [
                Action(type=ActionType.SPAWN_TRIAGE, event_id=event_id, bot_id=bot_id, details=details),
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

    def _handle_notification_trigger(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.SEND_NOTIFICATION, event_id=event_id, bot_id=bot_id)]

    def _handle_user_feedback(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        payload = json.loads(event.get("payload", "{}"))
        return [Action(type=ActionType.PROCESS_FEEDBACK, event_id=event_id, bot_id=bot_id, details=payload)]

    def _handle_coordinator_action(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        """Route coordinator action events to daily analysis queue.

        Coordinator events describe strategy interactions (tighten/loosen stops,
        sizing adjustments) and are consumed by the daily metrics pipeline to
        produce coordinator_impact.json.
        """
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_daily_snapshot(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_order(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_process_quality(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_bot_error(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return self._handle_error(event_id, bot_id, event)

    def _handle_post_exit(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_portfolio_rule(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_market_snapshot(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_exit_movement(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id)]

    def _handle_indicator_snapshot(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id,
                        details={"event_type": "indicator_snapshot", "payload": event.get("payload", "{}")})]

    def _handle_orderbook_context(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id,
                        details={"event_type": "orderbook_context", "payload": event.get("payload", "{}")})]

    def _handle_filter_decision(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id,
                        details={"event_type": "filter_decision", "payload": event.get("payload", "{}")})]

    def _handle_parameter_change(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        """Route parameter changes — safety-critical params get immediate alert."""
        payload = json.loads(event.get("payload", "{}")) if isinstance(event.get("payload"), str) else event.get("payload", {})
        param_name = payload.get("param_name", "")
        is_safety_critical = payload.get("is_safety_critical", False)

        _SAFETY_CRITICAL_PARAMS = {
            "risk_per_trade", "max_position_size", "kill_switch_enabled",
            "trailing_stop_pct", "max_drawdown_pct", "leverage_limit",
        }

        if is_safety_critical or param_name in _SAFETY_CRITICAL_PARAMS:
            return [Action(
                type=ActionType.ALERT_IMMEDIATE,
                event_id=event_id,
                bot_id=bot_id,
                details={"param_name": param_name, "safety_critical": True, **payload},
            )]
        return [Action(type=ActionType.QUEUE_FOR_DAILY, event_id=event_id, bot_id=bot_id,
                        details={"event_type": "parameter_change", "payload": event.get("payload", "{}")})]

    def _handle_unknown(self, event_id: str, bot_id: str, event: dict) -> list[Action]:
        return [Action(type=ActionType.LOG_UNKNOWN, event_id=event_id, bot_id=bot_id)]

    def get_error_rate_1h(self) -> float:
        """Return the total error count in the last hour across all keys."""
        return float(self._error_tracker.total_count())

    def record_daily_analysis(self, timestamp: str) -> None:
        """Record the timestamp of the last daily analysis run."""
        self.last_daily_analysis = timestamp

    def record_weekly_analysis(self, timestamp: str) -> None:
        """Record the timestamp of the last weekly analysis run."""
        self.last_weekly_analysis = timestamp

    _handlers: dict = {
        "trade": _handle_trade,
        "missed_opportunity": _handle_missed_opportunity,
        "error": _handle_error,
        "heartbeat": _handle_heartbeat,
        "daily_analysis_trigger": _handle_daily_analysis_trigger,
        "weekly_summary_trigger": _handle_weekly_summary_trigger,
        "wfo_trigger": _handle_wfo_trigger,
        "notification_trigger": _handle_notification_trigger,
        "coordinator_action": _handle_coordinator_action,
        "daily_snapshot": _handle_daily_snapshot,
        "order": _handle_order,
        "process_quality": _handle_process_quality,
        "bot_error": _handle_bot_error,
        "post_exit": _handle_post_exit,
        "portfolio_rule": _handle_portfolio_rule,
        "market_snapshot": _handle_market_snapshot,
        "exit_movement": _handle_exit_movement,
        "user_feedback": _handle_user_feedback,
        "parameter_change": _handle_parameter_change,
        "indicator_snapshot": _handle_indicator_snapshot,
        "orderbook_context": _handle_orderbook_context,
        "filter_decision": _handle_filter_decision,
    }
