# skills/autonomous_pipeline.py
"""Autonomous pipeline — suggestion-to-backtest-to-approval automation.

Processes new suggestions: filters actionable ones, backtests them,
creates approval requests, and sends Telegram notifications with
inline keyboard buttons.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from schemas.autonomous_pipeline import (
    ApprovalRequest,
    BacktestComparison,
    ParameterDefinition,
)
from skills.approval_tracker import ApprovalTracker
from skills.config_registry import ConfigRegistry
from skills.suggestion_backtester import SuggestionBacktester

logger = logging.getLogger(__name__)

_ACTIONABLE_TIERS = {"parameter", "filter"}
_MIN_CONFIDENCE = 0.5


class AutonomousPipeline:
    """Processes suggestions through backtest → approval → notification."""

    def __init__(
        self,
        config_registry: ConfigRegistry,
        backtester: SuggestionBacktester,
        approval_tracker: ApprovalTracker,
        suggestion_tracker: Any,  # SuggestionTracker
        telegram_bot: Any | None = None,
        telegram_renderer: Any | None = None,
        event_stream: Any | None = None,
    ) -> None:
        self._registry = config_registry
        self._backtester = backtester
        self._approval_tracker = approval_tracker
        self._suggestion_tracker = suggestion_tracker
        self._telegram_bot = telegram_bot
        self._telegram_renderer = telegram_renderer
        self._event_stream = event_stream

    async def process_new_suggestions(
        self,
        suggestion_ids: list[str],
        run_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """Process suggestions through backtest and create approval requests.

        Returns list of created ApprovalRequests.
        """
        results: list[ApprovalRequest] = []

        # Batch-load all suggestions once (O(n) instead of O(n*m))
        all_suggestions: dict[str, dict] = {}
        if self._suggestion_tracker:
            for s in self._suggestion_tracker.load_all():
                sid_key = s.get("suggestion_id", "")
                if sid_key:
                    all_suggestions[sid_key] = s

        for sid in suggestion_ids:
            try:
                result = await self._process_one(sid, run_id, all_suggestions)
                if result is not None:
                    results.append(result)
            except Exception:
                logger.exception("Pipeline error processing suggestion %s", sid)

        if results and self._event_stream:
            self._event_stream.broadcast("autonomous_pipeline_complete", {
                "run_id": run_id,
                "approval_requests_created": len(results),
            })

        return results

    async def _process_one(
        self,
        suggestion_id: str,
        run_id: str | None,
        all_suggestions: dict[str, dict] | None = None,
    ) -> ApprovalRequest | None:
        """Process a single suggestion through the pipeline."""
        # 1. Load suggestion record (use batch dict if available)
        if all_suggestions is not None:
            suggestion = all_suggestions.get(suggestion_id)
        else:
            suggestion = self._load_suggestion(suggestion_id)
        if suggestion is None:
            logger.debug("Suggestion %s not found in tracker", suggestion_id)
            return None

        # 2. Check actionability
        if not self._is_actionable(suggestion):
            logger.debug("Suggestion %s not actionable (tier=%s)", suggestion_id, suggestion.get("tier"))
            return None

        # 3. Resolve to parameters
        params = self._registry.resolve_suggestion_to_params(suggestion)
        if not params:
            logger.debug("No matching parameters for suggestion %s", suggestion_id)
            return None

        # 4. Process first matching parameter
        param = params[0]
        proposed_value = self._extract_proposed_value(suggestion, param)
        if proposed_value is None:
            logger.debug("Could not extract proposed value for suggestion %s", suggestion_id)
            return None

        # 5. Validate value
        valid, msg = self._registry.validate_value(param, proposed_value)
        if not valid:
            logger.info("Proposed value invalid for %s: %s", suggestion_id, msg)
            return None

        # 6. Run backtest
        comparison = await self._backtester.backtest_suggestion(
            suggestion_id=suggestion_id,
            bot_id=suggestion.get("bot_id", ""),
            param_name=param.param_name,
            current_value=param.current_value,
            proposed_value=proposed_value,
        )

        if not comparison.passes_safety:
            logger.info(
                "Backtest failed safety for %s: %s",
                suggestion_id, comparison.safety_notes,
            )
            return None

        # 7. Create approval request
        request_id = hashlib.sha256(
            f"{suggestion_id}:{param.param_name}:{proposed_value}".encode()
        ).hexdigest()[:12]

        request = ApprovalRequest(
            request_id=request_id,
            suggestion_id=suggestion_id,
            bot_id=suggestion.get("bot_id", ""),
            param_changes=[{
                "param_name": param.param_name,
                "current": param.current_value,
                "proposed": proposed_value,
            }],
            backtest_summary=comparison,
        )

        self._approval_tracker.create_request(request)
        logger.info("Created approval request %s for suggestion %s", request_id, suggestion_id)

        # 8. Send Telegram notification
        await self._send_telegram_notification(request)

        return request

    def _load_suggestion(self, suggestion_id: str) -> dict | None:
        """Load a suggestion record from the tracker."""
        if not self._suggestion_tracker:
            return None
        for s in self._suggestion_tracker.load_all():
            if s.get("suggestion_id") == suggestion_id:
                return s
        return None

    def _is_actionable(self, suggestion: dict) -> bool:
        """Check if suggestion is actionable: correct tier, sufficient confidence, not already queued."""
        tier = suggestion.get("tier", "")
        if tier not in _ACTIONABLE_TIERS:
            return False

        confidence = suggestion.get("confidence", 0.0)
        if confidence < _MIN_CONFIDENCE:
            return False

        # Check not already in approval queue
        suggestion_id = suggestion.get("suggestion_id", "")
        pending = self._approval_tracker.get_pending()
        if any(r.suggestion_id == suggestion_id for r in pending):
            return False

        return True

    def _extract_proposed_value(
        self,
        suggestion: dict,
        param: ParameterDefinition,
    ) -> Any | None:
        """Extract proposed value from suggestion.

        Priority:
        1. Structured field: suggestion["proposed_value"] (from AgentSuggestion schema)
        2. Regex extraction from free text (fallback)
        3. valid_range midpoint (last resort)
        """
        # 1. Try structured field first (from AgentSuggestion.proposed_value)
        structured_value = suggestion.get("proposed_value")
        if structured_value is not None:
            return self._cast_value(structured_value, param.value_type)

        # 2. Regex fallback on free text
        text = f"{suggestion.get('title', '')} {suggestion.get('description', '')}"
        param_variants = [
            param.param_name,
            param.param_name.replace("_", " "),
        ]

        for variant in param_variants:
            # "increase/decrease/set X to Y"
            pattern = rf"(?:increase|decrease|set|adjust)\s+{re.escape(variant)}\s+to\s+([0-9.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._cast_value(match.group(1), param.value_type)

            # "change X from A to B"
            pattern = rf"change\s+{re.escape(variant)}\s+from\s+[0-9.]+\s+to\s+([0-9.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._cast_value(match.group(1), param.value_type)

        # Try to extract any "to <number>" pattern
        match = re.search(r"\bto\s+([0-9]+\.?[0-9]*)\b", text)
        if match:
            return self._cast_value(match.group(1), param.value_type)

        # Fallback: midpoint of valid range
        if param.valid_range is not None:
            mid = (param.valid_range[0] + param.valid_range[1]) / 2
            return self._cast_value(mid, param.value_type)

        return None

    @staticmethod
    def _cast_value(value: Any, value_type: str) -> Any:
        try:
            if value_type == "int":
                return int(float(value))
            if value_type == "float":
                return float(value)
            if value_type == "bool":
                return str(value).lower() in ("true", "1", "yes")
            return value
        except (TypeError, ValueError):
            return None

    async def _send_telegram_notification(self, request: ApprovalRequest) -> None:
        """Send Telegram approval card if bot and renderer are available.

        Stores the returned message_id on the request for later editing.
        """
        if self._telegram_renderer is None or self._telegram_bot is None:
            return
        try:
            text, keyboard = self._telegram_renderer.render_approval_request(request)
            message_id = await self._telegram_bot.send_message(text, keyboard=keyboard)
            if message_id is not None:
                request.message_id = message_id
                self._approval_tracker.set_message_id(request.request_id, message_id)
        except Exception:
            logger.exception("Failed to send Telegram notification for %s", request.request_id)
