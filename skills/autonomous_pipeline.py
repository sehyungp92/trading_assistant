"""Autonomous pipeline for suggestion-to-approval automation."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from schemas.autonomous_pipeline import (
    ApprovalRequest,
    ChangeKind,
    FileChange,
    ParameterDefinition,
    RepoRiskTier,
)
from skills.approval_tracker import ApprovalTracker
from skills.config_registry import ConfigRegistry
from skills.repo_change_guard import RepoChangeGuard
from skills.suggestion_backtester import SuggestionBacktester

logger = logging.getLogger(__name__)

_ACTIONABLE_TIERS = {"parameter", "filter", "hypothesis"}
_MIN_CONFIDENCE = 0.5


class AutonomousPipeline:
    """Processes recorded suggestions into approval requests."""

    def __init__(
        self,
        config_registry: ConfigRegistry,
        backtester: SuggestionBacktester,
        approval_tracker: ApprovalTracker,
        suggestion_tracker: Any,
        telegram_bot: Any | None = None,
        telegram_renderer: Any | None = None,
        event_stream: Any | None = None,
        repo_change_guard: RepoChangeGuard | None = None,
    ) -> None:
        self._registry = config_registry
        self._backtester = backtester
        self._approval_tracker = approval_tracker
        self._suggestion_tracker = suggestion_tracker
        self._telegram_bot = telegram_bot
        self._telegram_renderer = telegram_renderer
        self._event_stream = event_stream
        self._repo_change_guard = repo_change_guard or RepoChangeGuard()

    async def process_new_suggestions(
        self,
        suggestion_ids: list[str],
        run_id: str | None = None,
    ) -> list[ApprovalRequest]:
        """Process new suggestions into approval requests."""
        results: list[ApprovalRequest] = []
        all_suggestions = self._load_all_suggestions()

        for suggestion_id in suggestion_ids:
            try:
                result = await self._process_one(suggestion_id, run_id, all_suggestions)
                if result is not None:
                    results.append(result)
            except Exception:
                logger.exception("Pipeline error processing suggestion %s", suggestion_id)

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
        all_suggestions: dict[str, dict],
    ) -> ApprovalRequest | None:
        suggestion = all_suggestions.get(suggestion_id)
        if suggestion is None:
            logger.debug("Suggestion %s not found in tracker", suggestion_id)
            return None
        if not self._is_actionable(suggestion):
            logger.debug("Suggestion %s not actionable", suggestion_id)
            return None

        tier = suggestion.get("tier", "")
        if tier in {"parameter", "filter"}:
            request = await self._build_parameter_request(suggestion_id, suggestion)
        elif tier == "hypothesis":
            request = self._build_structural_request(suggestion_id, suggestion)
        else:
            request = None

        if request is None:
            return None

        self._approval_tracker.create_request(request)
        logger.info("Created approval request %s for suggestion %s", request.request_id, suggestion_id)
        await self._send_telegram_notification(request)
        return request

    async def _build_parameter_request(
        self,
        suggestion_id: str,
        suggestion: dict,
    ) -> ApprovalRequest | None:
        params = self._registry.resolve_suggestion_to_params(suggestion)
        if not params:
            logger.debug("No matching parameters for suggestion %s", suggestion_id)
            return None

        param = params[0]
        proposed_value = self._extract_proposed_value(suggestion, param)
        if proposed_value is None:
            logger.debug("Could not extract proposed value for suggestion %s", suggestion_id)
            return None

        valid, message = self._registry.validate_value(param, proposed_value)
        if not valid:
            logger.info("Proposed value invalid for %s: %s", suggestion_id, message)
            return None

        comparison = await self._backtester.backtest_suggestion(
            suggestion_id=suggestion_id,
            bot_id=suggestion.get("bot_id", ""),
            param_name=param.param_name,
            current_value=param.current_value,
            proposed_value=proposed_value,
        )
        if not comparison.passes_safety:
            logger.info("Backtest failed safety for %s: %s", suggestion_id, comparison.safety_notes)
            return None

        profile = self._registry.get_profile(suggestion.get("bot_id", ""))
        if profile and self._repo_change_guard.blocked_paths(profile, [param.file_path]):
            logger.info(
                "Blocked parameter suggestion %s outside allowed_edit_paths: %s",
                suggestion_id,
                param.file_path,
            )
            return None
        risk_tier = self._risk_tier_for_files(profile, [param.file_path]) if profile else RepoRiskTier.REQUIRES_APPROVAL
        request_id = hashlib.sha256(
            f"{suggestion_id}:{param.param_name}:{proposed_value}".encode(),
        ).hexdigest()[:12]

        return ApprovalRequest(
            request_id=request_id,
            suggestion_id=suggestion_id,
            bot_id=suggestion.get("bot_id", ""),
            change_kind=ChangeKind.PARAMETER_CHANGE,
            title=f"Update {param.param_name}",
            summary=suggestion.get("title", ""),
            param_changes=[{
                "param_name": param.param_name,
                "current": param.current_value,
                "proposed": proposed_value,
            }],
            planned_files=[param.file_path],
            verification_commands=profile.verification_commands if profile else [],
            risk_tier=risk_tier,
            backtest_summary=comparison,
        )

    def _build_structural_request(
        self,
        suggestion_id: str,
        suggestion: dict,
    ) -> ApprovalRequest | None:
        context = suggestion.get("implementation_context") or {}
        file_changes = self._parse_file_changes(context.get("file_changes", []))
        implementation_notes = context.get("notes", "") or suggestion.get("description", "")
        if not file_changes and not implementation_notes:
            logger.debug("No implementation payload for structural suggestion %s", suggestion_id)
            return None

        profile = self._registry.get_profile(suggestion.get("bot_id", ""))
        planned_files = list(context.get("planned_files", []) or [])
        if not planned_files:
            planned_files = [file_change.file_path for file_change in file_changes]
        if profile and self._repo_change_guard.blocked_paths(profile, planned_files):
            logger.info(
                "Blocked structural suggestion %s outside allowed_edit_paths: %s",
                suggestion_id,
                planned_files,
            )
            return None
        risk_tier = (
            self._risk_tier_for_files(profile, planned_files)
            if profile and planned_files
            else RepoRiskTier.REQUIRES_APPROVAL
        )
        request_id = hashlib.sha256(
            f"{suggestion_id}:structural:{'|'.join(planned_files)}:{implementation_notes}".encode(),
        ).hexdigest()[:12]

        return ApprovalRequest(
            request_id=request_id,
            suggestion_id=suggestion_id,
            bot_id=suggestion.get("bot_id", ""),
            change_kind=ChangeKind.STRUCTURAL_CHANGE,
            title=suggestion.get("title", "Structural change"),
            summary=suggestion.get("description", ""),
            file_changes=file_changes,
            planned_files=planned_files,
            verification_commands=context.get("verification_commands", []) or (
                profile.verification_commands if profile else []
            ),
            risk_tier=risk_tier,
            draft_pr=risk_tier == RepoRiskTier.AUTO and bool(file_changes),
            implementation_notes=implementation_notes,
        )

    def _load_all_suggestions(self) -> dict[str, dict]:
        suggestions: dict[str, dict] = {}
        if not self._suggestion_tracker:
            return suggestions
        for suggestion in self._suggestion_tracker.load_all():
            suggestion_id = suggestion.get("suggestion_id", "")
            if suggestion_id:
                suggestions[suggestion_id] = suggestion
        return suggestions

    def _is_actionable(self, suggestion: dict) -> bool:
        tier = suggestion.get("tier", "")
        if tier not in _ACTIONABLE_TIERS:
            return False
        if suggestion.get("confidence", 0.0) < _MIN_CONFIDENCE:
            return False
        suggestion_id = suggestion.get("suggestion_id", "")
        return self._approval_tracker.find_latest_for_suggestion(suggestion_id) is None

    def _extract_proposed_value(
        self,
        suggestion: dict,
        param: ParameterDefinition,
    ) -> Any | None:
        structured_value = suggestion.get("proposed_value")
        if structured_value is not None:
            return self._cast_value(structured_value, param.value_type)

        text = f"{suggestion.get('title', '')} {suggestion.get('description', '')}"
        param_variants = [param.param_name, param.param_name.replace("_", " ")]
        for variant in param_variants:
            pattern = rf"(?:increase|decrease|set|adjust)\s+{re.escape(variant)}\s+to\s+([0-9.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._cast_value(match.group(1), param.value_type)

            pattern = rf"change\s+{re.escape(variant)}\s+from\s+[0-9.]+\s+to\s+([0-9.]+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._cast_value(match.group(1), param.value_type)

        match = re.search(r"\bto\s+([0-9]+\.?[0-9]*)\b", text)
        if match:
            return self._cast_value(match.group(1), param.value_type)
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

    def _risk_tier_for_files(
        self,
        profile,
        file_paths: list[str],
    ) -> RepoRiskTier:
        if profile is None:
            return RepoRiskTier.REQUIRES_APPROVAL
        result = self._repo_change_guard.check_paths(profile, file_paths)
        if result.tier.value >= 2:
            return RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
        if result.tier.value == 0:
            return RepoRiskTier.AUTO
        return RepoRiskTier.REQUIRES_APPROVAL

    @staticmethod
    def _parse_file_changes(raw_changes: list[dict]) -> list[FileChange]:
        parsed: list[FileChange] = []
        for raw_change in raw_changes:
            try:
                parsed.append(FileChange(**raw_change))
            except Exception:
                logger.warning("Skipping malformed file change payload", exc_info=True)
        return parsed

    async def _send_telegram_notification(self, request: ApprovalRequest) -> None:
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
