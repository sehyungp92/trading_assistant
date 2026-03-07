# skills/approval_handler.py
"""Approval handler — processes approve/reject/detail actions for suggestions.

Wired into Telegram callback router for inline keyboard button presses.
Orchestrates: ApprovalTracker → SuggestionTracker → FileChangeGenerator → PRBuilder.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from schemas.autonomous_pipeline import FileChange, PRRequest
from skills.approval_tracker import ApprovalTracker
from skills.config_registry import ConfigRegistry
from skills.file_change_generator import FileChangeGenerator
from skills.github_pr import PRBuilder

logger = logging.getLogger(__name__)


class ApprovalHandler:
    """Handles approve/reject/detail actions for suggestion approval requests."""

    def __init__(
        self,
        approval_tracker: ApprovalTracker,
        suggestion_tracker: Any,  # SuggestionTracker
        file_change_generator: FileChangeGenerator,
        pr_builder: PRBuilder,
        config_registry: ConfigRegistry,
        event_stream: Any | None = None,
        telegram_bot: Any | None = None,
    ) -> None:
        self._approval_tracker = approval_tracker
        self._suggestion_tracker = suggestion_tracker
        self._file_change_gen = file_change_generator
        self._pr_builder = pr_builder
        self._config_registry = config_registry
        self._event_stream = event_stream
        self._telegram_bot = telegram_bot

    async def handle_approve(self, request_id: str) -> str:
        """Approve a pending request, generate file changes, and create a PR.

        If PR creation fails, reverts approval to PENDING.
        """
        request = self._approval_tracker.get_by_id(request_id)
        if request is None:
            return f"Request {request_id} not found"

        try:
            approved = self._approval_tracker.approve(request_id)
        except ValueError as e:
            return str(e)

        # Mark linked suggestion as IMPLEMENTED
        if self._suggestion_tracker:
            try:
                self._suggestion_tracker.implement(approved.suggestion_id)
            except Exception:
                logger.warning("Failed to mark suggestion %s as IMPLEMENTED", approved.suggestion_id)

        # Generate file changes
        file_changes: list[FileChange] = []
        profile = self._config_registry.get_profile(approved.bot_id)
        if profile is None:
            return f"No config profile for bot {approved.bot_id}"

        repo_dir = Path(profile.repo_dir)
        for pc in approved.param_changes:
            param = self._config_registry.get_parameter(
                approved.bot_id, pc.get("param_name", ""),
            )
            if param is None:
                continue
            try:
                change = self._file_change_gen.generate_change(
                    param, pc.get("proposed"), repo_dir,
                )
                file_changes.append(change)
            except Exception as e:
                logger.warning("Failed to generate change for %s: %s", pc.get("param_name"), e)

        # Build PR
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        branch_name = f"ta/suggestion-{approved.suggestion_id[:8]}-{date_str}"
        title = f"Update {', '.join(pc.get('param_name', '?') for pc in approved.param_changes)}"

        pr_body = self._pr_builder.format_pr_body(
            PRRequest(
                approval_request_id=request_id,
                suggestion_id=approved.suggestion_id,
                bot_id=approved.bot_id,
                repo_dir=str(repo_dir),
                branch_name=branch_name,
                title=title,
                file_changes=file_changes,
            ),
            approved.backtest_summary,
        )

        pr_request = PRRequest(
            approval_request_id=request_id,
            suggestion_id=approved.suggestion_id,
            bot_id=approved.bot_id,
            repo_dir=str(repo_dir),
            branch_name=branch_name,
            title=title,
            body=pr_body,
            file_changes=file_changes,
        )

        result = await self._pr_builder.create_pr(pr_request)

        # Dedup: existing PR already covers this suggestion
        if result.existing_pr_url:
            await self._edit_approval_card(
                request, f"\u2705 APPROVED — Existing PR: {result.existing_pr_url}",
            )
            self._approval_tracker.set_pr_url(request_id, result.existing_pr_url)
            return f"Existing PR: {result.existing_pr_url}"

        if not result.success:
            # Rollback approval on PR failure
            self._approval_tracker.revert_to_pending(request_id)
            preflight_detail = ""
            if result.preflight and not result.preflight.passed:
                preflight_detail = f" (preflight: {'; '.join(result.preflight.reasons)})"
            logger.error("PR creation failed for %s: %s%s — reverting to PENDING", request_id, result.error, preflight_detail)
            return f"PR creation failed: {result.error}{preflight_detail}. Request reverted to PENDING."

        # Record PR URL
        self._approval_tracker.set_pr_url(request_id, result.pr_url or "")

        # Broadcast event
        if self._event_stream:
            self._event_stream.broadcast("suggestion_pr_created", {
                "request_id": request_id,
                "suggestion_id": approved.suggestion_id,
                "pr_url": result.pr_url,
            })

        # Edit original Telegram card to show approval result
        await self._edit_approval_card(
            request, f"\u2705 APPROVED — PR: {result.pr_url or 'created'}",
        )

        return f"PR created: {result.pr_url}"

    async def handle_reject(self, request_id: str, reason: str = "") -> str:
        """Reject a pending request."""
        try:
            rejected = self._approval_tracker.reject(request_id, reason)
        except ValueError as e:
            return str(e)

        # Mark suggestion as rejected
        if self._suggestion_tracker:
            try:
                self._suggestion_tracker.reject(
                    rejected.suggestion_id,
                    reason=reason or "rejected via Telegram",
                )
            except Exception:
                logger.warning("Failed to reject suggestion %s", rejected.suggestion_id)

        # Edit original Telegram card to show rejection
        rejected_req = self._approval_tracker.get_by_id(request_id)
        if rejected_req:
            await self._edit_approval_card(
                rejected_req, f"\u274c REJECTED — {reason or 'rejected via Telegram'}",
            )

        return f"Rejected request {request_id}"

    async def handle_detail(self, request_id: str) -> str:
        """Return extended backtest details for the request."""
        request = self._approval_tracker.get_by_id(request_id)
        if request is None:
            return f"Request {request_id} not found"

        lines = [f"Details for request {request_id}:"]
        lines.append(f"Bot: {request.bot_id}")
        lines.append(f"Status: {request.status.value}")

        if request.param_changes:
            lines.append("\nParameter Changes:")
            for pc in request.param_changes:
                lines.append(f"  {pc.get('param_name', '?')}: {pc.get('current', '?')} -> {pc.get('proposed', '?')}")

        bs = request.backtest_summary
        if bs:
            lines.append(f"\nBacktest ({bs.context.trade_count} trades, {bs.context.data_days} days):")
            lines.append(f"  Sharpe: {bs.baseline.sharpe_ratio:.2f} -> {bs.proposed.sharpe_ratio:.2f} ({bs.sharpe_change_pct:+.1f}%)")
            lines.append(f"  MaxDD: {bs.baseline.max_drawdown_pct:.1f}% -> {bs.proposed.max_drawdown_pct:.1f}% ({bs.max_dd_change_pct:+.1f}%)")
            lines.append(f"  PF: {bs.baseline.profit_factor:.2f} -> {bs.proposed.profit_factor:.2f} ({bs.profit_factor_change_pct:+.1f}%)")
            lines.append(f"  WR: {bs.baseline.win_rate:.1%} -> {bs.proposed.win_rate:.1%} ({bs.win_rate_change_pct:+.1f}%)")
            safety = "PASS" if bs.passes_safety else "FAIL"
            lines.append(f"  Safety: {safety}")

        return "\n".join(lines)

    async def _edit_approval_card(self, request, status_line: str) -> None:
        """Edit the original Telegram approval card to show the outcome."""
        if self._telegram_bot is None or not request.message_id:
            return
        try:
            text = (
                f"Suggestion {request.request_id}\n"
                f"Bot: {request.bot_id}\n"
                f"{status_line}"
            )
            await self._telegram_bot.edit_message(request.message_id, text)
        except Exception:
            logger.warning("Failed to edit approval card for %s", request.request_id)
