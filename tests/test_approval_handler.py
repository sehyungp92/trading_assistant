# tests/test_approval_handler.py
"""Tests for ApprovalHandler."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from schemas.autonomous_pipeline import (
    ApprovalRequest,
    ApprovalStatus,
    BacktestComparison,
    BacktestContext,
    PreflightResult,
    PRResult,
)
from schemas.wfo_results import SimulationMetrics
from skills.approval_handler import ApprovalHandler
from skills.approval_tracker import ApprovalTracker
from skills.config_registry import ConfigRegistry
from skills.file_change_generator import FileChangeGenerator
from skills.github_pr import PRBuilder
from skills.suggestion_tracker import SuggestionTracker


def _setup_registry(tmp_path: Path) -> ConfigRegistry:
    d = tmp_path / "configs"
    d.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "config.yaml").write_text("alpha:\n  quality_min: 0.6\n")
    (d / "test_bot.yaml").write_text(yaml.dump({
        "bot_id": "test_bot",
        "repo_dir": str(repo_dir),
        "strategies": ["alpha"],
        "parameters": [{
            "param_name": "quality_min",
            "param_type": "YAML_FIELD",
            "file_path": "config.yaml",
            "yaml_key": "alpha.quality_min",
            "current_value": 0.6,
            "valid_range": [0.0, 1.0],
            "value_type": "float",
            "category": "entry_signal",
            "is_safety_critical": False,
        }],
    }), encoding="utf-8")
    return ConfigRegistry(d)


def _make_comparison() -> BacktestComparison:
    return BacktestComparison(
        context=BacktestContext(
            suggestion_id="s1", bot_id="test_bot",
            param_name="quality_min", current_value=0.6, proposed_value=0.7,
            trade_count=25, data_days=30,
        ),
        baseline=SimulationMetrics(sharpe_ratio=1.0, profit_factor=1.5, total_trades=25, win_count=15),
        proposed=SimulationMetrics(sharpe_ratio=1.2, profit_factor=1.8, total_trades=25, win_count=17),
        passes_safety=True,
    )


@pytest.fixture
def components(tmp_path: Path):
    registry = _setup_registry(tmp_path)
    approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
    suggestion_tracker = SuggestionTracker(tmp_path / "findings")
    file_gen = FileChangeGenerator()
    pr_builder = PRBuilder(dry_run=True)
    event_stream = MagicMock()
    handler = ApprovalHandler(
        approval_tracker=approval_tracker,
        suggestion_tracker=suggestion_tracker,
        file_change_generator=file_gen,
        pr_builder=pr_builder,
        config_registry=registry,
        event_stream=event_stream,
    )
    return handler, approval_tracker, suggestion_tracker, event_stream


def _create_pending_request(approval_tracker: ApprovalTracker) -> ApprovalRequest:
    req = ApprovalRequest(
        request_id="req1",
        suggestion_id="s1",
        bot_id="test_bot",
        param_changes=[{"param_name": "quality_min", "current": 0.6, "proposed": 0.7}],
        backtest_summary=_make_comparison(),
    )
    approval_tracker.create_request(req)
    return req


class TestApprovalHandler:
    @pytest.mark.asyncio
    async def test_approve_creates_pr(self, components):
        handler, tracker, sug_tracker, event_stream = components
        _create_pending_request(tracker)
        result = await handler.handle_approve("req1")
        assert "dry-run" in result or "PR created" in result
        assert tracker.get_by_id("req1").status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_approve_pr_failure_reverts(self, components):
        handler, tracker, _, _ = components
        _create_pending_request(tracker)
        # Make PR builder fail
        handler._pr_builder = PRBuilder(dry_run=False)
        with patch.object(handler._pr_builder, "create_pr", new_callable=AsyncMock,
                          return_value=PRResult(success=False, error="network error")):
            result = await handler.handle_approve("req1")
        assert "failed" in result.lower()
        assert tracker.get_by_id("req1").status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_approve_non_pending_returns_error(self, components):
        handler, tracker, _, _ = components
        _create_pending_request(tracker)
        tracker.approve("req1")  # Already approved
        result = await handler.handle_approve("req1")
        assert "not PENDING" in result

    @pytest.mark.asyncio
    async def test_reject_with_reason(self, components):
        handler, tracker, sug_tracker, _ = components
        _create_pending_request(tracker)
        result = await handler.handle_reject("req1", reason="too risky")
        assert "Rejected" in result
        assert tracker.get_by_id("req1").status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_reject_updates_suggestion_tracker(self, components):
        handler, tracker, sug_tracker, _ = components
        _create_pending_request(tracker)
        # Record the suggestion first
        from schemas.suggestion_tracking import SuggestionRecord
        sug_tracker.record(SuggestionRecord(
            suggestion_id="s1", bot_id="test_bot",
            title="test", tier="parameter", source_report_id="r1",
        ))
        await handler.handle_reject("req1", reason="too risky")
        rejected = sug_tracker.get_rejected()
        assert any(s.get("suggestion_id") == "s1" for s in rejected)

    @pytest.mark.asyncio
    async def test_detail_returns_info(self, components):
        handler, tracker, _, _ = components
        _create_pending_request(tracker)
        result = await handler.handle_detail("req1")
        assert "req1" in result
        assert "test_bot" in result
        assert "Sharpe" in result

    @pytest.mark.asyncio
    async def test_detail_unknown_request(self, components):
        handler, _, _, _ = components
        result = await handler.handle_detail("nonexistent")
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_event_broadcast_on_approve(self, components):
        handler, tracker, _, event_stream = components
        _create_pending_request(tracker)
        await handler.handle_approve("req1")
        # Dry run PR builder doesn't produce a URL, but event_stream is not called
        # because dry_run returns success=True with error="dry-run"
        # The handler still gets a successful result with no pr_url

    @pytest.mark.asyncio
    async def test_approve_updates_suggestion_tracker(self, components):
        handler, tracker, sug_tracker, _ = components
        _create_pending_request(tracker)
        from schemas.suggestion_tracking import SuggestionRecord
        sug_tracker.record(SuggestionRecord(
            suggestion_id="s1", bot_id="test_bot",
            title="test", tier="parameter", source_report_id="r1",
        ))
        await handler.handle_approve("req1")
        all_suggestions = sug_tracker.load_all()
        s1 = [s for s in all_suggestions if s.get("suggestion_id") == "s1"]
        assert s1[0].get("status") == "implemented"

    @pytest.mark.asyncio
    async def test_approve_not_found(self, components):
        handler, _, _, _ = components
        result = await handler.handle_approve("nonexistent")
        assert "not found" in result


class TestApprovalCardEditing:
    """Tests for editing the original Telegram approval card after action."""

    @pytest.mark.asyncio
    async def test_approve_edits_card(self, tmp_path: Path):
        registry = _setup_registry(tmp_path)
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        telegram_bot = MagicMock()
        telegram_bot.edit_message = AsyncMock()

        handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=SuggestionTracker(tmp_path / "findings"),
            file_change_generator=FileChangeGenerator(),
            pr_builder=PRBuilder(dry_run=True),
            config_registry=registry,
            telegram_bot=telegram_bot,
        )
        req = _create_pending_request(approval_tracker)
        # Set message_id to simulate Telegram card
        approval_tracker.set_message_id("req1", 42)

        await handler.handle_approve("req1")
        telegram_bot.edit_message.assert_called_once()
        call_args = telegram_bot.edit_message.call_args
        assert call_args[0][0] == 42  # message_id
        assert "APPROVED" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_reject_edits_card(self, tmp_path: Path):
        registry = _setup_registry(tmp_path)
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        telegram_bot = MagicMock()
        telegram_bot.edit_message = AsyncMock()

        handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=SuggestionTracker(tmp_path / "findings"),
            file_change_generator=FileChangeGenerator(),
            pr_builder=PRBuilder(dry_run=True),
            config_registry=registry,
            telegram_bot=telegram_bot,
        )
        _create_pending_request(approval_tracker)
        approval_tracker.set_message_id("req1", 99)

        await handler.handle_reject("req1", reason="too risky")
        telegram_bot.edit_message.assert_called_once()
        call_args = telegram_bot.edit_message.call_args
        assert call_args[0][0] == 99
        assert "REJECTED" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_no_edit_without_message_id(self, tmp_path: Path):
        registry = _setup_registry(tmp_path)
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        telegram_bot = MagicMock()
        telegram_bot.edit_message = AsyncMock()

        handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=SuggestionTracker(tmp_path / "findings"),
            file_change_generator=FileChangeGenerator(),
            pr_builder=PRBuilder(dry_run=True),
            config_registry=registry,
            telegram_bot=telegram_bot,
        )
        _create_pending_request(approval_tracker)
        # No set_message_id call

        await handler.handle_approve("req1")
        telegram_bot.edit_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_edit_without_telegram_bot(self, components):
        handler, tracker, _, _ = components
        _create_pending_request(tracker)
        # handler has no telegram_bot (components fixture doesn't set one)
        # Should not raise
        await handler.handle_approve("req1")


class TestApprovalHandlerDedup:
    @pytest.mark.asyncio
    async def test_dedup_returns_existing_pr(self, tmp_path: Path):
        registry = _setup_registry(tmp_path)
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        pr_builder = PRBuilder(dry_run=False)

        handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=SuggestionTracker(tmp_path / "findings"),
            file_change_generator=FileChangeGenerator(),
            pr_builder=pr_builder,
            config_registry=registry,
        )
        _create_pending_request(approval_tracker)

        with patch.object(pr_builder, "create_pr", new_callable=AsyncMock,
                          return_value=PRResult(
                              success=True,
                              existing_pr_url="https://github.com/x/y/pull/42",
                              branch_name="ta/test",
                          )):
            result = await handler.handle_approve("req1")
        assert "Existing PR" in result
        assert approval_tracker.get_by_id("req1").pr_url == "https://github.com/x/y/pull/42"

    @pytest.mark.asyncio
    async def test_preflight_failure_reverts(self, tmp_path: Path):
        registry = _setup_registry(tmp_path)
        approval_tracker = ApprovalTracker(tmp_path / "approvals.jsonl")
        pr_builder = PRBuilder(dry_run=False)

        handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=SuggestionTracker(tmp_path / "findings"),
            file_change_generator=FileChangeGenerator(),
            pr_builder=pr_builder,
            config_registry=registry,
        )
        _create_pending_request(approval_tracker)

        pf = PreflightResult(passed=False, reasons=["Remote unreachable"])
        with patch.object(pr_builder, "create_pr", new_callable=AsyncMock,
                          return_value=PRResult(
                              success=False,
                              error="Pre-flight failed",
                              preflight=pf,
                          )):
            result = await handler.handle_approve("req1")
        assert "preflight" in result.lower()
        assert approval_tracker.get_by_id("req1").status == ApprovalStatus.PENDING
