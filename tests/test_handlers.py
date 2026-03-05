"""Tests for orchestrator/handlers.py — handler implementations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent_runner import AgentResult
from orchestrator.event_stream import EventStream
from orchestrator.handlers import Handlers
from orchestrator.orchestrator_brain import Action, ActionType
from schemas.notifications import NotificationPreferences, NotificationPriority


@pytest.fixture
def event_stream() -> EventStream:
    return EventStream()


@pytest.fixture
def mock_agent_runner():
    runner = AsyncMock()
    runner.invoke = AsyncMock(return_value=AgentResult(
        response="Analysis complete.",
        run_dir=Path("/tmp/test-run"),
        cost_usd=0.05,
        duration_ms=3000,
        session_id="test-session-123",
        success=True,
    ))
    return runner


@pytest.fixture
def mock_dispatcher():
    dispatcher = AsyncMock()
    dispatcher.dispatch = AsyncMock(return_value=[])
    return dispatcher


@pytest.fixture
def handlers(
    tmp_path: Path,
    mock_agent_runner,
    event_stream: EventStream,
    mock_dispatcher,
) -> Handlers:
    return Handlers(
        agent_runner=mock_agent_runner,
        event_stream=event_stream,
        dispatcher=mock_dispatcher,
        notification_prefs=NotificationPreferences(),
        curated_dir=tmp_path / "data" / "curated",
        memory_dir=tmp_path / "memory",
        runs_dir=tmp_path / "runs",
        source_root=tmp_path / "src",
        bots=["bot1", "bot2"],
        heartbeat_dir=tmp_path / "heartbeats",
        failure_log_path=tmp_path / "failure_log.jsonl",
    )


def _make_action(
    action_type: ActionType,
    bot_id: str = "bot1",
    details: dict | None = None,
) -> Action:
    return Action(
        type=action_type,
        event_id="evt-test-001",
        bot_id=bot_id,
        details=details,
    )


class TestDailyAnalysis:
    @pytest.mark.asyncio
    async def test_full_flow_with_passing_quality_gate(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """Quality gate PASS -> assemble -> invoke -> notify."""
        date = "2026-03-02"
        # Create curated data so quality gate passes
        for bot in ["bot1", "bot2"]:
            bot_dir = tmp_path / "data" / "curated" / date / bot
            bot_dir.mkdir(parents=True)
            for f in [
                "summary.json", "winners.json", "losers.json",
                "process_failures.json", "notable_missed.json",
                "regime_analysis.json", "filter_analysis.json",
                "root_cause_summary.json",
                "hourly_performance.json", "slippage_stats.json",
                "factor_attribution.json", "exit_efficiency.json",
            ]:
                (bot_dir / f).write_text("{}")
        # Portfolio risk card
        (tmp_path / "data" / "curated" / date / "portfolio_risk_card.json").write_text("{}")
        # Memory dir needs to exist for context builder
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)

        action = _make_action(ActionType.SPAWN_DAILY_ANALYSIS, details={"date": date})
        await handlers.handle_daily_analysis(action)

        mock_agent_runner.invoke.assert_called_once()
        call_kwargs = mock_agent_runner.invoke.call_args
        assert call_kwargs.kwargs.get("agent_type") or call_kwargs[1].get("agent_type", call_kwargs[0][0]) == "daily_analysis"
        mock_dispatcher.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_quality_gate_degraded_still_invokes_claude(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, event_stream: EventStream
    ):
        """Quality gate FAIL (degraded) -> still invoke Claude with available data."""
        action = _make_action(ActionType.SPAWN_DAILY_ANALYSIS, details={"date": "2026-03-02"})

        # No curated data -> quality gate will report degraded but can_proceed=True
        await handlers.handle_daily_analysis(action)

        # Agent SHOULD be invoked (graceful degradation)
        mock_agent_runner.invoke.assert_called_once()
        # Dispatcher called for the success notification
        assert mock_dispatcher.dispatch.call_count >= 1


class TestWeeklyAnalysis:
    @pytest.mark.asyncio
    async def test_full_flow(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """Metrics -> strategy -> assemble -> invoke."""
        week_start = "2026-02-23"
        week_end = "2026-03-01"
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_WEEKLY_SUMMARY,
            details={"week_start": week_start, "week_end": week_end},
        )
        await handlers.handle_weekly_analysis(action)

        # Agent should be invoked even with empty data (assembler handles gracefully)
        mock_agent_runner.invoke.assert_called_once()
        call_args = mock_agent_runner.invoke.call_args
        assert "weekly" in str(call_args)


class TestWFO:
    @pytest.mark.asyncio
    async def test_full_flow(
        self, handlers: Handlers, mock_agent_runner, tmp_path: Path
    ):
        """Runner -> report -> assemble -> invoke."""
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_WFO,
            bot_id="bot1",
            details={"bot_id": "bot1", "data_start": "2025-01-01", "data_end": "2026-03-01"},
        )
        await handlers.handle_wfo(action)

        # Agent should be invoked (WFO runner handles empty trades gracefully)
        mock_agent_runner.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_sends_critical_notification(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """WFO REJECT recommendation -> CRITICAL notification."""
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_WFO,
            bot_id="bot1",
            details={"bot_id": "bot1", "data_start": "2025-01-01", "data_end": "2026-03-01"},
        )
        await handlers.handle_wfo(action)

        # The WFO runner with empty trades will produce a REJECT recommendation
        # Verify notification was sent
        mock_dispatcher.dispatch.assert_called_once()
        payload = mock_dispatcher.dispatch.call_args[0][0]
        assert payload.priority == NotificationPriority.CRITICAL
        assert "REJECT" in payload.title


class TestTriage:
    @pytest.mark.asyncio
    async def test_known_fix_invokes_claude(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """KNOWN_FIX outcome -> context builder -> assembler -> invoke."""
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_TRIAGE,
            bot_id="bot1",
            details={
                "bot_id": "bot1",
                "error_type": "ImportError",
                "message": "No module named 'foo'",
                "stack_trace": "Traceback...\nImportError: No module named 'foo'",
                "source_file": "",
                "source_line": 0,
                "severity": "high",
            },
        )

        from schemas.bug_triage import (
            BugComplexity, BugSeverity, ErrorEvent, TriageOutcome, TriageResult,
        )

        mock_triage_result = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1",
                error_type="ImportError",
                message="No module named 'foo'",
                stack_trace="Traceback...",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
        )

        with (
            patch("skills.run_bug_triage.TriageRunner") as MockRunner,
            patch("skills.failure_log.FailureLog") as MockLog,
        ):
            MockRunner.return_value.triage.return_value = mock_triage_result
            MockLog.return_value.get_past_rejections.return_value = []
            await handlers.handle_triage(action)

        mock_agent_runner.invoke.assert_called_once()
        call_args = mock_agent_runner.invoke.call_args
        assert "triage" in str(call_args)

    @pytest.mark.asyncio
    async def test_needs_human_skips_claude_but_notifies(
        self, handlers: Handlers, mock_agent_runner, mock_dispatcher, tmp_path: Path
    ):
        """NEEDS_HUMAN -> no Claude invocation, but notify."""
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src").mkdir(parents=True, exist_ok=True)

        action = _make_action(
            ActionType.SPAWN_TRIAGE,
            bot_id="bot1",
            details={
                "bot_id": "bot1",
                "error_type": "RuntimeError",
                "message": "Complex state issue",
                "stack_trace": "Traceback...",
            },
        )

        from schemas.bug_triage import (
            BugComplexity, BugSeverity, ErrorEvent, TriageOutcome, TriageResult,
        )

        mock_result = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1",
                error_type="RuntimeError",
                message="Complex state issue",
                stack_trace="Traceback...",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.STATE_DEPENDENT,
            outcome=TriageOutcome.NEEDS_HUMAN,
        )

        with (
            patch("skills.run_bug_triage.TriageRunner") as MockRunner,
            patch("skills.failure_log.FailureLog") as MockLog,
        ):
            MockRunner.return_value.triage.return_value = mock_result
            MockLog.return_value.get_past_rejections.return_value = []
            await handlers.handle_triage(action)

        # Agent should NOT be invoked
        mock_agent_runner.invoke.assert_not_called()
        # But notification should be sent
        mock_dispatcher.dispatch.assert_called_once()
        payload = mock_dispatcher.dispatch.call_args[0][0]
        assert "needs_human" in payload.notification_type


class TestAlert:
    @pytest.mark.asyncio
    async def test_dispatches_critical(
        self, handlers: Handlers, mock_dispatcher
    ):
        """Alert -> immediate CRITICAL notification."""
        action = _make_action(
            ActionType.ALERT_IMMEDIATE,
            bot_id="bot1",
            details={"message": "Exchange API down", "severity": "CRITICAL"},
        )
        await handlers.handle_alert(action)

        mock_dispatcher.dispatch.assert_called_once()
        payload = mock_dispatcher.dispatch.call_args[0][0]
        assert payload.priority == NotificationPriority.CRITICAL
        assert "ALERT" in payload.title
        assert "bot1" in payload.title


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_writes_heartbeat_file(
        self, handlers: Handlers, tmp_path: Path
    ):
        """Heartbeat -> file write."""
        action = _make_action(ActionType.UPDATE_HEARTBEAT, bot_id="bot1")
        await handlers.handle_heartbeat(action)

        hb_path = tmp_path / "heartbeats" / "bot1.heartbeat"
        assert hb_path.exists()
        content = hb_path.read_text()
        # Should be a valid ISO timestamp
        datetime.fromisoformat(content)


class TestNotification:
    @pytest.mark.asyncio
    async def test_dispatches_payload(
        self, handlers: Handlers, mock_dispatcher
    ):
        """Notification action -> dispatcher."""
        action = _make_action(
            ActionType.SEND_NOTIFICATION,
            details={
                "notification_type": "custom_alert",
                "priority": "high",
                "title": "Test",
                "body": "Test body",
            },
        )
        await handlers.handle_notification(action)

        mock_dispatcher.dispatch.assert_called_once()
        payload = mock_dispatcher.dispatch.call_args[0][0]
        assert payload.notification_type == "custom_alert"
        assert payload.priority == NotificationPriority.HIGH


class TestErrorEmission:
    @pytest.mark.asyncio
    async def test_handler_error_emits_sse(
        self, handlers: Handlers, mock_agent_runner, event_stream: EventStream
    ):
        """Exception in handler -> SSE error event."""
        # Make agent runner raise
        mock_agent_runner.invoke.side_effect = RuntimeError("Unexpected error")

        # Create minimal curated data to pass quality gate
        curated_dir = handlers._curated_dir
        date = "2026-03-02"
        for bot in ["bot1", "bot2"]:
            bot_dir = curated_dir / date / bot
            bot_dir.mkdir(parents=True)
            for f in [
                "summary.json", "winners.json", "losers.json",
                "process_failures.json", "notable_missed.json",
                "regime_analysis.json", "filter_analysis.json",
                "root_cause_summary.json",
                "hourly_performance.json", "slippage_stats.json",
                "factor_attribution.json", "exit_efficiency.json",
            ]:
                (bot_dir / f).write_text("{}")
        (curated_dir / date / "portfolio_risk_card.json").write_text("{}")
        handlers._memory_dir.mkdir(parents=True, exist_ok=True)

        action = _make_action(ActionType.SPAWN_DAILY_ANALYSIS, details={"date": date})
        await handlers.handle_daily_analysis(action)

        # Verify SSE error event was broadcast
        recent = event_stream.get_recent()
        error_events = [e for e in recent if "error" in e.event_type]
        assert len(error_events) >= 1
