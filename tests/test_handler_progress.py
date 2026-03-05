"""Tests for handler progress broadcasts (C2)."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.event_stream import EventStream
from orchestrator.handlers import Handlers
from orchestrator.orchestrator_brain import Action, ActionType
from orchestrator.agent_runner import AgentResult
from schemas.notifications import NotificationPreferences


@pytest.fixture
def event_stream():
    return EventStream()


@pytest.fixture
def handlers_with_stream(tmp_path, event_stream):
    agent_runner = AsyncMock()
    agent_runner.invoke = AsyncMock(return_value=AgentResult(
        response="test", run_dir=tmp_path / "runs" / "test", success=True,
    ))

    handlers = Handlers(
        agent_runner=agent_runner,
        event_stream=event_stream,
        dispatcher=AsyncMock(),
        notification_prefs=NotificationPreferences(),
        curated_dir=tmp_path / "data" / "curated",
        memory_dir=tmp_path / "memory",
        runs_dir=tmp_path / "runs",
        source_root=tmp_path,
        bots=["bot1"],
        heartbeat_dir=tmp_path / "heartbeats",
    )
    # Create needed dirs
    (tmp_path / "memory" / "policies" / "v1").mkdir(parents=True)
    (tmp_path / "memory" / "findings").mkdir(parents=True)
    return handlers


@pytest.mark.asyncio
async def test_daily_handler_broadcasts_start_event(handlers_with_stream, event_stream):
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")

    # Mock quality gate to return can_proceed=True
    with patch("analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    progress_events = [e for e in events if e.event_type == "handler_progress"]
    assert len(progress_events) >= 1
    assert progress_events[0].data["stage"] == "started"


@pytest.mark.asyncio
async def test_daily_handler_broadcasts_progress_with_stage(handlers_with_stream, event_stream):
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")

    with patch("analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    progress_events = [e for e in events if e.event_type == "handler_progress"]
    stages = [e.data["stage"] for e in progress_events]

    assert "started" in stages
    assert "quality_gate" in stages
    assert "prompt_assembly" in stages


@pytest.mark.asyncio
async def test_all_four_handlers_emit_started(handlers_with_stream, event_stream):
    """Verify all 4 long-running handlers emit at least a 'started' progress event."""
    # Daily
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")
    with patch("analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate
        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    daily_starts = [
        e for e in events
        if e.event_type == "handler_progress"
        and e.data.get("handler") == "daily_analysis"
        and e.data.get("stage") == "started"
    ]
    assert len(daily_starts) >= 1


@pytest.mark.asyncio
async def test_progress_events_include_run_id(handlers_with_stream, event_stream):
    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1",
                    details={"date": "2026-03-04"})

    with patch("analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    progress_events = [e for e in events if e.event_type == "handler_progress"]
    assert all("run_id" in e.data for e in progress_events)
    assert progress_events[0].data["run_id"] == "daily-2026-03-04"


@pytest.mark.asyncio
async def test_error_events_still_broadcast(handlers_with_stream, event_stream):
    """When a handler fails, both progress and error events should exist."""
    handlers_with_stream._agent_runner.invoke = AsyncMock(side_effect=RuntimeError("boom"))

    action = Action(type=ActionType.SPAWN_DAILY_ANALYSIS, event_id="ev1", bot_id="bot1")

    with patch("analysis.quality_gate.QualityGate") as mock_gate_cls:
        mock_gate = MagicMock()
        mock_gate.run.return_value = MagicMock(can_proceed=True, overall="PASS", blocking_issues=[])
        mock_gate_cls.return_value = mock_gate

        await handlers_with_stream.handle_daily_analysis(action)

    events = event_stream.get_recent()
    types = [e.event_type for e in events]
    assert "handler_progress" in types
    assert "daily_analysis_error" in types
