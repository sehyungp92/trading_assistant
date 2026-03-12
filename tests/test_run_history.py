"""Tests for run history persistence (C0)."""
from __future__ import annotations

import json

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from orchestrator.event_stream import EventStream
from orchestrator.handlers import Handlers
from orchestrator.orchestrator_brain import Action, ActionType
from schemas.notifications import NotificationPreferences


@pytest.fixture
def handlers_with_tmp(tmp_path):
    run_history_path = tmp_path / "data" / "run_history.jsonl"
    event_stream = EventStream()
    handlers = Handlers(
        agent_runner=AsyncMock(),
        event_stream=event_stream,
        dispatcher=AsyncMock(),
        notification_prefs=NotificationPreferences(),
        curated_dir=tmp_path / "data" / "curated",
        memory_dir=tmp_path / "memory",
        runs_dir=tmp_path / "runs",
        source_root=tmp_path,
        bots=["bot1"],
        heartbeat_dir=tmp_path / "heartbeats",
        run_history_path=run_history_path,
    )
    return handlers, run_history_path


def test_record_run_creates_entry(handlers_with_tmp):
    handlers, path = handlers_with_tmp
    handlers._record_run("daily-2026-03-04", "daily_analysis", "running",
                         started_at="2026-03-04T22:30:00+00:00")

    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["run_id"] == "daily-2026-03-04"
    assert entry["status"] == "running"
    assert entry["handler"] == "daily_analysis"


def test_record_run_on_success(handlers_with_tmp):
    handlers, path = handlers_with_tmp
    handlers._record_run("daily-2026-03-04", "daily_analysis", "completed",
                         started_at="2026-03-04T22:30:00+00:00",
                         finished_at="2026-03-04T22:31:00+00:00",
                         duration_ms=60000)

    entry = json.loads(path.read_text().strip())
    assert entry["status"] == "completed"
    assert entry["duration_ms"] == 60000


def test_record_run_on_failure(handlers_with_tmp):
    handlers, path = handlers_with_tmp
    handlers._record_run("daily-2026-03-04", "daily_analysis", "failed",
                         error="Timeout after 600s")

    entry = json.loads(path.read_text().strip())
    assert entry["status"] == "failed"
    assert "Timeout" in entry["error"]


def test_task_records_include_duration(handlers_with_tmp):
    handlers, path = handlers_with_tmp
    handlers._record_run("wfo-bot1-2026-03-04", "wfo", "completed",
                         duration_ms=45000)

    entry = json.loads(path.read_text().strip())
    assert entry["duration_ms"] == 45000


def test_multiple_handlers_record_independently(handlers_with_tmp):
    handlers, path = handlers_with_tmp
    handlers._record_run("daily-2026-03-04", "daily_analysis", "completed")
    handlers._record_run("weekly-2026-03-01", "weekly_analysis", "completed")

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["agent_type"] == "daily_analysis"
    assert json.loads(lines[1])["agent_type"] == "weekly_analysis"
