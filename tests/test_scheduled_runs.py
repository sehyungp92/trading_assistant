"""Tests for the persisted scheduled run store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.scheduled_runs import ScheduledRunStore
from orchestrator.handlers import Handlers
from orchestrator.orchestrator_brain import Action, ActionType


@pytest.mark.asyncio
async def test_mark_started_and_completed_round_trip(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 3, 8, 8, 0, tzinfo=timezone.utc)
        await store.mark_started("weekly_summary", "global", scheduled_for)
        await store.mark_completed("weekly_summary", "global", scheduled_for)

        records = await store.get_records("weekly_summary", "global")
        assert len(records) == 1
        assert records[0].status == "completed"
        assert records[0].scheduled_for == scheduled_for
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mark_enqueued_is_not_completed(tmp_path: Path):
    """P1-3: trigger-only cron specs mark_enqueued — handler must signal mark_completed."""
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 6, 0, tzinfo=timezone.utc)
        await store.mark_started("daily_analysis", "global", scheduled_for)
        await store.mark_enqueued("daily_analysis", "global", scheduled_for)

        # Until the handler signals completion, is_completed must return False
        # so catch-up will re-fire the trigger.
        assert await store.is_completed("daily_analysis", "global", scheduled_for) is False
        assert await store.get_status("daily_analysis", "global", scheduled_for) == "enqueued"

        # Handler signals completion → status flips to completed.
        await store.mark_completed("daily_analysis", "global", scheduled_for)
        assert await store.is_completed("daily_analysis", "global", scheduled_for) is True
        assert await store.get_status("daily_analysis", "global", scheduled_for) == "completed"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_handler_scheduled_failure_marks_failed_not_completed(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 6, 0, tzinfo=timezone.utc)
        await store.mark_enqueued("daily_analysis", "global", scheduled_for)

        handler = Handlers.__new__(Handlers)
        handler._scheduled_run_store = store
        action = Action(
            type=ActionType.SPAWN_DAILY_ANALYSIS,
            event_id="daily-1",
            bot_id="scheduler",
            details={
                "__scheduled_run__": {
                    "job_key": "daily_analysis",
                    "scope_key": "global",
                    "scheduled_for": scheduled_for.isoformat(),
                },
            },
        )

        await handler._signal_scheduled_result(
            action,
            success=False,
            error="agent failed",
        )

        assert await store.get_status("daily_analysis", "global", scheduled_for) == "failed"
        assert await store.is_completed("daily_analysis", "global", scheduled_for) is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_handler_scheduled_success_marks_completed(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        scheduled_for = datetime(2026, 5, 10, 8, 0, tzinfo=timezone.utc)
        await store.mark_enqueued("weekly_summary", "global", scheduled_for)

        handler = Handlers.__new__(Handlers)
        handler._scheduled_run_store = store
        action = Action(
            type=ActionType.SPAWN_WEEKLY_SUMMARY,
            event_id="weekly-1",
            bot_id="scheduler",
            details={
                "__scheduled_run__": {
                    "job_key": "weekly_summary",
                    "scope_key": "global",
                    "scheduled_for": scheduled_for.isoformat(),
                },
            },
        )

        await handler._signal_scheduled_result(action, success=True)

        assert await store.get_status("weekly_summary", "global", scheduled_for) == "completed"
        assert await store.is_completed("weekly_summary", "global", scheduled_for) is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_baseline_round_trip(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        baseline = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        await store.set_baseline(baseline)
        assert await store.get_baseline() == baseline
    finally:
        await store.close()
