"""Tests for the persisted scheduled run store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.scheduled_runs import ScheduledRunStore


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
async def test_baseline_round_trip(tmp_path: Path):
    store = ScheduledRunStore(str(tmp_path / "scheduled_runs.db"))
    await store.initialize()
    try:
        baseline = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        await store.set_baseline(baseline)
        assert await store.get_baseline() == baseline
    finally:
        await store.close()
