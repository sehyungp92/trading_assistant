"""Tests for startup catch-up logic."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from orchestrator.catchup import StartupCatchup
from schemas.bot_config import BotConfig


@pytest.fixture
def history_path(tmp_path: Path) -> Path:
    return tmp_path / "run_history.jsonl"


class TestStartupCatchup:
    def test_no_history_triggers_daily_catchup(self, history_path):
        catchup = StartupCatchup(run_history_path=history_path)
        dailies = catchup.needs_daily_catchup()
        assert len(dailies) == 1
        assert "date" in dailies[0]

    def test_recent_run_no_catchup(self, history_path):
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        history_path.write_text(json.dumps({
            "run_id": f"daily-{yesterday}",
            "handler": "daily_analysis",
            "status": "completed",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }) + "\n")

        catchup = StartupCatchup(run_history_path=history_path)
        assert catchup.needs_daily_catchup() == []

    def test_old_run_triggers_catchup(self, history_path):
        old_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        history_path.write_text(json.dumps({
            "run_id": f"daily-{old_date}",
            "handler": "daily_analysis",
            "status": "completed",
            "started_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
        }) + "\n")

        catchup = StartupCatchup(run_history_path=history_path)
        dailies = catchup.needs_daily_catchup()
        # 3-day-old run means 2 missed days (day -2 and day -1)
        assert len(dailies) == 2

    def test_multi_tz_catchup(self, history_path):
        configs = {
            "btc_bot": BotConfig(bot_id="btc_bot"),
            "k_stock": BotConfig(
                bot_id="k_stock", timezone="Asia/Seoul",
                market_close_local="15:30",
            ),
        }
        catchup = StartupCatchup(
            run_history_path=history_path,
            bot_configs=configs,
        )
        dailies = catchup.needs_daily_catchup()
        # Two timezone groups means potentially two catch-ups
        assert len(dailies) >= 1
        for d in dailies:
            assert "date" in d
            assert "bots" in d

    def test_weekly_no_history(self, history_path):
        catchup = StartupCatchup(run_history_path=history_path)
        assert catchup.needs_weekly_catchup() is True

    def test_weekly_recent_run(self, history_path):
        # Simulate a weekly run this week
        now = datetime.now(timezone.utc)
        days_since_sunday = (now.weekday() + 1) % 7
        this_sunday = (now - timedelta(days=days_since_sunday)).strftime("%Y-%m-%d")
        history_path.write_text(json.dumps({
            "run_id": f"weekly-{this_sunday}",
            "handler": "weekly_analysis",
            "status": "completed",
            "started_at": now.isoformat(),
        }) + "\n")

        catchup = StartupCatchup(run_history_path=history_path)
        assert catchup.needs_weekly_catchup() is False

    def test_build_catchup_events(self, history_path):
        catchup = StartupCatchup(run_history_path=history_path)
        events = catchup.build_catchup_events()
        # Should include at least daily + weekly
        types = [e["event_type"] for e in events]
        assert "daily_analysis_trigger" in types
        assert "weekly_summary_trigger" in types

    def test_multi_day_gap_produces_multiple_catchups(self, history_path):
        """3-day gap should produce 3 catchup entries."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        history_path.write_text(json.dumps({
            "run_id": f"daily-{old_date}",
            "handler": "daily_analysis",
            "status": "completed",
            "started_at": (datetime.now(timezone.utc) - timedelta(days=4)).isoformat(),
        }) + "\n")

        catchup = StartupCatchup(run_history_path=history_path)
        dailies = catchup.needs_daily_catchup()
        # Should have 3 entries (days -3, -2, -1)
        assert len(dailies) == 3
        dates = [d["date"] for d in dailies]
        # Verify chronological order
        assert dates == sorted(dates)

    def test_multi_day_gap_capped_at_max(self, history_path):
        """Gaps larger than max_catchup_days are capped."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=20)).strftime("%Y-%m-%d")
        history_path.write_text(json.dumps({
            "run_id": f"daily-{old_date}",
            "handler": "daily_analysis",
            "status": "completed",
            "started_at": (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(),
        }) + "\n")

        catchup = StartupCatchup(run_history_path=history_path, max_catchup_days=5)
        dailies = catchup.needs_daily_catchup()
        assert len(dailies) == 5

    def test_multi_day_gap_with_bot_configs(self, history_path):
        """Multi-day gap with timezone-grouped bots."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        history_path.write_text(json.dumps({
            "run_id": f"daily-{old_date}",
            "handler": "daily_analysis",
            "status": "completed",
            "started_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
        }) + "\n")

        configs = {
            "btc_bot": BotConfig(bot_id="btc_bot"),
        }
        catchup = StartupCatchup(
            run_history_path=history_path,
            bot_configs=configs,
        )
        dailies = catchup.needs_daily_catchup()
        # Should have 2 entries (days -2 and -1)
        assert len(dailies) == 2
        for d in dailies:
            assert "bots" in d
            assert "date" in d

    def test_build_catchup_events_with_bots(self, history_path):
        configs = {
            "btc_bot": BotConfig(bot_id="btc_bot"),
        }
        catchup = StartupCatchup(
            run_history_path=history_path,
            bot_configs=configs,
        )
        events = catchup.build_catchup_events()
        daily_events = [e for e in events if e["event_type"] == "daily_analysis_trigger"]
        assert len(daily_events) >= 1
        payload = json.loads(daily_events[0]["payload"])
        assert "date" in payload
