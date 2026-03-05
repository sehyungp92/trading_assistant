# tests/test_weekly_prompt_assembler.py
"""Tests for the weekly prompt assembler."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from analysis.weekly_prompt_assembler import WeeklyPromptAssembler


@pytest.fixture
def setup_dirs(tmp_path: Path):
    """Create directory structure with weekly curated data and daily reports."""
    curated = tmp_path / "curated"
    memory = tmp_path / "memory"
    runs = tmp_path / "runs"

    # Weekly curated data
    weekly_dir = curated / "weekly" / "2026-02-23"
    weekly_dir.mkdir(parents=True)
    (weekly_dir / "weekly_summary.json").write_text(
        json.dumps({"week_start": "2026-02-23", "total_net_pnl": 500.0})
    )
    (weekly_dir / "refinement_report.json").write_text(
        json.dumps({"suggestions": [{"title": "Adjust RSI"}]})
    )
    (weekly_dir / "week_over_week.json").write_text(
        json.dumps({"pnl_delta": 100.0})
    )

    # 7 daily reports (just create stubs)
    week_start_dt = datetime.strptime("2026-02-23", "%Y-%m-%d")
    week_dates = [(week_start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    for date in week_dates:
        run_dir = runs / date / "daily-report"
        run_dir.mkdir(parents=True)
        (run_dir / "daily_report.md").write_text(f"# Daily Report {date}\nAll good.")

    # Portfolio risk cards (7 days)
    for date in week_dates:
        date_dir = curated / date
        date_dir.mkdir(parents=True)
        (date_dir / "portfolio_risk_card.json").write_text(
            json.dumps({"date": date, "concentration_score": 30.0})
        )

    # Memory policies
    policy_dir = memory / "policies" / "v1"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agents.md").write_text("You are the trading assistant.")
    (policy_dir / "trading_rules.md").write_text("Max drawdown 15%.")
    (policy_dir / "soul.md").write_text("Conservative approach.")

    # Corrections
    findings_dir = memory / "findings"
    findings_dir.mkdir(parents=True)
    (findings_dir / "corrections.jsonl").write_text(
        json.dumps({"correction_type": "positive_reinforcement", "raw_text": "good catch"})
        + "\n"
    )

    return curated, memory, runs


class TestWeeklyPromptAssembler:
    def test_assembles_complete_package(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1", "bot2"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()

        assert package.system_prompt
        assert package.task_prompt
        assert package.data
        assert package.instructions
        assert package.corrections is not None
        assert package.context_files is not None

    def test_system_prompt_includes_policies(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "trading assistant" in package.system_prompt
        assert "Max drawdown" in package.system_prompt

    def test_data_includes_weekly_summary(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "weekly_summary" in package.data
        assert package.data["weekly_summary"]["total_net_pnl"] == 500.0

    def test_data_includes_daily_reports(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "daily_reports" in package.data
        assert len(package.data["daily_reports"]) == 7

    def test_data_includes_risk_cards(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "portfolio_risk_cards" in package.data
        assert len(package.data["portfolio_risk_cards"]) == 7

    def test_task_prompt_mentions_weekly(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert "weekly" in package.task_prompt.lower()

    def test_corrections_loaded(self, setup_dirs):
        curated, memory, runs = setup_dirs
        assembler = WeeklyPromptAssembler(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        )
        package = assembler.assemble()
        assert len(package.corrections) == 1
