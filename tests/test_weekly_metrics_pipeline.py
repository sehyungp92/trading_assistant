# tests/test_weekly_metrics_pipeline.py
"""Tests for the weekly metrics aggregation pipeline."""
import json
from pathlib import Path

import pytest

from schemas.daily_metrics import BotDailySummary, RegimeAnalysis, FilterAnalysis, RootCauseSummary
from schemas.weekly_metrics import (
    BotWeeklySummary,
    WeeklySummary,
    WeekOverWeekComparison,
    ProcessQualityTrend,
    FilterWeeklySummary,
)
from skills.build_weekly_metrics import WeeklyMetricsBuilder


def _make_daily_summary(
    date: str, bot_id: str, net_pnl: float, total_trades: int = 10, **kwargs
) -> BotDailySummary:
    """Helper to create a BotDailySummary with sensible defaults."""
    wins = total_trades // 2 + (1 if net_pnl > 0 else 0)
    losses = total_trades - wins
    return BotDailySummary(
        date=date,
        bot_id=bot_id,
        total_trades=total_trades,
        win_count=wins,
        loss_count=losses,
        gross_pnl=net_pnl + 10.0,
        net_pnl=net_pnl,
        avg_win=abs(net_pnl) / max(wins, 1),
        avg_loss=-(abs(net_pnl) * 0.5) / max(losses, 1),
        avg_process_quality=kwargs.get("avg_process_quality", 75.0),
        missed_count=kwargs.get("missed_count", 2),
        missed_would_have_won=kwargs.get("missed_would_have_won", 1),
        error_count=kwargs.get("error_count", 0),
        uptime_pct=kwargs.get("uptime_pct", 100.0),
    )


_DATES = [
    "2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26",
    "2026-02-27", "2026-02-28", "2026-03-01",
]


class TestWeeklyMetricsBuilder:
    def test_build_bot_weekly_summary(self):
        dailies = [_make_daily_summary(d, "bot1", 50.0) for d in _DATES]
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        result = builder.build_bot_summary("bot1", dailies)
        assert result.bot_id == "bot1"
        assert result.total_trades == 70  # 10 * 7
        assert result.net_pnl == 350.0  # 50 * 7
        assert len(result.daily_pnl) == 7

    def test_build_bot_summary_empty(self):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        result = builder.build_bot_summary("bot1", [])
        assert result.total_trades == 0
        assert result.net_pnl == 0.0

    def test_build_portfolio_summary(self):
        bot1_dailies = [_make_daily_summary(d, "bot1", 50.0) for d in _DATES]
        bot2_dailies = [_make_daily_summary(d, "bot2", -20.0) for d in _DATES]
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1", "bot2"],
        )
        result = builder.build_portfolio_summary(
            {"bot1": bot1_dailies, "bot2": bot2_dailies}
        )
        assert result.total_net_pnl == 210.0  # 350 - 140
        assert len(result.bot_summaries) == 2

    def test_week_over_week_comparison(self):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        current = WeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            total_net_pnl=500.0,
            total_trades=60,
        )
        previous = WeeklySummary(
            week_start="2026-02-16",
            week_end="2026-02-22",
            total_net_pnl=400.0,
            total_trades=55,
        )
        wow = builder.compare_weeks(current, previous)
        assert wow.pnl_delta == 100.0
        assert wow.pnl_delta_pct == 25.0
        assert wow.trade_count_delta == 5

    def test_week_over_week_previous_zero_pnl(self):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        current = WeeklySummary(
            week_start="2026-02-23", week_end="2026-03-01", total_net_pnl=500.0
        )
        previous = WeeklySummary(
            week_start="2026-02-16", week_end="2026-02-22", total_net_pnl=0.0
        )
        wow = builder.compare_weeks(current, previous)
        assert wow.pnl_delta == 500.0
        assert wow.pnl_delta_pct == 0.0  # cannot compute % from zero

    def test_process_quality_trend(self):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        weekly_scores = [68.0, 72.0, 75.0, 80.0]
        root_causes = {"regime_mismatch": 12, "normal_loss": 25, "weak_signal": 5}
        trend = builder.compute_process_quality_trend(
            "bot1", weekly_scores, root_causes
        )
        assert trend.trend_direction == "improving"
        assert trend.current_avg == 80.0

    def test_process_quality_trend_degrading(self):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        weekly_scores = [80.0, 75.0, 72.0, 68.0]
        trend = builder.compute_process_quality_trend("bot1", weekly_scores, {})
        assert trend.trend_direction == "degrading"

    def test_process_quality_trend_stable(self):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        weekly_scores = [75.0, 76.0, 75.5, 75.0]
        trend = builder.compute_process_quality_trend("bot1", weekly_scores, {})
        assert trend.trend_direction == "stable"

    def test_filter_weekly_summary(self):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        daily_filters = [
            FilterAnalysis(
                bot_id="bot1",
                date=d,
                filter_block_counts={"volatility_filter": 3},
                filter_saved_pnl={"volatility_filter": 50.0},
                filter_missed_pnl={"volatility_filter": 80.0},
            )
            for d in _DATES
        ]
        result = builder.build_filter_weekly_summary("bot1", daily_filters)
        vol = next(f for f in result if f.filter_name == "volatility_filter")
        assert vol.total_blocks == 21  # 3 * 7

    def test_write_weekly_curated(self, tmp_path: Path):
        builder = WeeklyMetricsBuilder(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bots=["bot1"],
        )
        bot1_dailies = [_make_daily_summary(d, "bot1", 50.0) for d in _DATES]
        summary = builder.build_portfolio_summary({"bot1": bot1_dailies})
        output_dir = builder.write_weekly_curated(summary, tmp_path)

        assert (output_dir / "weekly_summary.json").exists()
        data = json.loads((output_dir / "weekly_summary.json").read_text())
        assert data["total_net_pnl"] == 350.0
