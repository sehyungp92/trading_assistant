"""Tests for the daily metrics data reduction pipeline."""
import json
from pathlib import Path

import pytest

from schemas.events import TradeEvent, MissedOpportunityEvent
from schemas.daily_metrics import (
    BotDailySummary,
    WinnerLoserRecord,
    ProcessFailureRecord,
    NotableMissedRecord,
    RegimeAnalysis,
    FilterAnalysis,
    AnomalyRecord,
    RootCauseSummary,
)
from skills.build_daily_metrics import DailyMetricsBuilder


def _make_trade(trade_id: str, bot_id: str, pnl: float, **kwargs) -> TradeEvent:
    """Helper to create a TradeEvent with sensible defaults."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    defaults = dict(
        trade_id=trade_id,
        bot_id=bot_id,
        pair="BTCUSDT",
        side="LONG",
        entry_time=now,
        exit_time=now,
        entry_price=50000.0,
        exit_price=50000.0 + pnl,
        position_size=1.0,
        pnl=pnl,
        pnl_pct=pnl / 50000.0 * 100,
        entry_signal="EMA cross",
        exit_reason="TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
        market_regime="trending_up",
        process_quality_score=85,
        root_causes=["normal_win"] if pnl > 0 else ["normal_loss"],
    )
    defaults.update(kwargs)
    return TradeEvent(**defaults)


def _make_missed(bot_id: str, pair: str, blocked_by: str, outcome_24h: float) -> MissedOpportunityEvent:
    return MissedOpportunityEvent(
        bot_id=bot_id,
        pair=pair,
        signal="RSI divergence",
        blocked_by=blocked_by,
        hypothetical_entry=50000.0,
        outcome_24h=outcome_24h,
        confidence=0.7,
        assumption_tags=["mid_fill"],
    )


class TestDailyMetricsBuilder:
    def test_build_summary_basic(self):
        trades = [
            _make_trade("t1", "bot1", 100.0),
            _make_trade("t2", "bot1", -50.0),
            _make_trade("t3", "bot1", 200.0),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        summary = builder.build_summary(trades)

        assert summary.total_trades == 3
        assert summary.win_count == 2
        assert summary.loss_count == 1
        assert summary.gross_pnl == 250.0

    def test_top_winners(self):
        trades = [_make_trade(f"t{i}", "bot1", (i + 1) * 10.0) for i in range(10)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        winners = builder.top_winners(trades, n=5)

        assert len(winners) == 5
        assert winners[0].pnl >= winners[1].pnl  # sorted descending

    def test_top_losers(self):
        trades = [_make_trade(f"t{i}", "bot1", -(i + 1) * 10.0) for i in range(10)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        losers = builder.top_losers(trades, n=5)

        assert len(losers) == 5
        assert losers[0].pnl <= losers[1].pnl  # sorted ascending (most negative first)

    def test_process_failures(self):
        trades = [
            _make_trade("t1", "bot1", -80.0, process_quality_score=45, root_causes=["regime_mismatch"]),
            _make_trade("t2", "bot1", 100.0, process_quality_score=90),
            _make_trade("t3", "bot1", -20.0, process_quality_score=55, root_causes=["weak_signal"]),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        failures = builder.process_failures(trades, threshold=60)

        assert len(failures) == 2
        assert all(f.process_quality_score < 60 for f in failures)

    def test_notable_missed(self):
        missed = [
            _make_missed("bot1", "BTCUSDT", "vol_filter", 500.0),
            _make_missed("bot1", "ETHUSDT", "spread_filter", 50.0),
            _make_missed("bot1", "SOLUSDT", "vol_filter", 800.0),
        ]
        trades = [_make_trade("t1", "bot1", 100.0)]  # avg_win = 100
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        notable = builder.notable_missed(missed, trades)

        # Only those with outcome > 2x avg_win (200)
        assert len(notable) == 2
        assert all(n.outcome_24h > 200 for n in notable)

    def test_regime_analysis(self):
        trades = [
            _make_trade("t1", "bot1", 100.0, market_regime="trending_up"),
            _make_trade("t2", "bot1", -50.0, market_regime="ranging"),
            _make_trade("t3", "bot1", 200.0, market_regime="trending_up"),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        ra = builder.regime_analysis(trades)

        assert ra.regime_pnl["trending_up"] == 300.0
        assert ra.regime_pnl["ranging"] == -50.0
        assert ra.regime_trade_count["trending_up"] == 2

    def test_filter_analysis(self):
        missed = [
            _make_missed("bot1", "BTCUSDT", "vol_filter", 500.0),
            _make_missed("bot1", "ETHUSDT", "vol_filter", -100.0),
            _make_missed("bot1", "SOLUSDT", "spread_filter", 200.0),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        fa = builder.filter_analysis(missed)

        assert fa.filter_block_counts["vol_filter"] == 2
        assert fa.filter_block_counts["spread_filter"] == 1

    def test_root_cause_summary(self):
        trades = [
            _make_trade("t1", "bot1", 100.0, root_causes=["normal_win"]),
            _make_trade("t2", "bot1", -50.0, root_causes=["regime_mismatch", "weak_signal"]),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        rcs = builder.root_cause_summary(trades)

        assert rcs.distribution["normal_win"] == 1
        assert rcs.distribution["regime_mismatch"] == 1
        assert rcs.distribution["weak_signal"] == 1

    def test_write_curated_files(self, tmp_path: Path):
        trades = [
            _make_trade("t1", "bot1", 100.0),
            _make_trade("t2", "bot1", -50.0),
        ]
        missed = [_make_missed("bot1", "BTCUSDT", "vol_filter", 500.0)]

        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        assert (output_dir / "summary.json").exists()
        assert (output_dir / "winners.json").exists()
        assert (output_dir / "losers.json").exists()
        assert (output_dir / "process_failures.json").exists()
        assert (output_dir / "notable_missed.json").exists()
        assert (output_dir / "regime_analysis.json").exists()
        assert (output_dir / "filter_analysis.json").exists()
        assert (output_dir / "root_cause_summary.json").exists()

        # Verify JSON is valid and round-trips
        summary_data = json.loads((output_dir / "summary.json").read_text())
        assert summary_data["bot_id"] == "bot1"


class TestDailyMetricsPipelineEnriched:
    """Tests for hourly_performance and slippage_stats integration."""

    def test_hourly_performance_method(self):
        from datetime import datetime, timezone

        trades = [
            _make_trade("t1", "bot1", 100.0, entry_time=datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)),
            _make_trade("t2", "bot1", -50.0, entry_time=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc)),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.hourly_performance(trades)

        assert result["bot_id"] == "bot1"
        assert result["date"] == "2026-03-01"
        assert len(result["buckets"]) == 2
        hours = {b["hour"] for b in result["buckets"]}
        assert hours == {9, 15}

    def test_slippage_stats_method(self):
        trades = [
            _make_trade("t1", "bot1", 100.0, spread_at_entry=5.0),
            _make_trade("t2", "bot1", -50.0, spread_at_entry=3.0, pair="ETHUSDT"),
        ]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.slippage_stats(trades)

        assert result["bot_id"] == "bot1"
        assert result["date"] == "2026-03-01"
        assert "BTCUSDT" in result["by_symbol"]
        assert "ETHUSDT" in result["by_symbol"]

    def test_writes_hourly_performance(self, tmp_path):
        from datetime import datetime, timezone

        trades = [
            _make_trade("t1", "bot1", 100.0, entry_time=datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)),
            _make_trade("t2", "bot1", -50.0, entry_time=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc)),
        ]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        hourly_path = output_dir / "hourly_performance.json"
        assert hourly_path.exists()
        data = json.loads(hourly_path.read_text())
        assert len(data["buckets"]) == 2
        assert data["bot_id"] == "bot1"

    def test_writes_slippage_stats(self, tmp_path):
        trades = [
            _make_trade("t1", "bot1", 100.0, spread_at_entry=5.0),
        ]
        missed: list = []
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        slippage_path = output_dir / "slippage_stats.json"
        assert slippage_path.exists()
        data = json.loads(slippage_path.read_text())
        assert data["bot_id"] == "bot1"
        assert "BTCUSDT" in data["by_symbol"]
        assert data["by_symbol"]["BTCUSDT"]["mean_bps"] == 5.0

    def test_write_curated_includes_all_enriched_files(self, tmp_path):
        """Verify write_curated produces both new files alongside existing ones."""
        from datetime import datetime, timezone

        trades = [
            _make_trade("t1", "bot1", 100.0,
                        entry_time=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
                        spread_at_entry=4.0),
            _make_trade("t2", "bot1", -50.0,
                        entry_time=datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc),
                        spread_at_entry=6.0),
        ]
        missed = [_make_missed("bot1", "BTCUSDT", "vol_filter", 500.0)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        output_dir = builder.write_curated(trades, missed, base_dir=tmp_path)

        # Original files still present
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "winners.json").exists()
        assert (output_dir / "regime_analysis.json").exists()

        # New enriched files present
        assert (output_dir / "hourly_performance.json").exists()
        assert (output_dir / "slippage_stats.json").exists()

    def test_hourly_performance_empty_trades(self):
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.hourly_performance([])
        assert result["buckets"] == []

    def test_slippage_stats_no_spread(self):
        """Trades with spread_at_entry=0 should produce empty distributions."""
        trades = [_make_trade("t1", "bot1", 100.0, spread_at_entry=0.0)]
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        result = builder.slippage_stats(trades)
        assert result["by_symbol"] == {}
        assert result["by_hour"] == {}
