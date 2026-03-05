"""Tests for HourlyAnalyzer — time-of-day performance buckets."""
from datetime import datetime

from schemas.events import TradeEvent
from schemas.hourly_performance import HourlyPerformance
from skills.hourly_analyzer import HourlyAnalyzer


def _make_trade(hour: int, pnl: float, quality: int = 80) -> TradeEvent:
    return TradeEvent(
        trade_id=f"t_{hour}_{pnl}",
        bot_id="bot1",
        pair="BTCUSDT",
        side="LONG",
        entry_time=datetime(2026, 3, 1, hour, 30, 0),
        exit_time=datetime(2026, 3, 1, hour + 1, 0, 0),
        entry_price=50000,
        exit_price=50000 + pnl,
        position_size=1.0,
        pnl=pnl,
        pnl_pct=pnl / 500,
        process_quality_score=quality,
    )


class TestHourlyAnalyzer:
    def test_basic_bucketing(self):
        trades = [
            _make_trade(9, 100.0),
            _make_trade(9, 50.0),
            _make_trade(15, -80.0),
        ]
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute(trades)

        assert isinstance(perf, HourlyPerformance)
        buckets_by_hour = {b.hour: b for b in perf.buckets}
        assert buckets_by_hour[9].trade_count == 2
        assert buckets_by_hour[9].pnl == 150.0
        assert buckets_by_hour[9].win_rate == 1.0
        assert buckets_by_hour[15].trade_count == 1
        assert buckets_by_hour[15].pnl == -80.0
        assert buckets_by_hour[15].win_rate == 0.0

    def test_avg_process_quality(self):
        trades = [
            _make_trade(14, 100.0, quality=90),
            _make_trade(14, -50.0, quality=60),
        ]
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute(trades)

        buckets_by_hour = {b.hour: b for b in perf.buckets}
        assert buckets_by_hour[14].avg_process_quality == 75.0

    def test_empty_trades(self):
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute([])
        assert perf.buckets == []

    def test_best_worst_hours(self):
        trades = [
            _make_trade(9, 200.0),
            _make_trade(12, -100.0),
            _make_trade(18, 50.0),
        ]
        analyzer = HourlyAnalyzer(bot_id="bot1", date="2026-03-01")
        perf = analyzer.compute(trades)
        assert perf.best_hour == 9
        assert perf.worst_hour == 12
