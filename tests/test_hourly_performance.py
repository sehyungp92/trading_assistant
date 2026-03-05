# tests/test_hourly_performance.py
"""Tests for hourly performance schemas."""
from schemas.hourly_performance import HourlyBucket, HourlyPerformance


class TestHourlyBucket:
    def test_creates_with_stats(self):
        bucket = HourlyBucket(
            hour=14,
            trade_count=8,
            pnl=250.0,
            win_rate=0.75,
            avg_process_quality=82.0,
        )
        assert bucket.hour == 14
        assert bucket.win_rate == 0.75

    def test_defaults(self):
        bucket = HourlyBucket(hour=0)
        assert bucket.trade_count == 0
        assert bucket.pnl == 0.0


class TestHourlyPerformance:
    def test_creates_with_buckets(self):
        perf = HourlyPerformance(
            bot_id="bot1",
            date="2026-03-01",
            buckets=[
                HourlyBucket(hour=9, trade_count=5, pnl=100.0, win_rate=0.8),
                HourlyBucket(hour=15, trade_count=3, pnl=-50.0, win_rate=0.33),
            ],
        )
        assert len(perf.buckets) == 2

    def test_best_hour_property(self):
        perf = HourlyPerformance(
            bot_id="bot1",
            date="2026-03-01",
            buckets=[
                HourlyBucket(hour=9, trade_count=5, pnl=100.0, win_rate=0.8),
                HourlyBucket(hour=15, trade_count=3, pnl=-50.0, win_rate=0.33),
                HourlyBucket(hour=20, trade_count=4, pnl=200.0, win_rate=0.75),
            ],
        )
        assert perf.best_hour == 20

    def test_worst_hour_property(self):
        perf = HourlyPerformance(
            bot_id="bot1",
            date="2026-03-01",
            buckets=[
                HourlyBucket(hour=9, pnl=100.0),
                HourlyBucket(hour=15, pnl=-50.0),
            ],
        )
        assert perf.worst_hour == 15

    def test_empty_buckets_returns_none(self):
        perf = HourlyPerformance(bot_id="bot1", date="2026-03-01")
        assert perf.best_hour is None
        assert perf.worst_hour is None
