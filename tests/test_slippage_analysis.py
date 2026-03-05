# tests/test_slippage_analysis.py
"""Tests for slippage analysis schemas."""
from schemas.slippage_analysis import (
    SlippageBucket,
    SlippageDistribution,
    SlippageTrend,
)


class TestSlippageBucket:
    def test_creates_with_stats(self):
        bucket = SlippageBucket(
            key="BTCUSDT",
            sample_count=50,
            mean_bps=4.2,
            median_bps=3.8,
            p75_bps=5.5,
            p95_bps=8.1,
        )
        assert bucket.mean_bps == 4.2
        assert bucket.sample_count == 50


class TestSlippageDistribution:
    def test_creates_per_symbol(self):
        dist = SlippageDistribution(
            bot_id="bot1",
            date="2026-03-01",
            by_symbol={
                "BTCUSDT": SlippageBucket(key="BTCUSDT", sample_count=50, mean_bps=4.2),
                "ETHUSDT": SlippageBucket(key="ETHUSDT", sample_count=30, mean_bps=6.1),
            },
        )
        assert len(dist.by_symbol) == 2

    def test_creates_per_hour(self):
        dist = SlippageDistribution(
            bot_id="bot1",
            date="2026-03-01",
            by_hour={
                "09": SlippageBucket(key="09", sample_count=10, mean_bps=3.0),
                "15": SlippageBucket(key="15", sample_count=15, mean_bps=5.0),
            },
        )
        assert dist.by_hour["15"].mean_bps == 5.0


class TestSlippageTrend:
    def test_creates_trend(self):
        trend = SlippageTrend(
            bot_id="bot1",
            symbol="BTCUSDT",
            weekly_mean_bps=[3.5, 4.0, 4.2, 4.8],
            trend_direction="increasing",
        )
        assert trend.trend_direction == "increasing"
