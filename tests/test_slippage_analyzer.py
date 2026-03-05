"""Tests for SlippageAnalyzer — computes per-symbol, per-hour slippage distributions."""
from datetime import datetime

from schemas.events import TradeEvent
from schemas.slippage_analysis import SlippageDistribution
from skills.slippage_analyzer import SlippageAnalyzer


def _make_trade(pair: str, entry_price: float, exit_price: float,
                spread_at_entry: float, entry_hour: int = 14) -> TradeEvent:
    return TradeEvent(
        trade_id=f"t_{pair}_{entry_hour}",
        bot_id="bot1",
        pair=pair,
        side="LONG",
        entry_time=datetime(2026, 3, 1, entry_hour, 0, 0),
        exit_time=datetime(2026, 3, 1, entry_hour + 1, 0, 0),
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=1.0,
        pnl=exit_price - entry_price,
        pnl_pct=((exit_price - entry_price) / entry_price) * 100,
        spread_at_entry=spread_at_entry,
    )


class TestSlippageAnalyzer:
    def test_compute_by_symbol(self):
        trades = [
            _make_trade("BTCUSDT", 50000, 50100, 5.0),
            _make_trade("BTCUSDT", 50200, 50300, 4.0),
            _make_trade("ETHUSDT", 3000, 3050, 8.0),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute(trades)

        assert isinstance(dist, SlippageDistribution)
        assert "BTCUSDT" in dist.by_symbol
        assert dist.by_symbol["BTCUSDT"].sample_count == 2
        assert "ETHUSDT" in dist.by_symbol
        assert dist.by_symbol["ETHUSDT"].sample_count == 1

    def test_compute_by_hour(self):
        trades = [
            _make_trade("BTCUSDT", 50000, 50100, 5.0, entry_hour=9),
            _make_trade("BTCUSDT", 50200, 50300, 3.0, entry_hour=9),
            _make_trade("BTCUSDT", 50400, 50500, 7.0, entry_hour=15),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute(trades)

        assert "09" in dist.by_hour
        assert dist.by_hour["09"].sample_count == 2
        assert "15" in dist.by_hour
        assert dist.by_hour["15"].sample_count == 1

    def test_slippage_bps_from_spread(self):
        """Spread at entry is the primary slippage signal."""
        trades = [
            _make_trade("BTCUSDT", 50000, 50100, 10.0),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute(trades)

        # Slippage = spread_at_entry (already in bps)
        assert dist.by_symbol["BTCUSDT"].mean_bps == 10.0

    def test_empty_trades(self):
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        dist = analyzer.compute([])
        assert dist.by_symbol == {}
        assert dist.by_hour == {}

    def test_export_for_cost_model(self):
        """Exports regime->bps mapping for WFO cost model empirical mode."""
        trades = [
            TradeEvent(
                trade_id="t1", bot_id="bot1", pair="BTCUSDT", side="LONG",
                entry_time=datetime(2026, 3, 1, 14, 0),
                exit_time=datetime(2026, 3, 1, 15, 0),
                entry_price=50000, exit_price=50100,
                position_size=1.0, pnl=100, pnl_pct=0.2,
                spread_at_entry=5.0, market_regime="trending_up",
            ),
            TradeEvent(
                trade_id="t2", bot_id="bot1", pair="BTCUSDT", side="LONG",
                entry_time=datetime(2026, 3, 1, 16, 0),
                exit_time=datetime(2026, 3, 1, 17, 0),
                entry_price=50200, exit_price=50300,
                position_size=1.0, pnl=100, pnl_pct=0.2,
                spread_at_entry=8.0, market_regime="ranging",
            ),
        ]
        analyzer = SlippageAnalyzer(bot_id="bot1", date="2026-03-01")
        regime_bps = analyzer.export_regime_bps(trades)
        assert "trending_up" in regime_bps
        assert regime_bps["trending_up"] == 5.0
        assert regime_bps["ranging"] == 8.0
