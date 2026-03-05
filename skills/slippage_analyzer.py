"""SlippageAnalyzer — computes per-symbol, per-hour slippage distributions.

Uses spread_at_entry from TradeEvent as the primary slippage signal.
Exports regime->bps mapping for WFO cost model empirical mode.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from schemas.events import TradeEvent
from schemas.slippage_analysis import SlippageBucket, SlippageDistribution


class SlippageAnalyzer:
    def __init__(self, bot_id: str, date: str) -> None:
        self._bot_id = bot_id
        self._date = date

    def compute(self, trades: list[TradeEvent]) -> SlippageDistribution:
        by_symbol: dict[str, list[float]] = defaultdict(list)
        by_hour: dict[str, list[float]] = defaultdict(list)

        for t in trades:
            bps = t.spread_at_entry
            if bps <= 0:
                continue
            by_symbol[t.pair].append(bps)
            hour_key = f"{t.entry_time.hour:02d}"
            by_hour[hour_key].append(bps)

        return SlippageDistribution(
            bot_id=self._bot_id,
            date=self._date,
            by_symbol={k: self._make_bucket(k, v) for k, v in by_symbol.items()},
            by_hour={k: self._make_bucket(k, v) for k, v in by_hour.items()},
        )

    def export_regime_bps(self, trades: list[TradeEvent]) -> dict[str, float]:
        """Export regime->mean_bps mapping for WFO cost model."""
        by_regime: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            if t.spread_at_entry > 0 and t.market_regime:
                by_regime[t.market_regime].append(t.spread_at_entry)
        return {
            regime: statistics.mean(values) for regime, values in by_regime.items()
        }

    @staticmethod
    def _make_bucket(key: str, values: list[float]) -> SlippageBucket:
        if not values:
            return SlippageBucket(key=key)
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return SlippageBucket(
            key=key,
            sample_count=n,
            mean_bps=statistics.mean(sorted_vals),
            median_bps=statistics.median(sorted_vals),
            p75_bps=sorted_vals[int(n * 0.75)] if n >= 4 else sorted_vals[-1],
            p95_bps=sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
        )
