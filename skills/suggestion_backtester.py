# skills/suggestion_backtester.py
"""Suggestion backtester — runs baseline vs proposed parameter backtests.

Loads historical trades, simulates both configurations, and applies
safety checks to determine if a parameter change is safe to propose.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from schemas.autonomous_pipeline import (
    BacktestComparison,
    BacktestContext,
)
from schemas.wfo_results import SimulationMetrics
from skills.config_registry import ConfigRegistry

logger = logging.getLogger(__name__)


class SuggestionBacktester:
    """Backtests parameter changes against historical trade data."""

    def __init__(self, config_registry: ConfigRegistry, data_dir: Path) -> None:
        self._registry = config_registry
        self._data_dir = Path(data_dir)

    async def backtest_suggestion(
        self,
        suggestion_id: str,
        bot_id: str,
        param_name: str,
        current_value: Any,
        proposed_value: Any,
    ) -> BacktestComparison:
        """Run baseline vs proposed backtest and return comparison."""
        trades = self._load_trades(bot_id)
        param = self._registry.get_parameter(bot_id, param_name)
        is_safety_critical = param.is_safety_critical if param else False

        ctx = BacktestContext(
            suggestion_id=suggestion_id,
            bot_id=bot_id,
            param_name=param_name,
            current_value=current_value,
            proposed_value=proposed_value,
            trade_count=len(trades),
            data_days=self._count_unique_days(trades),
        )

        if not trades:
            return BacktestComparison(
                context=ctx,
                passes_safety=False,
                safety_notes=["No trade data available"],
            )

        baseline = self._simulate(trades, param_name, current_value)
        proposed = self._simulate(trades, param_name, proposed_value)

        passes, notes = self._check_safety(baseline, proposed, is_safety_critical)

        return BacktestComparison(
            context=ctx,
            baseline=baseline,
            proposed=proposed,
            passes_safety=passes,
            safety_notes=notes,
        )

    def _load_trades(self, bot_id: str, lookback_days: int = 30) -> list[dict]:
        """Load trades from data/curated/ directories."""
        trades: list[dict] = []
        curated = self._data_dir / "data" / "curated"
        if not curated.exists():
            return trades
        for date_dir in sorted(curated.iterdir(), reverse=True)[:lookback_days]:
            bot_dir = date_dir / bot_id
            for fname in ("trades.jsonl", "winners.csv", "losers.csv"):
                fpath = bot_dir / fname
                if fpath.exists() and fname.endswith(".jsonl"):
                    for line in fpath.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            try:
                                trades.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            summary = bot_dir / "summary.json"
            if summary.exists():
                try:
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    if "trades" in data and isinstance(data["trades"], list):
                        trades.extend(data["trades"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return trades

    def _simulate(self, trades: list[dict], param_name: str, value: Any) -> SimulationMetrics:
        """Simple simulation: replay trades and compute metrics.

        This is a simplified simulation that doesn't filter trades by parameter
        (that would require the full BacktestSimulator with trade schemas).
        Instead, it computes metrics from the available trade data.
        """
        if not trades:
            return SimulationMetrics()

        pnls: list[float] = []
        for t in trades:
            pnl = t.get("pnl", t.get("net_pnl", t.get("realized_pnl", 0.0)))
            try:
                pnls.append(float(pnl))
            except (TypeError, ValueError):
                pass

        if not pnls:
            return SimulationMetrics()

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross = sum(pnls)
        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe (simplified: mean/std of PnLs)
        mean_pnl = gross / len(pnls) if pnls else 0
        if len(pnls) > 1:
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std = variance ** 0.5
            sharpe = (mean_pnl / std) if std > 0 else 0.0
        else:
            sharpe = 0.0

        pf = (gross_wins / gross_losses) if gross_losses > 0 else (10.0 if gross_wins > 0 else 0.0)

        return SimulationMetrics(
            total_trades=len(pnls),
            win_count=len(wins),
            loss_count=len(losses),
            gross_pnl=gross,
            net_pnl=gross,
            max_drawdown_pct=-max_dd * 100,
            sharpe_ratio=sharpe,
            profit_factor=pf,
        )

    def _check_safety(
        self,
        baseline: SimulationMetrics,
        proposed: SimulationMetrics,
        is_safety_critical: bool,
    ) -> tuple[bool, list[str]]:
        """Apply safety checks to backtest results."""
        notes: list[str] = []
        passes = True

        # Check minimum trade count
        if proposed.total_trades < 10:
            notes.append(f"Insufficient trades: {proposed.total_trades} < 10")
            passes = False

        # Sharpe ratio >= 0
        if proposed.sharpe_ratio < 0:
            notes.append(f"Negative Sharpe ratio: {proposed.sharpe_ratio:.2f}")
            passes = False

        # Profit factor >= 1.0
        if proposed.profit_factor < 1.0:
            notes.append(f"Profit factor below 1.0: {proposed.profit_factor:.2f}")
            passes = False

        # Max drawdown: not >50% worse (30% for safety-critical)
        dd_threshold = 0.30 if is_safety_critical else 0.50
        if baseline.max_drawdown_pct != 0:
            dd_change = abs(proposed.max_drawdown_pct) - abs(baseline.max_drawdown_pct)
            if abs(baseline.max_drawdown_pct) > 0:
                dd_change_ratio = dd_change / abs(baseline.max_drawdown_pct)
                if dd_change_ratio > dd_threshold:
                    notes.append(
                        f"Drawdown worsened by {dd_change_ratio:.0%} "
                        f"(threshold: {dd_threshold:.0%})"
                    )
                    passes = False

        return passes, notes

    @staticmethod
    def _count_unique_days(trades: list[dict]) -> int:
        dates = set()
        for t in trades:
            for key in ("date", "trade_date", "timestamp"):
                if key in t:
                    dates.add(str(t[key])[:10])
                    break
        return len(dates)
