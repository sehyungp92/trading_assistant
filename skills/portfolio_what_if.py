# skills/portfolio_what_if.py
"""Portfolio what-if analysis — simplified linear rescaling for allocation proposals.

Input: historical daily family PnL series + proposed allocation weights.
Method: multiply each family's daily PnL by (new_weight / old_weight).
Output: portfolio Calmar, Sharpe, max drawdown under new allocation.

Limitation: ignores interaction effects from coordination rules, capacity constraints.
Good enough for: allocation weight changes (~80% of proposals).
Not sufficient for: coordination rule changes, drawdown tier changes.
"""
from __future__ import annotations

import math


class PortfolioWhatIf:
    """Simplified what-if analysis for portfolio allocation changes."""

    def __init__(
        self,
        family_daily_pnl: dict[str, list[float]],
        current_weights: dict[str, float],
    ) -> None:
        """Initialize with historical daily PnL per family and current weights.

        Args:
            family_daily_pnl: family_name → list of daily PnL values (oldest first).
            current_weights: family_name → current allocation weight (0-1).
        """
        self._family_pnl = family_daily_pnl
        self._current_weights = current_weights

    def evaluate(self, proposed_weights: dict[str, float]) -> dict:
        """Evaluate a proposed allocation change.

        Returns dict with:
            current_sharpe, proposed_sharpe, current_calmar, proposed_calmar,
            current_max_drawdown, proposed_max_drawdown, calmar_delta, sharpe_delta
        """
        current_portfolio = self._build_portfolio_series(self._current_weights)
        proposed_portfolio = self._build_portfolio_series(proposed_weights)

        current_sharpe = self._sharpe(current_portfolio)
        proposed_sharpe = self._sharpe(proposed_portfolio)
        current_calmar = self._calmar(current_portfolio)
        proposed_calmar = self._calmar(proposed_portfolio)
        current_dd = self._max_drawdown(current_portfolio)
        proposed_dd = self._max_drawdown(proposed_portfolio)

        return {
            "current_sharpe": round(current_sharpe, 4),
            "proposed_sharpe": round(proposed_sharpe, 4),
            "sharpe_delta": round(proposed_sharpe - current_sharpe, 4),
            "current_calmar": round(current_calmar, 4),
            "proposed_calmar": round(proposed_calmar, 4),
            "calmar_delta": round(proposed_calmar - current_calmar, 4),
            "current_max_drawdown": round(current_dd, 4),
            "proposed_max_drawdown": round(proposed_dd, 4),
            "drawdown_delta": round(proposed_dd - current_dd, 4),
            "days_analyzed": len(current_portfolio),
            "method": "linear_rescaling",
        }

    def _build_portfolio_series(self, weights: dict[str, float]) -> list[float]:
        """Build daily portfolio PnL by rescaling family PnL series."""
        if not self._family_pnl:
            return []

        lengths = [len(v) for v in self._family_pnl.values() if v]
        if not lengths:
            return []
        max_len = max(lengths)
        portfolio = [0.0] * max_len

        for family, pnl_series in self._family_pnl.items():
            current_w = self._current_weights.get(family, 0.0)
            proposed_w = weights.get(family, current_w)

            if current_w <= 0:
                continue  # Can't rescale from zero

            scale = proposed_w / current_w

            for i, pnl in enumerate(pnl_series):
                portfolio[i] += pnl * scale

        return portfolio

    @staticmethod
    def _sharpe(returns: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _calmar(returns: list[float]) -> float:
        if not returns:
            return 0.0
        total = sum(returns)
        annualized = total * 252 / max(len(returns), 1)
        max_dd = PortfolioWhatIf._max_drawdown(returns)
        if max_dd <= 0:
            return 0.0
        return annualized / (max_dd * 100)

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
        if not returns:
            return 0.0
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in returns:
            equity += r
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd
