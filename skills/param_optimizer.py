# skills/param_optimizer.py
"""Parameter optimizer — grid search over parameter space with constraint filtering.

Evaluates every combination from the parameter space on in-sample trades,
ranks by the configured objective, and filters by constraints (max drawdown,
min trades per fold).
"""
from __future__ import annotations

import itertools

from schemas.events import TradeEvent, MissedOpportunityEvent
from schemas.wfo_config import OptimizationConfig, OptimizationObjective, ParameterSpace
from schemas.wfo_results import SimulationMetrics
from skills.backtest_simulator import BacktestSimulator


class ParamOptimizer:
    """Grid search optimizer over a parameter space."""

    def __init__(
        self,
        space: ParameterSpace,
        opt_config: OptimizationConfig,
        min_trades: int = 0,
    ) -> None:
        self._space = space
        self._opt = opt_config
        self._min_trades = min_trades

    def generate_grid(self) -> list[dict[str, float]]:
        """Generate all parameter combinations from the space."""
        if not self._space.parameters:
            return [{}]

        names = [p.name for p in self._space.parameters]
        grids = [p.grid_values for p in self._space.parameters]
        combos: list[dict[str, float]] = []
        for values in itertools.product(*grids):
            combos.append(dict(zip(names, values)))
        return combos

    def optimize(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        simulator: BacktestSimulator,
        cost_multiplier: float = 1.0,
    ) -> tuple[dict[str, float], SimulationMetrics, list[tuple[dict[str, float], SimulationMetrics]]]:
        """Find the best parameter set by grid search.

        Returns: (best_params, best_metrics, all_results)
        """
        grid = self.generate_grid()
        all_results: list[tuple[dict[str, float], SimulationMetrics]] = []

        for params in grid:
            metrics = simulator.simulate(trades, missed, params, cost_multiplier=cost_multiplier)
            all_results.append((params, metrics))

        # Filter by constraints
        valid = [
            (p, m) for p, m in all_results
            if m.total_trades >= self._min_trades
            and m.max_drawdown_pct <= self._opt.max_drawdown_constraint
        ]

        if not valid:
            return {}, SimulationMetrics(), all_results

        # Rank by objective
        best_params, best_metrics = max(valid, key=lambda x: self._objective_value(x[1]))
        return best_params, best_metrics, all_results

    def _objective_value(self, metrics: SimulationMetrics) -> float:
        obj = self._opt.objective
        if obj == OptimizationObjective.SHARPE:
            return metrics.sharpe_ratio
        elif obj == OptimizationObjective.SORTINO:
            return metrics.sortino_ratio
        elif obj == OptimizationObjective.CALMAR:
            return metrics.calmar_ratio
        elif obj == OptimizationObjective.PROFIT_FACTOR:
            return metrics.profit_factor
        return 0.0
