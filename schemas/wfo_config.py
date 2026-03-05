# schemas/wfo_config.py
"""WFO configuration schemas — defines the full parameter space for walk-forward optimization.

Matches the wfo_config.yaml structure from the roadmap (Phase 4.1).
Loaded by skills/run_wfo.py and consumed by the fold generator, optimizer,
cost model, leakage detector, and robustness tester.
"""
from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, computed_field


class WFOMethod(str, Enum):
    ANCHORED = "anchored"
    ROLLING = "rolling"


class OptimizationObjective(str, Enum):
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    PROFIT_FACTOR = "profit_factor"


class SlippageModel(str, Enum):
    FIXED = "fixed"
    SPREAD_PROPORTIONAL = "spread_proportional"
    EMPIRICAL = "empirical"


class OptimizationConfig(BaseModel):
    objective: OptimizationObjective = OptimizationObjective.CALMAR
    secondary: str = "max_drawdown"
    max_drawdown_constraint: float = 0.15


class CostModelConfig(BaseModel):
    fees_per_trade_bps: float = 7.0
    slippage_model: SlippageModel = SlippageModel.FIXED
    fixed_slippage_bps: float = 5.0
    slippage_source: str = ""
    spread_impact: bool = True
    reject_if_only_profitable_at_zero_cost: bool = True
    cost_sensitivity_test: bool = True
    cost_multipliers: list[float] = [1.0, 1.5, 2.0]


class LeakagePreventionConfig(BaseModel):
    strict_temporal_split: bool = True
    no_forward_fill_labels: bool = True
    feature_audit: bool = True


class RobustnessConfig(BaseModel):
    neighborhood_test: bool = True
    regime_stability: bool = True
    min_trades_per_fold: int = 30
    neighborhood_pct: float = 0.1
    min_profitable_regimes: int = 3
    total_regime_types: int = 4


class OutputConfig(BaseModel):
    param_recommendations: bool = True
    robustness_heatmap: bool = True
    equity_curves: bool = True
    regime_breakdown: bool = True
    cost_sensitivity_report: bool = True
    leakage_audit_log: bool = True


class ParameterDef(BaseModel):
    """A single tunable parameter with its search range."""

    name: str
    min_value: float
    max_value: float
    step: float
    current_value: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grid_values(self) -> list[float]:
        if self.step <= 0:
            return [self.min_value]
        values: list[float] = []
        v = self.min_value
        while v <= self.max_value + self.step * 1e-9:
            values.append(round(v, 10))
            v += self.step
        return values


class ParameterSpace(BaseModel):
    """The full parameter search space for a bot."""

    bot_id: str
    parameters: list[ParameterDef] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_combinations(self) -> int:
        if not self.parameters:
            return 0
        return math.prod(len(p.grid_values) for p in self.parameters)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_params(self) -> dict[str, float]:
        return {p.name: p.current_value for p in self.parameters}


class WFOConfig(BaseModel):
    """Complete walk-forward optimization configuration — maps to wfo_config.yaml."""

    bot_id: str
    method: WFOMethod = WFOMethod.ANCHORED
    in_sample_days: int = 180
    out_of_sample_days: int = 30
    step_days: int = 30
    min_folds: int = 6
    parameter_space: ParameterSpace
    optimization: OptimizationConfig = OptimizationConfig()
    cost_model: CostModelConfig = CostModelConfig()
    leakage_prevention: LeakagePreventionConfig = LeakagePreventionConfig()
    robustness: RobustnessConfig = RobustnessConfig()
    output: OutputConfig = OutputConfig()
