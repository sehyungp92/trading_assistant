# schemas/wfo_results.py
"""WFO result schemas — models for fold definitions, simulation results, and reports.

Produced by the WFO pipeline (skills/run_wfo.py) and consumed by the report builder
(analysis/wfo_report_builder.py) and prompt assembler (analysis/wfo_prompt_assembler.py).
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field


class FoldDefinition(BaseModel):
    """Temporal boundaries for one WFO fold."""

    fold_number: int
    is_start: str  # YYYY-MM-DD
    is_end: str
    oos_start: str
    oos_end: str


class SimulationMetrics(BaseModel):
    """Performance metrics from a simulation run."""

    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades_by_regime: dict[str, int] = {}
    pnl_by_regime: dict[str, float] = {}
    daily_pnl: dict[str, float] = {}  # date → PnL for equity curve

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expectancy(self) -> float:
        """win_rate × (avg_win / avg_loss) — the expectancy equation."""
        if self.avg_loss == 0 or self.total_trades == 0:
            return 0.0
        return self.win_rate * (self.avg_win / self.avg_loss)


class FoldResult(BaseModel):
    """Results for one WFO fold: best params from IS, performance on OOS."""

    fold: FoldDefinition
    best_params: dict[str, float]
    is_metrics: SimulationMetrics = SimulationMetrics()
    oos_metrics: SimulationMetrics = SimulationMetrics()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def oos_degradation_pct(self) -> float:
        """How much the objective metric dropped from IS to OOS. 0.0 = no drop, 1.0 = total loss."""
        is_val = self.is_metrics.calmar_ratio
        oos_val = self.oos_metrics.calmar_ratio
        if is_val == 0:
            return 0.0
        return (is_val - oos_val) / is_val


class CostSensitivityResult(BaseModel):
    """Simulation results at a specific cost multiplier."""

    cost_multiplier: float
    metrics: SimulationMetrics = SimulationMetrics()


class LeakageAuditEntry(BaseModel):
    """One entry in the leakage audit log — verifies a feature's temporal correctness."""

    feature_name: str
    computed_at: str  # ISO timestamp
    latest_data_used: str  # ISO timestamp
    passed: bool
    violation: str = ""


class RobustnessResult(BaseModel):
    """Results from neighborhood and regime stability tests."""

    neighborhood_scores: dict[str, float] = {}
    neighborhood_stable: bool = False
    regime_pnl: dict[str, float] = {}
    profitable_regime_count: int = 0
    regime_stable: bool = False
    robustness_score: float = 0.0  # 0–100


class WFORecommendation(str, Enum):
    ADOPT = "adopt"
    TEST_FURTHER = "test_further"
    REJECT = "reject"


class SafetyFlag(BaseModel):
    """A safety warning attached to WFO results."""

    flag_type: str  # low_conviction | likely_overfit | fragile
    description: str
    severity: str = "medium"  # low | medium | high


class WFOReport(BaseModel):
    """Complete WFO output — consumed by report builder and prompt assembler."""

    bot_id: str
    config_summary: dict = {}
    current_params: dict[str, float] = {}
    suggested_params: dict[str, float] = {}
    fold_results: list[FoldResult] = []
    aggregate_oos_metrics: SimulationMetrics = SimulationMetrics()
    cost_sensitivity: list[CostSensitivityResult] = []
    leakage_audit: list[LeakageAuditEntry] = []
    robustness: RobustnessResult = RobustnessResult()
    safety_flags: list[SafetyFlag] = []
    recommendation: WFORecommendation = WFORecommendation.REJECT
    recommendation_reasoning: str = ""
