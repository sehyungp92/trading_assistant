"""Regime-conditional analysis schemas — per-regime strategy metrics and allocations.

Used by analysis/strategy_engine.py to produce regime-aware allocation suggestions.
"""
from __future__ import annotations

from pydantic import BaseModel


class RegimeStrategyMetrics(BaseModel):
    """Performance metrics for a strategy in a specific regime."""

    bot_id: str
    strategy_id: str
    regime: str
    trade_count: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0  # avg PnL per trade
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_exit_efficiency: float = 0.0


class RegimeAllocation(BaseModel):
    """Suggested allocation for a specific regime."""

    regime: str
    allocations: dict[str, float] = {}  # "bot:strategy" → suggested_pct
    rationale: str = ""


class RegimeDistribution(BaseModel):
    """How much time/trades are spent in each regime."""

    regime: str
    pct_of_time: float = 0.0  # what % of trading time in this regime
    trend: str = ""  # "increasing" | "stable" | "decreasing"
    trade_count: int = 0


class RegimeConditionalReport(BaseModel):
    """Complete regime-conditional analysis for a week."""

    week_start: str
    week_end: str
    metrics: list[RegimeStrategyMetrics] = []
    optimal_allocations: list[RegimeAllocation] = []
    regime_distribution: list[RegimeDistribution] = []
    suggestions: list[dict] = []  # {regime, strategy, current_alloc, suggested_alloc, reason}
