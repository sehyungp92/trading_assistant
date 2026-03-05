"""Daily metrics schemas — Pydantic models for reduced/curated daily data.

These define the output format of the data reduction pipeline (skills/build_daily_metrics.py).
Claude sees these pre-scored, pre-classified structures — not raw trade data.
"""
from __future__ import annotations

from pydantic import BaseModel, computed_field


class BotDailySummary(BaseModel):
    """Aggregated daily stats for a single bot."""

    date: str  # YYYY-MM-DD
    bot_id: str
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_rolling_30d: float = 0.0
    sortino_rolling_30d: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    exposure_pct: float = 0.0
    missed_count: int = 0
    missed_would_have_won: int = 0
    error_count: int = 0
    uptime_pct: float = 100.0
    avg_process_quality: float = 100.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades

    @computed_field  # type: ignore[prop-decorator]
    @property
    def profit_factor(self) -> float:
        total_wins = self.avg_win * self.win_count
        total_losses = abs(self.avg_loss) * self.loss_count
        if total_losses == 0:
            return float("inf") if total_wins > 0 else 0.0
        return total_wins / total_losses


class WinnerLoserRecord(BaseModel):
    """A single notable winning or losing trade with full context."""

    trade_id: str
    bot_id: str
    pair: str
    side: str
    pnl: float
    pnl_pct: float = 0.0
    entry_signal: str = ""
    exit_reason: str = ""
    market_regime: str = ""
    process_quality_score: int = 100
    root_causes: list[str] = []
    entry_price: float = 0.0
    exit_price: float = 0.0
    atr_at_entry: float = 0.0


class ProcessFailureRecord(BaseModel):
    """A trade with process_quality_score < 60 — flagged for review."""

    trade_id: str
    bot_id: str
    pair: str
    process_quality_score: int
    root_causes: list[str]
    pnl: float
    entry_signal: str = ""
    market_regime: str = ""


class NotableMissedRecord(BaseModel):
    """A missed opportunity where outcome > 2× avg win."""

    bot_id: str
    pair: str
    signal: str
    blocked_by: str = ""
    hypothetical_entry: float = 0.0
    outcome_24h: float = 0.0
    confidence: float = 0.0
    assumption_tags: list[str] = []


class RegimeAnalysis(BaseModel):
    """PnL breakdown by market regime for one bot on one day."""

    bot_id: str
    date: str
    regime_pnl: dict[str, float] = {}
    regime_trade_count: dict[str, int] = {}
    regime_win_rate: dict[str, float] = {}


class FilterAnalysis(BaseModel):
    """Impact analysis for each filter: how many trades blocked, net PnL effect."""

    bot_id: str
    date: str
    filter_block_counts: dict[str, int] = {}
    filter_saved_pnl: dict[str, float] = {}
    filter_missed_pnl: dict[str, float] = {}


class AnomalyRecord(BaseModel):
    """A statistically unusual event detected in today's data."""

    bot_id: str
    date: str
    anomaly_type: str
    description: str
    severity: str = "medium"  # low | medium | high
    related_trades: list[str] = []


class RootCauseSummary(BaseModel):
    """Distribution of root causes across all trades for one bot on one day."""

    bot_id: str
    date: str
    distribution: dict[str, int] = {}
    total_trades: int = 0
