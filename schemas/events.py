"""Phase 0 event schemas — Pydantic models for trade instrumentation.

These define the data contracts between VPS bots, the relay, and the orchestrator.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING = "TRAILING"
    TIMEOUT = "TIMEOUT"
    MANUAL = "MANUAL"


class EventMetadata(BaseModel):
    """Attached to every event for traceability and clock alignment."""

    bot_id: str
    exchange_timestamp: datetime
    local_timestamp: datetime
    data_source_id: str
    event_type: str
    payload_key: str
    bar_id: Optional[str] = None
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def event_id(self) -> str:
        raw = f"{self.bot_id}|{self.exchange_timestamp.isoformat()}|{self.event_type}|{self.payload_key}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clock_skew_ms(self) -> int:
        delta = self.exchange_timestamp - self.local_timestamp
        return int(delta.total_seconds() * 1000)


class MarketSnapshot(BaseModel):
    snapshot_id: str = ""
    symbol: str = ""
    timestamp: Optional[datetime] = None
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    spread_bps: float = 0.0
    last_trade_price: float = 0.0
    volume_1m: float = 0.0
    atr_14: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0


class TradeEvent(BaseModel):
    """A completed trade emitted by a bot."""

    trade_id: str
    bot_id: str
    strategy_id: str = ""  # identifies strategy within multi-strategy bots
    pair: str
    event_metadata: Optional[EventMetadata] = None
    market_snapshot: Optional[MarketSnapshot] = None

    side: str  # LONG | SHORT
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float

    entry_signal: str = ""
    entry_signal_strength: float = 0.0
    exit_reason: str = ""
    market_regime: str = ""
    active_filters: list[str] = []
    blocked_by: Optional[str] = None

    atr_at_entry: float = 0.0
    volume_24h: float = 0.0
    spread_at_entry: float = 0.0
    funding_rate: float = 0.0
    open_interest_delta: float = 0.0

    process_quality_score: int = 100
    root_causes: list[str] = []
    evidence_refs: list[str] = []

    signal_factors: list[dict] | None = None
    post_exit_1h_price: float | None = None
    post_exit_4h_price: float | None = None

    # Intra-trade excursion tracking (populated by bots that implement bar-by-bar MFE/MAE)
    mfe_price: float | None = None
    mae_price: float | None = None
    mfe_pct: float | None = None
    mae_pct: float | None = None
    mfe_r: float | None = None
    mae_r: float | None = None
    exit_efficiency: float | None = None  # actual_pnl_pct / mfe_pct

    # 1.5: momentum_trader per-bar signal component values
    signal_evolution: list[dict] | None = None
    # 2.6: momentum_trader order fill details
    entry_fill_details: dict | None = None
    exit_fill_details: dict | None = None

    # stock_trader execution quality fields
    fees_paid: float = 0.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0
    entry_latency_ms: float = 0.0

    # stock_trader session/drawdown context
    session_type: str = ""  # e.g. "regular", "pre_market", "extended"
    drawdown_pct: float = 0.0

    # stock_trader signal tracing (links trade back to originating signal)
    signal_id: str = ""

    # stock_trader filter detail (passed_filters = filters the signal cleared;
    # distinct from active_filters which lists all filters that were active)
    passed_filters: list[str] | None = None
    filter_decisions: list[dict] | None = None

    # Macro regime context (from portfolio-level HMM classifier)
    macro_regime: str = ""  # G/R/S/D active at trade time
    stress_level_at_entry: float = 0.0  # P(stress) at trade time

    # Execution pipeline timing {signal_detected_at, intent_created_at, risk_checked_at, order_submitted_at, fill_received_at}
    execution_timestamps: dict | None = None
    # Position sizing decision context {target_risk_pct, account_equity, volatility_basis, sizing_model, unit_risk_usd, ...multipliers}
    sizing_inputs: dict | None = None
    # Full dict of active strategy parameter values at trade execution time
    strategy_params_at_entry: dict | None = None
    # Portfolio state at entry {exposure, direction, correlated_positions}
    portfolio_state_at_entry: dict | None = None
    # Comprehensive market condition dict from bot (supplements atr_at_entry, volume_24h, etc.)
    market_conditions_at_entry: dict | None = None
    # Post-exit move percentages (backfilled by bot, supplements post_exit_1h_price/4h_price)
    post_exit_1h_move_pct: float | None = None
    post_exit_4h_move_pct: float | None = None
    post_exit_backfill_status: str = ""  # pending | partial | complete


class MissedOpportunityEvent(BaseModel):
    event_metadata: Optional[EventMetadata] = None
    market_snapshot: Optional[MarketSnapshot] = None
    bot_id: str
    strategy_id: str = ""  # identifies strategy within multi-strategy bots
    pair: str
    signal: str
    signal_strength: float = 0.0
    blocked_by: str = ""
    hypothetical_entry: float = 0.0
    outcome_1h: Optional[float] = None
    outcome_4h: Optional[float] = None
    outcome_24h: Optional[float] = None
    would_have_hit_tp: Optional[bool] = None
    would_have_hit_sl: Optional[bool] = None
    confidence: float = 0.0
    assumption_tags: list[str] = []
    margin_pct: float | None = None  # how close to filter threshold (requires bot B4)

    # stock_trader extras
    signal_id: str = ""
    block_reason: str = ""  # freetext explanation (vs blocked_by = filter name)
    backfill_status: str = ""  # e.g. "completed", "pending", "failed"
    simulation_confidence: float = 0.0  # counterfactual sim confidence (vs confidence = signal confidence)


class DailySnapshot(BaseModel):
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
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    exposure_pct: float = 0.0
    missed_count: int = 0
    missed_would_have_won: int = 0
    regime_breakdown: dict = {}
    error_count: int = 0
    uptime_pct: float = 100.0
    avg_process_quality: float = 100.0
    root_cause_distribution: dict = {}
    per_strategy_summary: dict = {}
    overlay_state_summary: dict | None = None
    experiment_breakdown: dict | None = None  # 1.4: swing_trader per-experiment A/B stats

    # Macro regime context (from portfolio-level HMM classifier)
    regime_context: dict | None = None  # RegimeContext snapshot (macro_regime, confidence, stress, etc.)
    applied_regime_config: dict | None = None  # Active regime-adjusted portfolio config

    calmar_rolling_30d: float = 0.0


class RegimeTransitionEvent(BaseModel):
    """Emitted when the macro regime classifier changes state (G/R/S/D)."""

    bot_id: str
    event_metadata: Optional[EventMetadata] = None
    from_regime: str  # e.g. "G"
    to_regime: str  # e.g. "S"
    regime_confidence: float = 0.0  # confidence in new regime
    stress_level: float = 0.0  # stress at transition (observational, 41% FPR)
    timestamp: Optional[datetime] = None
