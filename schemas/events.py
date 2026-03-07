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


class MissedOpportunityEvent(BaseModel):
    event_metadata: Optional[EventMetadata] = None
    market_snapshot: Optional[MarketSnapshot] = None
    bot_id: str
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
