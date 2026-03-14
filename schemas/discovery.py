# schemas/discovery.py
"""Discovery schemas — novel patterns found by Claude from raw trade data."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class TradeReference(BaseModel):
    """Reference to a specific trade that supports a discovery."""
    date: str = ""
    bot_id: str = ""
    trade_id: str = ""
    pnl: float = 0.0
    regime: str = ""
    signal_strength: float = 0.0
    note: str = ""


class Discovery(BaseModel):
    """A novel pattern discovered by the analysis agent."""
    discovery_id: str = ""
    pattern_description: str
    evidence: list[TradeReference] = []
    proposed_root_cause: str = ""  # existing taxonomy or "novel"
    testable_hypothesis: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detector_coverage: str = ""  # which automated detector relates, or "novel"
    bot_id: str = ""
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class StrategyIdea(BaseModel):
    """A novel strategy concept grounded in evidence from discovery analysis."""
    idea_id: str = ""  # deterministic from hash of description
    title: str = ""  # short name (e.g., "Regime-Filtered ORB Reversal")
    description: str = ""  # how the strategy works
    edge_hypothesis: str = ""  # why it should work (grounded in data)
    evidence: list[TradeReference] = []  # supporting data points
    entry_logic: str = ""  # when to enter
    exit_logic: str = ""  # when to exit
    applicable_regimes: list[str] = []  # which market regimes
    applicable_bots: list[str] = []  # which bots could run it
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "proposed"  # proposed → under_review → testing → adopted → retired
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_discoveries: list[str] = []  # discovery_ids that informed this


class DiscoveryReport(BaseModel):
    """Collection of discoveries from a single agent invocation."""
    run_id: str = ""
    date: str = ""
    discoveries: list[Discovery] = []
    strategy_ideas: list[StrategyIdea] = []
    data_scope: str = ""  # e.g., "30d raw trades for bot_x"
