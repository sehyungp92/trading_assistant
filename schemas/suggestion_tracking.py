# schemas/suggestion_tracking.py
"""Suggestion tracking schemas — record and measure suggestion outcomes.

Closes the loop: suggestion proposed → accepted/rejected → implemented → measured.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SuggestionStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class SuggestionRecord(BaseModel):
    """A single suggestion with lifecycle tracking."""

    suggestion_id: str
    bot_id: str
    title: str
    tier: str  # parameter | filter | strategy_variant | hypothesis
    category: str = ""  # original category (exit_timing, filter_threshold, etc.)
    source_report_id: str
    description: str = ""
    status: SuggestionStatus = SuggestionStatus.PROPOSED
    proposed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
    rejection_reason: str = ""
    confidence: float = 0.0
    hypothesis_id: Optional[str] = None


class SuggestionOutcome(BaseModel):
    """Measured impact of an implemented suggestion."""

    suggestion_id: str
    implemented_date: str  # YYYY-MM-DD
    pnl_delta_7d: float = 0.0
    pnl_delta_30d: float = 0.0
    win_rate_delta_7d: float = 0.0
    win_rate_delta_30d: float = 0.0
    drawdown_delta_7d: float = 0.0
    drawdown_delta_30d: float = 0.0
    measured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def net_positive_7d(self) -> bool:
        return self.pnl_delta_7d > 0
