"""Schema for tracking historical allocation changes."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class AllocationSource(str, Enum):
    MANUAL = "manual"
    SUGGESTED = "suggested"


class AllocationRecord(BaseModel):
    """Single point-in-time allocation for a bot/strategy."""
    date: str  # YYYY-MM-DD
    bot_id: str
    strategy_id: str = ""
    allocation_pct: float
    unit_risk_pct: float = 0.0
    heat_cap_r: float = 0.0
    source: AllocationSource = AllocationSource.MANUAL
    reason: str = ""
