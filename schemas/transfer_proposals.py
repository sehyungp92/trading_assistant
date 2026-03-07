# schemas/transfer_proposals.py
"""Transfer proposal schemas — cross-bot pattern application candidates."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class TransferProposal(BaseModel):
    """A proposal to transfer a validated pattern from one bot to another."""

    pattern_id: str
    source_bot: str
    target_bot: str
    pattern_title: str
    category: str = ""
    compatibility_score: float = 0.0  # 0–1
    rationale: str = ""


class TransferOutcome(BaseModel):
    """Measured result of a pattern transfer to a target bot."""

    pattern_id: str
    source_bot: str
    target_bot: str
    transferred_at: str = ""
    measured_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    pnl_delta_7d: float = 0.0
    win_rate_delta_7d: float = 0.0
    verdict: Literal["positive", "neutral", "negative"] = "neutral"
