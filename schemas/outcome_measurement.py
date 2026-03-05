"""Automated suggestion outcome measurement schemas."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field


class Verdict(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class OutcomeMeasurement(BaseModel):
    """Before/after comparison for a implemented suggestion."""
    suggestion_id: str
    implemented_date: str
    measurement_date: str
    window_days: int
    pnl_before: float = 0.0
    pnl_after: float = 0.0
    win_rate_before: float = 0.0
    win_rate_after: float = 0.0
    drawdown_before: float = 0.0
    drawdown_after: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pnl_delta(self) -> float:
        return self.pnl_after - self.pnl_before

    @computed_field  # type: ignore[prop-decorator]
    @property
    def verdict(self) -> Verdict:
        # Positive if PnL improved by >10% and win rate didn't drop significantly
        if self.pnl_before == 0:
            return Verdict.NEUTRAL
        pnl_change = self.pnl_delta / abs(self.pnl_before)
        wr_change = self.win_rate_after - self.win_rate_before
        if pnl_change > 0.1 and wr_change >= -0.05:
            return Verdict.POSITIVE
        elif pnl_change < -0.1 or wr_change < -0.1:
            return Verdict.NEGATIVE
        return Verdict.NEUTRAL
