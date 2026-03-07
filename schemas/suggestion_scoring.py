# schemas/suggestion_scoring.py
"""Category-level scoring schemas for suggestion track record analysis."""
from __future__ import annotations

from pydantic import BaseModel


class CategoryScore(BaseModel):
    """Track record for a (bot_id, category) pair."""

    bot_id: str
    category: str
    win_rate: float = 0.0
    avg_pnl_delta: float = 0.0
    sample_size: int = 0
    confidence_multiplier: float = 1.0


class CategoryScorecard(BaseModel):
    """Aggregated category-level scores across all bots."""

    scores: list[CategoryScore] = []

    def get_score(self, bot_id: str, category: str) -> CategoryScore | None:
        for s in self.scores:
            if s.bot_id == bot_id and s.category == category:
                return s
        return None
