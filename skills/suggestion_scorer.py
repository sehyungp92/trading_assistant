# skills/suggestion_scorer.py
"""SuggestionScorer — computes per-category success rates from measured outcomes.

Reads outcomes.jsonl and suggestions.jsonl to build a CategoryScorecard that
tells the system which suggestion categories have positive track records.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from schemas.agent_response import CATEGORY_TO_TIER
from schemas.suggestion_scoring import CategoryScore, CategoryScorecard


# Build reverse mapping: tier → category, plus identity mappings for categories.
# "parameter" is an ambiguous tier (maps to exit_timing, stop_loss, signal, position_sizing)
# — we default it to "filter_threshold" for backward compatibility with existing data.
_TIER_TO_CATEGORY: dict[str, str] = {"parameter": "filter_threshold"}
for _cat, _tier in CATEGORY_TO_TIER.items():
    _TIER_TO_CATEGORY.setdefault(_tier, _cat)  # first category for that tier wins
    _TIER_TO_CATEGORY[_cat] = _cat  # identity for categories that are also tier names
# Additional legacy mappings
_TIER_TO_CATEGORY.setdefault("strategy_variant", "signal")
_TIER_TO_CATEGORY.setdefault("hypothesis", "structural")


class SuggestionScorer:
    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir

    def compute_scorecard(self) -> CategoryScorecard:
        """Compute per-(bot_id, category) win rates from outcomes and suggestions."""
        suggestions = self._load_suggestions()
        outcomes = self._load_outcomes()

        if not outcomes:
            return CategoryScorecard()

        # Build suggestion_id → (bot_id, category) mapping
        id_to_info: dict[str, tuple[str, str]] = {}
        for s in suggestions:
            sid = s.get("suggestion_id", "")
            bot_id = s.get("bot_id", "")
            category = s.get("category", "")
            if not category:
                tier = s.get("tier", "")
                category = _TIER_TO_CATEGORY.get(tier, tier)
            if sid:
                id_to_info[sid] = (bot_id, category)

        # Group outcomes by (bot_id, category)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for outcome in outcomes:
            sid = outcome.get("suggestion_id", "")
            info = id_to_info.get(sid)
            if info:
                groups[info].append(outcome)

        scores: list[CategoryScore] = []
        for (bot_id, category), group_outcomes in groups.items():
            positive = sum(1 for o in group_outcomes if o.get("pnl_delta_7d", 0) > 0)
            total = len(group_outcomes)
            win_rate = positive / total if total > 0 else 0.0
            avg_pnl = (
                sum(o.get("pnl_delta_7d", 0) for o in group_outcomes) / total
                if total > 0 else 0.0
            )
            # confidence_multiplier: penalize poor track records, don't penalize insufficient data
            if total >= 5:
                multiplier = max(0.3, win_rate)
            else:
                multiplier = 1.0

            scores.append(CategoryScore(
                bot_id=bot_id,
                category=category,
                win_rate=round(win_rate, 3),
                avg_pnl_delta=round(avg_pnl, 4),
                sample_size=total,
                confidence_multiplier=round(multiplier, 3),
            ))

        return CategoryScorecard(scores=scores)

    def _load_suggestions(self) -> list[dict]:
        return self._read_jsonl(self._findings_dir / "suggestions.jsonl")

    def _load_outcomes(self) -> list[dict]:
        return self._read_jsonl(self._findings_dir / "outcomes.jsonl")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records
