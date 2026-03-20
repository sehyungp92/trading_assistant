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
        """Compute per-(bot_id, category) win rates from outcomes and suggestions.

        Applies category_overrides.jsonl confidence multipliers when present.
        """
        suggestions = self._load_suggestions()
        outcomes = self._load_outcomes()
        category_overrides = self._load_category_overrides()

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

        # Group outcomes by (bot_id, category), filtering by measurement quality
        # then deduplicating by suggestion_id within each group (last-write-wins)
        # to prevent double-counting from legacy dual-write patterns.
        _HIGH_QUALITY = {"high", "medium"}
        raw_groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        for outcome in outcomes:
            # Only count outcomes with high or medium measurement quality
            quality = outcome.get("measurement_quality", "high")  # legacy defaults to high
            if quality not in _HIGH_QUALITY:
                continue
            sid = outcome.get("suggestion_id", "")
            info = id_to_info.get(sid)
            if info:
                raw_groups[info][sid] = outcome  # last-write-wins per suggestion_id
        groups: dict[tuple[str, str], list[dict]] = {
            k: list(v.values()) for k, v in raw_groups.items()
        }

        scores: list[CategoryScore] = []
        for (bot_id, category), group_outcomes in groups.items():
            positive = sum(1 for o in group_outcomes if self._is_positive(o))
            total = len(group_outcomes)
            win_rate = positive / total if total > 0 else 0.0
            avg_pnl = (
                sum(self._outcome_pnl_delta(o) for o in group_outcomes) / total
                if total > 0 else 0.0
            )
            # confidence_multiplier: penalize poor track records, don't penalize insufficient data
            override = category_overrides.get((bot_id, category))
            if override is not None:
                multiplier = override
            elif total >= 5:
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

    @staticmethod
    def _is_positive(outcome: dict) -> bool:
        """Check if an outcome is positive, preferring verdict field."""
        verdict = outcome.get("verdict", "")
        if verdict:
            return verdict == "positive"
        # Legacy fallback: check pnl_delta or pnl_delta_7d
        delta = outcome.get("pnl_delta")
        if delta is None:
            delta = outcome.get("pnl_delta_7d", 0)
        try:
            return float(delta) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _outcome_pnl_delta(outcome: dict) -> float:
        """Extract PnL delta from outcome, supporting both schemas."""
        delta = outcome.get("pnl_delta")
        if delta is None:
            delta = outcome.get("pnl_delta_7d", 0)
        try:
            return float(delta)
        except (TypeError, ValueError):
            return 0.0

    def apply_recalibration(self, discard_items: list) -> None:
        """Write category overrides from discard items.

        Discarded categories get confidence_multiplier = 0.3, which
        compute_scorecard() will pick up on next call.
        """
        overrides_path = self._findings_dir / "category_overrides.jsonl"
        overrides_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing overrides
        existing: dict[tuple[str, str], dict] = {}
        if overrides_path.exists():
            for line in overrides_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    try:
                        entry = json.loads(line)
                        key = (entry.get("bot_id", ""), entry.get("category", ""))
                        existing[key] = entry
                    except json.JSONDecodeError:
                        pass

        # Update/add overrides from discard items
        from datetime import datetime, timezone
        for item in discard_items:
            bot_id = item.bot_id if hasattr(item, "bot_id") else item.get("bot_id", "")
            category = item.category if hasattr(item, "category") else item.get("category", "")
            key = (bot_id, category)
            existing[key] = {
                "bot_id": bot_id,
                "category": category,
                "confidence_multiplier": 0.3,
                "reason": item.reason if hasattr(item, "reason") else item.get("reason", "discarded"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        # Rewrite file
        with open(overrides_path, "w", encoding="utf-8") as f:
            for entry in existing.values():
                f.write(json.dumps(entry) + "\n")

    def _load_category_overrides(self) -> dict[tuple[str, str], float]:
        """Load category confidence multiplier overrides."""
        path = self._findings_dir / "category_overrides.jsonl"
        if not path.exists():
            return {}
        overrides: dict[tuple[str, str], float] = {}
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                    key = (entry.get("bot_id", ""), entry.get("category", ""))
                    overrides[key] = entry.get("confidence_multiplier", 1.0)
                except json.JSONDecodeError:
                    pass
        return overrides

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
