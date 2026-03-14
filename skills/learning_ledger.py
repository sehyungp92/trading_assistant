# skills/learning_ledger.py
"""Learning ledger — JSONL-backed experiment log (autoresearch's results.tsv equivalent).

Tracks weekly ground truth snapshots, composite score deltas, and lessons learned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from schemas.learning_ledger import GroundTruthSnapshot, LearningLedgerEntry


class LearningLedger:
    """JSONL-backed weekly experiment log."""

    def __init__(self, findings_dir: Path) -> None:
        self._path = findings_dir / "learning_ledger.jsonl"
        self._findings_dir = findings_dir

    def record_week(
        self,
        week_start: str,
        week_end: str,
        gt_start: dict[str, GroundTruthSnapshot] | None = None,
        gt_end: dict[str, GroundTruthSnapshot] | None = None,
        bots: list[str] | None = None,
        *,
        composite_delta: dict[str, float] | None = None,
        net_improvement: bool | None = None,
        suggestions_proposed: int = 0,
        suggestions_accepted: int = 0,
        suggestions_implemented: int = 0,
        experiments_concluded: int = 0,
        discoveries_found: int = 0,
        what_worked: list[str] | None = None,
        what_failed: list[str] | None = None,
        lessons_for_next_week: list[str] | None = None,
    ) -> LearningLedgerEntry:
        """Record a week's learning data. Deduplicates by entry_id."""
        gt_start = gt_start or {}
        gt_end = gt_end or {}

        entry_id = hashlib.sha256(
            f"{week_start}:{week_end}".encode()
        ).hexdigest()[:12]

        # Check dedup
        existing = self._load_all()
        for e in existing:
            if e.get("entry_id") == entry_id:
                return LearningLedgerEntry(**e)

        # Compute composite deltas if not provided
        if composite_delta is None:
            composite_delta = {}
            all_bots = set(gt_start) | set(gt_end)
            for bot_id in all_bots:
                start_score = gt_start.get(bot_id)
                end_score = gt_end.get(bot_id)
                if start_score and end_score:
                    composite_delta[bot_id] = round(
                        end_score.composite_score - start_score.composite_score, 4,
                    )

        if net_improvement is None:
            net_improvement = sum(composite_delta.values()) > 0 if composite_delta else False

        entry = LearningLedgerEntry(
            entry_id=entry_id,
            week_start=week_start,
            week_end=week_end,
            ground_truth_start={
                k: v for k, v in gt_start.items()
            },
            ground_truth_end={
                k: v for k, v in gt_end.items()
            },
            composite_delta=composite_delta,
            net_improvement=net_improvement,
            suggestions_proposed=suggestions_proposed,
            suggestions_accepted=suggestions_accepted,
            suggestions_implemented=suggestions_implemented,
            experiments_concluded=experiments_concluded,
            discoveries_found=discoveries_found,
            what_worked=what_worked or [],
            what_failed=what_failed or [],
            lessons_for_next_week=lessons_for_next_week or [],
        )

        # Persist
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry

    def get_trend(self, weeks: int = 12) -> dict[str, list[float]]:
        """Return composite_score per bot over the last N weeks."""
        entries = self._load_all()
        # Sort by week_start ascending
        entries.sort(key=lambda e: e.get("week_start", ""))
        recent = entries[-weeks:] if len(entries) > weeks else entries

        trend: dict[str, list[float]] = {}
        for entry in recent:
            gt_end = entry.get("ground_truth_end", {})
            for bot_id, snapshot in gt_end.items():
                score = snapshot.get("composite_score", 0.5) if isinstance(snapshot, dict) else 0.5
                trend.setdefault(bot_id, []).append(score)

        return trend

    def get_lessons(self, weeks: int = 4) -> list[str]:
        """Return aggregated lessons from the last N weeks."""
        entries = self._load_all()
        entries.sort(key=lambda e: e.get("week_start", ""))
        recent = entries[-weeks:] if len(entries) > weeks else entries

        lessons: list[str] = []
        seen: set[str] = set()
        for entry in reversed(recent):  # most recent first
            for lesson in entry.get("lessons_for_next_week", []):
                normalized = lesson.strip().lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    lessons.append(lesson)
        return lessons

    def get_latest(self) -> LearningLedgerEntry | None:
        """Return the most recent ledger entry, or None."""
        entries = self._load_all()
        if not entries:
            return None
        entries.sort(key=lambda e: e.get("week_start", ""))
        return LearningLedgerEntry(**entries[-1])

    def _load_all(self) -> list[dict]:
        """Load all ledger entries from JSONL."""
        if not self._path.exists():
            return []
        entries: list[dict] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
