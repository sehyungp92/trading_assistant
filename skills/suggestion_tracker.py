# skills/suggestion_tracker.py
"""SuggestionTracker — records suggestions, tracks status, and measures outcomes.

Storage: Two JSONL files in store_dir:
  - suggestions.jsonl — one record per suggestion with lifecycle status
  - outcomes.jsonl — measured impacts of implemented suggestions
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from schemas.suggestion_tracking import (
    SuggestionOutcome,
    SuggestionRecord,
    SuggestionStatus,
)


class SuggestionTracker:
    def __init__(self, store_dir: Path) -> None:
        self._store_dir = store_dir
        self._suggestions_path = store_dir / "suggestions.jsonl"
        self._outcomes_path = store_dir / "outcomes.jsonl"

    def record(self, suggestion: SuggestionRecord) -> bool:
        """Record a suggestion. Returns False if suggestion_id already exists (dedup)."""
        existing = self.load_all()
        existing_ids = {s.get("suggestion_id") for s in existing}
        if suggestion.suggestion_id in existing_ids:
            return False
        self._store_dir.mkdir(parents=True, exist_ok=True)
        with open(self._suggestions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(suggestion.model_dump(mode="json"), default=str) + "\n")
        return True

    def reject(self, suggestion_id: str, reason: str = "") -> None:
        self._update_status(suggestion_id, SuggestionStatus.REJECTED, reason)

    def implement(self, suggestion_id: str) -> None:
        self._update_status(suggestion_id, SuggestionStatus.IMPLEMENTED)

    def record_outcome(self, outcome: SuggestionOutcome) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        with open(self._outcomes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(outcome.model_dump(mode="json"), default=str) + "\n")

    def load_all(self) -> list[dict]:
        return self._read_jsonl(self._suggestions_path)

    def load_outcomes(self) -> list[dict]:
        return self._read_jsonl(self._outcomes_path)

    def get_rejected(self, bot_id: str | None = None) -> list[dict]:
        suggestions = self.load_all()
        rejected = [s for s in suggestions if s.get("status") == SuggestionStatus.REJECTED.value]
        if bot_id:
            rejected = [s for s in rejected if s.get("bot_id") == bot_id]
        return rejected

    def _update_status(
        self, suggestion_id: str, status: SuggestionStatus, reason: str = ""
    ) -> None:
        from skills._atomic_write import atomic_rewrite_jsonl

        records = self.load_all()
        for rec in records:
            if rec["suggestion_id"] == suggestion_id:
                rec["status"] = status.value
                if reason:
                    rec["rejection_reason"] = reason
                rec["resolved_at"] = datetime.now(timezone.utc).isoformat()
        atomic_rewrite_jsonl(self._suggestions_path, records)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        import logging
        _logger = logging.getLogger(__name__)
        records: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    _logger.warning(
                        "Skipping malformed JSON at %s:%d", path.name, line_no,
                    )
                    continue
        return records
