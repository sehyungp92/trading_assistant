# skills/suggestion_tracker.py
"""SuggestionTracker — records suggestions, tracks status, and measures outcomes.

Storage: Two JSONL files in store_dir:
  - suggestions.jsonl — one record per suggestion with lifecycle status
  - outcomes.jsonl — measured impacts of implemented suggestions
"""
from __future__ import annotations

import json
import threading
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
        self._lock = threading.Lock()

    def record(self, suggestion: SuggestionRecord) -> bool:
        """Record a suggestion. Returns False if suggestion_id already exists (dedup)."""
        with self._lock:
            existing = self.load_all()
            existing_ids = {s.get("suggestion_id") for s in existing}
            if suggestion.suggestion_id in existing_ids:
                return False
            self._store_dir.mkdir(parents=True, exist_ok=True)
            with open(self._suggestions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(suggestion.model_dump(mode="json"), default=str) + "\n")
            return True

    def reject(self, suggestion_id: str, reason: str = "") -> None:
        self._update_status(suggestion_id, SuggestionStatus.REJECTED, reason=reason)

    def accept(
        self,
        suggestion_id: str,
        approval_request_id: str | None = None,
    ) -> None:
        self._update_status(
            suggestion_id,
            SuggestionStatus.ACCEPTED,
            approval_request_id=approval_request_id,
        )

    def mark_merged(
        self,
        suggestion_id: str,
        pr_url: str | None = None,
        deployment_id: str | None = None,
    ) -> None:
        self._update_status(
            suggestion_id,
            SuggestionStatus.MERGED,
            pr_url=pr_url,
            deployment_id=deployment_id,
        )

    def mark_deployed(
        self,
        suggestion_id: str,
        deployment_id: str | None = None,
    ) -> None:
        self._update_status(
            suggestion_id,
            SuggestionStatus.DEPLOYED,
            deployment_id=deployment_id,
        )

    def mark_measured(self, suggestion_id: str) -> None:
        self._update_status(suggestion_id, SuggestionStatus.MEASURED)

    def implement(self, suggestion_id: str) -> None:
        """Legacy helper retained for historical tooling.

        .. deprecated::
            Use ``accept()`` followed by ``mark_deployed()`` instead.
        """
        import warnings

        warnings.warn(
            "implement() is deprecated; use accept() + mark_deployed()",
            DeprecationWarning,
            stacklevel=2,
        )
        self._update_status(suggestion_id, SuggestionStatus.IMPLEMENTED)

    def record_outcome(self, outcome: SuggestionOutcome) -> None:
        with self._lock:
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

    def get_deployed_portfolio_count(self) -> int:
        """Count DEPLOYED suggestions with bot_id='PORTFOLIO'.

        Used to enforce concurrent deployment limit (max 1 portfolio change at a time).
        """
        suggestions = self.load_all()
        return sum(
            1 for s in suggestions
            if s.get("bot_id") == "PORTFOLIO"
            and s.get("status") == SuggestionStatus.DEPLOYED.value
        )

    def get_last_portfolio_proposal_date(self, proposal_type: str = "") -> str | None:
        """Get the date of the most recent portfolio proposal (any status).

        Args:
            proposal_type: Optional filter by category (e.g., 'portfolio_allocation').

        Returns:
            ISO date string of most recent proposal, or None if no history.
        """
        suggestions = self.load_all()
        portfolio = [
            s for s in suggestions
            if s.get("bot_id") == "PORTFOLIO"
        ]
        if proposal_type:
            portfolio = [s for s in portfolio if s.get("category") == proposal_type]
        if not portfolio:
            return None
        # Find most recent by proposed_at, created_at, or timestamp
        dates = []
        for s in portfolio:
            for key in ("proposed_at", "created_at", "timestamp", "accepted_at"):
                val = s.get(key)
                if val:
                    dates.append(val)
                    break
        return max(dates) if dates else None

    def _update_status(
        self,
        suggestion_id: str,
        status: SuggestionStatus,
        reason: str = "",
        approval_request_id: str | None = None,
        deployment_id: str | None = None,
        pr_url: str | None = None,
    ) -> None:
        from skills._atomic_write import atomic_rewrite_jsonl

        with self._lock:
            records = self.load_all()
            now = datetime.now(timezone.utc).isoformat()
            for rec in records:
                if rec["suggestion_id"] == suggestion_id:
                    rec["status"] = status.value
                    if status == SuggestionStatus.ACCEPTED:
                        rec["accepted_at"] = now
                    elif status == SuggestionStatus.MERGED:
                        rec["merged_at"] = now
                    elif status == SuggestionStatus.DEPLOYED:
                        rec["deployed_at"] = now
                    elif status == SuggestionStatus.MEASURED:
                        rec["measured_at"] = now
                    if reason:
                        rec["rejection_reason"] = reason
                    if approval_request_id:
                        rec["approval_request_id"] = approval_request_id
                    if deployment_id:
                        rec["deployment_id"] = deployment_id
                    if pr_url:
                        rec["pr_url"] = pr_url
                    if status in {
                        SuggestionStatus.REJECTED,
                        SuggestionStatus.MEASURED,
                    }:
                        rec["resolved_at"] = now
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
