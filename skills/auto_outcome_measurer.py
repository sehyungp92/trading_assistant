"""Automated outcome measurement for implemented suggestions.

Detects parameter changes from WFO reports, computes before/after metrics
from curated daily summaries, and records outcomes via SuggestionTracker.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from schemas.outcome_measurement import OutcomeMeasurement


class AutoOutcomeMeasurer:
    """Measures suggestion outcomes by comparing pre/post performance."""

    def __init__(
        self,
        curated_dir: Path,
        wfo_dir: Path | None = None,
    ) -> None:
        self._curated_dir = curated_dir
        self._wfo_dir = wfo_dir

    def measure(
        self,
        suggestion_id: str,
        bot_id: str,
        implemented_date: str,
        before_days: int = 7,
        after_days: int = 7,
    ) -> OutcomeMeasurement | None:
        """Compare performance before and after a suggestion was implemented."""
        impl_date = datetime.strptime(implemented_date, "%Y-%m-%d")
        today = datetime.now(timezone.utc)

        before_start = impl_date - timedelta(days=before_days)
        after_end = impl_date + timedelta(days=after_days)

        before_summaries = self._load_summaries(bot_id, before_start, impl_date)
        after_summaries = self._load_summaries(bot_id, impl_date, after_end)

        if not before_summaries or not after_summaries:
            return None

        before_pnl = sum(self._summary_pnl(s) for s in before_summaries)
        after_pnl = sum(self._summary_pnl(s) for s in after_summaries)

        before_wins = sum(s.get("win_count", 0) for s in before_summaries)
        before_total = sum(s.get("total_trades", 0) for s in before_summaries)
        after_wins = sum(s.get("win_count", 0) for s in after_summaries)
        after_total = sum(s.get("total_trades", 0) for s in after_summaries)

        before_wr = before_wins / before_total if before_total > 0 else 0
        after_wr = after_wins / after_total if after_total > 0 else 0

        before_dd = max((s.get("max_drawdown_pct", 0) for s in before_summaries), default=0)
        after_dd = max((s.get("max_drawdown_pct", 0) for s in after_summaries), default=0)

        return OutcomeMeasurement(
            suggestion_id=suggestion_id,
            implemented_date=implemented_date,
            measurement_date=today.strftime("%Y-%m-%d"),
            window_days=after_days,
            pnl_before=before_pnl,
            pnl_after=after_pnl,
            win_rate_before=before_wr,
            win_rate_after=after_wr,
            drawdown_before=before_dd,
            drawdown_after=after_dd,
        )

    def detect_parameter_changes(self, bot_id: str) -> list[dict]:
        """Detect parameter changes from WFO reports over time."""
        if not self._wfo_dir or not self._wfo_dir.exists():
            return []

        reports = []
        for f in sorted(self._wfo_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                if data.get("bot_id") == bot_id:
                    reports.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        changes = []
        for i in range(1, len(reports)):
            prev = reports[i - 1].get("suggested_params", {})
            curr = reports[i].get("suggested_params", {})
            if prev != curr:
                changes.append({
                    "date": reports[i].get("date", "unknown"),
                    "previous_params": prev,
                    "new_params": curr,
                })
        return changes

    @staticmethod
    def _summary_pnl(summary: dict) -> float:
        """Prefer net PnL because it reflects fees/slippage in live trading."""
        value = summary.get("net_pnl")
        if value is None:
            value = summary.get("gross_pnl", 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _load_summaries(
        self, bot_id: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Load daily summaries for a date range."""
        summaries = []
        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            summary_path = self._curated_dir / date_str / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    summaries.append(json.loads(summary_path.read_text()))
                except (json.JSONDecodeError, OSError):
                    pass
            current += timedelta(days=1)
        return summaries
