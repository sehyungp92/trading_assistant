# analysis/context_builder.py
"""Generic context builder — DRY policy and corrections loading for all prompt assemblers."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from schemas.memory import MemoryIndex
from schemas.prompt_package import PromptPackage

_POLICY_FILES = ["agents.md", "trading_rules.md", "soul.md"]
_FINDINGS_MAX_AGE_DAYS = 90
_FINDINGS_MAX_ENTRIES = 50


def _parse_timestamp(entry: dict) -> datetime | None:
    """Try to parse a timestamp from common fields."""
    for key in ("timestamp", "created_at", "date"):
        val = entry.get(key)
        if val and isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                pass
    return None


def _filter_by_bot(entries: list[dict], bot_id: str) -> list[dict]:
    """Filter entries by bot_id. Keeps entries that match or have no bot_id field."""
    if not bot_id:
        return entries
    result = []
    for entry in entries:
        entry_bot = entry.get("bot_id", "") or entry.get("target_id", "")
        # Keep entries that match the bot_id or are bot-agnostic (no bot_id field)
        if not entry_bot or bot_id in entry_bot:
            result.append(entry)
    return result


def _apply_temporal_window(
    entries: list[dict],
    max_age_days: int = _FINDINGS_MAX_AGE_DAYS,
    max_entries: int = _FINDINGS_MAX_ENTRIES,
) -> list[dict]:
    """Sort by recency, exclude entries older than max_age_days, cap at max_entries."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    # Separate entries with and without timestamps
    with_ts: list[tuple[datetime, dict]] = []
    without_ts: list[dict] = []

    for entry in entries:
        ts = _parse_timestamp(entry)
        if ts is not None:
            # Make timezone-aware if naive
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                with_ts.append((ts, entry))
        else:
            without_ts.append(entry)

    # Sort by timestamp descending (most recent first)
    with_ts.sort(key=lambda x: x[0], reverse=True)
    result = [e for _, e in with_ts] + without_ts

    return result[:max_entries]


class ContextBuilder:
    """Loads shared context (policies, corrections, metadata) used by all assemblers."""

    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = memory_dir

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    def build_system_prompt(self) -> str:
        """Load policy files from memory/policies/v1/ into a system prompt."""
        parts: list[str] = []
        policy_dir = self._memory_dir / "policies" / "v1"
        for name in _POLICY_FILES:
            path = policy_dir / name
            if path.exists():
                parts.append(f"--- {name} ---\n{path.read_text()}")
        return "\n\n".join(parts)

    def load_corrections(self, bot_id: str = "") -> list[dict]:
        """Load manual corrections from findings/corrections.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        If bot_id is provided, only returns corrections relevant to that bot.
        """
        corrections_path = self._memory_dir / "findings" / "corrections.jsonl"
        if not corrections_path.exists():
            return []
        corrections: list[dict] = []
        for line in corrections_path.read_text().strip().splitlines():
            if line.strip():
                corrections.append(json.loads(line))
        filtered = _filter_by_bot(corrections, bot_id) if bot_id else corrections
        return _apply_temporal_window(filtered)

    def load_failure_log(self, bot_id: str = "") -> list[dict]:
        """Load failure log entries from findings/failure-log.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        If bot_id is provided, only returns entries relevant to that bot.
        """
        path = self._memory_dir / "findings" / "failure-log.jsonl"
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))
        filtered = _filter_by_bot(entries, bot_id) if bot_id else entries
        return _apply_temporal_window(filtered)

    def load_rejected_suggestions(self) -> list[dict]:
        """Load rejected suggestions from findings/suggestions.jsonl."""
        path = self._memory_dir / "findings" / "suggestions.jsonl"
        if not path.exists():
            return []
        rejected: list[dict] = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "rejected":
                    rejected.append(rec)
        return rejected

    def load_outcome_measurements(self) -> list[dict]:
        """Load outcome measurements from findings/outcomes.jsonl."""
        path = self._memory_dir / "findings" / "outcomes.jsonl"
        if not path.exists():
            return []
        outcomes: list[dict] = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                outcomes.append(json.loads(line))
        return outcomes

    def load_allocation_history(self) -> list[dict]:
        """Load allocation history from findings/allocation_history.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        """
        path = self._memory_dir / "findings" / "allocation_history.jsonl"
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text().strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))
        return _apply_temporal_window(entries)

    def list_policy_files(self) -> list[str]:
        """List paths to included policy files (for context_files tracking)."""
        files: list[str] = []
        policy_dir = self._memory_dir / "policies" / "v1"
        for name in _POLICY_FILES:
            path = policy_dir / name
            if path.exists():
                files.append(str(path))
        return files

    def runtime_metadata(self) -> dict:
        """Return runtime metadata for the prompt package."""
        now = datetime.now(timezone.utc)
        return {
            "assembled_at": now.isoformat(),
            "timezone": "UTC",
        }

    @staticmethod
    def check_data_availability(
        index: MemoryIndex | None, bot_id: str, date: str,
    ) -> dict:
        """Check if curated data exists for a bot on a given date.

        Returns dict with: has_curated (bool), available_dates (list[str]).
        If index is None, returns unknown state.
        """
        if index is None:
            return {"has_curated": None, "available_dates": []}

        bot_dates = index.curated_dates_by_bot.get(bot_id, [])
        return {
            "has_curated": date in bot_dates,
            "available_dates": bot_dates,
        }

    def load_session_history(self, session_store, agent_type: str, days: int = 7) -> str:
        """Load recent session summaries as formatted text.

        Args:
            session_store: SessionStore instance.
            agent_type: Type of agent to load history for.
            days: Number of days to look back.

        Returns:
            Formatted string summarizing recent sessions, or empty string.
        """
        try:
            sessions = session_store.get_recent_sessions(agent_type, days=days)
        except Exception:
            return ""
        if not sessions:
            return ""

        lines = [f"Recent {agent_type} sessions (last {days} days):"]
        for s in sessions[:20]:  # cap to avoid context bloat
            duration = s.get("duration_ms", 0)
            summary = s.get("response_summary", "")[:100]
            lines.append(f"- {s.get('date', '?')}: {duration}ms — {summary}")
        return "\n".join(lines)

    def load_pattern_library(self, bot_id: str = "") -> list[dict]:
        """Load cross-bot pattern library entries.

        If bot_id is provided, only returns patterns relevant to that bot.
        """
        try:
            from skills.pattern_library import PatternLibrary

            lib = PatternLibrary(self._memory_dir / "findings")
            if bot_id:
                entries = lib.load_for_bot(bot_id)
            else:
                entries = lib.load_active()
            return [e.model_dump(mode="json") for e in entries]
        except Exception:
            return []

    def load_contradictions(
        self, date: str, bots: list[str], curated_dir: Path,
    ) -> list[dict]:
        """Load temporal contradictions across recent daily reports.

        Returns list of ContradictionItem dicts for prompt injection.
        """
        try:
            from skills.contradiction_detector import ContradictionDetector

            detector = ContradictionDetector(
                date=date, bots=bots, curated_dir=curated_dir,
            )
            report = detector.detect()
            return [item.model_dump(mode="json") for item in report.items]
        except Exception:
            return []

    def load_signal_factor_history(
        self, bot_id: str, date: str, findings_dir: Path,
    ) -> dict:
        """Load rolling signal factor analysis for a bot.

        Returns SignalFactorRollingReport as dict, or empty dict if insufficient data.
        """
        try:
            from skills.signal_factor_tracker import SignalFactorTracker

            tracker = SignalFactorTracker(findings_dir)
            report = tracker.compute_rolling(bot_id, date)
            if not report.factors:
                return {}
            return report.model_dump(mode="json")
        except Exception:
            return {}

    def load_correction_patterns(self) -> list[dict]:
        """Load extracted correction patterns from findings/correction_patterns.jsonl."""
        path = self._memory_dir / "findings" / "correction_patterns.jsonl"
        if not path.exists():
            return []
        patterns: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    patterns.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return patterns

    def load_consolidated_patterns(self) -> str:
        """Load patterns_consolidated.md if it exists."""
        path = self._memory_dir / "findings" / "patterns_consolidated.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def base_package(self, session_store=None, agent_type: str = "") -> PromptPackage:
        """Build a PromptPackage pre-filled with system prompt, corrections, and metadata.

        Args:
            session_store: Optional SessionStore for loading session history.
            agent_type: Agent type for session history filtering.
        """
        failure_log = self.load_failure_log()
        rejected_suggestions = self.load_rejected_suggestions()
        outcome_measurements = self.load_outcome_measurements()
        allocation_history = self.load_allocation_history()
        consolidated_patterns = self.load_consolidated_patterns()
        data: dict = {}
        if failure_log:
            data["failure_log"] = failure_log
        if rejected_suggestions:
            data["rejected_suggestions"] = rejected_suggestions
        if outcome_measurements:
            data["outcome_measurements"] = outcome_measurements
        if allocation_history:
            data["allocation_history"] = allocation_history
        if consolidated_patterns:
            data["consolidated_patterns"] = consolidated_patterns
        pattern_library = self.load_pattern_library()
        if pattern_library:
            data["pattern_library"] = pattern_library
        correction_patterns = self.load_correction_patterns()
        if correction_patterns:
            data["correction_patterns"] = correction_patterns
        if session_store and agent_type:
            session_history = self.load_session_history(session_store, agent_type)
            if session_history:
                data["session_history"] = session_history
        return PromptPackage(
            system_prompt=self.build_system_prompt(),
            corrections=self.load_corrections(),
            context_files=self.list_policy_files(),
            metadata=self.runtime_metadata(),
            data=data,
        )
