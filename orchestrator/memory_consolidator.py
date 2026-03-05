from __future__ import annotations
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from schemas.memory import ConsolidationSummary, MemoryIndex, PatternCount

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    """Consolidates JSONL findings into a summary when entries exceed a threshold."""

    def __init__(
        self,
        findings_dir: str | Path,
        threshold: int = 100,
        base_dir: str | Path | None = None,
    ) -> None:
        self._findings_dir = Path(findings_dir)
        self._threshold = threshold
        # base_dir is the project root for scanning curated/sessions/runs/heartbeats.
        # If not provided, assume findings_dir is inside {base_dir}/memory/findings/
        # and walk up to find the project root.
        if base_dir is not None:
            self._base_dir = Path(base_dir)
        else:
            self._base_dir = self._findings_dir.parent

    def needs_consolidation(self, filename: str = "corrections.jsonl") -> bool:
        """Check if a findings file exceeds the consolidation threshold."""
        path = self._findings_dir / filename
        if not path.exists():
            return False
        line_count = sum(1 for line in path.read_text().strip().splitlines() if line.strip())
        return line_count > self._threshold

    def consolidate(self, filename: str = "corrections.jsonl") -> ConsolidationSummary | None:
        """Aggregate a JSONL file into a summary. Returns None if below threshold."""
        path = self._findings_dir / filename
        if not path.exists():
            return None

        entries = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if len(entries) <= self._threshold:
            return None

        # Count by various dimensions
        bot_counter: Counter = Counter()
        error_type_counter: Counter = Counter()
        root_cause_counter: Counter = Counter()

        for entry in entries:
            if "bot_id" in entry:
                bot_counter[entry["bot_id"]] += 1
            if "error_type" in entry:
                error_type_counter[entry["error_type"]] += 1
            if "correction_type" in entry:
                error_type_counter[entry["correction_type"]] += 1
            if "root_cause" in entry:
                root_cause_counter[entry["root_cause"]] += 1
            for rc in entry.get("root_causes", []):
                root_cause_counter[rc] += 1

        summary = ConsolidationSummary(
            source_file=filename,
            total_entries=len(entries),
            top_bots=[PatternCount(category="bot", key=k, count=v) for k, v in bot_counter.most_common(10)],
            top_error_types=[PatternCount(category="error_type", key=k, count=v) for k, v in error_type_counter.most_common(10)],
            top_root_causes=[PatternCount(category="root_cause", key=k, count=v) for k, v in root_cause_counter.most_common(10)],
        )

        # Write consolidated summary
        output_path = self._findings_dir / "patterns_consolidated.md"
        self._write_markdown(summary, output_path)

        # Update memory index
        self.rebuild_index()

        return summary

    def rebuild_index(self) -> MemoryIndex:
        """Scan all data directories and rebuild memory/index.json.

        This is cheap (directory listing only, no file reads) and provides
        O(1) existence checks for "does data exist for X?" queries.
        """
        index = MemoryIndex()

        # 1. Curated data: data/curated/{date}/{bot_id}/
        curated_dir = self._base_dir / "data" / "curated"
        all_curated_dates: set[str] = set()
        if curated_dir.is_dir():
            for date_dir in sorted(curated_dir.iterdir()):
                if date_dir.is_dir() and _is_date_dir(date_dir.name):
                    all_curated_dates.add(date_dir.name)
                    for bot_dir in sorted(date_dir.iterdir()):
                        if bot_dir.is_dir() and bot_dir.name != "weekly":
                            index.curated_dates_by_bot.setdefault(
                                bot_dir.name, []
                            ).append(date_dir.name)

        # 2. Weekly curated: data/curated/weekly/{week_start}/
        weekly_dir = curated_dir / "weekly"
        if weekly_dir.is_dir():
            index.weekly_dates = sorted(
                d.name for d in weekly_dir.iterdir()
                if d.is_dir() and _is_date_dir(d.name)
            )

        # 3. Sessions: .assistant/sessions/{agent_type}/{date}/
        sessions_dir = self._base_dir / ".assistant" / "sessions"
        total_sessions = 0
        if sessions_dir.is_dir():
            for agent_dir in sorted(sessions_dir.iterdir()):
                if agent_dir.is_dir():
                    dates = sorted(
                        d.name for d in agent_dir.iterdir()
                        if d.is_dir() and (d / "sessions.jsonl").exists()
                    )
                    if dates:
                        index.sessions_by_agent_type[agent_dir.name] = dates
                        total_sessions += len(dates)

        # 4. Runs: runs/{run_id}/
        runs_dir = self._base_dir / "runs"
        if runs_dir.is_dir():
            index.run_ids = sorted(
                d.name for d in runs_dir.iterdir() if d.is_dir()
            )

        # 5. Findings: count lines per JSONL file in findings_dir
        total_findings = 0
        if self._findings_dir.is_dir():
            for f in self._findings_dir.glob("*.jsonl"):
                try:
                    count = sum(
                        1 for line in f.read_text(encoding="utf-8").strip().splitlines()
                        if line.strip()
                    )
                    index.findings_counts[f.name] = count
                    total_findings += count
                except OSError:
                    pass

        # 6. Heartbeats: heartbeats/{bot_id}.heartbeat
        heartbeats_dir = self._base_dir / "heartbeats"
        if heartbeats_dir.is_dir():
            for hb in heartbeats_dir.glob("*.heartbeat"):
                try:
                    index.heartbeat_last_seen[hb.stem] = hb.read_text(encoding="utf-8").strip()
                except OSError:
                    pass

        # Summary stats
        index.total_sessions = total_sessions
        index.total_findings = total_findings
        index.total_curated_days = len(all_curated_dates)
        index.last_consolidated = datetime.now(timezone.utc).isoformat()

        # Write index
        index_path = self._base_dir / "memory" / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps(index.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            "Memory index rebuilt: %d curated days, %d sessions, %d findings, %d runs",
            index.total_curated_days, index.total_sessions,
            index.total_findings, len(index.run_ids),
        )
        return index

    @staticmethod
    def load_index(base_dir: str | Path) -> MemoryIndex | None:
        """Load the memory index from disk. Returns None if not found."""
        path = Path(base_dir) / "memory" / "index.json"
        if not path.exists():
            return None
        try:
            return MemoryIndex.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            return None

    def _write_markdown(self, summary: ConsolidationSummary, path: Path) -> None:
        lines = [
            f"# Memory Consolidation Summary",
            f"",
            f"Consolidated at: {summary.consolidated_at.isoformat()}",
            f"Source: {summary.source_file}",
            f"Total entries: {summary.total_entries}",
            f"",
        ]

        if summary.top_bots:
            lines.append("## Top Bots")
            for p in summary.top_bots:
                lines.append(f"- {p.key}: {p.count}")
            lines.append("")

        if summary.top_error_types:
            lines.append("## Top Error Types")
            for p in summary.top_error_types:
                lines.append(f"- {p.key}: {p.count}")
            lines.append("")

        if summary.top_root_causes:
            lines.append("## Top Root Causes")
            for p in summary.top_root_causes:
                lines.append(f"- {p.key}: {p.count}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date_dir(name: str) -> bool:
    """Check if a directory name looks like a YYYY-MM-DD date."""
    return bool(_DATE_RE.match(name))
