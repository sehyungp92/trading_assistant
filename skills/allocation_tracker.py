"""AllocationTracker — records allocation changes to JSONL for historical tracking.

Storage: findings/allocation_history.jsonl (append-only JSONL).
"""
from __future__ import annotations

import json
from pathlib import Path

from schemas.allocation_history import AllocationRecord


class AllocationTracker:
    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir
        self._path = findings_dir / "allocation_history.jsonl"

    def record(self, entry: AllocationRecord) -> None:
        """Append an allocation record to the JSONL file."""
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry.model_dump(mode="json"), default=str) + "\n")

    def load_all(self) -> list[dict]:
        """Read all allocation records."""
        if not self._path.exists():
            return []
        records: list[dict] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
