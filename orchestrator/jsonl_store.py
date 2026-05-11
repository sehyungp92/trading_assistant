"""Small shared helpers for JSONL append stores."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable


_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[Path, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    key = path.expanduser().resolve(strict=False)
    with _LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Append JSON-serializable records under a process-wide per-path lock."""
    path = Path(path)
    records = list(records)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_for(path), open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")
