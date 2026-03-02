"""Deterministic monitoring loop — no LLM calls.

Runs on a cron schedule (e.g., every 10 minutes) and produces alerts when:
  - Agent tasks are stuck (running beyond timeout)
  - Run folders are missing expected output files
  - VPS sidecar heartbeats are stale
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field

from orchestrator.task_registry import TaskRegistry


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Alert:
    severity: AlertSeverity
    source: str  # which check produced this
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MonitoringCheck:
    """Individual monitoring checks. Compose as needed."""

    def __init__(
        self,
        registry: TaskRegistry | None = None,
        task_timeout_seconds: int = 3600,
        heartbeat_dir: str = "",
        heartbeat_max_age_seconds: int = 7200,
    ) -> None:
        self._registry = registry
        self._task_timeout = task_timeout_seconds
        self._heartbeat_dir = heartbeat_dir
        self._heartbeat_max_age = heartbeat_max_age_seconds

    async def check_stale_tasks(self) -> list[Alert]:
        """Find tasks stuck in RUNNING state beyond timeout."""
        if not self._registry:
            return []

        stale = await self._registry.find_stale(timeout_seconds=self._task_timeout)
        return [
            Alert(
                severity=AlertSeverity.HIGH,
                source="stale_task",
                message=f"Task '{task.id}' ({task.type}) has been running since {task.started_at}",
            )
            for task in stale
        ]

    def check_heartbeats(self) -> list[Alert]:
        """Check VPS sidecar heartbeat freshness."""
        if not self._heartbeat_dir:
            return []

        alerts: list[Alert] = []
        hb_path = Path(self._heartbeat_dir)

        for hb_file in hb_path.glob("*.heartbeat"):
            bot_id = hb_file.stem
            try:
                last_seen = datetime.fromisoformat(hb_file.read_text().strip())
                age = (datetime.now(timezone.utc) - last_seen).total_seconds()
                if age > self._heartbeat_max_age:
                    alerts.append(Alert(
                        severity=AlertSeverity.CRITICAL,
                        source="heartbeat",
                        message=f"Bot '{bot_id}' last heartbeat was {age / 3600:.1f}h ago",
                    ))
            except (ValueError, OSError) as e:
                alerts.append(Alert(
                    severity=AlertSeverity.HIGH,
                    source="heartbeat",
                    message=f"Cannot read heartbeat for '{bot_id}': {e}",
                ))

        return alerts

    def check_run_outputs(self, run_dir: str, expected_files: list[str]) -> list[Alert]:
        """Verify a run folder contains expected output files."""
        alerts: list[Alert] = []
        run_path = Path(run_dir)

        for filename in expected_files:
            if not (run_path / filename).exists():
                alerts.append(Alert(
                    severity=AlertSeverity.MEDIUM,
                    source="run_output",
                    message=f"Missing output file: {run_path / filename}",
                ))

        return alerts


class MonitoringLoop:
    """Orchestrates all monitoring checks and collects alerts."""

    def __init__(self, checks: list[MonitoringCheck]) -> None:
        self._checks = checks

    async def run_all(self) -> list[Alert]:
        """Run all checks and return combined alerts."""
        all_alerts: list[Alert] = []
        for check in self._checks:
            all_alerts.extend(await check.check_stale_tasks())
            all_alerts.extend(check.check_heartbeats())
        return sorted(all_alerts, key=lambda a: list(AlertSeverity).index(a.severity))
