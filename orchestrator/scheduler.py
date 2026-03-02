"""Scheduler — APScheduler configuration for periodic jobs.

Jobs:
  - worker: process pending events (every 60s)
  - monitoring: run health checks (every 10min)
  - relay_poll: pull events from relay VPS (every 5min)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Awaitable


@dataclass
class SchedulerConfig:
    monitoring_interval_minutes: int = 10
    worker_interval_seconds: int = 60
    relay_poll_interval_seconds: int = 300


def create_scheduler_jobs(
    config: SchedulerConfig,
    worker_fn: Callable[[], Awaitable[None]],
    monitoring_fn: Callable[[], Awaitable[None]],
    relay_fn: Callable[[], Awaitable[None]],
) -> list[dict]:
    """Build job definitions for APScheduler. Returns dicts, not APScheduler objects,
    so the caller can register them with their scheduler instance."""
    return [
        {
            "name": "worker",
            "func": worker_fn,
            "trigger": "interval",
            "seconds": config.worker_interval_seconds,
        },
        {
            "name": "monitoring",
            "func": monitoring_fn,
            "trigger": "interval",
            "seconds": config.monitoring_interval_minutes * 60,
        },
        {
            "name": "relay_poll",
            "func": relay_fn,
            "trigger": "interval",
            "seconds": config.relay_poll_interval_seconds,
        },
    ]
