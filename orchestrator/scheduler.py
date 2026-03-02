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
    daily_analysis_hour: int = 6  # UTC hour to run daily analysis
    daily_analysis_minute: int = 0
    weekly_analysis_day_of_week: str = "sun"  # day of week for weekly analysis
    weekly_analysis_hour: int = 8
    weekly_analysis_minute: int = 0


def create_scheduler_jobs(
    config: SchedulerConfig,
    worker_fn: Callable[[], Awaitable[None]],
    monitoring_fn: Callable[[], Awaitable[None]],
    relay_fn: Callable[[], Awaitable[None]],
    daily_analysis_fn: Callable[[], Awaitable[None]] | None = None,
    weekly_analysis_fn: Callable[[], Awaitable[None]] | None = None,
) -> list[dict]:
    """Build job definitions for APScheduler. Returns dicts, not APScheduler objects,
    so the caller can register them with their scheduler instance."""
    jobs = [
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

    if daily_analysis_fn is not None:
        jobs.append({
            "name": "daily_analysis",
            "func": daily_analysis_fn,
            "trigger": "cron",
            "hour": config.daily_analysis_hour,
            "minute": config.daily_analysis_minute,
        })

    if weekly_analysis_fn is not None:
        jobs.append({
            "name": "weekly_analysis",
            "func": weekly_analysis_fn,
            "trigger": "cron",
            "day_of_week": config.weekly_analysis_day_of_week,
            "hour": config.weekly_analysis_hour,
            "minute": config.weekly_analysis_minute,
        })

    return jobs
