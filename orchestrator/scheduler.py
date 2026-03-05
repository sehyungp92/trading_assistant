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
    monitoring_interval_minutes: int = 60
    worker_interval_seconds: int = 60
    relay_poll_interval_seconds: int = 300
    daily_analysis_hour: int = 6  # UTC hour to run daily analysis
    daily_analysis_minute: int = 0
    weekly_analysis_day_of_week: str = "sun"  # day of week for weekly analysis
    weekly_analysis_hour: int = 8
    weekly_analysis_minute: int = 0
    wfo_day_of_week: str = "sat"  # run WFO on Saturday
    wfo_hour: int = 2
    wfo_minute: int = 0
    stale_error_sweep_interval_seconds: int = 600  # sweep stale errors every 10 min
    stale_event_recovery_interval_seconds: int = 900  # recover stuck events every 15 min
    morning_scan_hour: int = 7
    morning_scan_minute: int = 0
    evening_report_hour: int = 22
    evening_report_minute: int = 0
    outcome_measurement_day_of_week: str = "sun"
    outcome_measurement_hour: int = 10
    outcome_measurement_minute: int = 0
    memory_consolidation_day_of_week: str = "sun"
    memory_consolidation_hour: int = 9
    memory_consolidation_minute: int = 0


def create_scheduler_jobs(
    config: SchedulerConfig,
    worker_fn: Callable[[], Awaitable[None]],
    monitoring_fn: Callable[[], Awaitable[None]],
    relay_fn: Callable[[], Awaitable[None]],
    daily_analysis_fn: Callable[[], Awaitable[None]] | None = None,
    weekly_analysis_fn: Callable[[], Awaitable[None]] | None = None,
    wfo_fn: Callable[[], Awaitable[None]] | None = None,
    stale_error_sweep_fn: Callable[[], Awaitable[None]] | None = None,
    stale_event_recovery_fn: Callable[[], Awaitable[None]] | None = None,
    morning_scan_fn: Callable[[], Awaitable[None]] | None = None,
    evening_report_fn: Callable[[], Awaitable[None]] | None = None,
    outcome_measurement_fn: Callable[[], Awaitable[None]] | None = None,
    memory_consolidation_fn: Callable[[], Awaitable[None]] | None = None,
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

    if wfo_fn is not None:
        jobs.append({
            "name": "wfo",
            "func": wfo_fn,
            "trigger": "cron",
            "day_of_week": config.wfo_day_of_week,
            "hour": config.wfo_hour,
            "minute": config.wfo_minute,
        })

    if stale_error_sweep_fn is not None:
        jobs.append({
            "name": "stale_error_sweep",
            "func": stale_error_sweep_fn,
            "trigger": "interval",
            "seconds": config.stale_error_sweep_interval_seconds,
        })

    if stale_event_recovery_fn is not None:
        jobs.append({
            "name": "stale_event_recovery",
            "func": stale_event_recovery_fn,
            "trigger": "interval",
            "seconds": config.stale_event_recovery_interval_seconds,
        })

    if morning_scan_fn is not None:
        jobs.append({
            "name": "morning_scan",
            "func": morning_scan_fn,
            "trigger": "cron",
            "hour": config.morning_scan_hour,
            "minute": config.morning_scan_minute,
        })

    if evening_report_fn is not None:
        jobs.append({
            "name": "evening_report",
            "func": evening_report_fn,
            "trigger": "cron",
            "hour": config.evening_report_hour,
            "minute": config.evening_report_minute,
        })

    if outcome_measurement_fn is not None:
        jobs.append({
            "name": "outcome_measurement",
            "func": outcome_measurement_fn,
            "trigger": "cron",
            "day_of_week": config.outcome_measurement_day_of_week,
            "hour": config.outcome_measurement_hour,
            "minute": config.outcome_measurement_minute,
        })

    if memory_consolidation_fn is not None:
        jobs.append({
            "name": "memory_consolidation",
            "func": memory_consolidation_fn,
            "trigger": "cron",
            "day_of_week": config.memory_consolidation_day_of_week,
            "hour": config.memory_consolidation_hour,
            "minute": config.memory_consolidation_minute,
        })

    return jobs
