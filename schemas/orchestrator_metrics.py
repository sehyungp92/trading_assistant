"""Orchestrator observability metrics schema."""
from __future__ import annotations

from pydantic import BaseModel


class OrchestratorMetrics(BaseModel):
    """Point-in-time snapshot of orchestrator operational state."""
    queue_depth: int = 0
    dead_letter_count: int = 0
    active_agents: int = 0
    error_rate_1h: float = 0.0
    uptime_seconds: float = 0.0
    last_daily_analysis: str | None = None
    last_weekly_analysis: str | None = None
