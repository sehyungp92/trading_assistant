# schemas/convergence.py
"""Convergence tracking schemas for learning loop health monitoring."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class DimensionStatus(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    OSCILLATING = "oscillating"
    INSUFFICIENT_DATA = "insufficient_data"


class ConvergenceDimension(BaseModel):
    """Status of a single convergence dimension."""

    name: str  # e.g. "composite_scores", "prediction_accuracy"
    status: DimensionStatus
    trend_value: float  # slope or delta
    window_weeks: int
    detail: str  # human-readable explanation


class ConvergenceReport(BaseModel):
    """Multi-dimensional convergence report for the learning system."""

    overall_status: DimensionStatus
    dimensions: list[ConvergenceDimension]
    oscillation_detected: bool
    weeks_analyzed: int
    recommendation: str  # e.g. "System converging — maintain current approach"
