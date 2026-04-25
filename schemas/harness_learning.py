"""Schemas for offline harness evaluation artifacts."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class HarnessVariant(BaseModel):
    """A named harness configuration variant for offline evaluation."""

    name: str
    prompt_patch: str = ""
    retrieval_mode: str = "baseline"
    validator_profile: str = "baseline"
    route_profile: str = "configured"
    enabled: bool = True


class HarnessEvalResult(BaseModel):
    """Offline benchmark comparison result for a harness variant."""

    variant_name: str
    benchmark_count: int = 0
    aggregate_score: float = 0.0
    per_source: dict[str, float] = Field(default_factory=dict)
    kept: bool = False
    rationale: str = ""
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
