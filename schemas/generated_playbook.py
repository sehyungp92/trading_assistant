"""Schemas for evidence-backed generated investigation playbooks."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class PlaybookStatus(str, Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    INACTIVE = "inactive"


class GeneratedPlaybook(BaseModel):
    """Advisory procedure generated from repeated evidence clusters."""

    playbook_id: str = ""
    workflow: str = ""
    title: str = ""
    trigger_tags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    provenance: str = ""
    status: PlaybookStatus = PlaybookStatus.ACTIVE
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: object) -> None:
        if not self.playbook_id:
            raw = f"{self.workflow}:{self.title}:{','.join(sorted(self.trigger_tags))}"
            self.playbook_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def match_score(self, workflow: str, tags: list[str]) -> float:
        """Return a simple overlap score for retrieval."""
        if self.status != PlaybookStatus.ACTIVE:
            return 0.0
        if workflow and self.workflow and self.workflow != workflow:
            return 0.0
        score = 0.0
        if workflow and self.workflow == workflow:
            score += 1.0
        trigger_set = set(self.trigger_tags)
        if trigger_set and tags:
            overlap = len(trigger_set & set(tags))
            score += overlap / max(len(trigger_set), 1)
        return score

    def to_prompt_text(self) -> str:
        """Render compact procedural guidance for prompt injection."""
        lines = [f"[PLAYBOOK] {self.title}"]
        if self.trigger_conditions:
            lines.append("Trigger: " + "; ".join(self.trigger_conditions))
        if self.required_evidence:
            lines.append("Required evidence: " + "; ".join(self.required_evidence))
        if self.steps:
            lines.append("Steps: " + " | ".join(self.steps))
        if self.failure_modes:
            lines.append("Failure modes: " + "; ".join(self.failure_modes))
        if self.provenance:
            lines.append(f"Provenance: {self.provenance}")
        return "\n".join(lines)
