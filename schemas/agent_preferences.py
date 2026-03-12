"""Schemas for agent runtime provider selection and preferences."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class AgentProvider(str, Enum):
    CLAUDE_MAX = "claude_max"
    CODEX_PRO = "codex_pro"
    ZAI_CODING_PLAN = "zai_coding_plan"
    OPENROUTER = "openrouter"


class AgentWorkflow(str, Enum):
    DAILY_ANALYSIS = "daily_analysis"
    WEEKLY_ANALYSIS = "weekly_analysis"
    WFO = "wfo"
    TRIAGE = "triage"


class AgentSelection(BaseModel):
    provider: AgentProvider
    model: str | None = None

    @model_validator(mode="after")
    def _normalize_model(self) -> AgentSelection:
        if self.model is not None:
            trimmed = self.model.strip()
            self.model = trimmed or None
        return self


class AgentPreferences(BaseModel):
    default: AgentSelection = Field(
        default_factory=lambda: AgentSelection(provider=AgentProvider.CLAUDE_MAX)
    )
    overrides: dict[AgentWorkflow, AgentSelection | None] = Field(default_factory=dict)


class ProviderReadiness(BaseModel):
    provider: AgentProvider
    available: bool
    runtime: str
    reason: str = ""


class AgentPreferencesView(BaseModel):
    default: AgentSelection
    overrides: dict[AgentWorkflow, AgentSelection | None] = Field(default_factory=dict)
    effective: dict[AgentWorkflow, AgentSelection] = Field(default_factory=dict)
    providers: list[ProviderReadiness] = Field(default_factory=list)
