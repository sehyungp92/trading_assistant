# schemas/agent_response.py
"""Structured response schemas for parsed Claude analysis outputs.

Every Claude invocation should emit a structured block that gets parsed into
ParsedAnalysis. This enables programmatic tracking, validation, and feedback.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from schemas.autonomous_pipeline import FileChange


class AgentPrediction(BaseModel):
    """A specific, measurable prediction about a bot's future performance."""

    bot_id: str
    metric: Literal["pnl", "win_rate", "drawdown", "sharpe"]
    direction: Literal["improve", "decline", "stable"]
    confidence: float = Field(ge=0.0, le=1.0)
    timeframe_days: int = 7
    reasoning: str = ""


class AgentSuggestion(BaseModel):
    """A structured suggestion extracted from Claude's analysis."""

    suggestion_id: str = ""
    bot_id: str
    category: str = ""  # exit_timing, filter_threshold, stop_loss, signal, structural, position_sizing, regime_gate
    title: str
    expected_impact: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_summary: str = ""
    proposed_value: Optional[float] = None  # numeric value for parameter suggestions
    target_param: Optional[str] = None  # parameter name this suggestion targets


class StructuralProposal(BaseModel):
    """A proposal for structural changes to a bot's architecture."""

    hypothesis_id: Optional[str] = None
    linked_suggestion_id: Optional[str] = None
    bot_id: str
    title: str
    description: str = ""
    reversibility: Literal["easy", "moderate", "hard"] = "moderate"
    evidence: str = ""
    estimated_complexity: Literal["low", "medium", "high"] = "medium"
    file_changes: list[FileChange] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    acceptance_criteria: list[dict] = Field(default_factory=list)


# Shared mapping from suggestion category → suggestion tier.
# Used in _record_agent_suggestions (handlers.py) and suggestion_scorer.py.
CATEGORY_TO_TIER: dict[str, str] = {
    "exit_timing": "parameter",
    "filter_threshold": "filter",
    "stop_loss": "parameter",
    "signal": "parameter",
    "structural": "hypothesis",
    "position_sizing": "parameter",
    "regime_gate": "filter",
}


class ParsedAnalysis(BaseModel):
    """Result of parsing Claude's structured output block."""

    predictions: list[AgentPrediction] = Field(default_factory=list)
    suggestions: list[AgentSuggestion] = Field(default_factory=list)
    structural_proposals: list[StructuralProposal] = Field(default_factory=list)
    raw_report: str = ""
    parse_success: bool = True
    raw_structured: Optional[dict] = None  # Full parsed JSON from STRUCTURED_OUTPUT block
