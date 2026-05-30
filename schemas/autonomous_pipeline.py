"""Compatibility exports for legacy autonomous-pipeline schema imports.

Canonical shared models now live in neutral schema modules. This file remains
temporarily so old autonomous tests and historical integrations can migrate
without changing model identity.
"""
from __future__ import annotations

from schemas.approval import (
    ApprovalRequest,
    ApprovalStatus,
    BacktestComparison,
    BacktestContext,
    RepoRiskTier,
)
from schemas.bot_profile import BotConfigProfile
from schemas.parameter_definition import ParameterDefinition, ParameterType
from schemas.repo_changes import (
    ChangeKind,
    FileChange,
    FileChangeMode,
    GitHubIssueRequest,
    GitHubIssueResult,
    PRRequest,
    PRResult,
    PRReviewStatus,
    PreflightResult,
    ReviewState,
)
from schemas.repo_task import RepoTaskContext
from schemas.simulation_metrics import SimulationMetrics

__all__ = [
    "ApprovalRequest",
    "ApprovalStatus",
    "BacktestComparison",
    "BacktestContext",
    "BotConfigProfile",
    "ChangeKind",
    "FileChange",
    "FileChangeMode",
    "GitHubIssueRequest",
    "GitHubIssueResult",
    "ParameterDefinition",
    "ParameterType",
    "PRRequest",
    "PRResult",
    "PRReviewStatus",
    "PreflightResult",
    "RepoRiskTier",
    "RepoTaskContext",
    "ReviewState",
    "SimulationMetrics",
]
