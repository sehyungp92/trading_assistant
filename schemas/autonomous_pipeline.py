# schemas/autonomous_pipeline.py
"""Autonomous pipeline schemas — suggestion-to-PR automation models.

Covers: parameter definitions, bot config profiles, backtest comparisons,
approval requests, file changes, and PR requests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from schemas.wfo_results import SimulationMetrics


class ParameterType(str, Enum):
    YAML_FIELD = "YAML_FIELD"
    PYTHON_CONSTANT = "PYTHON_CONSTANT"
    # FileChangeGenerator only supports YAML_FIELD currently.
    # PYTHON_CONSTANT is used in bot config definitions but not yet auto-modifiable.


class ParameterDefinition(BaseModel):
    """A tunable parameter in a bot's configuration."""

    param_name: str
    bot_id: str
    strategy_id: Optional[str] = None
    param_type: ParameterType
    file_path: str
    yaml_key: Optional[str] = None
    python_path: Optional[str] = None
    current_value: Any = None
    valid_range: Optional[tuple[float, float]] = None
    valid_values: Optional[list[Any]] = None
    value_type: Literal["int", "float", "bool", "str"] = "float"
    category: str = ""
    is_safety_critical: bool = False

    @model_validator(mode="after")
    def _check_type_fields(self) -> ParameterDefinition:
        if self.param_type == ParameterType.YAML_FIELD and not self.yaml_key:
            raise ValueError("yaml_key is required when param_type is YAML_FIELD")
        if self.param_type == ParameterType.PYTHON_CONSTANT and not self.python_path:
            raise ValueError("python_path is required when param_type is PYTHON_CONSTANT")
        if self.valid_range is not None and self.valid_range[0] >= self.valid_range[1]:
            raise ValueError("valid_range[0] must be less than valid_range[1]")
        return self


class BotConfigProfile(BaseModel):
    """Configuration profile for a single bot."""

    bot_id: str
    repo_url: str = ""
    repo_dir: str = ""
    parameters: list[ParameterDefinition] = []
    strategies: list[str] = []

    def get_parameter(self, param_name: str) -> Optional[ParameterDefinition]:
        for p in self.parameters:
            if p.param_name == param_name:
                return p
        return None

    def get_parameters_by_category(self, category: str) -> list[ParameterDefinition]:
        return [p for p in self.parameters if p.category == category]


class BacktestContext(BaseModel):
    """Context for a backtest comparison."""

    suggestion_id: str
    bot_id: str
    param_name: str
    current_value: Any = None
    proposed_value: Any = None
    trade_count: int = 0
    data_days: int = 0


class BacktestComparison(BaseModel):
    """Side-by-side comparison of baseline vs proposed backtest results."""

    context: BacktestContext
    baseline: SimulationMetrics = Field(default_factory=SimulationMetrics)
    proposed: SimulationMetrics = Field(default_factory=SimulationMetrics)
    sharpe_change_pct: float = 0.0
    max_dd_change_pct: float = 0.0
    profit_factor_change_pct: float = 0.0
    win_rate_change_pct: float = 0.0
    passes_safety: bool = False
    safety_notes: list[str] = []

    @model_validator(mode="after")
    def _compute_changes(self) -> BacktestComparison:
        self.sharpe_change_pct = _pct_change(
            self.baseline.sharpe_ratio, self.proposed.sharpe_ratio
        )
        self.max_dd_change_pct = _pct_change(
            self.baseline.max_drawdown_pct, self.proposed.max_drawdown_pct
        )
        self.profit_factor_change_pct = _pct_change(
            self.baseline.profit_factor, self.proposed.profit_factor
        )
        self.win_rate_change_pct = _pct_change(
            self.baseline.win_rate, self.proposed.win_rate
        )
        return self


def _pct_change(old: float, new: float) -> float:
    """Percentage change from old to new. Returns 0 if old is 0."""
    if old == 0:
        return 0.0
    return ((new - old) / abs(old)) * 100.0


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ApprovalRequest(BaseModel):
    """A request for human approval of a parameter change."""

    request_id: str
    suggestion_id: str
    bot_id: str
    param_changes: list[dict] = []
    backtest_summary: Optional[BacktestComparison] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    pr_url: Optional[str] = None
    message_id: Optional[int] = None


class FileChange(BaseModel):
    """A single file modification."""

    file_path: str
    original_content: str = ""
    new_content: str = ""
    diff_preview: str = ""


class PRRequest(BaseModel):
    """Request to create a PR in a bot repository."""

    approval_request_id: str
    suggestion_id: str
    bot_id: str
    repo_dir: str
    branch_name: str
    title: str
    body: str = ""
    file_changes: list[FileChange] = []


class PreflightResult(BaseModel):
    """Result of pre-flight checks before PR creation."""

    passed: bool = True
    checks: list[dict] = []
    reasons: list[str] = []


class ReviewState(str, Enum):
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"
    PENDING = "PENDING"


class PRReviewStatus(BaseModel):
    """Status of PR reviews from GitHub."""

    pr_number: int
    pr_url: str = ""
    review_state: ReviewState = ReviewState.PENDING
    reviewers: list[str] = []
    actionable_comments: list[str] = []
    needs_attention: bool = False
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class PRResult(BaseModel):
    """Result of a PR creation attempt."""

    success: bool
    pr_url: Optional[str] = None
    branch_name: str = ""
    error: Optional[str] = None
    pr_number: Optional[int] = None
    preflight: Optional[PreflightResult] = None
    existing_pr_url: Optional[str] = None
