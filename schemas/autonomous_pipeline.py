"""Autonomous pipeline schemas for repo-backed suggestion automation."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from schemas.wfo_results import SimulationMetrics


class ParameterType(str, Enum):
    YAML_FIELD = "YAML_FIELD"
    PYTHON_CONSTANT = "PYTHON_CONSTANT"


class ChangeKind(str, Enum):
    PARAMETER_CHANGE = "parameter_change"
    BUG_FIX = "bug_fix"
    STRUCTURAL_CHANGE = "structural_change"
    ROLLBACK = "rollback"


class RepoRiskTier(str, Enum):
    AUTO = "auto"
    REQUIRES_APPROVAL = "requires_approval"
    REQUIRES_DOUBLE_APPROVAL = "requires_double_approval"


class FileChangeMode(str, Enum):
    FILE_REPLACE = "file_replace"
    YAML_FIELD = "yaml_field"
    PYTHON_CONSTANT = "python_constant"
    UNIFIED_DIFF = "unified_diff"


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
    default_branch: str = "main"
    allowed_edit_paths: list[str] = Field(default_factory=list)
    structural_context_paths: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    parameters: list[ParameterDefinition] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)

    def get_parameter(self, param_name: str) -> Optional[ParameterDefinition]:
        for param in self.parameters:
            if param.param_name == param_name:
                return param
        return None

    def get_parameters_by_category(self, category: str) -> list[ParameterDefinition]:
        return [param for param in self.parameters if param.category == category]


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
    safety_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _compute_changes(self) -> BacktestComparison:
        self.sharpe_change_pct = _pct_change(
            self.baseline.sharpe_ratio, self.proposed.sharpe_ratio,
        )
        self.max_dd_change_pct = _pct_change(
            self.baseline.max_drawdown_pct, self.proposed.max_drawdown_pct,
        )
        self.profit_factor_change_pct = _pct_change(
            self.baseline.profit_factor, self.proposed.profit_factor,
        )
        self.win_rate_change_pct = _pct_change(
            self.baseline.win_rate, self.proposed.win_rate,
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


class RepoTaskContext(BaseModel):
    """Execution context for a repo mutation task."""

    task_id: str
    repo_url: str = ""
    repo_dir: str = ""
    default_branch: str = "main"
    repo_cache_dir: str = ""
    worktree_dir: str = ""
    artifact_dir: str = ""


class ApprovalRequest(BaseModel):
    """A request for human approval of a repo-backed change."""

    request_id: str
    suggestion_id: str
    bot_id: str
    change_kind: ChangeKind = ChangeKind.PARAMETER_CHANGE
    title: str = ""
    summary: str = ""
    param_changes: list[dict] = Field(default_factory=list)
    file_changes: list[FileChange] = Field(default_factory=list)
    backtest_summary: Optional[BacktestComparison] = None
    planned_files: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    risk_tier: RepoRiskTier = RepoRiskTier.REQUIRES_APPROVAL
    draft_pr: bool = False
    repo_task: Optional[RepoTaskContext] = None
    implementation_notes: str = ""
    issue_url: Optional[str] = None
    diff_summary: list[str] = Field(default_factory=list)
    approval_count: int = 0
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
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
    change_mode: FileChangeMode = FileChangeMode.FILE_REPLACE
    metadata: dict[str, Any] = Field(default_factory=dict)
    patch: str = ""
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
    change_kind: ChangeKind = ChangeKind.PARAMETER_CHANGE
    draft: bool = False
    verification_commands: list[str] = Field(default_factory=list)
    repo_task: Optional[RepoTaskContext] = None
    file_changes: list[FileChange] = Field(default_factory=list)


class GitHubIssueRequest(BaseModel):
    """Request to create or deduplicate a GitHub issue."""

    bot_id: str
    title: str
    body: str
    repo_dir: str
    labels: list[str] = Field(default_factory=list)
    dedupe_key: str = ""
    repo_task: Optional[RepoTaskContext] = None
    change_kind: ChangeKind = ChangeKind.BUG_FIX


class GitHubIssueResult(BaseModel):
    """Result of a GitHub issue creation attempt."""

    success: bool
    issue_url: Optional[str] = None
    issue_number: Optional[int] = None
    error: Optional[str] = None
    existing_issue_url: Optional[str] = None


class PreflightResult(BaseModel):
    """Result of pre-flight checks before PR creation."""

    passed: bool = True
    checks: list[dict] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


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
    reviewers: list[str] = Field(default_factory=list)
    actionable_comments: list[str] = Field(default_factory=list)
    needs_attention: bool = False
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
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
    diff_summary: list[str] = Field(default_factory=list)


ApprovalRequest.model_rebuild()
