"""Monthly validation run manifest schema.

The manifest is the frozen contract passed from trading_assistant to the
full-fidelity backtest/replay repository.  It intentionally contains paths and
versions, not imported strategy objects.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from schemas.monthly_optimizer import (
    MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION,
    PHASED_AUTO_RUNNER_CONTRACT_VERSION,
    SMOKE_REPAIR_RUNNER_CONTRACT_VERSION,
)
from schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION


class MonthlyRunMode(str, Enum):
    INCUMBENT_VALIDATION = "incumbent_validation"
    SMOKE_REPAIR = "smoke_repair"
    PHASED_AUTO = "phased_auto"
    STRUCTURAL_REVIEW = "structural_review"
    OUTCOME_MEASUREMENT = "outcome_measurement"


class MonthlyApprovalMode(str, Enum):
    NONE = "none"
    EXPERIMENT = "experiment"
    MANUAL_REQUIRED = "manual_required"


class MonthlyRunManifest(BaseModel):
    """Frozen evidence and execution manifest for a monthly validation run."""

    run_id: str
    run_month: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: MonthlyRunMode = MonthlyRunMode.INCUMBENT_VALIDATION
    bot_id: str
    strategy_id: str
    strategy_version: str = ""
    config_version: str = ""
    deployment_id: str = ""
    parameter_set_id: str = ""
    proposal_ids: list[str] = Field(default_factory=list)
    suggestion_ids: list[str] = Field(default_factory=list)
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    latest_month_start: date
    latest_month_end: date
    calibration_start: date | None = None
    calibration_end: date | None = None
    selection_oos_start: date | None = None
    selection_oos_end: date | None = None
    market_data_manifest_path: str
    telemetry_manifest_path: str
    backtest_repo_path: str = ""
    backtest_repo_commit_sha: str = ""
    trading_repo_path: str = ""
    trading_repo_branch: str = ""
    trading_repo_commit_sha: str = ""
    control_plane_commit_sha: str = ""
    backtest_command: list[str] = Field(default_factory=list)
    artifact_root: str
    strategy_plugin_id: str = ""
    strategy_plugin_contract_path: str = ""
    strategy_plugin_contract_version: str = ""
    round_id: str = ""
    prior_round_id: str = ""
    next_round_id: str = ""
    round_n_strategy_config_path: str = ""
    round_n_strategy_config_version: str = ""
    round_n_portfolio_config_path: str = ""
    round_n_portfolio_config_version: str = ""
    data_manifest_checksum: str = ""
    data_bundle_manifest_path: str = ""
    data_bundle_checksum: str = ""
    in_sample_start: date | None = None
    in_sample_end: date | None = None
    fold_manifest_path: str = ""
    rounds_manifest_path: str = ""
    end_of_round_diagnostics_path: str = ""
    candidate_workspace_root: str = ""
    candidate_workspace_key: str = ""
    candidate_workspace_manifest_path: str = ""
    candidate_attempt_id: str = ""
    candidate_attempt_number: int = 0
    candidate_attempt_status: str = ""
    retry_reason: str = ""
    stall_timeout_seconds: int = 0
    max_workers: int = 2
    score_component_cap: int = 7
    checkpoint_path: str = ""
    cache_path: str = ""
    outcome_prior_snapshot_path: str = ""
    monthly_search_brief_path: str = ""
    monthly_search_brief_id: str = ""
    source_weekly_signal_ids: list[str] = Field(default_factory=list)
    monthly_search_guidance: dict = Field(default_factory=dict)
    workflow_contract_path: str = ""
    workflow_contract_version: str = "monthly_incumbent_validation_v1"
    source_runner_contract_versions: list[str] = Field(
        default_factory=lambda: [
            SMOKE_REPAIR_RUNNER_CONTRACT_VERSION,
            PHASED_AUTO_RUNNER_CONTRACT_VERSION,
        ]
    )
    output_artifact_names: list[str] = Field(default_factory=list)
    required_json_schemas: list[str] = Field(default_factory=list)
    approval_mode: MonthlyApprovalMode = MonthlyApprovalMode.NONE
    expected_outputs: list[str] = Field(default_factory=list)
    manifest_version: str = "monthly_run_manifest_v1"

    @field_validator("run_month")
    @classmethod
    def _validate_run_month(cls, value: str) -> str:
        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError("run_month must be YYYY-MM")
        year, month = parts
        if len(year) != 4 or len(month) != 2:
            raise ValueError("run_month must be YYYY-MM")
        month_int = int(month)
        if not 1 <= month_int <= 12:
            raise ValueError("run_month month must be 01..12")
        return value

    @model_validator(mode="after")
    def _validate_windows(self) -> MonthlyRunManifest:
        if self.latest_month_end < self.latest_month_start:
            raise ValueError("latest_month_end must be >= latest_month_start")
        for start, end, name in (
            (self.calibration_start, self.calibration_end, "calibration"),
            (self.selection_oos_start, self.selection_oos_end, "selection_oos"),
            (self.in_sample_start, self.in_sample_end, "in_sample"),
        ):
            if (start is None) != (end is None):
                raise ValueError(f"{name}_start and {name}_end must be set together")
            if start and end and end < start:
                raise ValueError(f"{name}_end must be >= {name}_start")
        if self.in_sample_end and self.selection_oos_start and self.selection_oos_start <= self.in_sample_end:
            raise ValueError("selection_oos_start must be after in_sample_end")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.score_component_cap < 1 or self.score_component_cap > 7:
            raise ValueError("score_component_cap must be between 1 and 7")
        if self.stall_timeout_seconds < 0:
            raise ValueError("stall_timeout_seconds cannot be negative")
        if self.candidate_attempt_number < 0:
            raise ValueError("candidate_attempt_number cannot be negative")
        if self.mode == MonthlyRunMode.PHASED_AUTO and (
            not self.workflow_contract_version
            or self.workflow_contract_version == "monthly_incumbent_validation_v1"
        ):
            self.workflow_contract_version = MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION
        return self

    @property
    def manifest_id(self) -> str:
        raw = "|".join([
            self.run_id,
            self.run_month,
            self.bot_id,
            self.strategy_id,
            self.mode.value,
            self.latest_month_start.isoformat(),
            self.latest_month_end.isoformat(),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def artifact_path(self, name: str) -> Path:
        return Path(self.artifact_root) / name
