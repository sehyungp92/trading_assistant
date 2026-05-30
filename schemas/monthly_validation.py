"""Monthly validation result schemas."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION


class MonthlyValidationStatus(str, Enum):
    KEEP = "keep"
    WATCH = "watch"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"
    EXPERIMENT = "experiment"
    INSUFFICIENT_DATA = "insufficient_data"
    INSUFFICIENT_LINEAGE = "insufficient_lineage"
    UNSUPPORTED_NO_REPLAY_PLUGIN = "unsupported_no_replay_plugin"
    NO_CHANGE = "no_change"


class GapAttributionCategory(str, Enum):
    UNDER_TRADING = "under_trading"
    OUTLIER_LOSS = "outlier_loss"
    BROAD_DEGRADATION = "broad_degradation"
    EXECUTION_DRIFT = "execution_drift"
    SLIPPAGE_COST_DRIFT = "slippage_cost_drift"
    DATA_GAP = "data_gap"
    REGIME_MISMATCH = "regime_mismatch"
    HARMFUL_ACCEPTED_MUTATION = "harmful_accepted_mutation"
    FILTER_OVERREACH = "filter_overreach"
    ENTRY_SIGNAL_DECAY = "entry_signal_decay"
    EXIT_MISMATCH = "exit_mismatch"
    PORTFOLIO_CORRELATION_CROWDING = "portfolio_correlation_crowding"
    OPPORTUNITY_SCARCITY = "opportunity_scarcity"
    NONE = "none"


class GapAttribution(BaseModel):
    primary_category: GapAttributionCategory = GapAttributionCategory.NONE
    supporting_categories: list[GapAttributionCategory] = Field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    evidence_paths: list[str] = Field(default_factory=list)


class MonthlyValidationResult(BaseModel):
    run_id: str
    run_month: str
    bot_id: str
    strategy_id: str
    status: MonthlyValidationStatus
    objective_version: str = OBJECTIVE_WEIGHTS_VERSION
    telemetry_manifest_path: str = ""
    market_data_manifest_path: str = ""
    run_manifest_path: str = ""
    artifact_index_path: str = ""
    replay_parity_path: str = ""
    gap_attribution: GapAttribution = GapAttribution()
    monthly_report_path: str = ""
    strategy_change_record_id: str = ""
    blocking_reasons: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    candidate_summary_path: str = ""
    candidate_gate_report_path: str = ""
    approval_packet_paths: list[str] = Field(default_factory=list)
    approval_request_ids: list[str] = Field(default_factory=list)
    selected_candidate_count: int = 0
    rejected_candidate_count: int = 0
    gate_passed_candidate_count: int = 0
    approval_ready_candidate_count: int = 0
    model_review_path: str = ""
    model_review_validation_path: str = ""
    model_review_valid: bool | None = None
    model_review_issues: list[str] = Field(default_factory=list)
    model_review_provider: str = ""
    model_review_model: str = ""
    model_review_runtime: str = ""
    optimizer_sequence_result_path: str = ""
    optimizer_sequence_status: str = ""
    adopted_candidate_id: str = ""
    optimizer_no_adoption_reason: str = ""
    repair_request_path: str = ""
    repair_required: bool = False
    proposed_strategy_change_record_ids: list[str] = Field(default_factory=list)
    shadow: bool = True
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
