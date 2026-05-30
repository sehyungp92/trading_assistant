"""Monthly candidate ingestion, deterministic gates, and approval packets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.monthly_model_response_parser import parse_monthly_model_review
from analysis.monthly_model_response_validator import MonthlyModelResponseValidator
from schemas.approval import ApprovalRequest, RepoRiskTier
from schemas.backtest_artifacts import BacktestArtifactIndex
from schemas.decision_parity import DecisionParityReport
from schemas.market_data_manifest import MarketDataManifest
from schemas.monthly_candidates import (
    MonthlyApprovalEvidencePacket,
    MonthlyCandidateGateReport,
    MonthlyCandidateProcessingResult,
    MonthlyCandidateSource,
    MonthlyGateCheck,
    MonthlyGateSeverity,
    MonthlyImprovementCandidate,
    MonthlyRiskClassification,
)
from schemas.monthly_model_review import MonthlyModelReview, MonthlyModelValidationResult
from schemas.monthly_run_manifest import MonthlyRunManifest
from schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from schemas.proposal_ledger import (
    ProposalCandidate,
    ProposalEvaluation,
    ProposalKind,
    ProposalSource,
)
from schemas.repo_changes import ChangeKind, FileChange
from schemas.replay_parity import ReplayParityReport
from schemas.strategy_plugin_contract import StrategyPluginContract
from schemas.strategy_change_ledger import StrategyChangeRecord, StrategyChangeRecordType
from schemas.telemetry_manifest import TelemetryEligibility, TelemetryManifest
from skills.outcome_prior_store import OutcomePriorStore
from skills.proposal_ledger import make_proposal_id
from skills.search_allocation_policy import SearchAllocationPolicy

_APPROVAL_STATUSES = {
    MonthlyValidationStatus.REPAIR,
    MonthlyValidationStatus.ROLLBACK,
    MonthlyValidationStatus.EXPERIMENT,
    MonthlyValidationStatus.QUARANTINE,
}


class MonthlyCandidatePipeline:
    """Consumes backtest candidate artifacts and records approval-ready packets."""

    def __init__(
        self,
        *,
        approval_tracker: object | None = None,
        proposal_ledger: object | None = None,
        strategy_change_ledger: object | None = None,
        outcome_prior_store: OutcomePriorStore | None = None,
        min_trade_count: int = 10,
        max_outlier_win_concentration: float = 0.40,
    ) -> None:
        self.approval_tracker = approval_tracker
        self.proposal_ledger = proposal_ledger
        self.strategy_change_ledger = strategy_change_ledger
        self.outcome_prior_store = outcome_prior_store
        self.search_allocation_policy = (
            SearchAllocationPolicy(outcome_prior_store)
            if outcome_prior_store is not None else None
        )
        self.min_trade_count = min_trade_count
        self.max_outlier_win_concentration = max_outlier_win_concentration

    def process(
        self,
        *,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex,
        coverage: MarketDataManifest | None,
        telemetry: TelemetryManifest | None = None,
        parity_report: ReplayParityReport | None,
        artifact_root: Path,
        monthly_result_path: Path,
        shadow: bool,
        model_review_path: str = "",
    ) -> MonthlyCandidateProcessingResult:
        artifact_root = Path(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        if monthly_result.repair_required:
            summary_path = artifact_root / "candidate_generation_summary.json"
            result = MonthlyCandidateProcessingResult(
                run_id=monthly_result.run_id,
                run_month=monthly_result.run_month,
                bot_id=monthly_result.bot_id,
                strategy_id=monthly_result.strategy_id,
                selected_candidates=[],
                rejected_candidates=[],
                gate_reports=[],
                approval_packets=[],
                approval_request_ids=[],
                gate_passed_candidate_count=0,
                approval_ready_candidate_count=0,
                candidate_summary_path=str(summary_path),
                gate_report_path="",
                approval_packet_paths=[],
                model_review_path=model_review_path,
            )
            summary_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result
        selected = self._load_selected_candidates(
            artifact_index,
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
        )
        for candidate in selected:
            self._normalize_candidate_paths(candidate, artifact_index)
        if self.search_allocation_policy is not None:
            selected = self.search_allocation_policy.order_candidates(selected)
        rejected = self._load_rejected_candidates(artifact_index)
        model_validation_path = ""
        model_validation: MonthlyModelValidationResult | None = None
        if model_review_path:
            allowed_evidence = _dedupe([
                *monthly_result.evidence_paths,
                *[
                    path
                    for candidate in selected
                    for path in [*candidate.evidence_paths, *candidate.artifact_paths]
                ],
            ])
            _, model_validation, model_validation_path = self._load_and_validate_model_review(
                model_review_path=model_review_path,
                monthly_result=monthly_result,
                allowed_evidence_paths=allowed_evidence,
            )
        gate_reports: list[MonthlyCandidateGateReport] = []
        packets: list[MonthlyApprovalEvidencePacket] = []
        request_ids: list[str] = []
        packet_paths: list[str] = []

        for candidate in selected:
            gate_report = self.evaluate_candidate(
                candidate=candidate,
                monthly_result=monthly_result,
                artifact_index=artifact_index,
                coverage=coverage,
                telemetry=telemetry,
                parity_report=parity_report,
                model_validation=model_validation,
            )
            gate_reports.append(gate_report)
            packet = self.build_packet(
                candidate=candidate,
                gate_report=gate_report,
                monthly_result=monthly_result,
                coverage=coverage,
                telemetry=telemetry,
                parity_report=parity_report,
                monthly_result_path=monthly_result_path,
                model_review_path=model_review_path,
                model_validation=model_validation,
            )
            packet_path = artifact_root / f"approval_packet_{_safe_id(candidate.candidate_id)}.json"
            packet.approval_packet_path = str(packet_path)
            proposal_id = self._record_proposal(candidate, packet, gate_report, monthly_result)
            packet.proposal_id = proposal_id
            packet.suggestion_id = proposal_id

            if gate_report.passed and not shadow and self.approval_tracker is not None:
                request = self._build_approval_request(packet, candidate)
                strategy_record_id = self._record_strategy_change(candidate, packet, request)
                packet.request_id = request.request_id
                packet.strategy_change_record_id = strategy_record_id
                request.strategy_change_record_id = strategy_record_id
                self.approval_tracker.create_request(request)
                request_ids.append(request.request_id)
                packet.approval_ready = True
            else:
                packet.approval_ready = False
                packet.approval_suppressed_reasons = self._suppression_reasons(
                    gate_report,
                    shadow=shadow,
                    approval_tracker_present=self.approval_tracker is not None,
                )

            packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
            packet_paths.append(str(packet_path))
            packets.append(packet)

        gate_report_path = artifact_root / "candidate_gate_report.json"
        gate_report_path.write_text(
            json.dumps([report.model_dump(mode="json") for report in gate_reports], indent=2),
            encoding="utf-8",
        )
        summary_path = artifact_root / "candidate_generation_summary.json"
        result = MonthlyCandidateProcessingResult(
            run_id=monthly_result.run_id,
            run_month=monthly_result.run_month,
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
            selected_candidates=selected,
            rejected_candidates=rejected,
            gate_reports=gate_reports,
            approval_packets=packets,
            approval_request_ids=request_ids,
            gate_passed_candidate_count=sum(1 for report in gate_reports if report.passed),
            approval_ready_candidate_count=len(request_ids),
            candidate_summary_path=str(summary_path),
            gate_report_path=str(gate_report_path),
            approval_packet_paths=packet_paths,
            model_review_path=model_review_path,
            model_review_validation_path=model_validation_path,
            model_review_valid=model_validation.valid if model_validation is not None else None,
            model_review_issues=(
                [issue.message for issue in model_validation.issues]
                if model_validation is not None else []
            ),
        )
        summary_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def evaluate_candidate(
        self,
        *,
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex | None = None,
        coverage: MarketDataManifest | None = None,
        telemetry: TelemetryManifest,
        parity_report: ReplayParityReport | None,
        model_validation: MonthlyModelValidationResult | None = None,
    ) -> MonthlyCandidateGateReport:
        checks = [
            MonthlyGateCheck(
                name="monthly_status_allows_candidate",
                passed=monthly_result.status in _APPROVAL_STATUSES and not monthly_result.blocking_reasons,
                reason=(
                    f"monthly status {monthly_result.status.value} does not allow approval-ready candidates"
                    if monthly_result.status not in _APPROVAL_STATUSES
                    else "; ".join(monthly_result.blocking_reasons)
                ),
                evidence_paths=[monthly_result.run_manifest_path],
            ),
            MonthlyGateCheck(
                name="candidate_decision_allows_action",
                passed=candidate.decision.value in {"repair", "rollback", "experiment"},
                reason=(
                    ""
                    if candidate.decision.value in {"repair", "rollback", "experiment"}
                    else f"candidate decision is {candidate.decision.value}"
                ),
                evidence_paths=candidate.evidence_paths,
            ),
            MonthlyGateCheck(
                name="supported_candidate_source",
                passed=candidate.source in {
                    MonthlyCandidateSource.SMOKE_REPAIR,
                    MonthlyCandidateSource.PHASED_AUTO,
                },
                reason=(
                    ""
                    if candidate.source in {
                        MonthlyCandidateSource.SMOKE_REPAIR,
                        MonthlyCandidateSource.PHASED_AUTO,
                    }
                    else f"unsupported candidate source: {candidate.source.value}"
                ),
                evidence_paths=candidate.evidence_paths,
            ),
            self._runner_contract_gate(candidate),
            self._candidate_lineage_gate(candidate, monthly_result, artifact_index),
            self._candidate_workspace_gate(candidate),
            MonthlyGateCheck(
                name="candidate_evidence_paths",
                passed=bool(_existing_paths([*candidate.evidence_paths, *candidate.artifact_paths])),
                reason=(
                    ""
                    if _existing_paths([*candidate.evidence_paths, *candidate.artifact_paths])
                    else "candidate lacks existing replay evidence paths"
                ),
                evidence_paths=[*candidate.evidence_paths, *candidate.artifact_paths],
            ),
            self._candidate_artifact_containment_gate(candidate),
            MonthlyGateCheck(
                name="market_data_coverage",
                passed=coverage is not None and coverage.usable_for_authoritative_validation,
                reason="" if coverage is not None and coverage.usable_for_authoritative_validation else "market data is not authoritative",
                evidence_paths=[monthly_result.market_data_manifest_path],
            ),
            MonthlyGateCheck(
                name="telemetry_lineage",
                passed=(
                    telemetry is not None
                    and telemetry.authoritative_eligibility == TelemetryEligibility.AUTHORITATIVE
                ),
                reason=(
                    ""
                    if telemetry is not None
                    and telemetry.authoritative_eligibility == TelemetryEligibility.AUTHORITATIVE
                    else (
                        f"telemetry eligibility is {telemetry.authoritative_eligibility.value}"
                        if telemetry is not None else "telemetry manifest unavailable"
                    )
                ),
                evidence_paths=[monthly_result.telemetry_manifest_path],
            ),
            MonthlyGateCheck(
                name="replay_parity",
                passed=parity_report is not None and parity_report.eligible_for_authoritative_validation,
                reason=(
                    ""
                    if parity_report is not None and parity_report.eligible_for_authoritative_validation
                    else "replay parity is missing or not authoritative"
                ),
                evidence_paths=[monthly_result.replay_parity_path],
            ),
            self._approval_payload_gate(candidate),
            self._bool_gate(
                candidate,
                name="no_leakage",
                keys=("leakage_passed", "no_leakage", "fold_leakage_passed"),
                missing_reason="leakage check evidence is missing",
            ),
            self._improvement_gate(candidate),
            self._calibration_gate(candidate),
            self._trade_count_gate(candidate),
            self._bool_gate(
                candidate,
                name="realistic_costs",
                keys=("cost_gate_passed", "realistic_costs_passed", "slippage_cost_gate_passed"),
                missing_reason="realistic fee/slippage/funding evidence is missing",
            ),
            self._drawdown_gate(candidate),
            self._outlier_gate(candidate),
            self._bool_gate(
                candidate,
                name="risk_constraints",
                keys=("risk_constraints_passed", "portfolio_risk_constraints_passed"),
                missing_reason="portfolio/risk-constraint evidence is missing",
            ),
            self._phase_support_gate(candidate),
            self._decision_parity_gate(candidate, monthly_result),
            self._strategy_plugin_contract_gate(candidate, monthly_result),
            self._outcome_prior_gate(candidate),
            self._model_review_gate(candidate, model_validation),
        ]
        return MonthlyCandidateGateReport(
            candidate_id=candidate.candidate_id,
            source=candidate.source,
            checks=checks,
        )

    def build_packet(
        self,
        *,
        candidate: MonthlyImprovementCandidate,
        gate_report: MonthlyCandidateGateReport,
        monthly_result: MonthlyValidationResult,
        coverage: MarketDataManifest | None,
        telemetry: TelemetryManifest,
        parity_report: ReplayParityReport | None,
        monthly_result_path: Path,
        model_review_path: str = "",
        model_validation: MonthlyModelValidationResult | None = None,
    ) -> MonthlyApprovalEvidencePacket:
        artifact_paths = _dedupe([
            *monthly_result.evidence_paths,
            str(monthly_result_path),
            *candidate.evidence_paths,
            *candidate.artifact_paths,
        ])
        latest_delta = _candidate_float(candidate, "latest_month_oos_delta", "latest_month_objective_delta")
        calibration_delta = _candidate_float(candidate, "calibration_objective_delta", "calibration_delta")
        human_summary = (
            f"{candidate.source.value} candidate {candidate.candidate_id} for "
            f"{monthly_result.bot_id}/{monthly_result.strategy_id}: "
            f"objective_delta={candidate.objective_delta:+.4f}, "
            f"gates={'pass' if gate_report.passed else 'fail'}."
        )
        return MonthlyApprovalEvidencePacket(
            candidate_id=candidate.candidate_id,
            run_id=monthly_result.run_id,
            run_month=monthly_result.run_month,
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
            strategy_change_record_id=monthly_result.strategy_change_record_id,
            title=candidate.title,
            reason_for_change=monthly_result.gap_attribution.summary or candidate.description,
            incumbent_validation_summary=f"Monthly status: {monthly_result.status.value}",
            smoke_or_phased_evidence=f"{candidate.source.value}:{candidate.family}",
            objective_deltas={
                **candidate.objective_deltas,
                "candidate_objective_delta": candidate.objective_delta,
            },
            latest_month_behavior=(
                f"latest OOS delta {latest_delta:+.4f}"
                if latest_delta is not None else "latest OOS improvement evidence supplied by gate inputs"
            ),
            calibration_support=(
                f"calibration delta {calibration_delta:+.4f}"
                if calibration_delta is not None else "calibration support evidence supplied by gate inputs"
            ),
            data_coverage_status=(
                f"coverage={coverage.coverage_ratio:.3f}, usable={coverage.usable_for_authoritative_validation}"
                if coverage else "market data manifest unavailable"
            ),
            replay_parity_status=parity_report.status.value if parity_report else "missing",
            risk_classification=candidate.risk_classification,
            rollback_plan=candidate.rollback_plan,
            artifact_paths=artifact_paths,
            model_review_path=model_review_path,
            human_summary=human_summary,
            machine_readable_payload={
                "candidate": candidate.model_dump(mode="json"),
                "gate_report": gate_report.model_dump(mode="json"),
                "monthly_validation": monthly_result.model_dump(mode="json"),
                "telemetry_eligibility": telemetry.authoritative_eligibility.value,
                "model_review_validation": (
                    model_validation.model_dump(mode="json")
                    if model_validation is not None else None
                ),
            },
            approval_ready=False,
        )

    def _load_selected_candidates(
        self,
        artifact_index: BacktestArtifactIndex,
        *,
        bot_id: str,
        strategy_id: str,
    ) -> list[MonthlyImprovementCandidate]:
        raw = _load_json_artifact(artifact_index, "selected_candidates.json")
        default_source = _source_from_mode_decision(artifact_index)
        rows: list[dict[str, Any]]
        if isinstance(raw, list):
            rows = [item for item in raw if isinstance(item, dict)]
        elif isinstance(raw, dict):
            candidates = (
                raw.get("candidates")
                or raw.get("selected_candidates")
                or raw.get("selected")
                or raw.get("shortlist")
                or []
            )
            rows = [item for item in candidates if isinstance(item, dict)]
        else:
            rows = []
        return [
            MonthlyImprovementCandidate.from_raw(
                row,
                bot_id=bot_id,
                strategy_id=strategy_id,
                default_source=default_source,
            )
            for row in rows
        ]

    @staticmethod
    def _normalize_candidate_paths(
        candidate: MonthlyImprovementCandidate,
        artifact_index: BacktestArtifactIndex,
    ) -> None:
        root = Path(artifact_index.artifact_root)
        candidate.evidence_paths = _dedupe([
            _resolve_artifact_path(path, root)
            for path in candidate.evidence_paths
        ])
        candidate.artifact_paths = _dedupe([
            _resolve_artifact_path(path, root)
            for path in candidate.artifact_paths
        ])
        candidate.candidate_workspace_path = _resolve_artifact_path(
            candidate.candidate_workspace_path,
            root,
        )
        for attr in (
            "workflow_contract_path",
            "live_repo_patch_path",
            "backtest_adapter_patch_path",
            "config_patch_path",
            "decision_parity_report_path",
            "fold_manifest_path",
            "rounds_manifest_path",
            "end_of_round_diagnostics_path",
            "confirmatory_rerank_path",
            "checkpoint_path",
        ):
            resolved = _resolve_artifact_path(getattr(candidate, attr), root)
            setattr(candidate, attr, resolved)
            if resolved:
                candidate.artifact_paths.append(resolved)
        candidate.artifact_paths = _dedupe(candidate.artifact_paths)
        candidate.deterministic_gate_inputs.setdefault("artifact_root", str(root))

    def _load_rejected_candidates(self, artifact_index: BacktestArtifactIndex) -> list[dict[str, Any]]:
        rows = _load_jsonl_artifact(artifact_index, "rejected_candidates.jsonl")
        return rows

    def _load_and_validate_model_review(
        self,
        *,
        model_review_path: str,
        monthly_result: MonthlyValidationResult,
        allowed_evidence_paths: list[str],
    ) -> tuple[MonthlyModelReview | None, MonthlyModelValidationResult | None, str]:
        path = Path(model_review_path)
        if not path.exists():
            return None, None, ""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None, None, ""
        review = _parse_model_review_text(text)
        validation = MonthlyModelResponseValidator().validate(
            review,
            allowed_evidence_paths=[*allowed_evidence_paths, str(path)],
            expected_run_id=monthly_result.run_id,
            expected_bot_id=monthly_result.bot_id,
            expected_strategy_id=monthly_result.strategy_id,
        )
        validation_path = path.with_name("model_review_validation.json")
        validation_path.write_text(validation.model_dump_json(indent=2), encoding="utf-8")
        return review, validation, str(validation_path)

    def _record_proposal(
        self,
        candidate: MonthlyImprovementCandidate,
        packet: MonthlyApprovalEvidencePacket,
        gate_report: MonthlyCandidateGateReport,
        monthly_result: MonthlyValidationResult,
    ) -> str:
        proposal_id = make_proposal_id(
            _proposal_source(candidate.source),
            monthly_result.bot_id,
            _proposal_kind(candidate.change_kind),
            candidate.title,
            strategy_id=monthly_result.strategy_id,
            link_key=f"{monthly_result.run_id}:{candidate.candidate_id}",
        )
        if self.proposal_ledger is None:
            return proposal_id

        affected_parameters = [
            str(change.get("param_name") or change.get("parameter") or "")
            for change in candidate.param_changes
            if str(change.get("param_name") or change.get("parameter") or "")
        ]
        affected_files = _dedupe([
            *candidate.planned_files,
            *[
                str(change.get("file_path") or "")
                for change in candidate.file_changes
                if str(change.get("file_path") or "")
            ],
        ])
        candidate_record = ProposalCandidate(
            proposal_id=proposal_id,
            source=_proposal_source(candidate.source),
            kind=_proposal_kind(candidate.change_kind),
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
            lifecycle_stage=candidate.family,
            title=candidate.title,
            description=candidate.description or packet.reason_for_change,
            expected_mechanism=candidate.raw_payload.get("expected_mechanism", ""),
            affected_parameters=affected_parameters,
            affected_files=affected_files,
            acceptance_criteria=candidate.acceptance_criteria,
            evaluation_method=candidate.source.value,
            linked_diagnostics=[monthly_result.gap_attribution.primary_category.value],
            linked_run_id=monthly_result.run_id,
            suggestion_id=proposal_id,
        )
        self.proposal_ledger.record_candidate(candidate_record)
        self.proposal_ledger.record_evaluation(
            proposal_id,
            ProposalEvaluation(
                proposal_id=proposal_id,
                method="monthly_candidate_gates",
                summary=packet.human_summary,
                objective_score=candidate.objective_score or candidate.objective_delta,
                confidence=1.0 if gate_report.passed else 0.0,
                decision="approve" if gate_report.passed else "reject",
                decision_reason="; ".join(gate_report.blocking_reasons),
                evidence_paths=packet.artifact_paths,
            ),
        )
        return proposal_id

    def _record_strategy_change(
        self,
        candidate: MonthlyImprovementCandidate,
        packet: MonthlyApprovalEvidencePacket,
        request: ApprovalRequest,
    ) -> str:
        if self.strategy_change_ledger is None:
            return ""
        record = StrategyChangeRecord(
            bot_id=packet.bot_id,
            strategy_id=packet.strategy_id,
            record_type=StrategyChangeRecordType.PROPOSED_CHANGE,
            prior_config_version=str(candidate.raw_payload.get("prior_config_version") or ""),
            new_config_version=str(candidate.raw_payload.get("new_config_version") or candidate.raw_payload.get("proposed_config_version") or ""),
            mutation_diff={
                "candidate_id": candidate.candidate_id,
                "source": candidate.source.value,
                "family": candidate.family,
                "change_kind": candidate.change_kind,
                "param_changes": candidate.param_changes,
                "file_changes": candidate.file_changes,
                "proposed_changes": candidate.proposed_changes,
            },
            source_proposal_ids=[packet.proposal_id] if packet.proposal_id else [],
            source_suggestion_ids=[packet.suggestion_id] if packet.suggestion_id else [],
            approval_request_id=request.request_id,
            evidence_paths=packet.artifact_paths,
            objective_deltas=packet.objective_deltas,
            decision_reason=packet.human_summary,
            monthly_status="approval_ready",
            run_id=packet.run_id,
            run_month=packet.run_month,
        )
        self.strategy_change_ledger.record_proposed_change(record)
        return record.record_id

    def _build_approval_request(
        self,
        packet: MonthlyApprovalEvidencePacket,
        candidate: MonthlyImprovementCandidate,
    ) -> ApprovalRequest:
        request_id = hashlib.sha256(
            f"{packet.run_id}:{packet.candidate_id}:{packet.proposal_id}".encode(),
        ).hexdigest()[:12]
        file_changes = [
            FileChange.model_validate(change)
            for change in candidate.file_changes
            if isinstance(change, dict) and str(change.get("file_path") or "")
        ]
        planned_files = _dedupe([
            *candidate.planned_files,
            *[change.file_path for change in file_changes],
        ])
        return ApprovalRequest(
            request_id=request_id,
            suggestion_id=packet.proposal_id or packet.candidate_id,
            bot_id=packet.bot_id,
            strategy_id=packet.strategy_id,
            change_kind=_change_kind(candidate.change_kind, has_file_changes=bool(file_changes)),
            title=packet.title,
            summary=packet.human_summary,
            param_changes=candidate.param_changes,
            file_changes=file_changes,
            planned_files=planned_files,
            verification_commands=_string_list(candidate.raw_payload.get("verification_commands")),
            risk_tier=_risk_tier(packet.risk_classification),
            draft_pr=False,
            implementation_notes=json.dumps(packet.machine_readable_payload, indent=2, default=str),
            monthly_run_id=packet.run_id,
            monthly_run_month=packet.run_month,
            proposal_id=packet.proposal_id,
            evidence_paths=packet.artifact_paths,
            objective_deltas=packet.objective_deltas,
            risk_classification=packet.risk_classification.value,
            rollback_plan=packet.rollback_plan,
            approval_packet_path=packet.approval_packet_path,
            machine_readable_payload=packet.machine_readable_payload,
        )

    @staticmethod
    def _suppression_reasons(
        gate_report: MonthlyCandidateGateReport,
        *,
        shadow: bool,
        approval_tracker_present: bool,
    ) -> list[str]:
        reasons = list(gate_report.blocking_reasons)
        if shadow:
            reasons.append("monthly validation is running in shadow mode")
        if gate_report.passed and not approval_tracker_present:
            reasons.append("approval tracker is unavailable")
        return _dedupe(reasons)

    def _bool_gate(
        self,
        candidate: MonthlyImprovementCandidate,
        *,
        name: str,
        keys: tuple[str, ...],
        missing_reason: str,
    ) -> MonthlyGateCheck:
        value = _candidate_bool(candidate, *keys)
        return MonthlyGateCheck(
            name=name,
            passed=value is True,
            reason="" if value is True else (f"{name} did not pass" if value is False else missing_reason),
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _approval_payload_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        missing: list[str] = []
        if not candidate.rollback_plan:
            missing.append("rollback_plan")
        if not candidate.replay_or_experiment_plan:
            missing.append("replay_or_experiment_plan")
        if not candidate.acceptance_criteria:
            missing.append("acceptance_criteria")
        if not any([
            candidate.param_changes,
            candidate.file_changes,
            candidate.proposed_changes,
            candidate.planned_files,
        ]):
            missing.append("change_payload")
        return MonthlyGateCheck(
            name="approval_packet_payload",
            passed=not missing,
            reason="" if not missing else f"candidate is missing approval payload fields: {', '.join(missing)}",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _phase_support_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        if candidate.source != MonthlyCandidateSource.PHASED_AUTO:
            return MonthlyGateCheck(
                name="purged_fold_support",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="not required for non-phased-auto candidate",
                evidence_paths=candidate.evidence_paths,
            )
        explicit = _candidate_bool(
            candidate,
            "fold_support_passed",
            "purged_fold_support",
            "positive_fold_support",
            "folds_positive",
        )
        return MonthlyGateCheck(
            name="purged_fold_support",
            passed=explicit is True,
            reason="" if explicit is True else "phased-auto candidate lacks positive purged-fold support",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _decision_parity_gate(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
    ) -> MonthlyGateCheck:
        if not _is_structural_candidate(candidate):
            return MonthlyGateCheck(
                name="decision_parity_report",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="not required for non-structural candidate",
                evidence_paths=candidate.evidence_paths,
            )
        path = Path(candidate.decision_parity_report_path)
        if not candidate.decision_parity_report_path or not path.exists():
            return MonthlyGateCheck(
                name="decision_parity_report",
                passed=False,
                reason="structural candidate requires decision_parity_report_path",
                evidence_paths=candidate.evidence_paths,
            )
        try:
            report = DecisionParityReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            return MonthlyGateCheck(
                name="decision_parity_report",
                passed=False,
                reason=f"decision parity report is invalid: {exc}",
                evidence_paths=[str(path)],
            )
        reasons: list[str] = []
        manifest = _load_run_manifest(monthly_result.run_manifest_path)
        contract = _load_strategy_plugin_contract(manifest)
        if report.run_id != monthly_result.run_id:
            reasons.append("decision parity run_id does not match monthly run")
        if report.candidate_id != candidate.candidate_id:
            reasons.append("decision parity candidate_id does not match candidate")
        if not report.eligible_for_structural_approval:
            reasons.append("decision parity report is not pass")
        evidence_paths = _dedupe([
            *report.evidence_paths,
            *[
                evidence_path
                for check in report.checks
                for evidence_path in check.evidence_paths
            ],
        ])
        missing_evidence = [evidence_path for evidence_path in evidence_paths if not Path(evidence_path).exists()]
        if missing_evidence:
            reasons.append(
                "decision parity evidence paths do not exist: "
                + ", ".join(missing_evidence[:5])
            )
        root = (
            Path(monthly_result.artifact_index_path).parent
            if monthly_result.artifact_index_path else None
        )
        outside_evidence = _paths_outside_root(evidence_paths, root) if root is not None else []
        if outside_evidence:
            reasons.append(
                "decision parity evidence paths outside artifact_root: "
                + ", ".join(outside_evidence[:5])
            )
        if manifest and manifest.strategy_plugin_id and report.strategy_plugin_id != manifest.strategy_plugin_id:
            reasons.append("decision parity strategy_plugin_id does not match run manifest")
        if contract:
            if report.strategy_plugin_id != contract.plugin_id:
                reasons.append("decision parity strategy_plugin_id does not match plugin contract")
            if report.live_repo_commit_sha != contract.live_repo_commit_sha:
                reasons.append("decision parity live_repo_commit_sha does not match plugin contract")
            if report.backtest_adapter_commit_sha != contract.backtest_adapter_commit_sha:
                reasons.append("decision parity backtest_adapter_commit_sha does not match plugin contract")
        return MonthlyGateCheck(
            name="decision_parity_report",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[str(path), *evidence_paths],
        )

    @staticmethod
    def _strategy_plugin_contract_gate(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
    ) -> MonthlyGateCheck:
        if not _is_structural_candidate(candidate):
            return MonthlyGateCheck(
                name="strategy_plugin_contract_maturity",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="not required for non-structural candidate",
                evidence_paths=candidate.evidence_paths,
            )
        manifest = _load_run_manifest(monthly_result.run_manifest_path)
        contract_path = Path(manifest.strategy_plugin_contract_path) if manifest and manifest.strategy_plugin_contract_path else None
        if contract_path is None or not contract_path.exists():
            return MonthlyGateCheck(
                name="strategy_plugin_contract_maturity",
                passed=False,
                reason="structural candidate requires strategy plugin contract evidence",
                evidence_paths=[monthly_result.run_manifest_path],
            )
        try:
            contract = StrategyPluginContract.model_validate(
                json.loads(contract_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            return MonthlyGateCheck(
                name="strategy_plugin_contract_maturity",
                passed=False,
                reason=f"strategy plugin contract is invalid: {exc}",
                evidence_paths=[str(contract_path)],
            )
        reasons: list[str] = []
        if not contract.eligible_for_approval:
            reasons.append(f"strategy plugin contract maturity is {contract.maturity.value}")
        if manifest and manifest.strategy_plugin_id and contract.plugin_id != manifest.strategy_plugin_id:
            reasons.append("strategy plugin contract plugin_id does not match run manifest")
        return MonthlyGateCheck(
            name="strategy_plugin_contract_maturity",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[str(contract_path), *contract.parity_fixture_set],
        )

    @staticmethod
    def _runner_contract_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        version = str(
            candidate.deterministic_gate_inputs.get("runner_contract_version")
            or candidate.deterministic_gate_inputs.get("source_runner_contract_version")
            or candidate.workflow_contract_version
            or ""
        ).strip()
        expected = _runner_contract_version(candidate.source)
        passed = bool(expected and version == expected)
        return MonthlyGateCheck(
            name="source_runner_contract",
            passed=passed,
            reason="" if passed else f"candidate missing {expected or 'known'} runner contract",
            evidence_paths=[*candidate.evidence_paths, *candidate.artifact_paths],
        )

    @staticmethod
    def _candidate_lineage_gate(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex | None,
    ) -> MonthlyGateCheck:
        reasons: list[str] = []
        if candidate.run_id != monthly_result.run_id:
            reasons.append("candidate run_id does not match monthly run")
        run_manifest = _load_run_manifest(monthly_result.run_manifest_path)
        manifest_id = (
            artifact_index.manifest_id
            if artifact_index is not None and artifact_index.manifest_id
            else (run_manifest.manifest_id if run_manifest is not None else "")
        )
        if manifest_id and candidate.manifest_id != manifest_id:
            reasons.append("candidate manifest_id does not match artifact index")
        for field_name in ("round_id", "prior_round_id", "next_round_id"):
            if not str(getattr(candidate, field_name) or "").strip():
                reasons.append(f"candidate missing {field_name}")
        if not candidate.backtest_repo_commit_sha:
            reasons.append("candidate missing backtest_repo_commit_sha")
        if not (candidate.live_trading_repo_commit_sha or candidate.code_sha):
            reasons.append("candidate missing live_trading_repo_commit_sha")
        if not candidate.control_plane_commit_sha:
            reasons.append("candidate missing control_plane_commit_sha")
        return MonthlyGateCheck(
            name="candidate_lineage_contract",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[
                monthly_result.run_manifest_path,
                *candidate.evidence_paths,
                *candidate.artifact_paths,
            ],
        )

    @staticmethod
    def _candidate_workspace_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        supplied = any([
            candidate.candidate_workspace_key,
            candidate.candidate_workspace_path,
            candidate.candidate_attempt_id,
            candidate.candidate_attempt_status,
        ])
        if not supplied:
            return MonthlyGateCheck(
                name="candidate_workspace_attempt",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="candidate workspace/attempt metadata not supplied",
                evidence_paths=candidate.evidence_paths,
            )

        reasons: list[str] = []
        key = candidate.candidate_workspace_key
        if not key or key != _safe_id(key):
            reasons.append("candidate_workspace_key is missing or unsafe")
        status = candidate.candidate_attempt_status.strip().lower()
        if status not in {"completed", "succeeded", "success"}:
            reasons.append("candidate attempt did not complete successfully")
        path = Path(candidate.candidate_workspace_path)
        root = Path(str(candidate.deterministic_gate_inputs.get("artifact_root") or ""))
        if not candidate.candidate_workspace_path or not path.exists():
            reasons.append("candidate_workspace_path is missing or does not exist")
        elif root:
            try:
                path.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                reasons.append("candidate_workspace_path is outside artifact_root")
        if candidate.stall_timeout_seconds < 0:
            reasons.append("stall_timeout_seconds cannot be negative")

        return MonthlyGateCheck(
            name="candidate_workspace_attempt",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[candidate.candidate_workspace_path, *candidate.evidence_paths],
        )

    @staticmethod
    def _candidate_artifact_containment_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        paths = _dedupe([*candidate.evidence_paths, *candidate.artifact_paths])
        if not paths:
            return MonthlyGateCheck(
                name="candidate_artifact_containment",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="candidate has no evidence/artifact paths to contain",
                evidence_paths=[],
            )
        root = Path(str(candidate.deterministic_gate_inputs.get("artifact_root") or ""))
        outside = _paths_outside_root(paths, root) if root else paths
        return MonthlyGateCheck(
            name="candidate_artifact_containment",
            passed=not outside,
            reason=(
                ""
                if not outside
                else "candidate evidence/artifact paths outside artifact_root: "
                + ", ".join(outside[:5])
            ),
            evidence_paths=paths,
        )

    @staticmethod
    def _model_review_gate(
        candidate: MonthlyImprovementCandidate,
        validation: MonthlyModelValidationResult | None,
    ) -> MonthlyGateCheck:
        if validation is None:
            return MonthlyGateCheck(
                name="monthly_model_review",
                passed=False,
                reason="valid monthly model review is required before approval-ready candidates",
                evidence_paths=candidate.evidence_paths,
            )
        if not validation.valid:
            reasons = [issue.message for issue in validation.issues[:3]]
            return MonthlyGateCheck(
                name="monthly_model_review",
                passed=False,
                reason="monthly model review failed validation: " + "; ".join(reasons),
                evidence_paths=candidate.evidence_paths,
            )
        passed = candidate.candidate_id in set(validation.actionable_candidate_ids)
        return MonthlyGateCheck(
            name="monthly_model_review",
            passed=passed,
            reason="" if passed else "model review did not route this candidate as actionable",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _improvement_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "latest_month_oos_improvement", "latest_oos_improvement")
        delta = _candidate_float(candidate, "latest_month_oos_delta", "latest_month_objective_delta")
        passed = explicit if explicit is not None else (delta is not None and delta > 0)
        return MonthlyGateCheck(
            name="latest_month_oos_improvement",
            passed=passed is True,
            reason="" if passed is True else "latest-month selection-OOS improvement is missing or non-positive",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _calibration_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "calibration_support", "calibration_supported")
        delta = _candidate_float(candidate, "calibration_objective_delta", "calibration_delta")
        passed = explicit if explicit is not None else (delta is not None and delta > 0)
        return MonthlyGateCheck(
            name="calibration_support",
            passed=passed is True,
            reason="" if passed is True else "calibration support is missing or non-positive",
            evidence_paths=candidate.evidence_paths,
        )

    def _trade_count_gate(self, candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "sufficient_trade_count")
        sparse = _candidate_value(candidate, "sparse_sample_classification", "sample_classification")
        trade_count = _candidate_float(candidate, "trade_count", "selection_trade_count")
        passed = explicit if explicit is not None else False
        if explicit is None and trade_count is not None:
            passed = trade_count >= self.min_trade_count
        if not passed and str(sparse or "").strip():
            passed = True
        return MonthlyGateCheck(
            name="trade_count_or_sparse_classification",
            passed=passed,
            reason="" if passed else "trade count is insufficient and no sparse-sample classification was supplied",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _drawdown_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "drawdown_gate_passed", "no_material_drawdown_increase")
        delta = _candidate_float(candidate, "max_drawdown_delta_pct", "drawdown_delta_pct")
        passed = explicit if explicit is not None else (delta is not None and delta <= 0)
        return MonthlyGateCheck(
            name="drawdown_gate",
            passed=passed is True,
            reason="" if passed is True else "candidate materially increases max drawdown or lacks drawdown evidence",
            evidence_paths=candidate.evidence_paths,
        )

    def _outlier_gate(self, candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "outlier_concentration_passed", "no_outlier_dependency")
        concentration = _candidate_float(candidate, "outlier_win_concentration", "top_win_concentration")
        passed = explicit if explicit is not None else (
            concentration is not None and concentration <= self.max_outlier_win_concentration
        )
        return MonthlyGateCheck(
            name="outlier_concentration",
            passed=passed is True,
            reason="" if passed is True else "candidate depends on too few outlier wins or lacks outlier evidence",
            evidence_paths=candidate.evidence_paths,
        )

    def _outcome_prior_gate(self, candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        if self.search_allocation_policy is None:
            return MonthlyGateCheck(
                name="outcome_prior_controls",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="no outcome priors available",
                evidence_paths=candidate.evidence_paths,
            )
        requires_stronger = self.search_allocation_policy.requires_stronger_evidence(
            bot_id=candidate.bot_id,
            strategy_id=candidate.strategy_id,
            mutation_family=candidate.family,
            category=candidate.change_kind,
        )
        if not requires_stronger:
            return MonthlyGateCheck(
                name="outcome_prior_controls",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="no negative monthly prior for this family",
                evidence_paths=candidate.evidence_paths,
            )
        explicit = _candidate_bool(
            candidate,
            "stronger_evidence_passed",
            "negative_prior_override",
            "authoritative_prior_override",
        )
        return MonthlyGateCheck(
            name="outcome_prior_controls",
            passed=explicit is True,
            reason=(
                ""
                if explicit is True
                else "negative monthly priors require stronger validation evidence"
            ),
            evidence_paths=candidate.evidence_paths,
        )


def _load_json_artifact(index: BacktestArtifactIndex, name: str) -> Any:
    path = index.artifact_path(name)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_jsonl_artifact(index: BacktestArtifactIndex, name: str) -> list[dict[str, Any]]:
    path = index.artifact_path(name)
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    except Exception:
        return rows
    return rows


def _load_run_manifest(path: str) -> MonthlyRunManifest | None:
    if not path:
        return None
    try:
        return MonthlyRunManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return None


def _load_strategy_plugin_contract(manifest: MonthlyRunManifest | None) -> StrategyPluginContract | None:
    if not manifest or not manifest.strategy_plugin_contract_path:
        return None
    path = Path(manifest.strategy_plugin_contract_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        return StrategyPluginContract.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _is_structural_candidate(candidate: MonthlyImprovementCandidate) -> bool:
    return (
        candidate.change_kind == "structural_change"
        or bool(candidate.file_changes)
        or bool(candidate.live_repo_patch_path)
        or bool(candidate.backtest_adapter_patch_path)
    )


def _parse_model_review_text(text: str) -> MonthlyModelReview:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return parse_monthly_model_review(text)
    if isinstance(raw, dict):
        try:
            return MonthlyModelReview.model_validate(raw)
        except Exception:
            return parse_monthly_model_review(text)
    return parse_monthly_model_review(text)


def _runner_contract_version(source: MonthlyCandidateSource) -> str:
    if source == MonthlyCandidateSource.SMOKE_REPAIR:
        return "smoke_repair_runner_contract_v1"
    if source == MonthlyCandidateSource.PHASED_AUTO:
        return "phased_auto_runner_contract_v1"
    return ""


def _source_from_mode_decision(index: BacktestArtifactIndex) -> MonthlyCandidateSource:
    data = _load_json_artifact(index, "mode_decision.json")
    if not isinstance(data, dict):
        return MonthlyCandidateSource.UNKNOWN
    raw = (
        data.get("candidate_source")
        or data.get("mode")
        or data.get("routing")
        or data.get("decision")
        or data.get("status")
    )
    if not raw:
        return MonthlyCandidateSource.UNKNOWN
    normalized = str(raw).strip().lower().replace("-", "_")
    if normalized in {"smoke", "smoke_repair", "repair", "rollback"}:
        return MonthlyCandidateSource.SMOKE_REPAIR
    if normalized in {"phased", "phased_auto", "auto", "experiment"}:
        return MonthlyCandidateSource.PHASED_AUTO
    return MonthlyCandidateSource.UNKNOWN


def _resolve_artifact_path(path: str, root: Path) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate)


def _existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if path and Path(path).exists()]


def _paths_outside_root(paths: list[str], root: Path) -> list[str]:
    try:
        resolved_root = root.resolve()
    except OSError:
        return [path for path in paths if path]
    outside: list[str] = []
    for path in paths:
        if not path:
            continue
        try:
            Path(path).resolve().relative_to(resolved_root)
        except (OSError, ValueError):
            outside.append(path)
    return outside


def _candidate_value(candidate: MonthlyImprovementCandidate, *keys: str) -> Any:
    for key in keys:
        if key in candidate.deterministic_gate_inputs:
            return candidate.deterministic_gate_inputs[key]
        if key in candidate.raw_payload:
            return candidate.raw_payload[key]
        if key in candidate.objective_deltas:
            return candidate.objective_deltas[key]
    return None


def _candidate_bool(candidate: MonthlyImprovementCandidate, *keys: str) -> bool | None:
    value = _candidate_value(candidate, *keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "pass", "passed", "yes", "y"}:
            return True
        if lowered in {"false", "fail", "failed", "no", "n"}:
            return False
    return None


def _candidate_float(candidate: MonthlyImprovementCandidate, *keys: str) -> float | None:
    value = _candidate_value(candidate, *keys)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _proposal_source(source: MonthlyCandidateSource) -> ProposalSource:
    if source == MonthlyCandidateSource.SMOKE_REPAIR:
        return ProposalSource.MONTHLY_SMOKE_REPAIR
    if source == MonthlyCandidateSource.PHASED_AUTO:
        return ProposalSource.MONTHLY_PHASED_AUTO
    if source == MonthlyCandidateSource.MODEL_REVIEW:
        return ProposalSource.MONTHLY_MODEL_REVIEW
    return ProposalSource.DETERMINISTIC


def _proposal_kind(change_kind: str) -> ProposalKind:
    try:
        return ProposalKind(change_kind)
    except ValueError:
        if change_kind == "rollback":
            return ProposalKind.ROLLBACK
        return ProposalKind.STRUCTURAL_CHANGE


def _change_kind(change_kind: str, *, has_file_changes: bool) -> ChangeKind:
    raw = (change_kind or "").strip().lower()
    if raw == ChangeKind.ROLLBACK.value:
        return ChangeKind.ROLLBACK
    if raw == ChangeKind.BUG_FIX.value:
        return ChangeKind.BUG_FIX
    if raw == ChangeKind.STRUCTURAL_CHANGE.value or has_file_changes:
        return ChangeKind.STRUCTURAL_CHANGE
    return ChangeKind.PARAMETER_CHANGE


def _risk_tier(risk: MonthlyRiskClassification) -> RepoRiskTier:
    if risk in {MonthlyRiskClassification.HIGH, MonthlyRiskClassification.CRITICAL}:
        return RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
    return RepoRiskTier.REQUIRES_APPROVAL


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:64] or "candidate"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
