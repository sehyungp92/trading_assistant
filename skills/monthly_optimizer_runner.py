"""Control-plane runner for the Phase 4 monthly optimizer sequence.

The sibling backtest repository owns replay/search execution.  This module owns
the durable orchestration contract around that external runner: workspace and
attempt state, manifest freezing, artifact validation, and round_N+1 adoption
checks before the normal monthly candidate pipeline can trust the output.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from schemas.backtest_artifacts import (
    BacktestArtifactIndex,
    PHASE4_OOS_REPAIR_ARTIFACTS,
    PHASE4_OPTIMIZER_ARTIFACTS,
)
from schemas.data_bundle_manifest import DataBundleManifest, DataBundleStatus
from schemas.decision_parity import DecisionParityReport
from schemas.monthly_candidates import MonthlyCandidateSource, MonthlyImprovementCandidate
from schemas.monthly_optimizer import (
    CandidateAttemptRecord,
    CandidateAttemptState,
    CandidateWorkspaceManifest,
    ConfirmatoryRerank,
    FoldManifest,
    FoldSpec,
    MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION,
    MonthlyOptimizerSequenceResult,
    OptimizerExperimentPlan,
    OptimizerSequenceStatus,
    OptimizerStage,
    RoundsManifest,
    sanitize_workspace_key,
)
from schemas.monthly_run_manifest import MonthlyRunManifest, MonthlyRunMode
from schemas.strategy_plugin_contract import StrategyPluginContract
from skills.backtest_runner_client import BacktestRunnerClient


ACTIVE_ATTEMPT_STATES = {
    CandidateAttemptState.CLAIMED,
    CandidateAttemptState.RUNNING,
    CandidateAttemptState.RETRY_QUEUED,
}
TERMINAL_ATTEMPT_STATES = {
    CandidateAttemptState.RELEASED,
    CandidateAttemptState.SUCCEEDED,
    CandidateAttemptState.FAILED,
    CandidateAttemptState.TIMED_OUT,
    CandidateAttemptState.STALLED,
    CandidateAttemptState.CANCELED_BY_RECONCILIATION,
}

CORE_PHASE4_SEQUENCE_ARTIFACTS = [
    "fold_manifest.json",
    "rounds_manifest.json",
    "end_of_round_diagnostics.json",
    "llm_experiment_plan.json",
    "fold_validation.json",
    "confirmatory_rerank.json",
    "runner_observability.json",
]

ADOPTION_GATE_ARTIFACTS = [
    "leakage_report.json",
    "cost_sensitivity.json",
    "outlier_sensitivity.json",
    "portfolio_synergy.json",
]


@dataclass(frozen=True)
class CandidateCommandResult:
    attempt_id: str
    state: CandidateAttemptState
    return_code: int
    cwd: str
    stdout_path: str = ""
    stderr_path: str = ""
    error: str = ""


class CandidateWorkspaceManager:
    """Creates deterministic path-contained candidate workspaces."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        *,
        run_id: str,
        candidate_id: str,
        workspace_key: str = "",
        structural: bool = False,
    ) -> CandidateWorkspaceManifest:
        safe_key = sanitize_workspace_key(workspace_key or f"{run_id}-{candidate_id}")
        workspace_path = (self.workspace_root / safe_key).resolve()
        workspace_path.relative_to(self.workspace_root.resolve())
        workspace_path.mkdir(parents=True, exist_ok=True)
        manifest_path = workspace_path / "candidate_workspace_manifest.json"
        manifest = CandidateWorkspaceManifest(
            run_id=run_id,
            candidate_id=candidate_id,
            workspace_key=safe_key,
            workspace_root=str(self.workspace_root),
            workspace_path=str(workspace_path),
            manifest_path=str(manifest_path),
            structural=structural,
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest


class CandidateAttemptStore:
    """Append-only state log for Symphony-style candidate attempts."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CandidateAttemptRecord) -> CandidateAttemptRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record

    def load(self) -> list[CandidateAttemptRecord]:
        if not self.path.exists():
            return []
        records: list[CandidateAttemptRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(CandidateAttemptRecord.model_validate(json.loads(line)))
        return records

    def latest_by_attempt(self) -> dict[str, CandidateAttemptRecord]:
        latest: dict[str, CandidateAttemptRecord] = {}
        for record in self.load():
            latest[record.attempt_id] = record
        return latest

    def claim(
        self,
        *,
        run_id: str,
        candidate_id: str,
        workspace: CandidateWorkspaceManifest,
        manifest_id: str,
        stage: OptimizerStage = OptimizerStage.PHASED_AUTO,
        stall_timeout_seconds: int = 0,
        backtest_repo_commit_sha: str = "",
        trading_repo_commit_sha: str = "",
    ) -> CandidateAttemptRecord:
        records = self.load()
        for record in _latest_attempts(records).values():
            if (
                record.run_id == run_id
                and record.candidate_id == candidate_id
                and record.state in ACTIVE_ATTEMPT_STATES
            ):
                raise ValueError(
                    f"candidate {candidate_id} already has active attempt {record.attempt_id}"
                )
        attempt_number = 1 + max(
            (
                record.attempt_number
                for record in records
                if record.run_id == run_id and record.candidate_id == candidate_id
            ),
            default=0,
        )
        attempt_id = _attempt_id(run_id, candidate_id, attempt_number)
        return self.append(CandidateAttemptRecord(
            attempt_id=attempt_id,
            run_id=run_id,
            candidate_id=candidate_id,
            workspace_key=workspace.workspace_key,
            workspace_path=workspace.workspace_path,
            state=CandidateAttemptState.CLAIMED,
            stage=stage,
            attempt_number=attempt_number,
            stall_timeout_seconds=stall_timeout_seconds,
            manifest_id=manifest_id,
            backtest_repo_commit_sha=backtest_repo_commit_sha,
            trading_repo_commit_sha=trading_repo_commit_sha,
        ))

    def transition(
        self,
        attempt_id: str,
        state: CandidateAttemptState,
        *,
        reason: str = "",
        retry_reason: str = "",
        subprocess_pid: int | None = None,
        artifact_paths: list[str] | None = None,
        parity_status: str = "",
    ) -> CandidateAttemptRecord:
        latest = self.latest_by_attempt().get(attempt_id)
        if latest is None:
            raise ValueError(f"unknown attempt_id: {attempt_id}")
        retry_attempt = latest.retry_attempt
        if state == CandidateAttemptState.RETRY_QUEUED:
            retry_attempt += 1
        return self.append(latest.model_copy(update={
            "state": state,
            "reason": reason,
            "retry_reason": retry_reason or latest.retry_reason,
            "retry_attempt": retry_attempt,
            "subprocess_pid": subprocess_pid if subprocess_pid is not None else latest.subprocess_pid,
            "artifact_paths": artifact_paths if artifact_paths is not None else latest.artifact_paths,
            "parity_status": parity_status or latest.parity_status,
            "updated_at": datetime.now(timezone.utc),
        }))

    def mark_stalled(self, *, now: datetime | None = None) -> list[CandidateAttemptRecord]:
        current_time = now or datetime.now(timezone.utc)
        stalled: list[CandidateAttemptRecord] = []
        for record in self.latest_by_attempt().values():
            if record.state != CandidateAttemptState.RUNNING:
                continue
            if record.stall_timeout_seconds <= 0:
                continue
            age = current_time - record.updated_at
            if age.total_seconds() >= record.stall_timeout_seconds:
                stalled.append(self.transition(
                    record.attempt_id,
                    CandidateAttemptState.STALLED,
                    reason="attempt exceeded stall_timeout_seconds",
                ))
        return stalled

    def reconcile(
        self,
        *,
        manifest_id: str,
        backtest_repo_commit_sha: str = "",
        trading_repo_commit_sha: str = "",
        eligible_candidate_ids: set[str] | None = None,
    ) -> list[CandidateAttemptRecord]:
        canceled: list[CandidateAttemptRecord] = []
        for record in self.latest_by_attempt().values():
            if record.state not in ACTIVE_ATTEMPT_STATES:
                continue
            reason = ""
            if record.manifest_id and record.manifest_id != manifest_id:
                reason = "run manifest changed"
            elif (
                backtest_repo_commit_sha
                and record.backtest_repo_commit_sha
                and record.backtest_repo_commit_sha != backtest_repo_commit_sha
            ):
                reason = "backtest repo SHA drifted"
            elif (
                trading_repo_commit_sha
                and record.trading_repo_commit_sha
                and record.trading_repo_commit_sha != trading_repo_commit_sha
            ):
                reason = "trading repo SHA drifted"
            elif eligible_candidate_ids is not None and record.candidate_id not in eligible_candidate_ids:
                reason = "candidate is no longer eligible"
            if reason:
                canceled.append(self.transition(
                    record.attempt_id,
                    CandidateAttemptState.CANCELED_BY_RECONCILIATION,
                    reason=reason,
                ))
        return canceled

    @staticmethod
    def retry_backoff_seconds(retry_attempt: int, *, base_seconds: int = 60, max_seconds: int = 900) -> int:
        attempt = max(0, retry_attempt)
        return min(max_seconds, int(base_seconds * math.pow(2, attempt)))


class CandidateAttemptExecutor:
    """Runs candidate-local work with cwd pinned to the candidate workspace."""

    def __init__(self, store: CandidateAttemptStore) -> None:
        self.store = store

    def run(
        self,
        attempt: CandidateAttemptRecord,
        command: list[str],
        *,
        timeout_seconds: int | None = None,
    ) -> CandidateCommandResult:
        if not command:
            raise ValueError("candidate attempt command cannot be empty")
        workspace = Path(attempt.workspace_path).resolve()
        if not workspace.exists() or not workspace.is_dir():
            failed = self.store.transition(
                attempt.attempt_id,
                CandidateAttemptState.FAILED,
                reason="candidate workspace is missing",
            )
            return CandidateCommandResult(
                attempt_id=attempt.attempt_id,
                state=failed.state,
                return_code=-1,
                cwd=str(workspace),
                error="candidate workspace is missing",
            )

        running = self.store.transition(attempt.attempt_id, CandidateAttemptState.RUNNING)
        stdout_path = workspace / f"{sanitize_workspace_key(attempt.attempt_id)}.stdout.log"
        stderr_path = workspace / f"{sanitize_workspace_key(attempt.attempt_id)}.stderr.log"
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds or attempt.stall_timeout_seconds or None,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            timed_out = self.store.transition(
                running.attempt_id,
                CandidateAttemptState.TIMED_OUT,
                reason="candidate command timed out",
                artifact_paths=[str(stdout_path), str(stderr_path)],
            )
            return CandidateCommandResult(
                attempt_id=attempt.attempt_id,
                state=timed_out.state,
                return_code=-1,
                cwd=str(workspace),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error="candidate command timed out",
            )

        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        state = (
            CandidateAttemptState.SUCCEEDED
            if completed.returncode == 0
            else CandidateAttemptState.FAILED
        )
        final = self.store.transition(
            running.attempt_id,
            state,
            reason="" if completed.returncode == 0 else f"exit code {completed.returncode}",
            artifact_paths=[str(stdout_path), str(stderr_path)],
        )
        return CandidateCommandResult(
            attempt_id=attempt.attempt_id,
            state=final.state,
            return_code=completed.returncode,
            cwd=str(workspace),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )


class MonthlyOptimizerRunner:
    """Runs and validates the external Phase 4 optimizer contract."""

    def __init__(self, runner: BacktestRunnerClient | None = None) -> None:
        self.runner = runner or BacktestRunnerClient(timeout_seconds=6 * 3600)

    def prepare_manifest(self, manifest: MonthlyRunManifest) -> MonthlyRunManifest:
        updates: dict[str, Any] = {
            "mode": MonthlyRunMode.PHASED_AUTO,
            "workflow_contract_version": (
                manifest.workflow_contract_version
                if manifest.workflow_contract_version != "monthly_incumbent_validation_v1"
                else MONTHLY_OPTIMIZER_WORKFLOW_CONTRACT_VERSION
            ),
            "max_workers": manifest.max_workers or 2,
            "score_component_cap": min(manifest.score_component_cap or 7, 7),
            "expected_outputs": _dedupe([
                *manifest.expected_outputs,
                *CORE_PHASE4_SEQUENCE_ARTIFACTS,
            ]),
            "output_artifact_names": _dedupe([
                *manifest.output_artifact_names,
                *PHASE4_OPTIMIZER_ARTIFACTS,
                *PHASE4_OOS_REPAIR_ARTIFACTS,
            ]),
        }
        if manifest.selection_oos_start is None:
            updates["selection_oos_start"] = manifest.latest_month_start
            updates["selection_oos_end"] = manifest.latest_month_end
        if manifest.in_sample_start is None and manifest.calibration_start is not None:
            updates["in_sample_start"] = manifest.calibration_start
            updates["in_sample_end"] = manifest.calibration_end
        artifact_root = Path(manifest.artifact_root)
        if not manifest.fold_manifest_path:
            updates["fold_manifest_path"] = str(artifact_root / "fold_manifest.json")
        if not manifest.rounds_manifest_path:
            updates["rounds_manifest_path"] = str(artifact_root / "rounds_manifest.json")
        if not manifest.end_of_round_diagnostics_path:
            updates["end_of_round_diagnostics_path"] = str(
                artifact_root / "end_of_round_diagnostics.json"
            )
        if not manifest.candidate_workspace_root:
            updates["candidate_workspace_root"] = str(artifact_root / "workspaces")
        if not manifest.checkpoint_path:
            updates["checkpoint_path"] = str(artifact_root / "optimizer_checkpoint.json")
        if not manifest.cache_path:
            updates["cache_path"] = str(artifact_root / "optimizer_cache")
        return manifest.model_copy(update=updates)

    def run(
        self,
        manifest: MonthlyRunManifest,
        manifest_path: Path,
        *,
        write_manifest: bool = True,
    ) -> MonthlyOptimizerSequenceResult:
        frozen = self.prepare_manifest(manifest)
        manifest_path = Path(manifest_path)
        if write_manifest:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(frozen.model_dump_json(indent=2), encoding="utf-8")
        runner_result = self.runner.run(frozen, manifest_path)
        artifact_index = runner_result.artifact_index
        if not runner_result.success or artifact_index is None:
            return MonthlyOptimizerSequenceResult(
                run_id=frozen.run_id,
                status=OptimizerSequenceStatus.FAILED,
                blocking_reasons=[runner_result.error or "external optimizer runner failed"],
                artifact_index_path=str(Path(frozen.artifact_root) / "artifact_index.json"),
                evidence_paths=[str(manifest_path)],
            )
        return self.validate_artifacts(frozen, artifact_index, manifest_path=manifest_path)

    def validate_artifacts(
        self,
        manifest: MonthlyRunManifest,
        artifact_index: BacktestArtifactIndex,
        *,
        manifest_path: Path | None = None,
    ) -> MonthlyOptimizerSequenceResult:
        errors: list[str] = []
        evidence_paths: list[str] = []
        artifact_index_path = Path(artifact_index.artifact_root) / "artifact_index.json"
        if artifact_index_path.exists():
            evidence_paths.append(str(artifact_index_path))
        if manifest_path is not None:
            evidence_paths.append(str(manifest_path))

        missing = _missing_named_artifacts(artifact_index, CORE_PHASE4_SEQUENCE_ARTIFACTS)
        if missing:
            errors.append(f"missing core phase4 optimizer artifacts: {', '.join(missing)}")

        fold_manifest = _load_model(artifact_index, "fold_manifest.json", FoldManifest, errors)
        experiment_plan = _load_model(
            artifact_index,
            "llm_experiment_plan.json",
            OptimizerExperimentPlan,
            errors,
        )
        confirmatory = _load_model(
            artifact_index,
            "confirmatory_rerank.json",
            ConfirmatoryRerank,
            errors,
        )
        rounds_manifest = _load_model(artifact_index, "rounds_manifest.json", RoundsManifest, errors)
        selected = _load_candidates(artifact_index, "selected_candidates.json", errors)
        rejected = _load_jsonl(artifact_index, "rejected_candidates.jsonl", errors)
        attempts = _load_attempts(artifact_index, errors)

        _validate_manifest_alignment(
            manifest=manifest,
            fold_manifest=fold_manifest,
            experiment_plan=experiment_plan,
            confirmatory=confirmatory,
            rounds_manifest=rounds_manifest,
            errors=errors,
        )
        _validate_data_bundle_alignment(manifest, artifact_index, errors)
        _validate_attempts(
            attempts,
            manifest,
            errors,
            allow_empty=bool(confirmatory and confirmatory.no_adoption_reason and not selected),
        )
        _validate_runner_observability(artifact_index, manifest, attempts, errors)
        _validate_optimizer_decision(
            manifest=manifest,
            confirmatory=confirmatory,
            rounds_manifest=rounds_manifest,
            selected=selected,
            artifact_index=artifact_index,
            attempts=attempts,
            errors=errors,
        )

        repair_triggered = bool(confirmatory and confirmatory.repair_triggered)
        if repair_triggered:
            repair_missing = _missing_named_artifacts(artifact_index, PHASE4_OOS_REPAIR_ARTIFACTS)
            if repair_missing:
                errors.append(f"missing OOS repair artifacts: {', '.join(repair_missing)}")

        paths = {
            "fold_manifest_path": _artifact_path_str(artifact_index, "fold_manifest.json"),
            "experiment_plan_path": _artifact_path_str(artifact_index, "llm_experiment_plan.json"),
            "candidate_attempts_path": _artifact_path_str(artifact_index, "candidate_attempts.jsonl"),
            "runner_observability_path": _artifact_path_str(
                artifact_index,
                "runner_observability.json",
            ),
            "repair_ablation_matrix_path": _artifact_path_str(
                artifact_index,
                "repair_ablation_matrix.jsonl",
            ),
            "confirmatory_rerank_path": _artifact_path_str(artifact_index, "confirmatory_rerank.json"),
            "rounds_manifest_path": _artifact_path_str(artifact_index, "rounds_manifest.json"),
            "end_of_round_diagnostics_path": _artifact_path_str(
                artifact_index,
                "end_of_round_diagnostics.json",
            ),
        }
        evidence_paths.extend(path for path in paths.values() if path)
        if fold_manifest:
            evidence_paths.extend(_existing_paths([path for fold in fold_manifest.folds for path in fold.evidence_paths]))
        if experiment_plan:
            evidence_paths.extend(_existing_paths(experiment_plan.evidence_paths))
        if confirmatory:
            evidence_paths.extend(_existing_paths(confirmatory.evidence_paths))
        if rounds_manifest:
            evidence_paths.extend(_existing_paths([
                path for record in rounds_manifest.records for path in record.evidence_paths
            ]))

        no_adoption_reason = ""
        adopted_candidate_id = ""
        if confirmatory:
            adopted_candidate_id = confirmatory.adopted_candidate_id
            no_adoption_reason = confirmatory.no_adoption_reason
        status = OptimizerSequenceStatus.SUCCEEDED
        if errors:
            status = OptimizerSequenceStatus.BLOCKED
        elif no_adoption_reason:
            status = OptimizerSequenceStatus.NO_ADOPTION

        return MonthlyOptimizerSequenceResult(
            run_id=manifest.run_id,
            status=status,
            adopted_candidate_id=adopted_candidate_id,
            no_adoption_reason=no_adoption_reason,
            repair_triggered=repair_triggered,
            blocking_reasons=errors,
            artifact_index_path=str(artifact_index_path),
            selected_candidate_ids=[candidate.candidate_id for candidate in selected],
            rejected_candidate_count=len(rejected),
            evidence_paths=_dedupe(evidence_paths),
            **paths,
        )


def build_two_fold_manifest(
    *,
    run_id: str,
    run_month: str,
    in_sample_start: date,
    in_sample_end: date,
    selection_oos_start: date,
    selection_oos_end: date,
    embargo_days: int = 5,
    evidence_paths: list[str] | None = None,
) -> FoldManifest:
    """Build the monthly two-fold purged in-sample manifest."""
    total_days = (in_sample_end - in_sample_start).days + 1
    if total_days < 2:
        raise ValueError("two-fold phased-auto requires at least two in-sample days")
    split = in_sample_start + timedelta(days=(total_days // 2) - 1)
    second_start = split + timedelta(days=1)
    evidence = evidence_paths or []
    return FoldManifest(
        run_id=run_id,
        run_month=run_month,
        in_sample_start=in_sample_start,
        in_sample_end=in_sample_end,
        selection_oos_start=selection_oos_start,
        selection_oos_end=selection_oos_end,
        folds=[
            FoldSpec(
                fold_id="fold_1",
                training_start=second_start,
                training_end=in_sample_end,
                validation_start=in_sample_start,
                validation_end=split,
                embargo_days=embargo_days,
                evidence_paths=evidence,
            ),
            FoldSpec(
                fold_id="fold_2",
                training_start=in_sample_start,
                training_end=split,
                validation_start=second_start,
                validation_end=in_sample_end,
                embargo_days=embargo_days,
                evidence_paths=evidence,
            ),
        ],
    )


def _attempt_id(run_id: str, candidate_id: str, attempt_number: int) -> str:
    raw = f"{run_id}:{candidate_id}:{attempt_number}"
    return "attempt-" + sanitize_workspace_key(raw)


def _missing_named_artifacts(index: BacktestArtifactIndex, names: list[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        path = index.artifact_path(name)
        if path is None or not path.exists():
            missing.append(name)
    return missing


def _artifact_path_str(index: BacktestArtifactIndex, name: str) -> str:
    path = index.artifact_path(name)
    return str(path) if path is not None and path.exists() else ""


def _load_model(
    index: BacktestArtifactIndex,
    name: str,
    model_type: type[Any],
    errors: list[str],
) -> Any:
    path = index.artifact_path(name)
    if path is None or not path.exists():
        return None
    try:
        return model_type.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        errors.append(f"invalid {name}: {exc}")
        return None


def _load_candidates(
    index: BacktestArtifactIndex,
    name: str,
    errors: list[str],
) -> list[MonthlyImprovementCandidate]:
    path = index.artifact_path(name)
    if path is None or not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid {name}: {exc}")
        return []
    if isinstance(raw, dict):
        items = (
            raw.get("candidates")
            or raw.get("selected_candidates")
            or raw.get("selected")
            or raw.get("shortlist")
            or []
        )
    else:
        items = raw
    if not isinstance(items, list):
        errors.append(f"invalid {name}: expected list of candidates")
        return []
    return [
        MonthlyImprovementCandidate.from_raw(item)
        for item in items
        if isinstance(item, dict)
    ]


def _load_jsonl(index: BacktestArtifactIndex, name: str, errors: list[str]) -> list[dict[str, Any]]:
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
    except Exception as exc:
        errors.append(f"invalid {name}: {exc}")
    return rows


def _load_attempts(
    index: BacktestArtifactIndex,
    errors: list[str],
) -> list[CandidateAttemptRecord]:
    rows = _load_jsonl(index, "candidate_attempts.jsonl", errors)
    attempts: list[CandidateAttemptRecord] = []
    for row in rows:
        try:
            attempts.append(CandidateAttemptRecord.model_validate(row))
        except Exception as exc:
            errors.append(f"invalid candidate_attempts.jsonl row: {exc}")
    return attempts


def _latest_attempts(attempts: list[CandidateAttemptRecord]) -> dict[str, CandidateAttemptRecord]:
    latest: dict[str, CandidateAttemptRecord] = {}
    for attempt in attempts:
        current = latest.get(attempt.attempt_id)
        if current is None or attempt.updated_at >= current.updated_at:
            latest[attempt.attempt_id] = attempt
    return latest


def _validate_attempts(
    attempts: list[CandidateAttemptRecord],
    manifest: MonthlyRunManifest,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not attempts:
        if allow_empty:
            return
        errors.append("candidate_attempts.jsonl must include at least one attempt")
        return
    latest = _latest_attempts(attempts)
    attempt_ids_by_number: dict[tuple[str, str, int], set[str]] = {}
    for attempt in attempts:
        if attempt.run_id != manifest.run_id:
            errors.append(f"attempt {attempt.attempt_id} run_id does not match run manifest")
        if attempt.manifest_id and attempt.manifest_id != manifest.manifest_id:
            errors.append(f"attempt {attempt.attempt_id} manifest_id does not match run manifest")
        if (
            manifest.backtest_repo_commit_sha
            and attempt.backtest_repo_commit_sha
            and attempt.backtest_repo_commit_sha != manifest.backtest_repo_commit_sha
        ):
            errors.append(f"attempt {attempt.attempt_id} backtest repo SHA does not match run manifest")
        if (
            manifest.trading_repo_commit_sha
            and attempt.trading_repo_commit_sha
            and attempt.trading_repo_commit_sha != manifest.trading_repo_commit_sha
        ):
            errors.append(f"attempt {attempt.attempt_id} trading repo SHA does not match run manifest")
        number_key = (attempt.run_id, attempt.candidate_id, attempt.attempt_number)
        attempt_ids_by_number.setdefault(number_key, set()).add(attempt.attempt_id)
        if attempt.workspace_key != sanitize_workspace_key(attempt.workspace_key):
            errors.append(f"attempt {attempt.attempt_id} has unsafe workspace key")
        _validate_path_under_root(
            attempt.workspace_path,
            manifest.candidate_workspace_root or str(Path(manifest.artifact_root) / "workspaces"),
            f"attempt {attempt.attempt_id} workspace",
            errors,
        )
        for artifact_path in attempt.artifact_paths:
            _validate_path_under_any_root(
                artifact_path,
                [
                    manifest.artifact_root,
                    manifest.candidate_workspace_root or str(Path(manifest.artifact_root) / "workspaces"),
                ],
                f"attempt {attempt.attempt_id} artifact",
                errors,
            )
    for (_run_id, candidate_id, attempt_number), attempt_ids in attempt_ids_by_number.items():
        if len(attempt_ids) > 1:
            errors.append(
                "attempt_number collision for candidate "
                f"{candidate_id}: {attempt_number} used by attempts {', '.join(sorted(attempt_ids))}"
            )
    for attempt in latest.values():
        if attempt.state not in TERMINAL_ATTEMPT_STATES:
            errors.append(f"attempt {attempt.attempt_id} is not terminal: {attempt.state.value}")


def _validate_manifest_alignment(
    *,
    manifest: MonthlyRunManifest,
    fold_manifest: FoldManifest | None,
    experiment_plan: OptimizerExperimentPlan | None,
    confirmatory: ConfirmatoryRerank | None,
    rounds_manifest: RoundsManifest | None,
    errors: list[str],
) -> None:
    if fold_manifest is not None:
        if fold_manifest.run_id != manifest.run_id:
            errors.append("fold_manifest run_id does not match run manifest")
        if fold_manifest.run_month != manifest.run_month:
            errors.append("fold_manifest run_month does not match run manifest")
        if manifest.in_sample_start and fold_manifest.in_sample_start != manifest.in_sample_start:
            errors.append("fold_manifest in_sample_start does not match run manifest")
        if manifest.in_sample_end and fold_manifest.in_sample_end != manifest.in_sample_end:
            errors.append("fold_manifest in_sample_end does not match run manifest")
        if manifest.selection_oos_start and fold_manifest.selection_oos_start != manifest.selection_oos_start:
            errors.append("fold_manifest selection_oos_start does not match run manifest")
        if manifest.selection_oos_end and fold_manifest.selection_oos_end != manifest.selection_oos_end:
            errors.append("fold_manifest selection_oos_end does not match run manifest")
    if experiment_plan is not None:
        if experiment_plan.run_id != manifest.run_id:
            errors.append("llm_experiment_plan run_id does not match run manifest")
        _append_missing_paths(
            experiment_plan.evidence_paths,
            "llm_experiment_plan evidence",
            errors,
        )
        _validate_search_brief_consumed(manifest, experiment_plan, errors)
    if confirmatory is not None:
        if confirmatory.run_id != manifest.run_id:
            errors.append("confirmatory_rerank run_id does not match run manifest")
        _append_missing_paths(confirmatory.evidence_paths, "confirmatory_rerank evidence", errors)
    if rounds_manifest is not None:
        if rounds_manifest.run_id != manifest.run_id:
            errors.append("rounds_manifest run_id does not match run manifest")
        if rounds_manifest.bot_id and rounds_manifest.bot_id != manifest.bot_id:
            errors.append("rounds_manifest bot_id does not match run manifest")
        if rounds_manifest.strategy_id and rounds_manifest.strategy_id != manifest.strategy_id:
            errors.append("rounds_manifest strategy_id does not match run manifest")


def _validate_data_bundle_alignment(
    manifest: MonthlyRunManifest,
    artifact_index: BacktestArtifactIndex,
    errors: list[str],
) -> None:
    bundle_path = Path(manifest.data_bundle_manifest_path or manifest.market_data_manifest_path)
    if not bundle_path.exists():
        errors.append("data bundle manifest path does not exist")
        return
    try:
        bundle = DataBundleManifest.model_validate(json.loads(bundle_path.read_text(encoding="utf-8")))
    except Exception as exc:
        errors.append(f"invalid data bundle manifest: {exc}")
        return
    if bundle.status != DataBundleStatus.AUTHORITATIVE:
        errors.append(f"data bundle is not authoritative: {bundle.status.value}")
    expected_checksum = manifest.data_bundle_checksum or manifest.data_manifest_checksum
    if expected_checksum and bundle.bundle_checksum != expected_checksum:
        errors.append("data bundle checksum does not match run manifest")
    coverage_path = artifact_index.artifact_path("coverage_manifest.json")
    if coverage_path is None or not coverage_path.exists():
        return
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except Exception:
        return
    emitted_checksum = _find_checksum(coverage)
    if not emitted_checksum:
        errors.append("coverage_manifest.json must include data bundle checksum for optimizer runs")
    elif emitted_checksum != bundle.bundle_checksum:
        errors.append("coverage_manifest.json data bundle checksum does not match run manifest")


def _validate_search_brief_consumed(
    manifest: MonthlyRunManifest,
    experiment_plan: OptimizerExperimentPlan,
    errors: list[str],
) -> None:
    if not manifest.monthly_search_brief_path and not manifest.source_weekly_signal_ids:
        return
    expected_ids = {str(item) for item in manifest.source_weekly_signal_ids if str(item)}
    if not expected_ids:
        guidance_ids = manifest.monthly_search_guidance.get("source_weekly_signal_ids", [])
        expected_ids = {str(item) for item in guidance_ids if str(item)}
    guidance_has_content = bool(expected_ids) or any(
        manifest.monthly_search_guidance.get(key)
        for key in (
            "phase_order_hints",
            "priority_families",
            "seed_candidates",
            "negative_priors",
            "rollback_candidates",
        )
    )
    if manifest.monthly_search_brief_path:
        brief_path = Path(manifest.monthly_search_brief_path)
        if not brief_path.exists():
            errors.append("monthly_search_brief_path does not exist")
        elif guidance_has_content and str(brief_path) not in experiment_plan.evidence_paths:
            errors.append("llm_experiment_plan evidence_paths must cite monthly_search_brief_path")
    if expected_ids:
        consumed_ids = {str(item) for item in experiment_plan.source_weekly_signal_ids if str(item)}
        missing_ids = expected_ids - consumed_ids
        if missing_ids:
            errors.append(
                "llm_experiment_plan does not consume monthly search brief "
                f"source_weekly_signal_ids: {', '.join(sorted(missing_ids))}"
            )
    requirements = manifest.monthly_search_guidance.get("plan_requirements") or {}
    if not isinstance(requirements, dict):
        requirements = {}
    _validate_plan_requirement_families(
        experiment_plan,
        _string_set(requirements.get("candidate_families")),
        "candidate_families",
        errors,
    )
    _validate_plan_requirement_families(
        experiment_plan,
        _string_set(requirements.get("rollback_families")),
        "rollback_families",
        errors,
        require_caution=True,
    )
    _validate_plan_requirement_families(
        experiment_plan,
        _string_set(requirements.get("negative_prior_families")),
        "negative_prior_families",
        errors,
        require_caution=True,
    )


def _validate_plan_requirement_families(
    experiment_plan: OptimizerExperimentPlan,
    required: set[str],
    label: str,
    errors: list[str],
    *,
    require_caution: bool = False,
) -> None:
    if not required:
        return
    candidate_families = {
        str(item.get("family") or item.get("category") or "").strip().lower()
        for item in experiment_plan.candidate_families
        if isinstance(item, dict)
    }
    missing = {family for family in required if family.lower() not in candidate_families}
    if require_caution:
        caution_text = " ".join([
            *experiment_plan.gate_expectations,
            *experiment_plan.overfit_risks,
            json.dumps(experiment_plan.candidate_families, sort_keys=True, default=str),
        ]).lower()
        missing = {family for family in missing if family.lower() not in caution_text}
    if not missing:
        return
    errors.append(
        "llm_experiment_plan does not reflect monthly search brief "
        f"{label}: {', '.join(sorted(missing))}"
    )


def _validate_runner_observability(
    index: BacktestArtifactIndex,
    manifest: MonthlyRunManifest,
    attempts: list[CandidateAttemptRecord],
    errors: list[str],
) -> None:
    path = index.artifact_path("runner_observability.json")
    if path is None or not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid runner_observability.json: {exc}")
        return
    entries = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(entry, dict) for entry in entries):
        errors.append("runner_observability.json must contain an object or list of objects")
        return
    attempt_ids = {attempt.attempt_id for attempt in attempts}
    observed_attempt_ids: set[str] = set()
    for entry in entries:
        if str(entry.get("run_id") or "") != manifest.run_id:
            errors.append("runner_observability run_id does not match run manifest")
        attempt_id = str(entry.get("attempt_id") or "")
        if attempt_id:
            observed_attempt_ids.add(attempt_id)
        if attempt_id and attempt_ids and attempt_id not in attempt_ids:
            errors.append(f"runner_observability references unknown attempt_id: {attempt_id}")
        if attempt_ids and not str(entry.get("attempt_state") or ""):
            errors.append("runner_observability entries with attempts must include attempt_state")
    missing_observed_attempts = attempt_ids - observed_attempt_ids
    if missing_observed_attempts:
        errors.append(
            "runner_observability missing attempt entries: "
            + ", ".join(sorted(missing_observed_attempts))
        )


def _validate_optimizer_decision(
    *,
    manifest: MonthlyRunManifest,
    confirmatory: ConfirmatoryRerank | None,
    rounds_manifest: RoundsManifest | None,
    selected: list[MonthlyImprovementCandidate],
    artifact_index: BacktestArtifactIndex,
    attempts: list[CandidateAttemptRecord],
    errors: list[str],
) -> None:
    if confirmatory is None or rounds_manifest is None:
        return
    if confirmatory.adopted_candidate_id != rounds_manifest.adopted_candidate_id:
        errors.append("confirmatory rerank and rounds manifest adopted_candidate_id differ")
    if confirmatory.no_adoption_reason != rounds_manifest.no_adoption_reason:
        errors.append("confirmatory rerank and rounds manifest no_adoption_reason differ")
    if confirmatory.no_adoption_reason:
        if selected:
            errors.append("no-adoption optimizer result must not emit selected candidates")
        return
    if confirmatory.adopted_candidate_id:
        adoption_missing = _missing_named_artifacts(artifact_index, ADOPTION_GATE_ARTIFACTS)
        if adoption_missing:
            errors.append(f"missing adoption gate artifacts: {', '.join(adoption_missing)}")
        if len(selected) != 1:
            errors.append("optimizer adoption requires exactly one selected candidate")
            return
        candidate = selected[0]
        if candidate.candidate_id != confirmatory.adopted_candidate_id:
            errors.append("selected candidate is not the confirmatory adopted candidate")
        expected_source = (
            MonthlyCandidateSource.SMOKE_REPAIR
            if confirmatory.repair_triggered
            else MonthlyCandidateSource.PHASED_AUTO
        )
        if candidate.source != expected_source:
            errors.append(f"adopted candidate source must be {expected_source.value}")
        if candidate.source != confirmatory.primary_source:
            errors.append("adopted candidate source must match confirmatory_rerank.primary_source")
        _validate_candidate_lineage(candidate, manifest, errors)
        _validate_adopted_candidate_attempt(candidate, attempts, manifest, errors)
        if not _runner_contract_matches(candidate):
            errors.append("adopted candidate is missing the correct runner_contract_version")
        gate_inputs = candidate.deterministic_gate_inputs
        if gate_inputs.get("phase4_sequence_valid") is not True:
            errors.append("adopted candidate must set phase4_sequence_valid=true")
        if gate_inputs.get("round_n_plus_1_adopted") is not True:
            errors.append("adopted candidate must set round_n_plus_1_adopted=true")
        if gate_inputs.get("end_of_round_diagnostics_saved") is not True:
            errors.append("adopted candidate must prove end-of-round diagnostics were saved")
        if gate_inputs.get("live_backtest_parity_aligned") is not True:
            errors.append("adopted candidate must prove live/backtest parity alignment")
        for attr, name in (
            ("fold_manifest_path", "fold_manifest_path"),
            ("rounds_manifest_path", "rounds_manifest_path"),
            ("end_of_round_diagnostics_path", "end_of_round_diagnostics_path"),
            ("confirmatory_rerank_path", "confirmatory_rerank_path"),
        ):
            if not getattr(candidate, attr):
                errors.append(f"adopted candidate missing {name}")
        if candidate.score_component_count and candidate.score_component_count > manifest.score_component_cap:
            errors.append("adopted candidate exceeds score_component_cap")
        if _is_structural_candidate(candidate):
            _validate_structural_candidate(candidate, artifact_index, manifest, errors)
        _validate_path_under_root(
            candidate.candidate_workspace_path,
            manifest.candidate_workspace_root or str(Path(manifest.artifact_root) / "workspaces"),
            "adopted candidate workspace",
            errors,
        )
        if candidate.decision_parity_report_path:
            _validate_path_under_root(
                candidate.decision_parity_report_path,
                artifact_index.artifact_root,
                "adopted candidate decision parity report",
                errors,
            )


def _validate_adopted_candidate_attempt(
    candidate: MonthlyImprovementCandidate,
    attempts: list[CandidateAttemptRecord],
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> None:
    latest = _latest_attempts(attempts)
    if not candidate.candidate_attempt_id:
        errors.append("adopted candidate must include candidate_attempt_id")
        return
    attempt = latest.get(candidate.candidate_attempt_id)
    if attempt is None:
        errors.append("adopted candidate candidate_attempt_id is missing from attempt ledger")
        return
    if attempt.run_id != manifest.run_id:
        errors.append("adopted candidate attempt run_id does not match run manifest")
    if not attempt.manifest_id:
        errors.append("adopted candidate attempt missing manifest_id")
    elif attempt.manifest_id != manifest.manifest_id:
        errors.append("adopted candidate attempt manifest_id does not match run manifest")
    if attempt.candidate_id != candidate.candidate_id:
        errors.append("adopted candidate attempt belongs to a different candidate")
    if attempt.state != CandidateAttemptState.SUCCEEDED:
        errors.append(f"adopted candidate attempt must be succeeded, got {attempt.state.value}")
    if candidate.candidate_attempt_status and candidate.candidate_attempt_status != attempt.state.value:
        errors.append("adopted candidate attempt status does not match latest attempt ledger state")


def _validate_candidate_lineage(
    candidate: MonthlyImprovementCandidate,
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> None:
    if candidate.run_id != manifest.run_id:
        errors.append("adopted candidate run_id does not match run manifest")
    if candidate.manifest_id != manifest.manifest_id:
        errors.append("adopted candidate manifest_id does not match run manifest")
    for field_name in ("round_id", "prior_round_id", "next_round_id"):
        if not str(getattr(candidate, field_name) or "").strip():
            errors.append(f"adopted candidate missing {field_name}")
    if manifest.round_id and candidate.prior_round_id != manifest.round_id:
        errors.append("adopted candidate prior_round_id does not match run manifest round_id")
    if manifest.next_round_id and candidate.next_round_id != manifest.next_round_id:
        errors.append("adopted candidate next_round_id does not match run manifest")
    if not candidate.backtest_repo_commit_sha:
        errors.append("adopted candidate missing backtest_repo_commit_sha")
    elif (
        manifest.backtest_repo_commit_sha
        and candidate.backtest_repo_commit_sha != manifest.backtest_repo_commit_sha
    ):
        errors.append("adopted candidate backtest repo SHA does not match run manifest")
    live_sha = candidate.live_trading_repo_commit_sha or candidate.code_sha
    if not live_sha:
        errors.append("adopted candidate missing live_trading_repo_commit_sha")
    elif manifest.trading_repo_commit_sha and live_sha != manifest.trading_repo_commit_sha:
        errors.append("adopted candidate live trading repo SHA does not match run manifest")
    if not candidate.control_plane_commit_sha:
        errors.append("adopted candidate missing control_plane_commit_sha")
    elif (
        manifest.control_plane_commit_sha
        and candidate.control_plane_commit_sha != manifest.control_plane_commit_sha
    ):
        errors.append("adopted candidate control-plane SHA does not match run manifest")


def _runner_contract_matches(candidate: MonthlyImprovementCandidate) -> bool:
    version = str(
        candidate.deterministic_gate_inputs.get("runner_contract_version")
        or candidate.deterministic_gate_inputs.get("source_runner_contract_version")
        or candidate.workflow_contract_version
        or ""
    )
    expected = {
        MonthlyCandidateSource.SMOKE_REPAIR: "smoke_repair_runner_contract_v1",
        MonthlyCandidateSource.PHASED_AUTO: "phased_auto_runner_contract_v1",
    }.get(candidate.source, "")
    return bool(expected and version == expected)


def _is_structural_candidate(candidate: MonthlyImprovementCandidate) -> bool:
    return (
        candidate.change_kind == "structural_change"
        or bool(candidate.file_changes)
        or bool(candidate.live_repo_patch_path)
        or bool(candidate.backtest_adapter_patch_path)
    )


def _validate_structural_candidate(
    candidate: MonthlyImprovementCandidate,
    artifact_index: BacktestArtifactIndex,
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> None:
    missing = [
        name for name, value in (
            ("live_repo_patch_path", candidate.live_repo_patch_path),
            ("backtest_adapter_patch_path", candidate.backtest_adapter_patch_path),
            ("decision_parity_report_path", candidate.decision_parity_report_path),
        )
        if not value
    ]
    if missing:
        errors.append(f"structural candidate missing required lineage: {', '.join(missing)}")
    for label, path in (
        ("live repo patch", candidate.live_repo_patch_path),
        ("backtest adapter patch", candidate.backtest_adapter_patch_path),
        ("config patch", candidate.config_patch_path),
        ("decision parity report", candidate.decision_parity_report_path),
    ):
        if path:
            _validate_path_under_root(path, artifact_index.artifact_root, label, errors)
            if not Path(path).exists():
                errors.append(f"structural candidate {label} path does not exist")
    if candidate.decision_parity_report_path and Path(candidate.decision_parity_report_path).exists():
        try:
            report = DecisionParityReport.model_validate(
                json.loads(Path(candidate.decision_parity_report_path).read_text(encoding="utf-8"))
            )
        except Exception as exc:
            errors.append(f"structural candidate decision parity report is invalid: {exc}")
            return
        if report.run_id != candidate.run_id:
            errors.append("structural candidate decision parity run_id does not match candidate")
        if report.candidate_id != candidate.candidate_id:
            errors.append("structural candidate decision parity candidate_id does not match candidate")
        if not report.eligible_for_structural_approval:
            errors.append("structural candidate decision parity report is not pass")
        _validate_decision_parity_evidence(report, artifact_index, errors)
        if manifest.strategy_plugin_id and report.strategy_plugin_id != manifest.strategy_plugin_id:
            errors.append("structural candidate decision parity strategy_plugin_id does not match run manifest")
        contract = _load_strategy_plugin_contract(manifest, errors)
        if contract is None:
            errors.append("structural candidate requires strategy plugin contract evidence")
        else:
            if not contract.eligible_for_approval:
                errors.append("structural candidate strategy plugin contract is not approval-ready")
            if report.strategy_plugin_id != contract.plugin_id:
                errors.append("structural candidate decision parity strategy_plugin_id does not match plugin contract")
            if report.live_repo_commit_sha != contract.live_repo_commit_sha:
                errors.append("structural candidate decision parity live repo SHA does not match plugin contract")
            if report.backtest_adapter_commit_sha != contract.backtest_adapter_commit_sha:
                errors.append("structural candidate decision parity backtest adapter SHA does not match plugin contract")


def _load_strategy_plugin_contract(
    manifest: MonthlyRunManifest,
    errors: list[str],
) -> StrategyPluginContract | None:
    if not manifest.strategy_plugin_contract_path:
        return None
    path = Path(manifest.strategy_plugin_contract_path)
    if not path.exists() or not path.is_file():
        errors.append("strategy plugin contract path is missing")
        return None
    try:
        return StrategyPluginContract.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        errors.append(f"strategy plugin contract is invalid: {exc}")
        return None


def _validate_decision_parity_evidence(
    report: DecisionParityReport,
    artifact_index: BacktestArtifactIndex,
    errors: list[str],
) -> None:
    evidence_paths = _dedupe([
        *report.evidence_paths,
        *[
            path
            for check in report.checks
            for path in check.evidence_paths
        ],
    ])
    for path in evidence_paths:
        _validate_path_under_root(path, artifact_index.artifact_root, "decision parity evidence", errors)
        if not Path(path).exists():
            errors.append(f"decision parity evidence path does not exist: {path}")


def _validate_path_under_root(path: str, root: str, label: str, errors: list[str]) -> None:
    if not path:
        return
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        errors.append(f"{label} is outside configured root")


def _validate_path_under_any_root(
    path: str,
    roots: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not path:
        return
    resolved = Path(path).resolve()
    for root in roots:
        try:
            resolved.relative_to(Path(root).resolve())
            return
        except (OSError, ValueError):
            continue
    errors.append(f"{label} is outside configured root")


def _append_missing_paths(paths: list[str], label: str, errors: list[str]) -> None:
    missing = [path for path in paths if path and not Path(path).exists()]
    if missing:
        errors.append(f"{label} paths do not exist: {', '.join(missing[:5])}")


def _existing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if path and Path(path).exists()]


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, list | tuple | set):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _find_checksum(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("bundle_checksum", "data_bundle_checksum", "checksum"):
            if value.get(key):
                return str(value[key])
        for item in value.values():
            found = _find_checksum(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_checksum(item)
            if found:
                return found
    return ""


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
