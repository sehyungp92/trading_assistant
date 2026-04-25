"""Handler implementations — wire worker action types to full pipelines.

Each handler implements: data preparation -> prompt assembly -> agent invocation
-> post-processing -> notification dispatch.
"""
from __future__ import annotations

import json
import logging
import hashlib
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel

from analysis.context_builder import ContextBuilder
from orchestrator.agent_runner import AgentRunner, AgentResult
from orchestrator.event_stream import EventStream
from orchestrator.memory_consolidator import MemoryConsolidator
from orchestrator.orchestrator_brain import Action, OrchestratorBrain
from orchestrator.worker import Worker
from schemas.notifications import (
    NotificationPayload,
    NotificationPreferences,
    NotificationPriority,
)

logger = logging.getLogger(__name__)

# Minimum total trades across all bots to justify a full agent-runtime invocation.
# Days with fewer trades get a deterministic summary instead.
_MIN_TRADES_FOR_ANALYSIS = 3
_INSTRUMENTATION_READINESS_THRESHOLD = 0.4  # warn when a bot's overall readiness is below this


class Handlers:
    """Handler implementations for all worker action types."""

    def __init__(
        self,
        agent_runner: AgentRunner,
        event_stream: EventStream,
        dispatcher: object,  # NotificationDispatcher
        notification_prefs: NotificationPreferences,
        curated_dir: Path,
        memory_dir: Path,
        runs_dir: Path,
        source_root: Path,
        bots: list[str],
        raw_data_dir: Path | None = None,
        heartbeat_dir: Path | None = None,
        failure_log_path: Path | None = None,
        worker: Worker | None = None,
        brain: OrchestratorBrain | None = None,
        run_history_path: Path | None = None,
        suggestion_tracker: object | None = None,
        autonomous_pipeline: object | None = None,
        approval_handler: object | None = None,
        approval_tracker: object | None = None,
        pr_builder: object | None = None,
        config_registry: object | None = None,
        repo_workspace_manager: object | None = None,
        deployment_monitor: object | None = None,
        threshold_learner: object | None = None,
        experiment_manager: object | None = None,
        experiment_config_gen: object | None = None,
        bot_configs: dict | None = None,
        reliability_tracker: object | None = None,
        structural_experiment_tracker: object | None = None,
        strategy_registry: object | None = None,
        run_index: object | None = None,
    ) -> None:
        self._agent_runner = agent_runner
        self._event_stream = event_stream
        self._dispatcher = dispatcher
        self._notification_prefs = notification_prefs
        self._curated_dir = Path(curated_dir)
        self._raw_data_dir = (
            Path(raw_data_dir) if raw_data_dir is not None else self._curated_dir.parent / "raw"
        )
        self._memory_dir = Path(memory_dir)
        self._runs_dir = Path(runs_dir)
        self._source_root = Path(source_root)
        self._bots = bots
        self._heartbeat_dir = Path(heartbeat_dir) if heartbeat_dir else self._runs_dir.parent / "heartbeats"
        self._failure_log_path = failure_log_path or (self._runs_dir.parent / "data" / "failure_log.jsonl")
        self._worker = worker
        self._brain = brain
        self._run_history_path = run_history_path or (self._runs_dir.parent / "data" / "run_history.jsonl")
        self._suggestion_tracker = suggestion_tracker
        self._autonomous_pipeline = autonomous_pipeline
        self._approval_handler = approval_handler
        self._approval_tracker = approval_tracker
        self._pr_builder = pr_builder
        self._config_registry = config_registry
        self._repo_workspace_manager = repo_workspace_manager
        self._deployment_monitor = deployment_monitor
        self._threshold_learner = threshold_learner
        self._experiment_manager = experiment_manager
        self._experiment_config_gen = experiment_config_gen
        self._bot_configs = bot_configs
        self._reliability_tracker = reliability_tracker
        self._structural_experiment_tracker = structural_experiment_tracker
        self._strategy_registry = strategy_registry
        self._run_index = run_index

    async def handle_daily_analysis(self, action: Action) -> None:
        """Run the daily analysis pipeline: quality gate -> assemble -> invoke -> notify."""
        details = action.details or {}
        date = details.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        bots = details.get("bots", self._bots)
        run_scope = details.get("run_scope", "")
        run_id = f"daily-{date}"
        if run_scope:
            run_id = f"{run_id}-{hashlib.sha256(run_scope.encode('utf-8')).hexdigest()[:8]}"
        start_time = datetime.now(timezone.utc)
        self._record_run(run_id, "daily_analysis", "running", started_at=start_time.isoformat())

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "daily_analysis",
            })

            self._rebuild_daily_curated_from_raw(date, bots)

            # Data availability pre-check via MemoryIndex
            index = MemoryConsolidator.load_index(self._runs_dir.parent)
            for bot in bots:
                avail = ContextBuilder.check_data_availability(index, bot, date)
                if avail["has_curated"] is False:
                    logger.warning("No curated data for %s on %s — analysis may be incomplete", bot, date)

            # Quality gate
            from analysis.quality_gate import QualityGate

            gate = QualityGate(
                report_id=run_id,
                date=date,
                expected_bots=bots,
                curated_dir=self._curated_dir,
            )
            checklist = gate.run()

            if not checklist.can_proceed:
                logger.warning("Quality gate blocked for %s: %s", run_id, checklist.blocking_issues)
                self._event_stream.broadcast("daily_analysis_blocked", {
                    "date": date,
                    "blocking_issues": checklist.blocking_issues,
                })
                await self._notify(
                    notification_type="daily_report_blocked",
                    priority=NotificationPriority.LOW,
                    title=f"Daily report {date} — blocked",
                    body=f"Quality gate blocked: {', '.join(checklist.blocking_issues)}",
                )
                return

            if checklist.overall == "FAIL":
                logger.warning("Quality gate FAIL (degraded) for %s: %s", run_id, checklist.blocking_issues)
                self._event_stream.broadcast("daily_analysis_degraded", {
                    "date": date,
                    "blocking_issues": checklist.blocking_issues,
                    "data_completeness": checklist.data_completeness,
                })

            # Minimum-data threshold: skip the agent runtime if insufficient data
            total_trades = self._count_daily_trades(date)
            if total_trades < _MIN_TRADES_FOR_ANALYSIS:
                logger.info(
                    "Only %d trades on %s (min %d) — producing deterministic summary",
                    total_trades, date, _MIN_TRADES_FOR_ANALYSIS,
                )
                completeness = getattr(checklist, "data_completeness", 0.0)
                try:
                    completeness_str = f"{completeness:.0%}"
                except (TypeError, ValueError):
                    completeness_str = str(completeness)
                body = (
                    f"Daily summary for {date}: {total_trades} trade(s) across {len(bots)} bot(s). "
                    f"Insufficient data for full analysis (minimum {_MIN_TRADES_FOR_ANALYSIS} trades required). "
                    f"Data completeness: {completeness_str}."
                )
                self._record_run(
                    run_id, "daily_analysis", "skipped",
                    started_at=start_time.isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                )
                self._write_run_report(run_id, "daily_report.md", body, mirror_response=True)
                await self._notify(
                    notification_type="daily_report",
                    priority=NotificationPriority.LOW,
                    title=f"Daily Summary — {date} (light day)",
                    body=body,
                )
                return

            # Collect event batching counts
            event_counts: dict = {}
            if self._worker:
                event_counts = self._worker.get_and_reset_daily_counts()

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "quality_gate", "handler": "daily_analysis",
            })

            # Run deterministic triage to identify significant events
            from analysis.daily_triage import DailyTriage
            from analysis.prompt_assembler import DailyPromptAssembler

            ctx = ContextBuilder(self._memory_dir, curated_dir=self._curated_dir)

            # Instrumentation readiness check — warn (never block) for under-instrumented bots
            try:
                readiness = ctx.load_instrumentation_readiness(bots)
                low_readiness_bots = {
                    bid: r.get("overall_score", 0)
                    for bid, r in readiness.items()
                    if r.get("overall_score", 0) < _INSTRUMENTATION_READINESS_THRESHOLD
                }
                if low_readiness_bots:
                    logger.warning(
                        "Low instrumentation readiness: %s",
                        ", ".join(f"{b}={s:.0%}" for b, s in low_readiness_bots.items()),
                    )
                    self._event_stream.broadcast("instrumentation_readiness_low", {
                        "date": date, "bots": low_readiness_bots,
                    })
            except Exception:
                logger.debug("Instrumentation readiness check skipped", exc_info=True)

            active_suggestions = ctx.load_active_suggestions()
            triage = DailyTriage(
                curated_dir=self._curated_dir,
                date=date,
                bots=bots,
                active_suggestions=active_suggestions,
            )
            triage_report = triage.run()

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "triage", "handler": "daily_analysis",
                "significant_events": len(triage_report.significant_events),
                "focus_questions": len(triage_report.focus_questions),
            })

            # Assemble prompt with triage-driven focus
            assembler = DailyPromptAssembler(
                date=date,
                bots=bots,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                bot_configs=self._bot_configs,
                strategy_registry=self._strategy_registry,
                run_index=self._run_index,
            )
            package = assembler.assemble(
                triage_report=triage_report,
                session_store=self._agent_runner.session_store,
            )

            # Include event counts in prompt metadata
            if event_counts:
                package.metadata["event_counts"] = event_counts

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "daily_analysis",
            })

            # Invoke the configured agent runtime
            result = await self._agent_runner.invoke(
                agent_type="daily_analysis",
                prompt_package=package,
                run_id=run_id,
                allowed_tools=["Read", "Grep", "Glob"],
            )

            # Parse structured response
            final_report = result.response
            if result.success:
                from analysis.response_parser import parse_response

                parsed = parse_response(result.response)
                # Save parsed analysis alongside raw response
                try:
                    run_dir = result.run_dir or (self._runs_dir / run_id)
                    Path(run_dir).mkdir(parents=True, exist_ok=True)
                    (Path(run_dir) / "parsed_analysis.json").write_text(
                        parsed.model_dump_json(indent=2), encoding="utf-8",
                    )
                except Exception:
                    logger.error("Failed to save parsed analysis for %s", run_id, exc_info=True)

                if not parsed.parse_success:
                    logger.warning("No structured output block found in %s response", run_id)
                elif parsed.fallback_used:
                    logger.info("Fallback markdown extraction used for %s — structured block was missing", run_id)

                # Validate and annotate response
                final_report, validation = self._validate_and_annotate(
                    parsed,
                    date,
                    provider=result.provider,
                    model=result.effective_model,
                    run_id=run_id,
                    agent_type="daily_analysis",
                    bot_ids=package.metadata.get("bot_ids", ""),
                )
                self._persist_validator_notes(result.run_dir or (self._runs_dir / run_id), validation)
                self._refresh_run_index_entry(
                    run_id=run_id,
                    agent_type="daily_analysis",
                    run_dir=result.run_dir or (self._runs_dir / run_id),
                    provider=result.provider,
                    model=result.effective_model,
                    prompt_package=package,
                    success=result.success,
                    duration_ms=result.duration_ms,
                    cost_usd=result.cost_usd,
                )

                # Fallback: if validation failed but we have suggestions, record them unvalidated
                if validation is None and parsed.suggestions:
                    from analysis.response_validator import ValidationResult
                    validation = ValidationResult(
                        approved_suggestions=parsed.suggestions,
                        approved_predictions=parsed.predictions,
                    )
                    logger.warning("Validation failed for %s — recording unvalidated suggestions", run_id)

                # Record approved agent suggestions to tracker
                agent_suggestion_ids = self._record_agent_suggestions(
                    validation, run_id, parsed,
                    provider=result.provider, model=result.effective_model,
                )

                # Record learning card feedback from validation signal
                self._record_learning_card_feedback_targeted(validation, package)

                # Run autonomous pipeline on recorded suggestions
                await self._run_autonomous_pipeline(agent_suggestion_ids, run_id)

                # Record predictions
                self._record_predictions(date, parsed.predictions)

            # Record completion timestamp
            if self._brain:
                self._brain.record_daily_analysis(datetime.now(timezone.utc).isoformat())

            # Broadcast + notify
            self._event_stream.broadcast("daily_analysis_complete", {
                "date": date,
                "success": result.success,
                "run_dir": str(result.run_dir),
            })

            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            status = "completed" if result.success else "failed"
            self._record_run(
                run_id, "daily_analysis", status,
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed,
                error=result.error if not result.success else "",
            )

            if result.success:
                self._write_run_report(run_id, "daily_report.md", final_report)
                await self._notify(
                    notification_type="daily_report",
                    priority=NotificationPriority.NORMAL,
                    title=f"Daily Report — {date}",
                    body=final_report[:2000],
                )
            else:
                await self._notify(
                    notification_type="daily_report_error",
                    priority=NotificationPriority.HIGH,
                    title=f"Daily report {date} — error",
                    body=f"Agent failed: {result.error}",
                )

        except Exception as exc:
            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self._record_run(
                run_id, "daily_analysis", "failed",
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed, error=str(exc),
            )
            logger.exception("Daily analysis handler failed for %s", run_id)
            self._event_stream.broadcast("daily_analysis_error", {
                "date": date,
                "error": str(exc),
            })

    async def handle_weekly_analysis(self, action: Action) -> None:
        """Run the weekly analysis pipeline: metrics -> strategy -> simulations -> assemble -> invoke -> notify."""
        details = action.details or {}
        week_start = details.get("week_start", "")
        week_end = details.get("week_end", "")
        run_id = f"weekly-{week_start}"
        start_time = datetime.now(timezone.utc)
        self._record_run(run_id, "weekly_analysis", "running", started_at=start_time.isoformat())

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "weekly_analysis",
            })

            # Data availability pre-check
            index = MemoryConsolidator.load_index(self._runs_dir.parent)
            if index:
                from datetime import timedelta as _td
                start = datetime.strptime(week_start, "%Y-%m-%d")
                dates_in_week = [(start + _td(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                for bot in self._bots:
                    avail_dates = index.curated_dates_by_bot.get(bot, [])
                    count = sum(1 for d in dates_in_week if d in avail_dates)
                    if count < 5:
                        logger.warning(
                            "Only %d/7 daily data days for %s in week %s", count, bot, week_start,
                        )

            # Build weekly metrics
            from skills.build_weekly_metrics import WeeklyMetricsBuilder

            builder = WeeklyMetricsBuilder(
                week_start=week_start,
                week_end=week_end,
                bots=self._bots,
            )

            # Load daily summaries from curated dir
            from schemas.daily_metrics import BotDailySummary

            dailies_by_bot: dict[str, list] = {}
            for bot in self._bots:
                dailies_by_bot[bot] = self._load_bot_dailies(bot, week_start, week_end)

            portfolio_summary = builder.build_portfolio_summary(dailies_by_bot)
            builder.write_weekly_curated(portfolio_summary, self._curated_dir)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "metrics_build", "handler": "weekly_analysis",
            })

            # Load most recent signal_health.json per bot from the week
            signal_health_data: dict[str, dict] = {}
            from datetime import timedelta as _td
            start_dt = datetime.strptime(week_start, "%Y-%m-%d")
            dates_in_week = [(start_dt + _td(days=i)).strftime("%Y-%m-%d") for i in range(7)]
            for bot in self._bots:
                for date_str in reversed(dates_in_week):  # most recent first
                    sh_path = self._curated_dir / date_str / bot / "signal_health.json"
                    if sh_path.exists():
                        signal_health_data[bot] = json.loads(sh_path.read_text())
                        break

            # Load factor rolling data per bot
            factor_rolling_data: dict[str, list[dict]] = {}
            findings_dir = self._memory_dir / "findings"
            if findings_dir.exists():
                try:
                    from skills.signal_factor_tracker import SignalFactorTracker

                    tracker = SignalFactorTracker(findings_dir)
                    for bot in self._bots:
                        report = tracker.compute_rolling(bot, week_end)
                        if report.factors:
                            factor_rolling_data[bot] = [
                                f.model_dump(mode="json") for f in report.factors
                            ]
                except Exception:
                    logger.warning("Failed to load factor rolling data", exc_info=True)

            weekly_evidence = self._load_weekly_strategy_evidence(
                week_start=week_start,
                week_end=week_end,
                bot_summaries=portfolio_summary.bot_summaries,
                signal_health_data=signal_health_data,
                factor_rolling_data=factor_rolling_data,
            )

            # Run strategy engine
            from analysis.strategy_engine import StrategyEngine

            # Compute category scorecard and per-detector confidence
            _scorecard = None
            _scorer_inst = None
            _detector_confidence: dict[str, float] = {}
            try:
                from skills.suggestion_scorer import SuggestionScorer as _Scorer
                _scorer_inst = _Scorer(self._context_builder.memory_dir / "findings")
                _scorecard = _scorer_inst.compute_scorecard()
                _detector_confidence = _scorer_inst.compute_detector_confidence()
            except Exception:
                logger.debug("Could not load category scorecard / detector confidence")

            # Load recent suggestions per bot for anti-oscillation
            _recent_suggestions: list[dict] = []
            if self._suggestion_tracker:
                for bid in portfolio_summary.bot_summaries:
                    _recent_suggestions.extend(
                        self._suggestion_tracker.get_recent_by_bot(bid, weeks=4)
                    )

            # Load convergence report for oscillation dampening
            _convergence_report: dict = {}
            try:
                _convergence_report = self._context_builder.load_convergence_report()
            except Exception:
                pass

            # Compute category value map for optimization allocation
            _category_value_map: dict = {}
            if _scorer_inst:
                try:
                    _category_value_map = _scorer_inst.compute_category_value_map()
                except Exception:
                    pass

            engine = StrategyEngine(
                week_start=week_start, week_end=week_end,
                threshold_learner=self._threshold_learner,
                strategy_registry=self._strategy_registry,
                category_scorecard=_scorecard,
                detector_confidence=_detector_confidence,
                recent_suggestions=_recent_suggestions,
                convergence_report=_convergence_report,
                category_value_map=_category_value_map,
            )
            refinement_report = engine.build_report(
                portfolio_summary.bot_summaries,
                **weekly_evidence,
            )

            # Run portfolio-level detectors (Phase 2 inner loop)
            try:
                portfolio_suggestions = self._run_portfolio_detectors(
                    engine, week_start, week_end, portfolio_summary,
                )
                if portfolio_suggestions:
                    refinement_report.suggestions.extend(portfolio_suggestions)
                    logger.info(
                        "Portfolio detectors produced %d suggestions", len(portfolio_suggestions),
                    )
            except Exception:
                logger.warning("Portfolio detectors failed — skipping", exc_info=True)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "strategy_engine", "handler": "weekly_analysis",
            })

            # Run simulation skills based on strategy engine findings
            simulation_results = self._run_weekly_simulations(
                refinement_report, week_start, week_end,
            )

            # Run allocation analyses
            allocation_results = self._run_allocation_analyses(
                portfolio_summary, week_start, week_end,
            )

            if simulation_results:
                weekly_evidence = self._load_weekly_strategy_evidence(
                    week_start=week_start,
                    week_end=week_end,
                    bot_summaries=portfolio_summary.bot_summaries,
                    signal_health_data=signal_health_data,
                    factor_rolling_data=factor_rolling_data,
                    simulation_results=simulation_results,
                )
                refinement_report = engine.build_report(
                    portfolio_summary.bot_summaries,
                    **weekly_evidence,
                )

            # Write refinement report
            weekly_dir = self._curated_dir / "weekly" / week_start
            weekly_dir.mkdir(parents=True, exist_ok=True)
            refinement_path = weekly_dir / "refinement_report.json"
            refinement_path.write_text(
                json.dumps(refinement_report.model_dump(mode="json"), indent=2, default=str),
                encoding="utf-8",
            )

            # Record suggestions from strategy engine to SuggestionTracker
            suggestion_ids = self._record_suggestions(
                getattr(refinement_report, "suggestions", []), run_id,
                category_scorecard=_scorecard,
            )

            # Run autonomous pipeline on recorded suggestions
            await self._run_autonomous_pipeline(suggestion_ids, run_id)

            # Ingest experiment variant data from curated experiment_data.json files
            experiment_results = []
            if self._experiment_manager is not None:
                from datetime import timedelta as _td_exp
                start_exp = datetime.strptime(week_start, "%Y-%m-%d")
                dates_in_week_exp = [(start_exp + _td_exp(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                for bot in self._bots:
                    for date_str in dates_in_week_exp:
                        exp_path = self._curated_dir / date_str / bot / "experiment_data.json"
                        if not exp_path.exists():
                            continue
                        try:
                            exp_data = json.loads(exp_path.read_text())
                            for exp_id, variants in exp_data.items():
                                for variant_name, variant_data in variants.items():
                                    trades = variant_data.get("trades", [])
                                    if trades:
                                        self._experiment_manager.ingest_variant_data(
                                            exp_id, variant_name, trades,
                                        )
                        except Exception:
                            logger.warning(
                                "Failed to ingest experiment data from %s", exp_path,
                            )

                # Check auto-conclusion and analyze
                active_experiments = self._experiment_manager.get_active()
                for exp in active_experiments:
                    try:
                        if self._experiment_manager.check_auto_conclusion(exp.experiment_id):
                            result = self._experiment_manager.analyze_experiment(exp.experiment_id)
                            self._experiment_manager.conclude_experiment(exp.experiment_id, result)
                            experiment_results.append(result)
                            self._event_stream.broadcast("experiment_concluded", {
                                "experiment_id": exp.experiment_id,
                                "recommendation": result.recommendation,
                            })
                    except Exception:
                        logger.warning("Experiment check failed for %s", exp.experiment_id)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "simulations", "handler": "weekly_analysis",
            })

            # Run weekly triage for focused analytical questions
            from analysis.weekly_triage import WeeklyTriage
            from analysis.weekly_prompt_assembler import WeeklyPromptAssembler

            ctx_weekly = ContextBuilder(self._memory_dir, curated_dir=self._curated_dir)
            reliable_outcomes, _low_q = ctx_weekly.load_outcome_measurements()
            weekly_triage = WeeklyTriage(
                curated_dir=self._curated_dir,
                end_date=week_end,
                bots=self._bots,
                active_suggestions=ctx_weekly.load_active_suggestions(),
                outcome_measurements=reliable_outcomes,
                prediction_accuracy=ctx_weekly.load_prediction_accuracy(),
            )
            weekly_triage_report = weekly_triage.run()

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "triage", "handler": "weekly_analysis",
                "anomalies": len(weekly_triage_report.anomalies),
            })

            # Assemble prompt with triage-driven focus
            assembler = WeeklyPromptAssembler(
                week_start=week_start,
                week_end=week_end,
                bots=self._bots,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                runs_dir=self._runs_dir,
                bot_configs=self._bot_configs,
                strategy_registry=self._strategy_registry,
                run_index=self._run_index,
            )
            package = assembler.assemble(
                triage_report=weekly_triage_report,
                session_store=self._agent_runner.session_store,
            )

            # Inject simulation results into prompt data
            if simulation_results:
                package.data.update({"simulation_results": simulation_results})

            # Inject allocation analysis results
            if allocation_results:
                package.data.update({"allocation_analysis": allocation_results})

            # Inject suggestion ID mapping so the agent can reference them in the report
            if suggestion_ids:
                package.metadata["suggestion_ids"] = suggestion_ids

            # Weekly retrospective — compare last week's predictions to actual outcomes
            retro_builder = None
            try:
                from skills.retrospective_builder import RetrospectiveBuilder

                retro_builder = RetrospectiveBuilder(
                    runs_dir=self._runs_dir,
                    curated_dir=self._curated_dir,
                    memory_dir=self._memory_dir,
                )
                retrospective = retro_builder.build(week_start, week_end)
                if retrospective.predictions_reviewed > 0:
                    package.data["weekly_retrospective"] = retrospective.model_dump(mode="json")
            except Exception:
                logger.warning("Retrospective builder failed — skipping")

            # Build retrospective synthesis (keep/discard verdicts for dynamic prompt evolution)
            try:
                if retro_builder is None:
                    from skills.retrospective_builder import RetrospectiveBuilder as _RB
                    retro_builder = _RB(
                        runs_dir=self._runs_dir,
                        curated_dir=self._curated_dir,
                        memory_dir=self._memory_dir,
                    )
                synthesis = retro_builder.build_synthesis(week_start, week_end)
                has_content = (
                    synthesis.what_worked
                    or synthesis.what_failed
                    or synthesis.discard
                    or synthesis.lessons
                )
                if has_content:
                    package.data["last_week_synthesis"] = synthesis.model_dump(mode="json")

                # Persist discard items as category overrides (recalibration)
                if synthesis.discard:
                    try:
                        from skills.suggestion_scorer import SuggestionScorer
                        scorer = SuggestionScorer(self._memory_dir / "findings")
                        scorer.apply_recalibration(synthesis.discard)
                        logger.info("Recalibrated %d categories from synthesis", len(synthesis.discard))
                    except Exception:
                        logger.warning("Category recalibration failed")
            except Exception:
                logger.warning("Retrospective synthesis failed — skipping")

            # Record forecast data and compute meta-analysis
            try:
                from skills.forecast_tracker import ForecastTracker
                from schemas.forecast_tracking import ForecastRecord

                forecast_tracker = ForecastTracker(self._memory_dir / "findings")
                retro_data = package.data.get("weekly_retrospective")
                if retro_data:
                    by_bot = {}
                    by_type: dict[str, list[float]] = {}
                    for pred in retro_data.get("predictions", []):
                        bid = pred.get("bot_id", "")
                        accuracy = pred.get("accuracy", "")
                        score = {
                            "correct": 1.0,
                            "partially_correct": 0.5,
                            "incorrect": 0.0,
                        }.get(accuracy)
                        if score is None:
                            continue
                        if bid:
                            by_bot.setdefault(bid, []).append(score)
                        metric = pred.get("metric", "") or pred.get("prediction_type", "")
                        if metric:
                            by_type.setdefault(metric, []).append(score)
                    forecast_record = ForecastRecord(
                        week_start=week_start,
                        week_end=week_end,
                        predictions_reviewed=retro_data.get("predictions_reviewed", 0),
                        correct_predictions=retro_data.get("correct", 0),
                        accuracy=retro_data.get("accuracy_pct", 0.0) / 100.0,
                        by_bot={b: sum(v) / len(v) for b, v in by_bot.items() if v},
                        by_type={m: sum(v) / len(v) for m, v in by_type.items() if v},
                    )
                    forecast_tracker.record_week(forecast_record)
                meta = forecast_tracker.compute_meta_analysis()
                if meta.weeks_analyzed > 0:
                    package.data["forecast_meta_analysis"] = meta.model_dump(mode="json")
            except Exception:
                logger.error("Forecast tracking failed — skipping", exc_info=True)

            # Inject outcome-derived lessons into learning ledger
            try:
                from skills.learning_ledger import LearningLedger

                _ledger = LearningLedger(self._memory_dir / "findings")
                outcome_lessons: list[str] = []
                outcomes_data = package.data.get("outcome_measurements", [])
                for om in outcomes_data:
                    verdict = om.get("verdict", "")
                    sid = om.get("suggestion_id", "")
                    if verdict in ("positive", "negative"):
                        outcome_lessons.append(
                            f"Suggestion {sid}: {verdict} outcome "
                            f"(PnL delta={om.get('pnl_delta', 0):.2f})"
                        )
                if outcome_lessons:
                    _ledger.record_outcome_lessons(week_start, outcome_lessons)
            except Exception:
                logger.debug("Outcome-derived lesson injection failed")

            # Correction pattern extraction — surface recurring human correction patterns
            try:
                from skills.correction_pattern_extractor import CorrectionPatternExtractor

                ctx = ContextBuilder(self._memory_dir)
                corrections = ctx.load_corrections()
                if corrections:
                    extractor = CorrectionPatternExtractor(min_occurrences=2)
                    pattern_report = extractor.extract(corrections)
                    if pattern_report.patterns:
                        package.data["correction_patterns"] = [
                            p.model_dump(mode="json") for p in pattern_report.patterns
                        ]
                        # Persist to findings for use in future base_package()
                        patterns_path = self._memory_dir / "findings" / "correction_patterns.jsonl"
                        patterns_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(patterns_path, "w", encoding="utf-8") as f:
                            for p in pattern_report.patterns:
                                f.write(json.dumps(p.model_dump(mode="json")) + "\n")

                        # Synthesize persistent correction patterns into learning ledger lessons
                        try:
                            from skills.learning_ledger import LearningLedger as _LL
                            correction_lessons = []
                            for p in pattern_report.patterns[:5]:
                                if p.count >= 3:
                                    target = getattr(p, "target", "") or ""
                                    correction_lessons.append(
                                        f"[correction] {p.description}. "
                                        f"Adjust analysis approach for {target}."
                                    )
                            if correction_lessons:
                                _corr_ledger = _LL(self._memory_dir / "findings")
                                _corr_ledger.record_outcome_lessons(
                                    week_start, correction_lessons, source="corrections",
                                )
                        except Exception:
                            logger.debug("Correction lesson synthesis failed")
            except Exception:
                logger.error("Correction pattern extraction failed — skipping", exc_info=True)

            # Structural hypotheses — JSONL-backed library with adaptive lifecycle
            try:
                from skills.hypothesis_library import HypothesisLibrary, get_relevant as _get_relevant_legacy

                hypothesis_lib = HypothesisLibrary(self._memory_dir / "findings")
                active_hypotheses = hypothesis_lib.get_active()

                # Keyword-match from strategy suggestions (same logic as legacy)
                suggestions_list = getattr(refinement_report, "suggestions", [])
                keyword_map = {
                    "signal": "signal_decay", "decay": "signal_decay", "alpha": "signal_decay",
                    "filter": "filter_over_blocking", "block": "filter_over_blocking",
                    "exit": "exit_timing", "premature": "exit_timing", "stop": "exit_timing",
                    "slippage": "adverse_fills", "fill": "adverse_fills",
                    "regime": "regime_breakdown",
                    "correlation": "correlation_crowding", "crowding": "correlation_crowding",
                    "diversif": "correlation_crowding",
                }
                matched_categories: set[str] = set()
                for suggestion in suggestions_list:
                    title = (getattr(suggestion, "title", "") or "").lower()
                    description = (getattr(suggestion, "description", "") or "").lower()
                    text = f"{title} {description}"
                    for keyword, category in keyword_map.items():
                        if keyword in text:
                            matched_categories.add(category)

                # Merge: keyword-matched + high-effectiveness active hypotheses
                seen_ids: set[str] = set()
                merged: list[dict] = []
                for h in active_hypotheses:
                    if h.category in matched_categories or h.effectiveness > 0.3 or h.status == "candidate":
                        if h.id not in seen_ids:
                            seen_ids.add(h.id)
                            merged.append({
                                "id": h.id, "title": h.title, "category": h.category,
                                "description": h.description, "evidence_required": h.evidence_required,
                                "reversibility": h.reversibility, "estimated_complexity": h.estimated_complexity,
                                "effectiveness": round(h.effectiveness, 3),
                                "times_proposed": h.times_proposed,
                            })
                if merged:
                    package.data["structural_hypotheses"] = merged
            except Exception:
                logger.error("Hypothesis library matching failed — skipping", exc_info=True)

            # Cross-bot transfer proposals
            try:
                from skills.pattern_library import PatternLibrary
                from skills.transfer_proposal_builder import TransferProposalBuilder

                lib = PatternLibrary(self._memory_dir / "findings")
                builder = TransferProposalBuilder(
                    pattern_library=lib,
                    curated_dir=self._curated_dir,
                    bots=self._bots,
                    findings_dir=self._memory_dir / "findings",
                    strategy_registry=self._strategy_registry,
                )
                proposals = builder.build_proposals()
                if proposals:
                    package.data["transfer_proposals"] = [
                        p.model_dump(mode="json") for p in proposals
                    ]
            except Exception:
                logger.error("Transfer proposal building failed — skipping", exc_info=True)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "weekly_analysis",
            })

            # Invoke the configured agent runtime
            result = await self._agent_runner.invoke(
                agent_type="weekly_analysis",
                prompt_package=package,
                run_id=run_id,
                allowed_tools=["Read", "Grep", "Glob"],
            )

            # Parse structured response
            final_report = result.response
            if result.success:
                from analysis.response_parser import parse_response

                parsed = parse_response(result.response)
                try:
                    run_dir = result.run_dir or (self._runs_dir / run_id)
                    Path(run_dir).mkdir(parents=True, exist_ok=True)
                    (Path(run_dir) / "parsed_analysis.json").write_text(
                        parsed.model_dump_json(indent=2), encoding="utf-8",
                    )
                except Exception:
                    logger.error("Failed to save parsed analysis for %s", run_id, exc_info=True)

                if not parsed.parse_success:
                    logger.warning("No structured output block found in %s response", run_id)
                elif parsed.fallback_used:
                    logger.info("Fallback markdown extraction used for %s — structured block was missing", run_id)

                # Validate and annotate response
                final_report, validation = self._validate_and_annotate(
                    parsed,
                    week_start,
                    provider=result.provider,
                    model=result.effective_model,
                    run_id=run_id,
                    agent_type="weekly_analysis",
                    bot_ids=package.metadata.get("bot_ids", ""),
                )
                self._persist_validator_notes(result.run_dir or (self._runs_dir / run_id), validation)
                self._refresh_run_index_entry(
                    run_id=run_id,
                    agent_type="weekly_analysis",
                    run_dir=result.run_dir or (self._runs_dir / run_id),
                    provider=result.provider,
                    model=result.effective_model,
                    prompt_package=package,
                    success=result.success,
                    duration_ms=result.duration_ms,
                    cost_usd=result.cost_usd,
                )

                # Fallback: if validation failed but we have suggestions, record them unvalidated
                if validation is None and (parsed.suggestions or parsed.portfolio_proposals):
                    from analysis.response_validator import ValidationResult
                    validation = ValidationResult(
                        approved_suggestions=parsed.suggestions,
                        approved_predictions=parsed.predictions,
                        approved_portfolio_proposals=parsed.portfolio_proposals,
                    )
                    logger.warning("Validation failed for %s — recording unvalidated suggestions", run_id)

                # Record approved agent suggestions to tracker
                weekly_agent_ids = self._record_agent_suggestions(
                    validation, run_id, parsed,
                    provider=result.provider, model=result.effective_model,
                )

                # Record learning card feedback from validation signal
                self._record_learning_card_feedback_targeted(validation, package)

                # Record approved portfolio proposals
                self._record_portfolio_proposals(validation, run_id)

                # Run autonomous pipeline on recorded suggestions
                await self._run_autonomous_pipeline(weekly_agent_ids, run_id)

                # Record predictions
                self._record_predictions(week_start, parsed.predictions)

                # Wire hypothesis lifecycle for structural proposals
                self._update_hypothesis_lifecycle(parsed, suggestion_ids)

                # Extract and record patterns for cross-bot transfer
                self._extract_and_record_patterns(parsed, self._bots, suggestion_ids)

            # Record completion timestamp
            if self._brain:
                self._brain.record_weekly_analysis(datetime.now(timezone.utc).isoformat())

            self._event_stream.broadcast("weekly_analysis_complete", {
                "week_start": week_start,
                "success": result.success,
            })

            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            status = "completed" if result.success else "failed"
            self._record_run(
                run_id, "weekly_analysis", status,
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed,
            )

            if result.success:
                self._write_run_report(run_id, "weekly_report.md", final_report)
                await self._notify(
                    notification_type="weekly_report",
                    priority=NotificationPriority.NORMAL,
                    title=f"Weekly Report — {week_start} to {week_end}",
                    body=final_report[:2000],
                )

        except Exception as exc:
            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self._record_run(
                run_id, "weekly_analysis", "failed",
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed, error=str(exc),
            )
            logger.exception("Weekly analysis handler failed for %s", run_id)
            self._event_stream.broadcast("weekly_analysis_error", {
                "week_start": week_start,
                "error": str(exc),
            })

    async def handle_wfo(self, action: Action) -> None:
        """Run the WFO pipeline: runner -> report -> assemble -> invoke -> notify."""
        details = action.details or {}
        bot_id = details.get("bot_id", action.bot_id)
        data_end = details.get("data_end", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        run_id = f"wfo-{bot_id}-{data_end}"
        start_time = datetime.now(timezone.utc)
        self._record_run(run_id, "wfo", "running", started_at=start_time.isoformat())

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "wfo",
            })

            from schemas.wfo_config import WFOConfig
            from skills.run_wfo import WFORunner
            from analysis.wfo_report_builder import WFOReportBuilder
            from analysis.wfo_prompt_assembler import WFOPromptAssembler

            # Load WFO config
            from schemas.wfo_config import ParameterSpace

            config_path = self._curated_dir.parent / "wfo_configs" / f"{bot_id}.yaml"
            if config_path.exists():
                import yaml
                config = WFOConfig(**yaml.safe_load(config_path.read_text()))
            else:
                config = WFOConfig(
                    bot_id=bot_id,
                    parameter_space=ParameterSpace(bot_id=bot_id),
                )

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "config_load", "handler": "wfo",
            })

            # Load trades from curated data (filtered by date range)
            data_start = details.get("data_start", "")
            if not data_start:
                required_span_days = (
                    config.in_sample_days
                    + config.out_of_sample_days
                    + config.step_days * (config.min_folds - 1)
                )
                data_start = (
                    datetime.strptime(data_end, "%Y-%m-%d") - timedelta(days=required_span_days)
                ).strftime("%Y-%m-%d")
            trades, missed = self._load_trades_for_wfo(bot_id, date_start=data_start, date_end=data_end)

            # Run WFO pipeline
            runner = WFORunner(config)
            report = runner.run(trades, missed, data_start, data_end)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "wfo_run", "handler": "wfo",
            })

            # Write output
            output_dir = self._runs_dir / run_id
            runner.write_output(report, output_dir)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "report_build", "handler": "wfo",
            })

            # Build markdown report
            markdown = WFOReportBuilder().build_markdown(report)
            (output_dir / "wfo_report.md").write_text(markdown, encoding="utf-8")

            # Assemble prompt for provider review
            assembler = WFOPromptAssembler(
                bot_id=bot_id,
                memory_dir=self._memory_dir,
                wfo_output_dir=output_dir,
            )
            package = assembler.assemble(session_store=self._agent_runner.session_store)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "agent_invocation", "handler": "wfo",
            })

            # Invoke the configured agent runtime
            result = await self._agent_runner.invoke(
                agent_type="wfo",
                prompt_package=package,
                run_id=run_id,
                max_turns=3,
                allowed_tools=["Read", "Grep", "Glob"],
            )

            self._event_stream.broadcast("wfo_complete", {
                "bot_id": bot_id,
                "recommendation": report.recommendation.value,
                "success": result.success,
            })

            # Notify — CRITICAL if REJECT
            from schemas.wfo_results import WFORecommendation

            priority = (
                NotificationPriority.CRITICAL
                if report.recommendation == WFORecommendation.REJECT
                else NotificationPriority.NORMAL
            )
            await self._notify(
                notification_type="wfo_report",
                priority=priority,
                title=f"WFO {bot_id} — {report.recommendation.value.upper()}",
                body=result.response[:2000] if result.success else markdown[:2000],
            )
            finished_at = datetime.now(timezone.utc)
            self._record_run(
                run_id,
                "wfo",
                "completed",
                started_at=start_time.isoformat(),
                finished_at=finished_at.isoformat(),
                duration_ms=int((finished_at - start_time).total_seconds() * 1000),
            )

        except Exception as exc:
            logger.exception("WFO handler failed for %s", run_id)
            self._event_stream.broadcast("wfo_error", {
                "bot_id": bot_id,
                "error": str(exc),
            })
            finished_at = datetime.now(timezone.utc)
            self._record_run(
                run_id,
                "wfo",
                "failed",
                started_at=start_time.isoformat(),
                finished_at=finished_at.isoformat(),
                duration_ms=int((finished_at - start_time).total_seconds() * 1000),
                error=str(exc),
            )

    async def handle_triage(self, action: Action) -> None:
        """Run the triage pipeline: classify -> context -> assemble -> invoke -> notify."""
        details = action.details or {}
        bot_id = details.get("bot_id", action.bot_id)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"triage-{bot_id}-{timestamp}"

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "triage",
            })

            from analysis.triage_response_parser import parse_triage_response
            from schemas.bug_triage import ErrorEvent, TriageOutcome
            from skills.run_bug_triage import TriageRunner
            from skills.triage_context_builder import TriageContextBuilder
            from analysis.triage_prompt_assembler import TriagePromptAssembler

            event = ErrorEvent(
                bot_id=bot_id,
                error_type=details.get("error_type", "Unknown"),
                message=details.get("message", ""),
                stack_trace=details.get("stack_trace", ""),
                source_file=details.get("source_file", ""),
                source_line=details.get("source_line", 0),
                context=details.get("context", {}),
            )

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "classification", "handler": "triage",
            })

            triage_runner = TriageRunner(
                source_root=self._source_root,
                failure_log_path=self._failure_log_path,
            )
            triage_result = triage_runner.triage(event)

            self._event_stream.broadcast("triage_classified", {
                "bot_id": bot_id,
                "severity": triage_result.severity.value,
                "complexity": triage_result.complexity.value,
                "outcome": triage_result.outcome.value,
            })

            # Record recurrence against open reliability interventions
            if self._reliability_tracker is not None:
                try:
                    from schemas.reliability_learning import BugClass
                    bug_class = self._map_error_to_bug_class(
                        triage_result.error_event.category.value
                        if triage_result.error_event.category else "unknown"
                    )
                    matched = self._reliability_tracker.record_recurrence(
                        bot_id, bug_class, event.error_type,
                    )
                    if matched:
                        logger.info(
                            "Recurrence matched intervention %s for %s/%s",
                            matched, bot_id, bug_class.value,
                        )
                except Exception:
                    logger.warning("Failed to check reliability recurrence for %s", bot_id)

            agent_response = ""
            repair_proposal = None
            if triage_result.outcome in (TriageOutcome.KNOWN_FIX, TriageOutcome.NEEDS_INVESTIGATION):
                self._event_stream.broadcast("handler_progress", {
                    "run_id": run_id, "stage": "context_build", "handler": "triage",
                })
                ctx_builder = TriageContextBuilder(source_root=self._source_root)
                from skills.failure_log import FailureLog
                failure_log = FailureLog(self._failure_log_path)
                past_rejections = failure_log.get_past_rejections(
                    error_type=event.error_type, limit=5,
                )
                from schemas.bug_triage import ErrorCategory

                category = triage_result.error_event.category or ErrorCategory.UNKNOWN
                context = ctx_builder.build(
                    event,
                    triage_result.severity,
                    category,
                    past_rejections,
                )

                assembler = TriagePromptAssembler(memory_dir=self._memory_dir)
                package = assembler.assemble(
                    context, triage_result.severity, triage_result.complexity,
                    session_store=self._agent_runner.session_store,
                    bot_id=bot_id,
                )

                result = await self._agent_runner.invoke(
                    agent_type="triage",
                    prompt_package=package,
                    run_id=run_id,
                    allowed_tools=["Read", "Bash", "Grep", "Glob"],
                )

                if result.success:
                    agent_response = result.response
                    repair_proposal = parse_triage_response(result.response)

                    # Record reliability intervention for successful triage fix
                    if self._reliability_tracker is not None and repair_proposal:
                        try:
                            from schemas.reliability_learning import (
                                BugClass as _BugClass,
                                ReliabilityIntervention,
                            )
                            _bug_class = self._map_error_to_bug_class(
                                triage_result.error_event.category.value
                                if triage_result.error_event.category else "unknown"
                            )
                            from skills.reliability_tracker import ReliabilityTracker
                            intervention = ReliabilityIntervention(
                                intervention_id=ReliabilityTracker.generate_id(
                                    bot_id, _bug_class.value,
                                    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                ),
                                bot_id=bot_id,
                                bug_class=_bug_class,
                                error_category=event.error_type,
                                triage_run_id=run_id,
                                fix_description=(
                                    getattr(repair_proposal, "fix_plan", "")
                                    or getattr(repair_proposal, "issue_title", "")
                                )[:200],
                            )
                            self._reliability_tracker.record_intervention(intervention)
                        except Exception:
                            logger.warning("Failed to record reliability intervention for %s", run_id)

            if triage_result.outcome == TriageOutcome.KNOWN_FIX:
                handled = await self._handle_known_fix_triage(
                    triage_result,
                    repair_proposal,
                    agent_response,
                )
                if not handled:
                    await self._notify(
                        notification_type="triage_result",
                        priority=self._severity_to_priority(triage_result.severity),
                        title=f"Triage [{triage_result.severity.value.upper()}] {bot_id}",
                        body=(agent_response or triage_result.suggested_fix or event.message)[:2000],
                    )
            elif triage_result.outcome == TriageOutcome.NEEDS_INVESTIGATION:
                handled = await self._handle_investigation_triage(
                    triage_result,
                    repair_proposal,
                    agent_response,
                )
                if not handled:
                    await self._notify(
                        notification_type="triage_result",
                        priority=self._severity_to_priority(triage_result.severity),
                        title=f"Triage [{triage_result.severity.value.upper()}] {bot_id}",
                        body=(agent_response or event.message)[:2000],
                    )
            elif triage_result.outcome == TriageOutcome.NEEDS_HUMAN:
                await self._notify(
                    notification_type="triage_needs_human",
                    priority=NotificationPriority.HIGH,
                    title=f"Triage [{triage_result.severity.value.upper()}] {bot_id} — needs human",
                    body=f"{event.error_type}: {event.message}",
                )

        except Exception as exc:
            logger.exception("Triage handler failed for %s", run_id)
            self._event_stream.broadcast("triage_error", {
                "bot_id": bot_id,
                "error": str(exc),
            })

    async def handle_alert(self, action: Action) -> None:
        """Dispatch a CRITICAL alert immediately (bypasses quiet hours)."""
        details = action.details or {}
        await self._notify(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title=f"ALERT: {action.bot_id}",
            body=details.get("message", json.dumps(details)),
        )

    async def handle_heartbeat(self, action: Action) -> None:
        """Write heartbeat timestamp for a bot."""
        self._heartbeat_dir.mkdir(parents=True, exist_ok=True)
        hb_path = self._heartbeat_dir / f"{action.bot_id}.heartbeat"
        hb_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    async def handle_notification(self, action: Action) -> None:
        """Build and dispatch a notification from action details."""
        details = action.details or {}
        await self._notify(
            notification_type=details.get("notification_type", "general"),
            priority=NotificationPriority(details.get("priority", "normal")),
            title=details.get("title", ""),
            body=details.get("body", ""),
        )

    async def handle_discovery_analysis(self, action: Action) -> None:
        """Run the discovery analysis: raw data exploration for novel patterns."""
        details = action.details or {}
        date = details.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        bots = details.get("bots", self._bots)
        run_id = f"discovery-{date}"
        start_time = datetime.now(timezone.utc)
        self._record_run(run_id, "discovery_analysis", "running", started_at=start_time.isoformat())

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "discovery_analysis",
            })

            from analysis.discovery_prompt_assembler import DiscoveryPromptAssembler

            assembler = DiscoveryPromptAssembler(
                date=date,
                bots=bots,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                bot_configs=self._bot_configs,
            )
            package = assembler.assemble(session_store=self._agent_runner.session_store)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "discovery_analysis",
            })

            # Discovery agent gets higher max_turns and file access tools
            result = await self._agent_runner.invoke(
                agent_type="discovery_analysis",
                prompt_package=package,
                run_id=run_id,
                allowed_tools=["Read", "Grep", "Glob"],
                max_turns=15,
            )

            if result.success:
                from analysis.response_parser import parse_response

                parsed = parse_response(result.response)

                # Parse discoveries from structured output
                discoveries = []
                if parsed.raw_structured and "discoveries" in parsed.raw_structured:
                    discoveries = parsed.raw_structured["discoveries"]

                # Persist discoveries to findings
                if discoveries:
                    discoveries_path = self._memory_dir / "findings" / "discoveries.jsonl"
                    discoveries_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(discoveries_path, "a", encoding="utf-8") as f:
                        for d in discoveries:
                            d["run_id"] = run_id
                            d["date"] = date
                            d["discovered_at"] = datetime.now(timezone.utc).isoformat()
                            f.write(json.dumps(d) + "\n")

                    # Add novel hypotheses to hypothesis library
                    try:
                        from skills.hypothesis_library import HypothesisLibrary

                        hypothesis_lib = HypothesisLibrary(self._memory_dir / "findings")
                        for d in discoveries:
                            if d.get("testable_hypothesis") and d.get("confidence", 0) >= 0.5:
                                hypothesis_lib.add_candidate(
                                    title=d.get("pattern_description", "")[:100],
                                    description=d.get("testable_hypothesis", ""),
                                    category=d.get("proposed_root_cause", "novel"),
                                )
                    except Exception:
                        logger.warning("Failed to add discovery hypotheses")

                    self._event_stream.broadcast("discoveries_recorded", {
                        "run_id": run_id, "count": len(discoveries),
                    })

                # Parse strategy ideas from structured output
                strategy_ideas = []
                if parsed.raw_structured and "strategy_ideas" in parsed.raw_structured:
                    strategy_ideas = parsed.raw_structured["strategy_ideas"]

                if strategy_ideas:
                    import hashlib as _hashlib
                    ideas_path = self._memory_dir / "findings" / "strategy_ideas.jsonl"
                    ideas_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(ideas_path, "a", encoding="utf-8") as f:
                        for idea in strategy_ideas:
                            # Deterministic ID from description
                            desc = idea.get("description", "")
                            idea["idea_id"] = _hashlib.sha256(desc.encode()).hexdigest()[:12]
                            idea["proposed_at"] = datetime.now(timezone.utc).isoformat()
                            idea["run_id"] = run_id
                            idea["status"] = "proposed"
                            f.write(json.dumps(idea) + "\n")

                    # High-confidence ideas → structural experiment records
                    if self._structural_experiment_tracker is not None:
                        from schemas.structural_experiment import (
                            AcceptanceCriteria,
                            ExperimentRecord,
                        )

                        for idea in strategy_ideas:
                            if idea.get("confidence", 0) >= 0.7:
                                try:
                                    exp_id = "exp_" + _hashlib.sha256(
                                        f"{run_id}:{idea.get('bot_id', 'unknown')}:{idea.get('title', '')}".encode()
                                    ).hexdigest()[:12]

                                    # Extract criteria from the idea itself if available
                                    idea_criteria: list[AcceptanceCriteria] = []
                                    for raw_c in idea.get("acceptance_criteria", []):
                                        if isinstance(raw_c, dict) and raw_c.get("metric"):
                                            try:
                                                idea_criteria.append(AcceptanceCriteria(**raw_c))
                                            except Exception:
                                                pass
                                    if not idea_criteria:
                                        # Fallback: infer from applicable_regimes and edge_hypothesis
                                        default_metric = "pnl"
                                        default_window = 14
                                        default_min_trades = 20
                                        # Longer window for regime-specific strategies
                                        if idea.get("applicable_regimes") and len(idea.get("applicable_regimes", [])) <= 2:
                                            default_window = 30
                                            default_min_trades = 15
                                        idea_criteria = [
                                            AcceptanceCriteria(
                                                metric=default_metric,
                                                direction="improve",
                                                minimum_change=0.0,
                                                observation_window_days=default_window,
                                                minimum_trade_count=default_min_trades,
                                            ),
                                        ]

                                    experiment = ExperimentRecord(
                                        experiment_id=exp_id,
                                        bot_id=idea.get("bot_id", "unknown"),
                                        title=idea.get("title", "Strategy idea"),
                                        description=idea.get("description", ""),
                                        proposal_run_id=run_id,
                                        acceptance_criteria=idea_criteria,
                                    )
                                    self._structural_experiment_tracker.record_experiment(experiment)
                                    logger.info("Recorded structural experiment %s for strategy idea %s", exp_id, idea.get("idea_id"))
                                    await self._notify(
                                        "structural_experiment_proposed",
                                        NotificationPriority.NORMAL,
                                        f"Structural Experiment Proposed: {experiment.title}",
                                        (f"Bot: {experiment.bot_id}\n"
                                         f"ID: {exp_id}\n"
                                         f"Description: {experiment.description[:200]}"),
                                    )
                                except Exception:
                                    logger.warning("Failed to create experiment for strategy idea %s", idea.get("idea_id"))

                    self._event_stream.broadcast("strategy_ideas_proposed", {
                        "run_id": run_id, "count": len(strategy_ideas),
                    })

                # Process structural proposals from discovery
                if parsed.structural_proposals:
                    self._update_hypothesis_lifecycle(parsed, {})
                    self._record_structural_experiments(parsed.structural_proposals, run_id)

                self._write_run_report(run_id, "discovery_report.md", result.response)

            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            status = "completed" if result.success else "failed"
            self._record_run(
                run_id, "discovery_analysis", status,
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self._record_run(
                run_id, "discovery_analysis", "failed",
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed, error=str(exc),
            )
            logger.exception("Discovery analysis handler failed for %s", run_id)

    async def handle_feedback(self, action: Action) -> None:
        """Process user feedback from Telegram/Discord callbacks."""
        details = action.details or {}
        text = details.get("text", "")
        report_id = details.get("report_id", "unknown")
        if not text:
            return

        from analysis.feedback_handler import FeedbackHandler

        handler = FeedbackHandler(report_id=report_id)
        correction = handler.parse(text)
        corrections_path = self._memory_dir / "findings" / "corrections.jsonl"
        handler.write_correction(correction, corrections_path)

        # Record allocation changes when user approves allocation recommendations
        from schemas.corrections import CorrectionType

        if correction.correction_type == CorrectionType.ALLOCATION_CHANGE:
            self._record_allocation_change(correction, details)

        # Route suggestion accept/reject to SuggestionTracker
        if self._suggestion_tracker and correction.target_id:
            if correction.correction_type == CorrectionType.SUGGESTION_ACCEPT:
                accepted = False
                if self._approval_tracker and self._approval_handler:
                    pending = self._approval_tracker.find_pending_for_suggestion(
                        correction.target_id,
                    )
                    if pending is not None:
                        await self._approval_handler.handle_approve(pending.request_id)
                        updated = self._approval_tracker.get_by_id(pending.request_id)
                        accepted = (
                            updated is not None
                            and updated.status.value == "APPROVED"
                        )
                if not accepted and not (
                    self._approval_tracker
                    and self._approval_tracker.find_pending_for_suggestion(correction.target_id)
                ):
                    self._suggestion_tracker.accept(correction.target_id)
                    accepted = True
                if accepted:
                    self._event_stream.broadcast("suggestion_accepted", {
                        "suggestion_id": correction.target_id,
                    })
                    self._update_hypothesis_from_feedback(
                        correction.target_id, accepted=True,
                    )
            elif correction.correction_type == CorrectionType.SUGGESTION_REJECT:
                routed = False
                if self._approval_tracker and self._approval_handler:
                    pending = self._approval_tracker.find_pending_for_suggestion(
                        correction.target_id,
                    )
                    if pending is not None:
                        await self._approval_handler.handle_reject(
                            pending.request_id,
                            reason=text[:200],
                        )
                        routed = True
                if not routed:
                    self._suggestion_tracker.reject(correction.target_id, text[:200])
                self._event_stream.broadcast("suggestion_rejected", {
                    "suggestion_id": correction.target_id,
                })
                self._update_hypothesis_from_feedback(
                    correction.target_id, accepted=False,
                )

        await self._notify(
            notification_type="feedback_received",
            priority=NotificationPriority.LOW,
            title="Feedback recorded",
            body=f"Correction type: {correction.correction_type.value}",
        )

    async def _handle_known_fix_triage(
        self,
        triage_result,
        repair_proposal,
        agent_response: str,
    ) -> bool:
        if (
            repair_proposal is None
            or self._approval_tracker is None
            or self._config_registry is None
        ):
            return False

        from schemas.autonomous_pipeline import ApprovalRequest, ChangeKind, RepoRiskTier
        from skills.repo_change_guard import RepoChangeGuard

        profile = self._config_registry.get_profile(triage_result.error_event.bot_id)
        if profile is None:
            return False

        file_changes = list(repair_proposal.file_changes)
        planned_files = repair_proposal.candidate_files or [
            file_change.file_path for file_change in file_changes
        ]
        if not planned_files and not (repair_proposal.fix_plan or agent_response):
            return False

        guard = RepoChangeGuard()
        blocked_paths = guard.blocked_paths(profile, planned_files) if planned_files else []
        if blocked_paths:
            await self._notify(
                notification_type="triage_result",
                priority=self._severity_to_priority(triage_result.severity),
                title=f"Triage [{triage_result.severity.value.upper()}] {triage_result.error_event.bot_id}",
                body=(
                    "Blocked bug-fix proposal outside allowed_edit_paths: "
                    f"{', '.join(blocked_paths)}"
                )[:2000],
            )
            return True
        risk_tier = RepoRiskTier.REQUIRES_APPROVAL
        if planned_files:
            permission_result = guard.check_paths(profile, planned_files)
            risk_tier = self._permission_tier_to_risk(permission_result.tier)

        import hashlib

        request_id = hashlib.sha256(
            (
                f"{triage_result.error_event.bot_id}:{triage_result.error_event.error_type}:"
                f"{triage_result.error_event.message}:{'|'.join(planned_files)}"
            ).encode(),
        ).hexdigest()[:12]

        request = self._approval_tracker.get_by_id(request_id)
        if request is None:
            request = ApprovalRequest(
                request_id=request_id,
                suggestion_id=request_id,
                bot_id=triage_result.error_event.bot_id,
                change_kind=ChangeKind.BUG_FIX,
                title=repair_proposal.issue_title or f"Fix {triage_result.error_event.error_type}",
                summary=repair_proposal.fix_plan or triage_result.error_event.message,
                file_changes=file_changes,
                planned_files=planned_files,
                verification_commands=profile.verification_commands,
                risk_tier=risk_tier,
                draft_pr=risk_tier == RepoRiskTier.AUTO,
                implementation_notes=repair_proposal.risk_notes,
            )
            self._approval_tracker.create_request(request)
        elif request.pr_url:
            triage_result.pr_url = request.pr_url
            self._record_triage_followup(triage_result)
            await self._notify(
                notification_type="triage_result",
                priority=self._severity_to_priority(triage_result.severity),
                title=f"Triage [{triage_result.severity.value.upper()}] {request.bot_id}",
                body=(
                    f"Existing bug-fix PR: {request.pr_url}\n\n"
                    f"{(repair_proposal.fix_plan or agent_response or triage_result.error_event.message)[:1700]}"
                ),
            )
            return True

        self._event_stream.broadcast("triage_bug_fix_request_created", {
            "request_id": request.request_id,
            "bot_id": request.bot_id,
            "risk_tier": request.risk_tier.value,
            "planned_files": request.planned_files,
        })

        if request.risk_tier == RepoRiskTier.AUTO and self._approval_handler is not None:
            approval_message = await self._approval_handler.handle_approve(request.request_id)
            updated = self._approval_tracker.get_by_id(request.request_id)
            if updated and updated.pr_url:
                triage_result.pr_url = updated.pr_url
                self._record_triage_followup(triage_result)
            auto_body = approval_message
            if updated and updated.pr_url:
                auto_body = repair_proposal.fix_plan or agent_response or approval_message
            await self._notify(
                notification_type="triage_result",
                priority=self._severity_to_priority(triage_result.severity),
                title=f"Triage [{triage_result.severity.value.upper()}] {request.bot_id}",
                body=auto_body[:2000],
            )
            return True

        await self._notify(
            notification_type="triage_result",
            priority=self._severity_to_priority(triage_result.severity),
            title=f"Triage [{triage_result.severity.value.upper()}] {request.bot_id}",
            body=(
                f"Created {request.risk_tier.value} bug-fix request `{request.request_id}` "
                f"for {', '.join(planned_files)}.\n\n"
                f"{(repair_proposal.fix_plan or agent_response)[:1600]}"
            ),
        )
        return True

    async def _handle_investigation_triage(
        self,
        triage_result,
        repair_proposal,
        agent_response: str,
    ) -> bool:
        if repair_proposal is None or self._pr_builder is None or self._config_registry is None:
            return False

        from schemas.autonomous_pipeline import GitHubIssueRequest

        profile = self._config_registry.get_profile(triage_result.error_event.bot_id)
        if profile is None:
            return False

        repo_task = None
        repo_dir = None
        try:
            if self._repo_workspace_manager is not None:
                repo_task = self._repo_workspace_manager.prepare_workspace(
                    profile,
                    f"triage-issue-{triage_result.error_event.bot_id}-{triage_result.timestamp.strftime('%Y%m%d%H%M%S')}",
                )
                repo_dir = Path(repo_task.worktree_dir)
            elif profile.repo_dir:
                repo_dir = Path(profile.repo_dir)
            if repo_dir is None:
                return False

            category = triage_result.error_event.category.value if triage_result.error_event.category else "unknown"
            dedupe_key = (
                f"{triage_result.error_event.bot_id}:{triage_result.error_event.error_type}:"
                f"{triage_result.error_event.message[:80]}"
            )
            issue_request = GitHubIssueRequest(
                bot_id=triage_result.error_event.bot_id,
                title=repair_proposal.issue_title or f"Investigate {triage_result.error_event.error_type}",
                body=repair_proposal.issue_body or agent_response or triage_result.error_event.message,
                repo_dir=str(repo_dir),
                labels=[
                    "trading-assistant",
                    f"severity/{triage_result.severity.value}",
                    f"category/{category}",
                ],
                dedupe_key=dedupe_key,
                repo_task=repo_task,
            )
            result = await self._pr_builder.create_issue(issue_request)
        finally:
            if repo_task and self._repo_workspace_manager is not None:
                try:
                    self._repo_workspace_manager.cleanup(repo_task)
                except Exception:
                    logger.warning("Failed to clean up triage issue workspace %s", repo_task.task_id)

        issue_url = result.issue_url or result.existing_issue_url
        if not result.success or not issue_url:
            return False

        triage_result.github_issue_url = issue_url
        self._record_triage_followup(triage_result)
        self._event_stream.broadcast("triage_issue_created", {
            "bot_id": triage_result.error_event.bot_id,
            "issue_url": issue_url,
            "existing_issue_url": result.existing_issue_url or "",
        })
        await self._notify(
            notification_type="triage_result",
            priority=self._severity_to_priority(triage_result.severity),
            title=f"Triage [{triage_result.severity.value.upper()}] {triage_result.error_event.bot_id}",
            body=(
                f"Investigation issue: {issue_url}\n\n"
                f"{(repair_proposal.fix_plan or agent_response or triage_result.error_event.message)[:1700]}"
            ),
        )
        return True

    def _record_triage_followup(self, triage_result) -> None:
        try:
            from skills.failure_log import FailureLog

            FailureLog(self._failure_log_path).record_triage(triage_result)
        except Exception:
            logger.warning("Failed to append triage follow-up", exc_info=True)

    @staticmethod
    def _permission_tier_to_risk(permission_tier) -> str:
        from schemas.autonomous_pipeline import RepoRiskTier
        from schemas.permissions import PermissionTier

        if permission_tier == PermissionTier.AUTO:
            return RepoRiskTier.AUTO
        if permission_tier == PermissionTier.REQUIRES_DOUBLE_APPROVAL:
            return RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
        return RepoRiskTier.REQUIRES_APPROVAL

    def _record_allocation_change(self, correction, details: dict) -> None:
        """Persist an approved allocation change via AllocationTracker."""
        try:
            from skills.allocation_tracker import AllocationTracker
            from schemas.allocation_history import AllocationRecord, AllocationSource

            tracker = AllocationTracker(self._memory_dir / "findings")

            # Extract allocation details from the action payload if provided
            allocations = details.get("allocations", [])
            date = details.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

            if allocations:
                for alloc in allocations:
                    tracker.record(AllocationRecord(
                        date=date,
                        bot_id=alloc.get("bot_id", ""),
                        strategy_id=alloc.get("strategy_id", ""),
                        allocation_pct=alloc.get("allocation_pct", 0.0),
                        source=AllocationSource.MANUAL,
                        reason=f"Approved via feedback: {correction.raw_text[:100]}",
                    ))
            else:
                # Record the approval event even without specific allocations
                tracker.record(AllocationRecord(
                    date=date,
                    bot_id=details.get("bot_id", "unknown"),
                    allocation_pct=0.0,
                    source=AllocationSource.MANUAL,
                    reason=f"Allocation approved: {correction.raw_text[:200]}",
                ))
        except Exception:
            logger.error("Failed to record allocation change from feedback", exc_info=True)

    def _extract_and_record_patterns(
        self, parsed, bots: list[str], suggestion_ids: dict[str, str] | None = None,
    ) -> None:
        """Extract patterns from structural proposals and record in PatternLibrary."""
        if not parsed.structural_proposals:
            return
        try:
            from schemas.pattern_library import PatternCategory
            from skills.pattern_library import PatternLibrary, PatternEntry, PatternStatus

            lib = PatternLibrary(self._memory_dir / "findings")

            # Map proposal categories to PatternCategory
            category_map = {
                "signal": PatternCategory.ENTRY_SIGNAL,
                "signal_decay": PatternCategory.ENTRY_SIGNAL,
                "filter": PatternCategory.FILTER,
                "filter_over_blocking": PatternCategory.FILTER,
                "exit_timing": PatternCategory.EXIT_RULE,
                "exit": PatternCategory.EXIT_RULE,
                "adverse_fills": PatternCategory.RISK_MANAGEMENT,
                "regime_breakdown": PatternCategory.REGIME_GATE,
                "regime": PatternCategory.REGIME_GATE,
                "correlation_crowding": PatternCategory.COORDINATION,
                "position_sizing": PatternCategory.POSITION_SIZING,
                "structural": PatternCategory.ENTRY_SIGNAL,
            }

            existing = lib.load_all()
            existing_titles = {e.title for e in existing}

            for proposal in parsed.structural_proposals:
                # Dedup by title
                if proposal.title in existing_titles:
                    continue

                # Determine category from proposal fields
                raw_cat = ""
                if proposal.hypothesis_id:
                    # Try to infer from hypothesis category
                    try:
                        from skills.hypothesis_library import HypothesisLibrary
                        hyp_lib = HypothesisLibrary(self._memory_dir / "findings")
                        for h in hyp_lib.get_all_records():
                            if h.id == proposal.hypothesis_id:
                                raw_cat = h.category
                                break
                    except Exception:
                        pass
                if not raw_cat:
                    # Infer from title keywords
                    title_lower = proposal.title.lower()
                    for keyword, cat_str in [
                        ("filter", "filter"), ("exit", "exit"),
                        ("signal", "signal"), ("regime", "regime"),
                        ("sizing", "position_sizing"), ("stop", "exit"),
                    ]:
                        if keyword in title_lower:
                            raw_cat = cat_str
                            break

                cat = category_map.get(raw_cat, PatternCategory.ENTRY_SIGNAL)
                target_bots = [b for b in bots if b != proposal.bot_id]

                linked_sid = proposal.linked_suggestion_id or ""

                entry = PatternEntry(
                    title=proposal.title,
                    category=cat,
                    status=PatternStatus.PROPOSED,
                    source_bot=proposal.bot_id,
                    target_bots=target_bots,
                    description=proposal.description,
                    evidence=proposal.evidence,
                    linked_suggestion_id=linked_sid,
                )
                lib.add(entry)
                existing_titles.add(proposal.title)
                logger.info("Recorded pattern from structural proposal: %s", proposal.title)
        except Exception:
            logger.error("Failed to extract and record patterns", exc_info=True)

    def _update_hypothesis_from_feedback(
        self, suggestion_id: str, accepted: bool,
    ) -> None:
        """Update HypothesisLibrary when a suggestion linked to a hypothesis is accepted/rejected."""
        if not self._suggestion_tracker:
            return
        try:
            from skills.hypothesis_library import HypothesisLibrary

            # Find the suggestion to get its hypothesis_id
            all_suggestions = self._suggestion_tracker.load_all()
            hypothesis_id = None
            for s in all_suggestions:
                if s.get("suggestion_id") == suggestion_id:
                    hypothesis_id = s.get("hypothesis_id")
                    break

            if not hypothesis_id:
                return

            lib = HypothesisLibrary(self._memory_dir / "findings")
            if accepted:
                lib.record_acceptance(hypothesis_id)
            else:
                lib.record_rejection(hypothesis_id)
        except Exception:
            logger.error("Failed to update hypothesis lifecycle for suggestion %s", suggestion_id, exc_info=True)

    # --- Private helpers ---

    def _validate_and_annotate(
        self,
        parsed,
        date_or_week: str,
        provider: str = "",
        model: str = "",
        run_id: str = "",
        agent_type: str = "",
        bot_ids: str | list[str] | None = None,
    ):
        """Run response validation and return annotated report text + validation result.

        Returns:
            tuple[str, ValidationResult | None]: (annotated_report, validation_result)
        """
        try:
            from analysis.response_validator import ResponseValidator
            from skills.suggestion_scorer import SuggestionScorer

            ctx = ContextBuilder(self._memory_dir, curated_dir=self._curated_dir)
            rejected = ctx.load_rejected_suggestions()
            forecast_meta = ctx.load_forecast_meta()

            scorer = SuggestionScorer(self._memory_dir / "findings")
            scorecard = scorer.compute_scorecard()

            hypothesis_tr = ctx.load_hypothesis_track_record()
            prediction_accuracy = ctx.load_prediction_accuracy()
            recalibrations = ctx.load_recalibrations()

            # Load current macro regime for confidence adjustment
            macro_regime_ctx = ctx.load_macro_regime_context()
            current_macro_regime = macro_regime_ctx.get("macro_regime", "") if macro_regime_ctx else ""

            validator = ResponseValidator(
                rejected_suggestions=rejected,
                forecast_meta=forecast_meta,
                category_scorecard=scorecard,
                hypothesis_track_record=hypothesis_tr,
                prediction_accuracy=prediction_accuracy,
                recalibrations=recalibrations,
                current_macro_regime=current_macro_regime,
            )
            validation = validator.validate(parsed)

            final_report = parsed.raw_report
            if validation.validator_notes:
                final_report += "\n\n---\n## Validator Notes\n" + validation.validator_notes

            # Log validation results (with blocked details for learning signal)
            try:
                log_path = self._memory_dir / "findings" / "validation_log.jsonl"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                import json as _json
                blocked_details = [
                    {
                        "title": b.suggestion.title,
                        "reason": b.reason,
                        "bot_id": b.suggestion.bot_id,
                        "category": getattr(b.suggestion, "category", "") or "",
                    }
                    for b in validation.blocked_suggestions
                ]
                entry = {
                    "date": date_or_week,
                    "approved_count": len(validation.approved_suggestions),
                    "blocked_count": len(validation.blocked_suggestions),
                    "blocked_details": blocked_details,
                    "provider": provider,
                    "model": model,
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "bot_ids": bot_ids or "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(entry) + "\n")
            except Exception:
                logger.error("Failed to log validation results", exc_info=True)

            return final_report, validation
        except Exception:
            logger.warning("Response validation failed — using raw report")
            return parsed.raw_report, None

    def _persist_validator_notes(self, run_dir: Path, validation_result) -> None:
        """Persist validator notes as a run artifact for RunIndex FTS."""
        if validation_result is None:
            return
        notes = getattr(validation_result, "validator_notes", "") or ""
        if not notes:
            return
        try:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "validator_notes.md").write_text(
                notes,
                encoding="utf-8",
            )
        except Exception:
            logger.error("Failed to persist validator notes for %s", run_dir, exc_info=True)

    def _refresh_run_index_entry(
        self,
        *,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        provider: str = "",
        model: str = "",
        prompt_package=None,
        success: bool = True,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Refresh RunIndex after handlers persist parsed output and validator notes."""
        try:
            self._agent_runner.refresh_run_index(
                run_id=run_id,
                agent_type=agent_type,
                run_dir=Path(run_dir),
                provider=provider,
                model=model,
                prompt_package=prompt_package,
                success=success,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
            )
        except Exception:
            logger.debug("Failed to refresh RunIndex entry for %s", run_id)

    def _record_learning_card_feedback(
        self, validation_result, prompt_package,
    ) -> None:
        """Compatibility shim for older callers; delegates to targeted feedback."""
        self._record_learning_card_feedback_targeted(validation_result, prompt_package)

    def _record_learning_card_feedback_targeted(
        self, validation_result, prompt_package,
    ) -> None:
        """Record feedback only for retrieved cards that match validated tags."""
        if validation_result is None:
            return
        card_ids = (prompt_package.metadata or {}).get("_learning_card_ids")
        if not card_ids:
            return

        approved_tags = self._feedback_structured_tags(validation_result.approved_suggestions)
        blocked_tags = self._feedback_structured_tags(
            [b.suggestion for b in validation_result.blocked_suggestions],
            reasons=[b.reason for b in validation_result.blocked_suggestions],
        )
        if not approved_tags and not blocked_tags:
            return

        try:
            from skills.learning_card_store import LearningCardStore

            store = LearningCardStore(self._memory_dir / "findings")
            index = store.load()
            for card_id in card_ids:
                card = index.get(card_id)
                if card is None:
                    continue
                card_tags = set(card.tags)
                matched_approved = bool(card_tags & approved_tags)
                matched_blocked = bool(card_tags & blocked_tags)
                if matched_approved and not matched_blocked:
                    index.record_feedback(card_id, True)
                elif matched_blocked and not matched_approved:
                    index.record_feedback(card_id, False)
            store.save(index)
        except Exception:
            logger.error("Failed to record targeted learning card feedback", exc_info=True)

    @staticmethod
    def _feedback_structured_tags(suggestions: list, reasons: list[str] | None = None) -> set[str]:
        tags: set[str] = set()
        for suggestion in suggestions or []:
            category = str(getattr(suggestion, "category", "") or "").strip()
            if category:
                tag = Handlers._feedback_tag("category", category)
                if tag:
                    tags.add(tag)
        for reason in reasons or []:
            tag = Handlers._feedback_tag("reason", reason)
            if tag:
                tags.add(tag)
        return tags

    @staticmethod
    def _feedback_tag(prefix: str, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        chars: list[str] = []
        prev_sep = False
        for char in text:
            if char.isalnum():
                chars.append(char)
                prev_sep = False
            elif not prev_sep:
                chars.append("_")
                prev_sep = True
        slug = "".join(chars).strip("_")
        return f"{prefix}:{slug}" if slug else ""

    def _record_predictions(self, date_or_week: str, predictions: list) -> None:
        """Record structured predictions from a parsed response."""
        if not predictions:
            return
        try:
            from skills.prediction_tracker import PredictionTracker

            tracker = PredictionTracker(self._memory_dir / "findings")
            tracker.record_predictions(date_or_week, predictions)
        except Exception:
            logger.error("Failed to record predictions for %s", date_or_week, exc_info=True)

    def _update_hypothesis_lifecycle(self, parsed, suggestion_ids: dict) -> None:
        """Update hypothesis lifecycle based on parsed structural proposals."""
        if not parsed.structural_proposals:
            return
        try:
            from skills.hypothesis_library import HypothesisLibrary

            lib = HypothesisLibrary(self._memory_dir / "findings")
            for proposal in parsed.structural_proposals:
                if proposal.hypothesis_id:
                    lib.record_proposal(proposal.hypothesis_id)
        except Exception:
            logger.error("Failed to update hypothesis lifecycle", exc_info=True)

    def _run_portfolio_detectors(
        self, engine, week_start: str, week_end: str, portfolio_summary,
    ) -> list:
        """Run portfolio-level detectors from the strategy engine.

        Returns list of StrategySuggestion objects with tier=PORTFOLIO.
        """
        results: list = []

        # Load family snapshots for the week
        family_snapshots: dict = {}
        family_allocations: dict = {}
        correlation_matrix: dict = {}

        try:
            weekly_dir = self._curated_dir / "weekly" / week_start
            snap_path = weekly_dir / "allocation_analysis.json"
            if snap_path.exists():
                alloc_data = json.loads(snap_path.read_text(encoding="utf-8"))
                family_allocations = alloc_data.get("current_allocations", {})

            # Load family daily snapshots aggregated over the week
            from datetime import timedelta as _td_pf
            start_dt = datetime.strptime(week_start, "%Y-%m-%d")
            for i in range(7):
                date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                snap_path = self._curated_dir / date_str / "portfolio" / "family_snapshots.json"
                if snap_path.exists():
                    daily_snaps = json.loads(snap_path.read_text(encoding="utf-8"))
                    if isinstance(daily_snaps, list):
                        for snap in daily_snaps:
                            fam = snap.get("family", "")
                            if fam:
                                family_snapshots.setdefault(fam, []).append(snap)

            # Load correlation matrix from latest risk card
            for i in range(6, -1, -1):
                date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                risk_path = self._curated_dir / date_str / "portfolio_risk_card.json"
                if risk_path.exists():
                    risk_data = json.loads(risk_path.read_text(encoding="utf-8"))
                    correlation_matrix = risk_data.get("correlation_matrix", {})
                    if correlation_matrix:
                        break
        except Exception:
            logger.warning("Failed to load portfolio detector inputs", exc_info=True)

        # Aggregate daily snapshots into per-family summary dicts
        aggregated_families: dict = {}
        for fam, daily_list in family_snapshots.items():
            total_pnl = sum(s.get("total_net_pnl", 0.0) for s in daily_list)
            total_trades = sum(s.get("trade_count", 0) for s in daily_list)
            total_wins = sum(s.get("win_count", 0) for s in daily_list)
            aggregated_families[fam] = {
                "total_net_pnl": total_pnl,
                "trade_count": total_trades,
                "win_count": total_wins,
                "days": len(daily_list),
            }

        # Run each detector
        if aggregated_families and family_allocations:
            try:
                results.extend(engine.detect_family_imbalance(
                    aggregated_families, family_allocations,
                ))
            except Exception:
                logger.warning("detect_family_imbalance failed", exc_info=True)

        if correlation_matrix and family_allocations:
            try:
                results.extend(engine.detect_correlation_concentration(
                    correlation_matrix, family_allocations,
                ))
            except Exception:
                logger.warning("detect_correlation_concentration failed", exc_info=True)

        # Drawdown tier miscalibration
        if self._strategy_registry and hasattr(self._strategy_registry, "portfolio"):
            tiers = getattr(self._strategy_registry.portfolio, "drawdown_tiers", [])
            if tiers:
                try:
                    # Collect historical drawdowns from risk cards across the week
                    hist_drawdowns: list[float] = []
                    for i in range(7):
                        date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                        risk_path = self._curated_dir / date_str / "portfolio_risk_card.json"
                        if risk_path.exists():
                            rdata = json.loads(risk_path.read_text(encoding="utf-8"))
                            dd = rdata.get("max_drawdown_pct", 0.0)
                            if isinstance(dd, (int, float)):
                                hist_drawdowns.append(float(dd))
                    if hist_drawdowns:
                        results.extend(engine.detect_drawdown_tier_miscalibration(
                            hist_drawdowns, tiers,
                        ))
                except Exception:
                    logger.warning("detect_drawdown_tier_miscalibration failed", exc_info=True)

        # Coordination gaps
        try:
            concurrent_path = self._curated_dir / "weekly" / week_start / "concurrent_position_analysis.json"
            if concurrent_path.exists():
                concurrent_data = json.loads(concurrent_path.read_text(encoding="utf-8"))
                coord_config = None
                if self._strategy_registry and hasattr(self._strategy_registry, "coordination"):
                    coord_config = self._strategy_registry.coordination.model_dump(mode="json")
                results.extend(engine.detect_coordination_gaps(
                    concurrent_data, coord_config,
                ))
        except Exception:
            logger.warning("detect_coordination_gaps failed", exc_info=True)

        # Heat cap utilization
        if self._strategy_registry and hasattr(self._strategy_registry, "portfolio"):
            heat_cap = self._strategy_registry.portfolio.heat_cap_R
            if heat_cap > 0:
                try:
                    daily_heat: list[float] = []
                    for i in range(7):
                        date_str = (start_dt + _td_pf(days=i)).strftime("%Y-%m-%d")
                        risk_path = self._curated_dir / date_str / "portfolio_risk_card.json"
                        if risk_path.exists():
                            rdata = json.loads(risk_path.read_text(encoding="utf-8"))
                            heat = rdata.get("total_heat_R") or rdata.get("total_exposure", 0)
                            if isinstance(heat, (int, float)):
                                daily_heat.append(float(heat))
                    if daily_heat:
                        results.extend(engine.detect_heat_cap_utilization(
                            daily_heat, heat_cap,
                        ))
                except Exception:
                    logger.warning("detect_heat_cap_utilization failed", exc_info=True)

        # Drawdown correlation across families
        if family_snapshots and len(family_snapshots) >= 2:
            try:
                from itertools import accumulate
                from skills.compute_portfolio_risk import PortfolioRiskComputer

                family_equity: dict[str, list[float]] = {}
                for fam, daily_list in family_snapshots.items():
                    # Convert daily PnL to cumulative equity curve —
                    # _drawdown_series() computes drawdown from peak and
                    # expects an equity curve, not individual daily values.
                    family_equity[fam] = list(accumulate(
                        s.get("total_net_pnl", 0.0) for s in daily_list
                    ))
                dd_corr = PortfolioRiskComputer.compute_drawdown_correlation(family_equity)
                weekly_dir = self._curated_dir / "weekly" / week_start
                weekly_dir.mkdir(parents=True, exist_ok=True)
                (weekly_dir / "drawdown_correlation.json").write_text(
                    json.dumps(dd_corr, indent=2, default=str), encoding="utf-8"
                )
            except Exception:
                logger.warning("compute_drawdown_correlation failed", exc_info=True)

        return results

    def _record_portfolio_proposals(self, validation, run_id: str) -> None:
        """Record approved portfolio proposals as SuggestionRecords.

        Enforces cadence gate and concurrent deployment limit.
        """
        if not self._suggestion_tracker or validation is None:
            return
        proposals = getattr(validation, "approved_portfolio_proposals", [])
        if not proposals:
            return

        import hashlib
        from schemas.suggestion_tracking import SuggestionRecord

        # Concurrent deployment check: max 1 DEPLOYED portfolio change
        try:
            deployed_count = self._suggestion_tracker.get_deployed_portfolio_count()
            if deployed_count >= 1:
                logger.info(
                    "Skipping portfolio proposal recording: %d already deployed", deployed_count,
                )
                return
        except Exception:
            logger.warning("Portfolio deployment count check failed — proceeding cautiously")

        for idx, proposal in enumerate(proposals):
            ptype = getattr(proposal, "proposal_type", "unknown")
            ptype_str = ptype.value if hasattr(ptype, "value") else str(ptype)

            # Cadence gate
            if not self._check_portfolio_cadence(ptype_str):
                logger.info("Cadence gate blocked portfolio proposal: %s", ptype_str)
                continue

            # Run what-if analysis for allocation proposals
            what_if_result = None
            if ptype_str == "allocation_rebalance":
                try:
                    from skills.portfolio_what_if import PortfolioWhatIf

                    proposed = getattr(proposal, "proposed_config", {}) or {}
                    current = getattr(proposal, "current_config", {}) or {}
                    # Build family PnL series from curated snapshots
                    family_pnl = self._load_family_pnl_for_what_if()
                    # Try trade-level loading for enriched metrics
                    family_trades = None
                    if self._strategy_registry:
                        try:
                            family_trades = self._load_family_trades_for_what_if()
                        except Exception:
                            logger.warning("Trade-level loading failed, using daily aggregates")
                    if (family_pnl or family_trades) and current:
                        what_if = PortfolioWhatIf(
                            family_daily_pnl=family_pnl or {},
                            current_weights=current,
                            family_trades=family_trades,
                        )
                        what_if_result = what_if.evaluate(proposed)
                    if what_if_result and what_if_result.get("calmar_delta", 0) < 0:
                        logger.info(
                            "What-if shows negative Calmar delta for allocation proposal — skipping",
                        )
                        continue
                except Exception:
                    logger.warning("Portfolio what-if failed — recording without it")

            raw = f"{run_id}:portfolio:{idx}:{ptype_str}"
            suggestion_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

            detection_ctx: dict = {}
            if what_if_result:
                detection_ctx["what_if_result"] = what_if_result
            detection_ctx["current_config"] = getattr(proposal, "current_config", {})
            detection_ctx["proposed_config"] = getattr(proposal, "proposed_config", {})

            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id="PORTFOLIO",
                title=f"Portfolio: {ptype_str}",
                tier="portfolio",
                category=f"portfolio_{ptype_str}" if not ptype_str.startswith("portfolio_") else ptype_str,
                source_report_id=run_id,
                description=getattr(proposal, "evidence_summary", "") or "",
                confidence=float(getattr(proposal, "confidence", 0.5) or 0.5),
                detection_context=detection_ctx if detection_ctx else None,
            )
            recorded = self._suggestion_tracker.record(record)
            if recorded is not False:
                logger.info("Recorded portfolio proposal %s: %s", suggestion_id, ptype_str)
                self._event_stream.broadcast("portfolio_proposal_recorded", {
                    "run_id": run_id, "proposal_type": ptype_str,
                    "suggestion_id": suggestion_id,
                })

    def _check_portfolio_cadence(self, proposal_type: str) -> bool:
        """Check if enough time has passed since the last portfolio proposal of this type.

        Returns True if the cadence gate allows a new proposal.
        """
        if not self._suggestion_tracker:
            return True

        try:
            # Allocation: 30 days; risk/drawdown: 90 days
            if "allocation" in proposal_type or "coordination" in proposal_type:
                min_days = 30
            else:
                min_days = 90

            last_date = self._suggestion_tracker.get_last_portfolio_proposal_date(
                proposal_type=proposal_type,
            )
            if not last_date:
                return True  # No history — allow

            last_dt = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last_dt).days
            if elapsed < min_days:
                logger.info(
                    "Portfolio cadence gate: %s last proposed %d days ago (min %d)",
                    proposal_type, elapsed, min_days,
                )
                return False
            return True
        except Exception:
            logger.warning("Portfolio cadence check failed — allowing proposal")
            return True

    def _load_family_pnl_for_what_if(self, lookback_days: int = 60) -> dict[str, list[float]]:
        """Load family daily PnL series for what-if analysis."""
        from datetime import timedelta as _td_wif

        family_pnl: dict[str, list[float]] = {}
        end = datetime.now(timezone.utc)
        for d in range(lookback_days):
            date_str = (end - _td_wif(days=d)).strftime("%Y-%m-%d")
            snap_path = self._curated_dir / date_str / "portfolio" / "family_snapshots.json"
            if not snap_path.exists():
                continue
            try:
                snaps = json.loads(snap_path.read_text(encoding="utf-8"))
                for snap in snaps:
                    fam = snap.get("family", "")
                    if fam:
                        family_pnl.setdefault(fam, []).append(
                            snap.get("total_net_pnl", 0.0),
                        )
            except (json.JSONDecodeError, OSError):
                continue
        # Reverse so oldest first
        for fam in family_pnl:
            family_pnl[fam].reverse()
        return family_pnl

    def _load_family_trades_for_what_if(
        self, lookback_days: int = 60,
    ) -> dict[str, list]:
        """Load per-family trade-level data for enriched what-if analysis.

        Scans curated directories for trades.jsonl files, groups trades by
        family using strategy_registry bot_id → family mapping.
        Returns family_name → list[TradeEvent].
        """
        from datetime import timedelta as _td_trades
        from schemas.events import TradeEvent

        if not self._strategy_registry:
            return {}

        # Build bot_id → family mapping
        bot_to_family: dict[str, str] = {}
        for _sid, profile in self._strategy_registry.strategies.items():
            if profile.bot_id and profile.family:
                bot_to_family[profile.bot_id] = profile.family

        if not bot_to_family:
            return {}

        family_trades: dict[str, list] = {}
        end = datetime.now(timezone.utc)

        for d in range(lookback_days):
            date_str = (end - _td_trades(days=d)).strftime("%Y-%m-%d")
            date_dir = self._curated_dir / date_str
            if not date_dir.is_dir():
                continue

            for bot_id, family in bot_to_family.items():
                trades_file = date_dir / bot_id / "trades.jsonl"
                if not trades_file.exists():
                    continue
                for line in trades_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            trade = TradeEvent(**json.loads(line))
                            family_trades.setdefault(family, []).append(trade)
                        except Exception:
                            logger.warning("Bad trade record in %s", trades_file)

        logger.info(
            "Loaded %d families with %d total trades for what-if",
            len(family_trades),
            sum(len(ts) for ts in family_trades.values()),
        )
        return family_trades

    def _record_suggestions(
        self, suggestions: list, run_id: str,
        category_scorecard=None,
    ) -> dict[str, str]:
        """Convert StrategySuggestions to SuggestionRecords and persist via tracker.

        Applies scorecard pre-validation: skips suggestions in categories with
        poor track records (win_rate < 0.3, sample_size >= 5) to prevent
        category leakage when scorecard was unavailable during build_report().

        Returns a mapping of suggestion_id → title for metadata injection.
        """
        if not self._suggestion_tracker or not suggestions:
            return {}

        import hashlib
        from schemas.agent_response import CATEGORY_TO_TIER as _C2T
        from schemas.suggestion_tracking import SuggestionRecord

        id_map: dict[str, str] = {}
        for idx, suggestion in enumerate(suggestions):
            title = getattr(suggestion, "title", "") or ""
            bot_id = getattr(suggestion, "bot_id", "") or ""
            tier = getattr(suggestion, "tier", "parameter")
            description = getattr(suggestion, "description", "") or ""

            # Pre-validation: skip suggestions in categories with poor track record
            if category_scorecard is not None:
                tier_val = str(tier.value) if hasattr(tier, "value") else str(tier)
                _skip = False
                for _score in getattr(category_scorecard, "scores", []):
                    if _score.bot_id != bot_id:
                        continue
                    _cat_tier = _C2T.get(_score.category, _score.category)
                    if _cat_tier == tier_val and _score.sample_size >= 5 and _score.win_rate < 0.3:
                        logger.info(
                            "Skipping strategy suggestion in poor category %s/%s (win_rate=%.0f%%, n=%d)",
                            bot_id, _score.category, _score.win_rate * 100, _score.sample_size,
                        )
                        _skip = True
                        break
                if _skip:
                    continue

            # Deterministic ID: SHA256(run_id + index + title)[:12]
            raw = f"{run_id}:{idx}:{title}"
            suggestion_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

            category_str = str(getattr(suggestion, "category", "") or "")
            confidence = float(getattr(suggestion, "confidence", 0.0) or 0.0)
            det_ctx = getattr(suggestion, "detection_context", None)
            det_ctx_dict = None
            if isinstance(det_ctx, BaseModel):
                det_ctx_dict = det_ctx.model_dump(mode="json")
            # Derive category from detector_name when not set on suggestion
            if not category_str and det_ctx:
                from analysis.strategy_engine import StrategyEngine
                _det_name = ""
                if isinstance(det_ctx, BaseModel):
                    _det_name = getattr(det_ctx, "detector_name", "")
                elif isinstance(det_ctx, dict):
                    _det_name = det_ctx.get("detector_name", "")
                if _det_name:
                    category_str = StrategyEngine._DETECTOR_TO_CATEGORY.get(_det_name, "")
            # Extract target_param and proposed_value from detection context
            target_param = None
            proposed_value = None
            expected_impact = ""
            if det_ctx:
                if isinstance(det_ctx, BaseModel):
                    target_param = getattr(det_ctx, "threshold_name", None)
                elif isinstance(det_ctx, dict):
                    target_param = det_ctx.get("threshold_name")
            raw_suggested = getattr(suggestion, "suggested_value", None)
            if raw_suggested is not None:
                try:
                    proposed_value = float(raw_suggested)
                except (TypeError, ValueError):
                    pass
            if description:
                expected_impact = description[:200]

            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                title=title,
                tier=str(tier.value) if hasattr(tier, "value") else str(tier),
                category=category_str,
                source_report_id=run_id,
                description=description,
                confidence=confidence,
                target_param=target_param,
                proposed_value=proposed_value,
                expected_impact=expected_impact,
                detection_context=det_ctx_dict,
            )

            recorded = self._suggestion_tracker.record(record)
            if recorded is not False:  # record() returns None (old) or bool (new)
                id_map[suggestion_id] = title
                logger.info("Recorded suggestion %s: %s", suggestion_id, title)

        if id_map:
            self._event_stream.broadcast("suggestions_recorded", {
                "run_id": run_id, "count": len(id_map),
            })

        return id_map

    def _record_agent_suggestions(
        self, validation_result, run_id: str, parsed=None,
        provider: str = "", model: str = "",
    ) -> dict[str, str]:
        """Record approved suggestions from validation into SuggestionTracker.

        Maps AgentSuggestion → SuggestionRecord with deterministic IDs and hypothesis linking.
        Only approved (not blocked) suggestions are recorded.

        Returns a mapping of suggestion_id → title.
        """
        if not self._suggestion_tracker or validation_result is None:
            return {}
        if not validation_result.approved_suggestions:
            return {}

        import hashlib
        from schemas.suggestion_tracking import SuggestionRecord

        # Build structural proposal lookup by explicit linked suggestion id.
        structural_context_map: dict[str, dict] = {}
        fallback_context_by_bot: dict[str, list[dict]] = {}
        if parsed and parsed.structural_proposals:
            for proposal in parsed.structural_proposals:
                context = {
                    "notes": proposal.description,
                    "file_changes": [
                        fc.model_dump(mode="json")
                        for fc in getattr(proposal, "file_changes", [])
                    ],
                    "verification_commands": list(
                        getattr(proposal, "verification_commands", []) or []
                    ),
                    "hypothesis_id": proposal.hypothesis_id,
                }
                if proposal.linked_suggestion_id:
                    structural_context_map[proposal.linked_suggestion_id] = context
                    continue
                fallback_context_by_bot.setdefault(proposal.bot_id or "", []).append(context)

        approved_counts_by_bot: dict[str, int] = {}
        for suggestion in validation_result.approved_suggestions:
            bot_id = suggestion.bot_id or ""
            approved_counts_by_bot[bot_id] = approved_counts_by_bot.get(bot_id, 0) + 1

        # Validate suggestions with backtesting before recording
        from skills.suggestion_validator import SuggestionValidator

        validator = SuggestionValidator(curated_dir=self._curated_dir)

        id_map: dict[str, str] = {}
        for idx, suggestion in enumerate(validation_result.approved_suggestions):
            title = suggestion.title or ""
            bot_id = suggestion.bot_id or ""
            category = suggestion.category or "parameter"

            raw = f"{run_id}:agent:{idx}:{title}"
            existing_id = getattr(suggestion, "suggestion_id", "")
            suggestion_id = existing_id if isinstance(existing_id, str) and existing_id else hashlib.sha256(raw.encode()).hexdigest()[:12]

            # Map category to tier using shared mapping
            from schemas.agent_response import CATEGORY_TO_TIER

            tier = CATEGORY_TO_TIER.get(category, "parameter")

            confidence = float(getattr(suggestion, "confidence", 0.0) or 0.0)

            # Run suggestion validation (backtest replay)
            validation_evidence = None
            val_result = None
            try:
                val_result = validator.validate(
                    suggestion_id=suggestion_id,
                    bot_id=bot_id,
                    category=category,
                    target_param=getattr(suggestion, "target_param", None),
                    proposed_value=getattr(suggestion, "proposed_value", None),
                    title=title,
                )
                validation_evidence = val_result.evidence.model_dump(mode="json")
                if val_result.degradation_detected:
                    logger.warning(
                        "Suggestion %s shows degradation in backtest: improvement=%.1f%%",
                        suggestion_id, val_result.evidence.improvement_pct,
                    )
            except Exception:
                logger.warning("Suggestion validation failed for %s — recording anyway", suggestion_id)

            structural_context = structural_context_map.get(suggestion_id)
            if structural_context is None:
                fallback_contexts = fallback_context_by_bot.get(bot_id, [])
                # Preserve precise explicit links, but still allow the legacy
                # one-suggestion/one-proposal hypothesis mapping for same-bot runs.
                if fallback_contexts and approved_counts_by_bot.get(bot_id, 0) == 1 and len(fallback_contexts) == 1:
                    structural_context = fallback_contexts[0]

            # Merge validation evidence into detection_context
            detection_ctx = {}
            if provider:
                detection_ctx["source_provider"] = provider
            if model:
                detection_ctx["source_model"] = model
            if validation_evidence:
                detection_ctx["validation_evidence"] = validation_evidence
            if val_result and val_result.requires_review:
                detection_ctx["requires_review"] = True

            # Extract target fields from AgentSuggestion
            _raw_tp = getattr(suggestion, "target_param", None)
            agent_target_param = str(_raw_tp) if isinstance(_raw_tp, str) else None
            agent_proposed_value = None
            raw_pv = getattr(suggestion, "proposed_value", None)
            if raw_pv is not None:
                try:
                    agent_proposed_value = float(raw_pv)
                except (TypeError, ValueError):
                    pass
            _raw_ei = getattr(suggestion, "expected_impact", None)
            agent_expected_impact = str(_raw_ei) if isinstance(_raw_ei, str) else ""

            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                title=title,
                tier=tier,
                category=category,
                source_report_id=run_id,
                description=suggestion.evidence_summary or "",
                confidence=confidence,
                target_param=agent_target_param,
                proposed_value=agent_proposed_value,
                expected_impact=agent_expected_impact,
                hypothesis_id=(
                    structural_context.get("hypothesis_id")
                    if structural_context is not None
                    else None
                ),
                implementation_context=structural_context,
                detection_context=detection_ctx if detection_ctx else None,
            )

            recorded = self._suggestion_tracker.record(record)
            if recorded is not False:
                id_map[suggestion_id] = title
                logger.info("Recorded agent suggestion %s: %s", suggestion_id, title)

        if id_map:
            self._event_stream.broadcast("agent_suggestions_recorded", {
                "run_id": run_id, "count": len(id_map),
            })

        # Record structural experiments from approved proposals with acceptance_criteria
        if parsed and parsed.structural_proposals:
            self._record_structural_experiments(parsed.structural_proposals, run_id)

        return id_map

    def _record_structural_experiments(self, proposals, run_id: str) -> None:
        """Convert structural proposals with acceptance_criteria into experiment records."""
        if not self._structural_experiment_tracker:
            return

        import hashlib

        from schemas.structural_experiment import AcceptanceCriteria, ExperimentRecord

        for proposal in proposals:
            if not proposal.acceptance_criteria:
                continue
            # Validate criteria have at least a metric field
            valid_criteria: list[AcceptanceCriteria] = []
            for raw_c in proposal.acceptance_criteria:
                if isinstance(raw_c, dict) and raw_c.get("metric"):
                    try:
                        valid_criteria.append(AcceptanceCriteria(**raw_c))
                    except Exception:
                        pass
            if not valid_criteria:
                continue

            exp_id = "exp_" + hashlib.sha256(
                f"{run_id}:{proposal.bot_id}:{proposal.title}".encode()
            ).hexdigest()[:12]

            experiment = ExperimentRecord(
                experiment_id=exp_id,
                bot_id=proposal.bot_id,
                title=proposal.title,
                description=proposal.description,
                hypothesis_id=proposal.hypothesis_id,
                suggestion_id=proposal.linked_suggestion_id,
                proposal_run_id=run_id,
                acceptance_criteria=valid_criteria,
            )
            recorded = self._structural_experiment_tracker.record_experiment(experiment)
            if recorded:
                logger.info("Recorded structural experiment %s: %s", exp_id, proposal.title)

    async def _run_autonomous_pipeline(self, suggestion_ids: dict[str, str], run_id: str) -> None:
        """Run the autonomous pipeline on newly recorded suggestions (if enabled)."""
        if not self._autonomous_pipeline or not suggestion_ids:
            return
        try:
            await self._autonomous_pipeline.process_new_suggestions(
                suggestion_ids=list(suggestion_ids.keys()),
                run_id=run_id,
            )
        except Exception:
            logger.exception("Autonomous pipeline failed — analysis unaffected")

    def _record_run(
        self, run_id: str, agent_type: str, status: str,
        started_at: str = "", finished_at: str = "", error: str = "",
        duration_ms: int = 0,
    ) -> None:
        """Append a run history entry to the JSONL log."""
        try:
            self._run_history_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "run_id": run_id,
                "agent_type": agent_type,
                "handler": agent_type,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_ms": duration_ms,
                "error": error,
            }
            with open(self._run_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            logger.warning("Failed to write run history for %s", run_id)

    async def _notify(
        self,
        notification_type: str,
        priority: NotificationPriority,
        title: str,
        body: str,
    ) -> None:
        """Build a NotificationPayload and dispatch it."""
        payload = NotificationPayload(
            notification_type=notification_type,
            priority=priority,
            title=title,
            body=body,
        )
        hour_utc = datetime.now(timezone.utc).hour
        try:
            results = await self._dispatcher.dispatch(payload, self._notification_prefs, hour_utc)
            for r in results:
                if not r.success:
                    logger.warning(
                        "Notification delivery failed on %s: %s",
                        r.channel.value, r.error,
                    )
        except Exception:
            logger.exception("Notification dispatch failed for %s", notification_type)

    def _severity_to_priority(self, severity) -> NotificationPriority:
        """Map BugSeverity to NotificationPriority."""
        from schemas.bug_triage import BugSeverity

        return {
            BugSeverity.CRITICAL: NotificationPriority.CRITICAL,
            BugSeverity.HIGH: NotificationPriority.HIGH,
            BugSeverity.MEDIUM: NotificationPriority.NORMAL,
            BugSeverity.LOW: NotificationPriority.LOW,
        }.get(severity, NotificationPriority.NORMAL)

    @staticmethod
    def _map_error_to_bug_class(error_category: str) -> "BugClass":
        """Map error category string to BugClass enum."""
        from schemas.reliability_learning import BugClass

        mapping = {
            "connection": BugClass.CONNECTION,
            "network": BugClass.CONNECTION,
            "timeout": BugClass.CONNECTION,
            "data": BugClass.DATA_INTEGRITY,
            "data_integrity": BugClass.DATA_INTEGRITY,
            "parse": BugClass.DATA_INTEGRITY,
            "timing": BugClass.TIMING,
            "schedule": BugClass.TIMING,
            "config": BugClass.CONFIG,
            "configuration": BugClass.CONFIG,
            "logic": BugClass.LOGIC,
            "assertion": BugClass.LOGIC,
            "dependency": BugClass.DEPENDENCY,
            "import": BugClass.DEPENDENCY,
            "resource": BugClass.RESOURCE,
            "memory": BugClass.RESOURCE,
            "disk": BugClass.RESOURCE,
        }
        cat_lower = error_category.lower()
        for key, cls in mapping.items():
            if key in cat_lower:
                return cls
        return BugClass.UNKNOWN

    def _run_weekly_simulations(
        self,
        refinement_report: object,
        week_start: str,
        week_end: str,
    ) -> dict:
        """Run simulation skills for all bots unconditionally.

        Runs FilterSensitivityAnalyzer, CounterfactualSimulator, and
        ExitStrategySimulator for every bot that has data, regardless of
        whether the strategy engine flagged specific issues. This ensures
        healthy bots still get proactive "what if" analysis.

        Strategy engine suggestions are used to enrich context (e.g. extracting
        a specific regime for counterfactual analysis) but do NOT gate whether
        simulations run.
        """
        results: dict = {}

        try:
            from skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer
            from skills.counterfactual_simulator import CounterfactualSimulator
            from skills.exit_strategy_simulator import ExitStrategySimulator
            from schemas.exit_simulation import ExitStrategyConfig, ExitStrategyType, ExitSweepResult

            suggestions = getattr(refinement_report, "suggestions", [])
            counterfactual = CounterfactualSimulator()
            exit_sim = ExitStrategySimulator()

            # Extract regime hints from suggestions (used to enrich counterfactual, not to gate)
            regime_hints: dict[str, str] = {}
            for suggestion in suggestions:
                bot_id = getattr(suggestion, "bot_id", None) or ""
                title = getattr(suggestion, "title", "") or ""
                if "regime" in title.lower() and bot_id:
                    regime_hints[bot_id] = getattr(suggestion, "regime", None) or "ranging"

            # Run all simulations for every known bot
            for bot_id in self._bots:
                trades, missed = self._load_trades_for_week(bot_id, week_start, week_end)

                # FilterSensitivity — needs missed opportunities
                if missed:
                    try:
                        analyzer = FilterSensitivityAnalyzer(bot_id=bot_id, date=week_start)
                        report = analyzer.analyze(missed)
                        results[f"filter_sensitivity_{bot_id}"] = report.model_dump(mode="json")
                    except Exception:
                        logger.warning("FilterSensitivity failed for %s", bot_id)

                # Counterfactual — needs trades or missed
                if trades or missed:
                    try:
                        regime = regime_hints.get(bot_id, "ranging")
                        sim_result = counterfactual.simulate_regime_gate(trades, missed, regime)
                        results[f"counterfactual_{bot_id}"] = sim_result.model_dump(mode="json")
                    except Exception:
                        logger.warning("Counterfactual failed for %s", bot_id)

                # ExitStrategy sweep — test all 12 default configs
                if trades:
                    try:
                        sweep_results = exit_sim.sweep(trades)
                        best = max(sweep_results, key=lambda r: r.improvement)
                        sweep_out = ExitSweepResult(
                            bot_id=bot_id,
                            configs_tested=len(sweep_results),
                            baseline_pnl=best.baseline_pnl,
                            results=sweep_results,
                            best_strategy=best.strategy,
                            best_improvement=best.improvement,
                        )
                        results[f"exit_sweep_{bot_id}"] = sweep_out.model_dump(mode="json")
                    except Exception:
                        logger.warning("ExitStrategy sweep failed for %s", bot_id)

                # FilterInteraction — analyze filter pair co-activation patterns
                if trades or missed:
                    try:
                        from skills.filter_interaction_analyzer import FilterInteractionAnalyzer

                        fi_analyzer = FilterInteractionAnalyzer(bot_id=bot_id, date=week_start)
                        fi_report = fi_analyzer.analyze(trades, missed)
                        if fi_report.pairs:
                            results[f"filter_interaction_{bot_id}"] = fi_report.model_dump(mode="json")
                    except Exception:
                        logger.warning("FilterInteraction failed for %s", bot_id)

        except Exception:
            logger.warning("Simulation skills import failed — skipping simulations")

        return results

    def _run_allocation_analyses(
        self,
        portfolio_summary: object,
        week_start: str,
        week_end: str,
    ) -> dict:
        """Run portfolio allocation, synergy, and proportion optimization analyses."""
        results: dict = {}

        try:
            from schemas.weekly_metrics import WeeklySummary
            from skills.portfolio_allocator import PortfolioAllocator
            from skills.synergy_analyzer import SynergyAnalyzer
            from skills.strategy_proportion_optimizer import StrategyProportionOptimizer

            bot_summaries = getattr(portfolio_summary, "bot_summaries", {})
            if not bot_summaries:
                return results

            # 1. Synergy analysis first (we need the correlation matrix for allocation)
            per_strat = {
                bid: s.per_strategy_summary
                for bid, s in bot_summaries.items()
            }
            synergy = SynergyAnalyzer(week_start, week_end)
            synergy_report = synergy.compute(per_strat)
            results["synergy_analysis"] = synergy_report.model_dump(mode="json")

            # Also compute intra-bot synergy for each bot with multiple strategies
            intra_bot_results: dict = {}
            for bid, strats in per_strat.items():
                if len(strats) >= 2:
                    intra_report = synergy.compute_intra_bot(bid, strats)
                    intra_bot_results[bid] = intra_report.model_dump(mode="json")
            if intra_bot_results:
                results["intra_bot_synergy"] = intra_bot_results

            # 2. Portfolio allocation (cross-bot) with correlation matrix from synergy
            from skills.allocation_tracker import AllocationTracker
            from skills.drift_analyzer import DriftAnalyzer

            allocator = PortfolioAllocator(week_start, week_end)
            n_bots = len(bot_summaries)
            tracker = AllocationTracker(self._memory_dir / "findings")
            latest_actuals = tracker.get_latest_actuals()
            if latest_actuals:
                default_pct = 100.0 / n_bots if n_bots > 0 else 0.0
                current = {bid: latest_actuals.get(bid, default_pct) for bid in bot_summaries}
                total = sum(current.values())
                if total > 0 and abs(total - 100.0) > 0.1:
                    current = {bid: pct / total * 100.0 for bid, pct in current.items()}
            else:
                current = {bid: 100.0 / n_bots for bid in bot_summaries} if n_bots > 0 else {}
            bot_correlation = synergy.compute_bot_correlation_matrix(per_strat)
            alloc_report = allocator.compute(bot_summaries, current, correlation_matrix=bot_correlation)
            results["portfolio_allocation"] = alloc_report.model_dump(mode="json")

            # Record allocation snapshot and compute drift trend
            snapshot = DriftAnalyzer.compute_snapshot(alloc_report, current)
            tracker.record_snapshot(snapshot)
            all_snapshots = tracker.load_snapshots()
            drift_trend = DriftAnalyzer.compute_drift_trend(all_snapshots)
            results["allocation_drift"] = {
                "current_snapshot": snapshot.model_dump(mode="json"),
                "trend": drift_trend,
            }

            # 3. Proportion optimization (intra-bot)
            optimizer = StrategyProportionOptimizer(week_start, week_end)
            proportion_report = optimizer.compute(per_strat)
            results["proportion_optimization"] = proportion_report.model_dump(mode="json")

            # 4. Structural analysis
            from skills.structural_analyzer import StructuralAnalyzer

            structural = StructuralAnalyzer(week_start, week_end)
            structural_report = structural.compute(per_strat)
            results["structural_analysis"] = structural_report.model_dump(mode="json")

            # Write to weekly curated dir for prompt assembler
            weekly_dir = self._curated_dir / "weekly" / week_start
            weekly_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            (weekly_dir / "structural_analysis.json").write_text(
                _json.dumps(results["structural_analysis"], indent=2, default=str),
                encoding="utf-8",
            )

            if "allocation_drift" in results:
                (weekly_dir / "allocation_drift.json").write_text(
                    _json.dumps(results["allocation_drift"], indent=2, default=str),
                    encoding="utf-8",
                )

            # 5. Regime-conditional metrics
            from analysis.strategy_engine import StrategyEngine as _SE

            engine = _SE(
                week_start=week_start, week_end=week_end,
                threshold_learner=self._threshold_learner,
            )
            trades_by_bot: dict[str, list] = {}
            for bid in bot_summaries:
                trades, _ = self._load_trades_for_week(bid, week_start, week_end)
                trades_by_bot[bid] = trades

            regime_report = engine.compute_regime_conditional_metrics(per_strat, trades_by_bot)
            results["regime_conditional_analysis"] = regime_report.model_dump(mode="json")
            (weekly_dir / "regime_conditional_analysis.json").write_text(
                _json.dumps(results["regime_conditional_analysis"], indent=2, default=str),
                encoding="utf-8",
            )

            # 6. Interaction analysis (swing_trader only)
            if "swing_trader" in bot_summaries:
                from skills.interaction_analyzer import InteractionAnalyzer
                from schemas.interaction_analysis import CoordinatorAction

                ia = InteractionAnalyzer(week_start, week_end, bot_id="swing_trader")
                coord_events = self._load_coordinator_events(week_start, week_end)
                swing_trades = trades_by_bot.get("swing_trader", [])
                interaction_report = ia.compute(coord_events, swing_trades)
                results["interaction_analysis"] = interaction_report.model_dump(mode="json")
                (weekly_dir / "interaction_analysis.json").write_text(
                    _json.dumps(results["interaction_analysis"], indent=2, default=str),
                    encoding="utf-8",
                )

        except Exception:
            logger.warning("Allocation analyses failed — skipping")

        return results

    def _load_trades_for_week(
        self, bot_id: str, week_start: str, week_end: str,
    ) -> tuple:
        """Load trade and missed opportunity events for a bot within a date range."""
        from datetime import timedelta
        from schemas.events import TradeEvent, MissedOpportunityEvent

        trades: list[TradeEvent] = []
        missed: list[MissedOpportunityEvent] = []
        start = datetime.strptime(week_start, "%Y-%m-%d")
        end = datetime.strptime(week_end, "%Y-%m-%d")

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            bot_dir = self._curated_dir / date_str / bot_id

            if bot_dir.is_dir():
                trades_file = bot_dir / "trades.jsonl"
                if trades_file.exists():
                    for line in trades_file.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                trades.append(TradeEvent(**json.loads(line)))
                            except Exception:
                                pass

                missed_file = bot_dir / "missed.jsonl"
                if missed_file.exists():
                    for line in missed_file.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                missed.append(MissedOpportunityEvent(**json.loads(line)))
                            except Exception:
                                pass

            current += timedelta(days=1)

        return (trades, missed)

    def _load_coordinator_events(
        self, week_start: str, week_end: str,
    ) -> list:
        """Load coordinator action events for swing_trader within a date range."""
        from datetime import timedelta
        from schemas.interaction_analysis import CoordinatorAction

        events: list[CoordinatorAction] = []
        start = datetime.strptime(week_start, "%Y-%m-%d")
        end = datetime.strptime(week_end, "%Y-%m-%d")

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            coord_file = self._curated_dir / date_str / "swing_trader" / "coordinator_impact.json"
            if not coord_file.exists():
                current += timedelta(days=1)
                continue

            try:
                data = json.loads(coord_file.read_text())
                for evt in data.get("events", []):
                    try:
                        events.append(CoordinatorAction(**evt))
                    except Exception:
                        logger.warning("Skipping malformed coordinator event in %s", coord_file)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not read coordinator file %s", coord_file)

            current += timedelta(days=1)

        return events

    def _load_bot_dailies(self, bot_id: str, week_start: str, week_end: str) -> list:
        """Load BotDailySummary objects from curated dir for a date range."""
        from datetime import timedelta
        from schemas.daily_metrics import BotDailySummary

        start = datetime.strptime(week_start, "%Y-%m-%d")
        dailies = []
        for i in range(7):
            date_str = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            summary_path = self._curated_dir / date_str / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text())
                    dailies.append(BotDailySummary(**data))
                except (json.JSONDecodeError, Exception):
                    logger.warning("Could not load daily summary for %s on %s", bot_id, date_str)
        return dailies

    def _load_weekly_strategy_evidence(
        self,
        week_start: str,
        week_end: str,
        bot_summaries: dict[str, object],
        signal_health_data: dict[str, dict] | None = None,
        factor_rolling_data: dict[str, list[dict]] | None = None,
        simulation_results: dict | None = None,
    ) -> dict:
        """Aggregate curated weekly evidence into strategy-engine inputs."""
        bot_ids = list(bot_summaries.keys())
        trades_by_bot: dict[str, list] = {}
        missed_by_bot: dict[str, list] = {}
        for bot_id in bot_ids:
            trades, missed = self._load_trades_for_week(bot_id, week_start, week_end)
            trades_by_bot[bot_id] = trades
            missed_by_bot[bot_id] = missed

        evidence = {
            "filter_summaries": self._aggregate_weekly_filter_summaries(
                week_start, week_end, bot_ids, missed_by_bot,
            ),
            "regime_trends": self._aggregate_weekly_regime_trends(week_end, bot_ids),
            "rolling_sharpe": self._aggregate_rolling_sharpe(week_end, bot_ids),
            "signal_correlations": self._aggregate_signal_correlations(week_end, bot_ids),
            "hourly_buckets": self._aggregate_hourly_buckets(week_start, week_end, bot_ids),
            "correlation_summaries": self._build_bot_correlation_summaries(bot_summaries),
            "drawdown_data": self._aggregate_drawdown_data(week_end, trades_by_bot),
            "signal_health": signal_health_data or None,
            "factor_rolling": factor_rolling_data or None,
            "filter_interactions": self._load_filter_interactions_from_simulations(
                simulation_results or {},
            ),
            "orderbook_stats": self._aggregate_orderbook_stats(week_start, week_end, bot_ids),
        }

        # Load macro regime data from most recent portfolio curated file
        macro_regime_data = None
        end_dt = datetime.strptime(week_end, "%Y-%m-%d")
        for i in range(7):
            date_str = (end_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            mr_path = self._curated_dir / date_str / "portfolio" / "macro_regime_analysis.json"
            if mr_path.exists():
                try:
                    macro_regime_data = json.loads(mr_path.read_text(encoding="utf-8"))
                    if macro_regime_data:
                        break
                except (json.JSONDecodeError, OSError):
                    pass
        evidence["macro_regime_data"] = macro_regime_data

        # Aggregate exit efficiency data for detect_exit_timing_issues
        exit_efficiency_data: dict[str, dict] = {}
        for bot_id in bot_ids:
            daily_effs: list[dict] = []
            for date_str in self._iter_date_range(week_start, week_end):
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "exit_efficiency.json"
                )
                if isinstance(data, dict) and data.get("total_trades_with_data", 0) > 0:
                    daily_effs.append(data)
            if daily_effs:
                avg_eff = statistics.mean(d["avg_efficiency"] for d in daily_effs)
                avg_premature = statistics.mean(d["premature_exit_pct"] for d in daily_effs)
                exit_efficiency_data[bot_id] = {
                    "avg_exit_efficiency": avg_eff,
                    "premature_exit_pct": avg_premature,
                }
        evidence["exit_efficiency_data"] = exit_efficiency_data or None

        # Aggregate enriched instrumentation curated files
        evidence["execution_latency"] = self._aggregate_curated_file(
            "execution_latency.json", week_start, week_end, bot_ids,
        )
        evidence["sizing_data"] = self._aggregate_curated_file(
            "sizing_analysis.json", week_start, week_end, bot_ids,
        )
        evidence["param_correlations"] = self._aggregate_curated_file(
            "param_outcome_correlation.json", week_start, week_end, bot_ids,
        )
        evidence["portfolio_context"] = self._aggregate_curated_file(
            "portfolio_context.json", week_start, week_end, bot_ids,
        )

        return {key: value for key, value in evidence.items() if value}

    def _aggregate_weekly_filter_summaries(
        self,
        week_start: str,
        week_end: str,
        bot_ids: list[str],
        missed_by_bot: dict[str, list],
    ) -> dict[str, list]:
        """Aggregate daily filter analysis and missed events into weekly summaries."""
        from schemas.weekly_metrics import FilterWeeklySummary

        aggregated: dict[str, dict[str, dict[str, float]]] = {}
        for date_str in self._iter_date_range(week_start, week_end):
            for bot_id in bot_ids:
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "filter_analysis.json"
                )
                if not isinstance(data, dict):
                    continue
                counts = data.get("filter_block_counts", {})
                saved = data.get("filter_saved_pnl", {})
                missed_pnl = data.get("filter_missed_pnl", {})
                if not isinstance(counts, dict):
                    counts = {}
                if not isinstance(saved, dict):
                    saved = {}
                if not isinstance(missed_pnl, dict):
                    missed_pnl = {}

                for filter_name in set(counts) | set(saved) | set(missed_pnl):
                    record = aggregated.setdefault(bot_id, {}).setdefault(
                        filter_name,
                        {
                            "total_blocks": 0.0,
                            "blocks_that_would_have_won": 0.0,
                            "blocks_that_would_have_lost": 0.0,
                            "net_impact_pnl": 0.0,
                        },
                    )
                    record["total_blocks"] += float(counts.get(filter_name, 0) or 0)
                    record["net_impact_pnl"] += (
                        float(saved.get(filter_name, 0.0) or 0.0)
                        - float(missed_pnl.get(filter_name, 0.0) or 0.0)
                    )

        for bot_id, missed_events in missed_by_bot.items():
            for event in missed_events:
                filter_name = getattr(event, "blocked_by", "") or ""
                if not filter_name:
                    continue
                record = aggregated.setdefault(bot_id, {}).setdefault(
                    filter_name,
                    {
                        "total_blocks": 0.0,
                        "blocks_that_would_have_won": 0.0,
                        "blocks_that_would_have_lost": 0.0,
                        "net_impact_pnl": 0.0,
                    },
                )
                if getattr(event, "outcome_24h", 0.0) > 0:
                    record["blocks_that_would_have_won"] += 1.0
                else:
                    record["blocks_that_would_have_lost"] += 1.0

        results: dict[str, list] = {}
        for bot_id, filters in aggregated.items():
            summaries: list[FilterWeeklySummary] = []
            for filter_name, record in sorted(filters.items()):
                total_blocks = int(record["total_blocks"])
                classified = (
                    int(record["blocks_that_would_have_won"])
                    + int(record["blocks_that_would_have_lost"])
                )
                confidence = min(1.0, classified / total_blocks) if total_blocks > 0 else 0.0
                summaries.append(
                    FilterWeeklySummary(
                        bot_id=bot_id,
                        filter_name=filter_name,
                        total_blocks=total_blocks,
                        blocks_that_would_have_won=int(record["blocks_that_would_have_won"]),
                        blocks_that_would_have_lost=int(record["blocks_that_would_have_lost"]),
                        net_impact_pnl=round(record["net_impact_pnl"], 4),
                        confidence=round(confidence, 4),
                    )
                )
            if summaries:
                results[bot_id] = summaries
        return results

    def _aggregate_weekly_regime_trends(
        self, week_end: str, bot_ids: list[str], lookback_weeks: int = 4,
    ) -> dict[str, list]:
        """Build regime trends from the last few weeks of daily regime analysis."""
        from schemas.weekly_metrics import RegimePerformanceTrend

        end_dt = datetime.strptime(week_end, "%Y-%m-%d")
        history: dict[str, dict[str, dict[str, list[float] | list[int]]]] = {}

        for week_offset in range(lookback_weeks - 1, -1, -1):
            window_end = end_dt - timedelta(days=week_offset * 7)
            window_start = window_end - timedelta(days=6)
            weekly_data: dict[str, dict[str, dict[str, float]]] = {}

            for date_str in self._iter_date_range(
                window_start.strftime("%Y-%m-%d"),
                window_end.strftime("%Y-%m-%d"),
            ):
                for bot_id in bot_ids:
                    data = self._load_json_file(
                        self._curated_dir / date_str / bot_id / "regime_analysis.json"
                    )
                    if not isinstance(data, dict):
                        continue
                    regime_pnl = data.get("regime_pnl", {})
                    regime_trade_count = data.get("regime_trade_count", {})
                    regime_win_rate = data.get("regime_win_rate", {})
                    if not isinstance(regime_pnl, dict):
                        regime_pnl = {}
                    if not isinstance(regime_trade_count, dict):
                        regime_trade_count = {}
                    if not isinstance(regime_win_rate, dict):
                        regime_win_rate = {}

                    for regime_name in (
                        set(regime_pnl) | set(regime_trade_count) | set(regime_win_rate)
                    ):
                        record = weekly_data.setdefault(bot_id, {}).setdefault(
                            regime_name,
                            {"pnl": 0.0, "trade_count": 0.0, "weighted_wins": 0.0},
                        )
                        trade_count = float(regime_trade_count.get(regime_name, 0) or 0)
                        record["pnl"] += float(regime_pnl.get(regime_name, 0.0) or 0.0)
                        record["trade_count"] += trade_count
                        record["weighted_wins"] += (
                            float(regime_win_rate.get(regime_name, 0.0) or 0.0) * trade_count
                        )

            for bot_id in bot_ids:
                known_regimes = set(history.get(bot_id, {})) | set(weekly_data.get(bot_id, {}))
                for regime_name in known_regimes:
                    payload = weekly_data.get(bot_id, {}).get(
                        regime_name,
                        {"pnl": 0.0, "trade_count": 0.0, "weighted_wins": 0.0},
                    )
                    trend = history.setdefault(bot_id, {}).setdefault(
                        regime_name,
                        {
                            "weekly_pnl": [],
                            "weekly_trade_count": [],
                            "weekly_win_rate": [],
                        },
                    )
                    trade_count = int(payload["trade_count"])
                    win_rate = (
                        float(payload["weighted_wins"]) / float(payload["trade_count"])
                        if payload["trade_count"] > 0 else 0.0
                    )
                    trend["weekly_pnl"].append(round(float(payload["pnl"]), 4))
                    trend["weekly_trade_count"].append(trade_count)
                    trend["weekly_win_rate"].append(round(win_rate, 4))

        results: dict[str, list] = {}
        for bot_id, regimes in history.items():
            entries: list[RegimePerformanceTrend] = []
            for regime_name, payload in sorted(regimes.items()):
                trade_counts = payload["weekly_trade_count"]
                if not any(trade_counts):
                    continue
                entries.append(
                    RegimePerformanceTrend(
                        bot_id=bot_id,
                        regime=regime_name,
                        weekly_pnl=list(payload["weekly_pnl"]),
                        weekly_win_rate=list(payload["weekly_win_rate"]),
                        weekly_trade_count=list(trade_counts),
                    )
                )
            if entries:
                results[bot_id] = entries
        return results

    def _aggregate_rolling_sharpe(
        self, week_end: str, bot_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute 30d/60d/90d rolling Sharpe from daily summaries."""
        results: dict[str, dict[str, float]] = {}
        for bot_id in bot_ids:
            pnls = self._load_recent_daily_metric_series(bot_id, week_end, "net_pnl", 90)
            if not pnls:
                continue
            results[bot_id] = {
                "30d": round(self._compute_sharpe(pnls[-30:]), 4),
                "60d": round(self._compute_sharpe(pnls[-60:]), 4),
                "90d": round(self._compute_sharpe(pnls[-90:]), 4),
            }
        return results

    def _aggregate_signal_correlations(
        self, week_end: str, bot_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute short/long signal-to-outcome correlation baselines."""
        results: dict[str, dict[str, float]] = {}
        for bot_id in bot_ids:
            corr_30d = self._load_recent_signal_correlation(bot_id, week_end, 30)
            corr_90d = self._load_recent_signal_correlation(bot_id, week_end, 90)
            if corr_30d is None and corr_90d is None:
                continue
            results[bot_id] = {
                "30d": round(corr_30d or 0.0, 4),
                "90d": round(corr_90d if corr_90d is not None else (corr_30d or 0.0), 4),
            }
        return results

    def _aggregate_hourly_buckets(
        self, week_start: str, week_end: str, bot_ids: list[str],
    ) -> dict[str, list]:
        """Aggregate hourly buckets across the current week."""
        from schemas.hourly_performance import HourlyBucket

        aggregated: dict[str, dict[int, dict[str, float]]] = {}
        for date_str in self._iter_date_range(week_start, week_end):
            for bot_id in bot_ids:
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "hourly_performance.json"
                )
                if not isinstance(data, dict):
                    continue
                buckets = data.get("buckets", [])
                if not isinstance(buckets, list):
                    continue
                for bucket in buckets:
                    if not isinstance(bucket, dict):
                        continue
                    hour = int(bucket.get("hour", 0) or 0)
                    trade_count = int(bucket.get("trade_count", 0) or 0)
                    record = aggregated.setdefault(bot_id, {}).setdefault(
                        hour,
                        {
                            "trade_count": 0.0,
                            "pnl": 0.0,
                            "win_count": 0.0,
                            "process_quality_sum": 0.0,
                        },
                    )
                    record["trade_count"] += trade_count
                    record["pnl"] += float(bucket.get("pnl", 0.0) or 0.0)
                    record["win_count"] += (
                        float(bucket.get("win_rate", 0.0) or 0.0) * trade_count
                    )
                    record["process_quality_sum"] += (
                        float(bucket.get("avg_process_quality", 0.0) or 0.0) * trade_count
                    )

        results: dict[str, list] = {}
        for bot_id, hours in aggregated.items():
            buckets: list[HourlyBucket] = []
            for hour, record in sorted(hours.items()):
                trade_count = int(record["trade_count"])
                if trade_count <= 0:
                    continue
                buckets.append(
                    HourlyBucket(
                        hour=hour,
                        trade_count=trade_count,
                        pnl=round(record["pnl"], 4),
                        win_rate=round(record["win_count"] / trade_count, 4),
                        avg_process_quality=round(
                            record["process_quality_sum"] / trade_count, 4,
                        ),
                    )
                )
            if buckets:
                results[bot_id] = buckets
        return results

    def _build_bot_correlation_summaries(self, bot_summaries: dict[str, object]) -> list:
        """Compute bot-level weekly correlation summaries from daily PnL series."""
        from schemas.weekly_metrics import CorrelationSummary

        bot_ids = sorted(bot_summaries)
        all_dates: set[str] = set()
        daily_pnl_by_bot: dict[str, dict[str, float]] = {}
        for bot_id, summary in bot_summaries.items():
            daily = getattr(summary, "daily_pnl", {}) or {}
            if not isinstance(daily, dict):
                daily = {}
            daily_pnl_by_bot[bot_id] = {str(k): float(v) for k, v in daily.items()}
            all_dates.update(daily.keys())

        if not all_dates:
            return []

        ordered_dates = sorted(all_dates)
        summaries: list[CorrelationSummary] = []
        for idx, bot_a in enumerate(bot_ids):
            for bot_b in bot_ids[idx + 1:]:
                series_a = [daily_pnl_by_bot.get(bot_a, {}).get(date, 0.0) for date in ordered_dates]
                series_b = [daily_pnl_by_bot.get(bot_b, {}).get(date, 0.0) for date in ordered_dates]
                active_pairs = [
                    (left, right)
                    for left, right in zip(series_a, series_b)
                    if left != 0.0 and right != 0.0
                ]
                same_direction = 0.0
                if active_pairs:
                    same_direction = (
                        sum(1 for left, right in active_pairs if left * right > 0)
                        / len(active_pairs)
                    )
                corr = self._pearson(series_a, series_b)
                summaries.append(
                    CorrelationSummary(
                        bot_a=bot_a,
                        bot_b=bot_b,
                        rolling_30d_correlation=round(corr, 4),
                        weekly_pnl_correlation=round(corr, 4),
                        same_direction_pct=round(same_direction, 4),
                    )
                )
        return summaries

    def _aggregate_drawdown_data(
        self, week_end: str, trades_by_bot: dict[str, list],
    ) -> dict[str, dict]:
        """Compute drawdown concentration inputs for the strategy engine."""
        from skills.drawdown_analyzer import DrawdownAnalyzer

        results: dict[str, dict] = {}
        for bot_id, trades in trades_by_bot.items():
            if not trades:
                continue
            report = DrawdownAnalyzer(bot_id=bot_id, date=week_end).compute(trades)
            loss_pcts = [abs(float(t.pnl_pct)) for t in trades if getattr(t, "pnl", 0.0) < 0]
            results[bot_id] = {
                "largest_single_loss_pct": round(report.largest_single_loss_pct, 4),
                "max_drawdown_pct": round(report.max_drawdown_pct, 4),
                "avg_loss_pct": round(
                    (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.0,
                    4,
                ),
            }
        return results

    def _load_filter_interactions_from_simulations(self, simulation_results: dict) -> dict[str, list]:
        """Pull filter interaction evidence from simulator outputs when present."""
        interactions: dict[str, list] = {}
        for key, value in simulation_results.items():
            if not key.startswith("filter_interaction_") or not isinstance(value, dict):
                continue
            bot_id = str(value.get("bot_id", "") or key.removeprefix("filter_interaction_"))
            pairs = value.get("pairs", [])
            if bot_id and isinstance(pairs, list) and pairs:
                interactions[bot_id] = pairs
        return interactions

    def _aggregate_orderbook_stats(
        self, week_start: str, week_end: str, bot_ids: list[str],
    ) -> dict[str, dict]:
        """Aggregate orderbook context stats across the current week."""
        aggregated: dict[str, dict[str, dict[str, float]]] = {}
        for date_str in self._iter_date_range(week_start, week_end):
            for bot_id in bot_ids:
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / "orderbook_stats.json"
                )
                if not isinstance(data, dict):
                    continue
                by_context = data.get("by_context", {})
                if not isinstance(by_context, dict):
                    continue
                for context_name, context_data in by_context.items():
                    if not isinstance(context_data, dict):
                        continue
                    count = int(context_data.get("count", 0) or 0)
                    if count <= 0:
                        continue
                    record = aggregated.setdefault(bot_id, {}).setdefault(
                        context_name,
                        {
                            "count": 0.0,
                            "spread_sum": 0.0,
                            "imbalance_sum": 0.0,
                        },
                    )
                    record["count"] += count
                    record["spread_sum"] += (
                        float(context_data.get("spread_stats", {}).get("mean", 0.0) or 0.0)
                        * count
                    )
                    record["imbalance_sum"] += (
                        float(context_data.get("imbalance_stats", {}).get("mean", 0.0) or 0.0)
                        * count
                    )

        results: dict[str, dict] = {}
        for bot_id, contexts in aggregated.items():
            by_context: dict[str, dict] = {}
            for context_name, record in contexts.items():
                count = int(record["count"])
                if count <= 0:
                    continue
                by_context[context_name] = {
                    "count": count,
                    "spread_stats": {"mean": round(record["spread_sum"] / count, 4)},
                    "imbalance_stats": {"mean": round(record["imbalance_sum"] / count, 4)},
                }
            if by_context:
                results[bot_id] = {"by_context": by_context}
        return results

    def _aggregate_curated_file(
        self,
        filename: str,
        week_start: str,
        week_end: str,
        bot_ids: list[str],
    ) -> dict[str, dict] | None:
        """Load a named curated JSON file per bot, using most recent day's data."""
        result: dict[str, dict] = {}
        for bot_id in bot_ids:
            # Walk backwards through the week to find most recent data
            for date_str in reversed(self._iter_date_range(week_start, week_end)):
                data = self._load_json_file(
                    self._curated_dir / date_str / bot_id / filename
                )
                if isinstance(data, dict) and data.get("coverage", 0) > 0:
                    result[bot_id] = data
                    break
        return result or None

    def _load_recent_daily_metric_series(
        self, bot_id: str, end_date: str, metric_key: str, max_points: int,
    ) -> list[float]:
        """Load recent metric values from daily summaries for rolling windows."""
        values: list[float] = []
        for date_str in self._list_available_dates(end=end_date):
            summary = self._load_json_file(self._curated_dir / date_str / bot_id / "summary.json")
            if not isinstance(summary, dict):
                continue
            value = summary.get(metric_key)
            if not isinstance(value, (int, float)):
                continue
            values.append(float(value))
        return values[-max_points:]

    def _load_recent_signal_correlation(
        self, bot_id: str, end_date: str, lookback_days: int,
    ) -> float | None:
        """Load weighted average signal correlation over a rolling lookback."""
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=lookback_days - 1)
        weighted_total = 0.0
        total_weight = 0.0

        for date_str in self._iter_date_range(
            start_dt.strftime("%Y-%m-%d"), end_date,
        ):
            data = self._load_json_file(
                self._curated_dir / date_str / bot_id / "signal_health.json"
            )
            if not isinstance(data, dict):
                continue
            components = data.get("components", [])
            if not isinstance(components, list):
                continue
            for component in components:
                if not isinstance(component, dict):
                    continue
                correlation = float(component.get("win_correlation", 0.0) or 0.0)
                weight = max(int(component.get("trade_count", 0) or 0), 1)
                weighted_total += correlation * weight
                total_weight += weight

        if total_weight <= 0:
            return None
        return weighted_total / total_weight

    @staticmethod
    def _compute_sharpe(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        try:
            mean = statistics.mean(values)
            std = statistics.stdev(values)
        except statistics.StatisticsError:
            return 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)

    @staticmethod
    def _pearson(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or len(left) < 2:
            return 0.0
        try:
            mean_left = statistics.mean(left)
            mean_right = statistics.mean(right)
        except statistics.StatisticsError:
            return 0.0
        numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
        denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
        denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
        if denom_left == 0 or denom_right == 0:
            return 0.0
        return numerator / (denom_left * denom_right)

    def _list_available_dates(self, start: str = "", end: str = "") -> list[str]:
        """List curated date directories in lexical date order."""
        if not self._curated_dir.exists():
            return []
        dates: list[str] = []
        for entry in sorted(self._curated_dir.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if len(name) != 10 or name[4] != "-" or name[7] != "-":
                continue
            if start and name < start:
                continue
            if end and name > end:
                continue
            dates.append(name)
        return dates

    def _iter_date_range(self, start: str, end: str) -> list[str]:
        """Return inclusive YYYY-MM-DD date strings for a range."""
        current = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        dates: list[str] = []
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    @staticmethod
    def _merge_daily_snapshots(snapshots: list[dict]) -> dict:
        """Merge per-strategy daily snapshots into one combined snapshot."""
        if not snapshots:
            return {}
        if len(snapshots) == 1:
            return snapshots[0]

        merged = dict(snapshots[-1])  # non-additive account-level fields from latest

        # Additive integers
        for key in ("total_trades", "win_count", "loss_count",
                    "missed_count", "missed_would_have_won", "error_count"):
            merged[key] = sum(int(s.get(key, 0) or 0) for s in snapshots)

        # Additive floats
        for key in ("gross_pnl", "net_pnl"):
            merged[key] = sum(float(s.get(key, 0.0) or 0.0) for s in snapshots)

        # Count-weighted averages
        total_wins = merged["win_count"]
        total_losses = merged["loss_count"]
        total_trades = merged["total_trades"]

        if total_wins > 0:
            merged["avg_win"] = sum(
                int(s.get("win_count", 0) or 0) * float(s.get("avg_win", 0.0) or 0.0)
                for s in snapshots
            ) / total_wins
        if total_losses > 0:
            merged["avg_loss"] = sum(
                int(s.get("loss_count", 0) or 0) * float(s.get("avg_loss", 0.0) or 0.0)
                for s in snapshots
            ) / total_losses
        if total_trades > 0:
            merged["avg_process_quality"] = sum(
                int(s.get("total_trades", 0) or 0) * float(s.get("avg_process_quality", 0.0) or 0.0)
                for s in snapshots
            ) / total_trades
            merged["win_rate"] = merged["win_count"] / total_trades * 100
        else:
            merged["win_rate"] = 0.0

        # Union per_strategy_summary dicts
        combined_pss: dict = {}
        for s in snapshots:
            pss = s.get("per_strategy_summary", {})
            if isinstance(pss, dict):
                combined_pss.update(pss)
        if combined_pss:
            merged["per_strategy_summary"] = combined_pss

        # Sum root_cause_distribution counts
        combined_rc: dict = {}
        for s in snapshots:
            rc = s.get("root_cause_distribution", {})
            if isinstance(rc, dict):
                for cause, count in rc.items():
                    combined_rc[cause] = combined_rc.get(cause, 0) + int(count or 0)
        if combined_rc:
            merged["root_cause_distribution"] = combined_rc

        return merged

    @staticmethod
    def _load_json_file(path: Path) -> dict | list | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _rebuild_daily_curated_from_raw(self, date: str, bots: list[str] | None = None) -> None:
        """Build canonical daily curated files from assistant-owned raw JSONL."""
        if not self._raw_data_dir.exists():
            return

        from schemas.events import MissedOpportunityEvent, TradeEvent
        from skills.build_daily_metrics import DailyMetricsBuilder

        findings_dir = self._memory_dir / "findings"
        target_bots = bots or self._bots
        for bot_id in target_bots:
            bot_raw = self._raw_data_dir / date / bot_id
            if not bot_raw.exists():
                continue

            trade_records = self._load_raw_json_records(bot_raw, "trade")
            missed_records = self._load_raw_json_records(bot_raw, "missed_opportunity")
            trades = self._validate_raw_models(TradeEvent, trade_records, "trade", bot_id, date)
            missed = self._validate_raw_models(
                MissedOpportunityEvent, missed_records, "missed_opportunity", bot_id, date,
            )

            kwargs: dict = {}
            for event_type, param_name in {
                "filter_decision": "filter_decision_events",
                "indicator_snapshot": "indicator_snapshot_events",
                "orderbook_context": "orderbook_context_events",
                "parameter_change": "parameter_change_events",
                "order": "order_events",
                "process_quality": "process_quality_events",
                "stop_adjustment": "stop_adjustment_events",
                "post_exit": "post_exit_events",
            }.items():
                events = self._load_raw_json_records(bot_raw, event_type)
                if events:
                    kwargs[param_name] = events

            daily_snapshots = self._load_raw_json_records(bot_raw, "daily_snapshot")
            if daily_snapshots:
                kwargs["daily_snapshot"] = self._merge_daily_snapshots(daily_snapshots)

            coordinator_events = self._load_raw_json_records(bot_raw, "coordinator_action")
            if coordinator_events:
                kwargs["coordination_events"] = coordinator_events

            if not trades and not missed and not kwargs:
                continue

            try:
                bot_timezone = "UTC"
                if self._bot_configs and bot_id in self._bot_configs:
                    bot_timezone = getattr(self._bot_configs[bot_id], "timezone", "UTC")
                builder = DailyMetricsBuilder(date=date, bot_id=bot_id, bot_timezone=bot_timezone)
                builder.write_curated(
                    trades=trades,
                    missed=missed,
                    base_dir=self._curated_dir,
                    findings_dir=findings_dir,
                    **kwargs,
                )
            except Exception:
                logger.warning(
                    "Failed to rebuild curated data for %s/%s from raw events",
                    date, bot_id,
                    exc_info=True,
                )

        # ── Portfolio-level curated files ──────────────────────────────
        try:
            from schemas.daily_metrics import BotDailySummary
            from skills.build_daily_metrics import (
                build_concurrent_position_analysis,
                build_family_snapshots,
                build_macro_regime_analysis,
                build_portfolio_rules_summary,
                build_sector_exposure,
            )
            from skills.compute_portfolio_risk import PortfolioRiskComputer
            from skills.portfolio_metrics_tracker import PortfolioMetricsTracker

            # Collect all trade records, portfolio_rule events, and daily snapshots across bots
            all_trade_records: list[dict] = []
            all_rule_events: list[dict] = []
            all_daily_snapshots: list[dict] = []
            for bot_id in target_bots:
                bot_raw = self._raw_data_dir / date / bot_id
                if not bot_raw.exists():
                    continue
                all_trade_records.extend(self._load_raw_json_records(bot_raw, "trade"))
                all_rule_events.extend(self._load_raw_json_records(bot_raw, "portfolio_rule_check"))
                all_daily_snapshots.extend(self._load_raw_json_records(bot_raw, "daily_snapshot"))

            # Load BotDailySummary from just-written per-bot summary.json files
            bot_summaries: list[BotDailySummary] = []
            for bot_id in target_bots:
                summary_path = self._curated_dir / date / bot_id / "summary.json"
                if summary_path.exists():
                    try:
                        raw = json.loads(summary_path.read_text(encoding="utf-8"))
                        bot_summaries.append(BotDailySummary.model_validate(raw))
                    except Exception:
                        logger.warning("Failed to load summary for %s/%s", date, bot_id)

            if all_trade_records or all_rule_events or bot_summaries:
                portfolio_dir = self._curated_dir / date / "portfolio"
                portfolio_dir.mkdir(parents=True, exist_ok=True)

                # 1. rule_blocks_summary.json
                rules_summary = build_portfolio_rules_summary(all_rule_events)
                (portfolio_dir / "rule_blocks_summary.json").write_text(
                    json.dumps(rules_summary, indent=2, default=str), encoding="utf-8",
                )

                # 2. family_snapshots.json (must be BEFORE portfolio_rolling_metrics)
                family_snaps = build_family_snapshots(bot_summaries, self._strategy_registry)
                (portfolio_dir / "family_snapshots.json").write_text(
                    json.dumps(family_snaps, indent=2, default=str), encoding="utf-8",
                )

                # 3. concurrent_position_analysis.json
                concurrent = build_concurrent_position_analysis(all_trade_records)
                (portfolio_dir / "concurrent_position_analysis.json").write_text(
                    json.dumps(concurrent, indent=2, default=str), encoding="utf-8",
                )

                # 4. sector_exposure.json
                sector_exp = build_sector_exposure(all_trade_records)
                (portfolio_dir / "sector_exposure.json").write_text(
                    json.dumps(sector_exp, indent=2, default=str), encoding="utf-8",
                )

                # 4b. macro_regime_analysis.json
                # Unwrap payload if snapshots have event wrapper
                unwrapped_snapshots = [
                    s.get("payload", s) for s in all_daily_snapshots
                ]
                macro_regime = build_macro_regime_analysis(unwrapped_snapshots, date)
                if macro_regime:
                    (portfolio_dir / "macro_regime_analysis.json").write_text(
                        json.dumps(macro_regime, indent=2, default=str), encoding="utf-8",
                    )

                # 5. portfolio_rolling_metrics.json (reads family_snapshots.json)
                try:
                    tracker = PortfolioMetricsTracker(self._curated_dir)
                    rolling = tracker.compute(date)
                    (portfolio_dir / "portfolio_rolling_metrics.json").write_text(
                        json.dumps(rolling.model_dump(mode="json"), indent=2, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    logger.warning("Failed to compute portfolio rolling metrics for %s", date, exc_info=True)

                # 6. portfolio_risk_card.json
                try:
                    position_details: dict[str, list[dict]] = {}
                    for evt in all_trade_records:
                        payload = evt.get("payload", evt)
                        bid = payload.get("bot_id", "")
                        if bid:
                            position_details.setdefault(bid, []).append({
                                "symbol": payload.get("pair", ""),
                                "direction": payload.get("side", "LONG"),
                                "exposure_pct": payload.get("exposure_pct", 0.0),
                            })

                    historical_pnl: dict[str, list[float]] = {}
                    for hbot_id in target_bots:
                        pnls: list[float] = []
                        for d in range(20):
                            past = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=d)).strftime("%Y-%m-%d")
                            sp = self._curated_dir / past / hbot_id / "summary.json"
                            if sp.exists():
                                try:
                                    pnls.append(json.loads(sp.read_text(encoding="utf-8")).get("net_pnl", 0.0))
                                except (json.JSONDecodeError, OSError):
                                    pass
                        if pnls:
                            pnls.reverse()
                            historical_pnl[hbot_id] = pnls

                    sector_map: dict[str, str] = {}
                    for evt in all_trade_records:
                        payload = evt.get("payload", evt)
                        sym = payload.get("pair", "")
                        sec = payload.get("sector", "")
                        if sym and sec:
                            sector_map[sym] = sec

                    computer = PortfolioRiskComputer(
                        date=date,
                        bot_summaries=bot_summaries,
                        position_details=position_details,
                        historical_pnl=historical_pnl,
                        sector_map=sector_map,
                    )
                    risk_card = computer.compute()
                    (self._curated_dir / date / "portfolio_risk_card.json").write_text(
                        json.dumps(risk_card.model_dump(mode="json"), indent=2, default=str),
                        encoding="utf-8",
                    )
                except Exception:
                    logger.warning("Failed to compute portfolio risk card for %s", date, exc_info=True)
        except Exception:
            logger.warning("Failed to rebuild portfolio curated files for %s", date, exc_info=True)

    def _build_enriched_curated(self, date: str, bots: list[str] | None = None) -> None:
        """Compatibility wrapper for the old enriched-curated rebuild entrypoint."""
        self._rebuild_daily_curated_from_raw(date, bots)

    def _load_raw_json_records(self, bot_raw: Path, event_type: str) -> list[dict]:
        """Load JSON/JSONL raw event records, skipping malformed lines."""
        records: list[dict] = []
        candidate_paths = [
            bot_raw / f"{event_type}.jsonl",
            bot_raw / f"{event_type}.json",
        ]
        for path in candidate_paths:
            if not path.exists():
                continue
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    records.append(data)
                elif isinstance(data, list):
                    records.extend(d for d in data if isinstance(d, dict))
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    records.append(data)
        return records

    def _validate_raw_models(
        self,
        model_cls: type[BaseModel],
        records: list[dict],
        event_type: str,
        bot_id: str,
        date: str,
    ) -> list[BaseModel]:
        validated: list[BaseModel] = []
        for record in records:
            try:
                validated.append(model_cls.model_validate(record))
            except Exception:
                logger.warning(
                    "Skipping malformed raw %s event for %s on %s",
                    event_type, bot_id, date,
                )
        return validated

    def _write_run_report(
        self,
        run_id: str,
        report_name: str,
        content: str,
        mirror_response: bool = False,
    ) -> None:
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / report_name).write_text(content, encoding="utf-8")
        response_path = run_dir / "response.md"
        if mirror_response or not response_path.exists():
            response_path.write_text(content, encoding="utf-8")

    def _get_latest_heartbeat_time(self, bot_id: str) -> datetime | None:
        """Get the latest heartbeat timestamp for a bot from heartbeat files."""
        hb_file = self._heartbeat_dir / f"{bot_id}.heartbeat"
        if hb_file.exists():
            try:
                raw = hb_file.read_text(encoding="utf-8").strip()
                if raw:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except (ValueError, OSError):
                pass

        hb_file = self._heartbeat_dir / f"{bot_id}.json"
        if not hb_file.exists():
            return None
        try:
            data = json.loads(hb_file.read_text(encoding="utf-8"))
            ts = data.get("last_seen") or data.get("timestamp")
            if ts:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (json.JSONDecodeError, ValueError, OSError):
            pass
        return None

    def _count_daily_trades(self, date: str) -> int:
        """Count total trades across all bots for a given date."""
        count = 0
        for bot_id in self._bots:
            trades_file = self._curated_dir / date / bot_id / "trades.jsonl"
            if trades_file.exists():
                try:
                    count += sum(
                        1 for line in trades_file.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                except OSError:
                    pass
        return count

    def _load_trades_for_wfo(
        self, bot_id: str, date_start: str = "", date_end: str = "",
    ) -> tuple:
        """Load trade and missed opportunity events for WFO from curated data.

        Scans curated_dir/YYYY-MM-DD/bot_id/ for trades.jsonl and missed.jsonl.
        If date_start/date_end are provided, only directories within that range are loaded.
        Returns (trades, missed) tuple of lists.
        """
        from schemas.events import TradeEvent, MissedOpportunityEvent

        trades: list[TradeEvent] = []
        missed: list[MissedOpportunityEvent] = []

        if not self._curated_dir.exists():
            return ([], [])

        for date_dir in sorted(self._curated_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            dir_name = date_dir.name
            if date_start and dir_name < date_start:
                continue
            if date_end and dir_name > date_end:
                continue
            bot_dir = date_dir / bot_id
            if not bot_dir.is_dir():
                continue

            trades_file = bot_dir / "trades.jsonl"
            if trades_file.exists():
                for line in trades_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            trades.append(TradeEvent(**json.loads(line)))
                        except Exception:
                            logger.warning("Bad trade record in %s", trades_file)

            missed_file = bot_dir / "missed.jsonl"
            if missed_file.exists():
                for line in missed_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            missed.append(MissedOpportunityEvent(**json.loads(line)))
                        except Exception:
                            logger.warning("Bad missed opp record in %s", missed_file)

        logger.info("Loaded %d trades, %d missed for WFO bot %s", len(trades), len(missed), bot_id)
        return (trades, missed)

    async def _check_deployments(self) -> None:
        """Periodic deployment monitoring check."""
        if not self._deployment_monitor:
            return
        from schemas.deployment_monitoring import DeploymentStatus

        for deployment in self._deployment_monitor.get_monitoring():
            try:
                if deployment.status == DeploymentStatus.PENDING_MERGE:
                    merged = await self._deployment_monitor.check_merge_status(
                        deployment.deployment_id,
                    )
                    if (
                        merged
                        and self._suggestion_tracker is not None
                        and deployment.suggestion_id
                    ):
                        self._suggestion_tracker.mark_merged(
                            deployment.suggestion_id,
                            pr_url=deployment.pr_url,
                            deployment_id=deployment.deployment_id,
                        )
                    # Also check for stale pending
                    self._deployment_monitor.check_stale_pending(deployment.deployment_id)
                elif deployment.status == DeploymentStatus.MERGED:
                    # Record pre-deploy metrics if not yet captured
                    if deployment.pre_deploy_metrics is None:
                        snapshot = self._deployment_monitor.collect_metrics_snapshot(deployment.bot_id)
                        if snapshot:
                            self._deployment_monitor.record_pre_deploy_metrics(
                                deployment.deployment_id, snapshot,
                            )
                    # Require heartbeat confirmation before marking DEPLOYED
                    latest_hb = self._get_latest_heartbeat_time(deployment.bot_id)
                    if self._deployment_monitor.is_heartbeat_confirmed(
                        deployment.deployment_id, latest_hb,
                    ):
                        self._deployment_monitor.mark_deployed(
                            deployment.deployment_id,
                            detected_at=latest_hb,
                        )
                        if (
                            self._suggestion_tracker is not None
                            and deployment.suggestion_id
                        ):
                            self._suggestion_tracker.mark_deployed(
                                deployment.suggestion_id,
                                deployment_id=deployment.deployment_id,
                            )
                        logger.info(
                            "Deployment %s confirmed by heartbeat, monitoring started",
                            deployment.deployment_id,
                        )
                    elif self._deployment_monitor.check_merged_timeout(deployment.deployment_id):
                        logger.warning(
                            "Deployment %s timed out waiting for heartbeat — STALE",
                            deployment.deployment_id,
                        )
                        await self._notify(
                            notification_type="alert",
                            priority=NotificationPriority.HIGH,
                            title=f"Deployment Stale — {deployment.bot_id}",
                            body=(
                                f"Deployment {deployment.deployment_id} merged but no heartbeat "
                                f"received within 6 hours. Check bot status."
                            ),
                        )
                elif deployment.status == DeploymentStatus.DEPLOYED:
                    snapshot = self._deployment_monitor.collect_metrics_snapshot(deployment.bot_id)
                    if snapshot:
                        self._deployment_monitor.record_post_deploy_metrics(
                            deployment.deployment_id, snapshot,
                        )
                    if self._deployment_monitor.check_monitoring_window_expired(deployment.deployment_id):
                        logger.info(
                            "Deployment %s monitoring complete — no regression",
                            deployment.deployment_id,
                        )
                    elif self._deployment_monitor.check_regression(deployment.deployment_id):
                        logger.warning(
                            "Regression detected for deployment %s",
                            deployment.deployment_id,
                        )
                        result = await self._deployment_monitor.create_rollback_pr(
                            deployment.deployment_id,
                        )
                        record = self._deployment_monitor.get_by_id(deployment.deployment_id)
                        await self._notify(
                            notification_type="alert",
                            priority=NotificationPriority.CRITICAL,
                            title=f"Regression Detected — {deployment.bot_id}",
                            body=(
                                f"Deployment {deployment.deployment_id} shows regression.\n"
                                f"Details: {record.regression_details if record else 'unknown'}\n"
                                f"Rollback PR: {result.pr_url if result and result.success else 'failed'}"
                            ),
                        )
            except Exception:
                logger.exception("Deployment check failed for %s", deployment.deployment_id)
