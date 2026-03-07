"""Handler implementations — wire worker action types to full pipelines.

Each handler implements: data preparation -> prompt assembly -> agent invocation
-> post-processing -> notification dispatch.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

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

# Minimum total trades across all bots to justify a full Claude analysis invocation.
# Days with fewer trades get a deterministic summary instead.
_MIN_TRADES_FOR_ANALYSIS = 3


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
        heartbeat_dir: Path | None = None,
        failure_log_path: Path | None = None,
        worker: Worker | None = None,
        brain: OrchestratorBrain | None = None,
        run_history_path: Path | None = None,
        suggestion_tracker: object | None = None,
        autonomous_pipeline: object | None = None,
        deployment_monitor: object | None = None,
        threshold_learner: object | None = None,
    ) -> None:
        self._agent_runner = agent_runner
        self._event_stream = event_stream
        self._dispatcher = dispatcher
        self._notification_prefs = notification_prefs
        self._curated_dir = Path(curated_dir)
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
        self._deployment_monitor = deployment_monitor
        self._threshold_learner = threshold_learner

    async def handle_daily_analysis(self, action: Action) -> None:
        """Run the daily analysis pipeline: quality gate -> assemble -> invoke -> notify."""
        date = (action.details or {}).get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        run_id = f"daily-{date}"
        start_time = datetime.now(timezone.utc)
        self._record_run(run_id, "daily_analysis", "running", started_at=start_time.isoformat())

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "daily_analysis",
            })

            # Data availability pre-check via MemoryIndex
            index = MemoryConsolidator.load_index(self._runs_dir.parent)
            for bot in self._bots:
                avail = ContextBuilder.check_data_availability(index, bot, date)
                if avail["has_curated"] is False:
                    logger.warning("No curated data for %s on %s — analysis may be incomplete", bot, date)

            # Quality gate
            from analysis.quality_gate import QualityGate

            gate = QualityGate(
                report_id=run_id,
                date=date,
                expected_bots=self._bots,
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

            # Minimum-data threshold: skip Claude if insufficient data
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
                    f"Daily summary for {date}: {total_trades} trade(s) across {len(self._bots)} bot(s). "
                    f"Insufficient data for full analysis (minimum {_MIN_TRADES_FOR_ANALYSIS} trades required). "
                    f"Data completeness: {completeness_str}."
                )
                self._record_run(
                    run_id, "daily_analysis", "skipped",
                    started_at=start_time.isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                )
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

            # Assemble prompt
            from analysis.prompt_assembler import DailyPromptAssembler

            assembler = DailyPromptAssembler(
                date=date,
                bots=self._bots,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
            )
            package = assembler.assemble()

            # Include event counts in prompt metadata
            if event_counts:
                package.metadata["event_counts"] = event_counts

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "daily_analysis",
            })

            # Invoke Claude
            result = await self._agent_runner.invoke(
                agent_type="daily_analysis",
                prompt_package=package,
                run_id=run_id,
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

                # Validate and annotate response
                final_report, validation = self._validate_and_annotate(parsed, date)

                # Fallback: if validation failed but we have suggestions, record them unvalidated
                if validation is None and parsed.suggestions:
                    from analysis.response_validator import ValidationResult
                    validation = ValidationResult(
                        approved_suggestions=parsed.suggestions,
                        approved_predictions=parsed.predictions,
                    )
                    logger.warning("Validation failed for %s — recording unvalidated suggestions", run_id)

                # Record approved agent suggestions to tracker
                agent_suggestion_ids = self._record_agent_suggestions(validation, run_id, parsed)

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

            # Run strategy engine
            from analysis.strategy_engine import StrategyEngine

            engine = StrategyEngine(
                week_start=week_start, week_end=week_end,
                threshold_learner=self._threshold_learner,
            )
            refinement_report = engine.build_report(
                portfolio_summary.bot_summaries,
                signal_health=signal_health_data if signal_health_data else None,
                factor_rolling=factor_rolling_data if factor_rolling_data else None,
            )

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
            )

            # Run autonomous pipeline on recorded suggestions
            await self._run_autonomous_pipeline(suggestion_ids, run_id)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "simulations", "handler": "weekly_analysis",
            })

            # Assemble prompt
            from analysis.weekly_prompt_assembler import WeeklyPromptAssembler

            assembler = WeeklyPromptAssembler(
                week_start=week_start,
                week_end=week_end,
                bots=self._bots,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                runs_dir=self._runs_dir,
            )
            package = assembler.assemble()

            # Inject simulation results into prompt data
            if simulation_results:
                package.data.update({"simulation_results": simulation_results})

            # Inject allocation analysis results
            if allocation_results:
                package.data.update({"allocation_analysis": allocation_results})

            # Inject suggestion ID mapping so Claude can reference them in the report
            if suggestion_ids:
                package.metadata["suggestion_ids"] = suggestion_ids

            # Weekly retrospective — compare last week's predictions to actual outcomes
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

            # Record forecast data and compute meta-analysis
            try:
                from skills.forecast_tracker import ForecastTracker
                from schemas.forecast_tracking import ForecastRecord

                forecast_tracker = ForecastTracker(self._memory_dir / "findings")
                retro_data = package.data.get("weekly_retrospective")
                if retro_data:
                    by_bot = {}
                    by_type: dict[str, list[int]] = {}
                    for pred in retro_data.get("predictions", []):
                        bid = pred.get("bot_id", "")
                        if bid:
                            by_bot.setdefault(bid, []).append(1 if pred.get("correct") else 0)
                        metric = pred.get("metric", "")
                        if metric:
                            by_type.setdefault(metric, []).append(1 if pred.get("correct") else 0)
                    forecast_record = ForecastRecord(
                        week_start=week_start,
                        week_end=week_end,
                        predictions_reviewed=retro_data.get("predictions_reviewed", 0),
                        correct_predictions=retro_data.get("correct_predictions", 0),
                        accuracy=retro_data.get("accuracy", 0.0),
                        by_bot={b: sum(v) / len(v) for b, v in by_bot.items() if v},
                        by_type={m: sum(v) / len(v) for m, v in by_type.items() if v},
                    )
                    forecast_tracker.record_week(forecast_record)
                meta = forecast_tracker.compute_meta_analysis()
                if meta.weeks_analyzed > 0:
                    package.data["forecast_meta_analysis"] = meta.model_dump(mode="json")
            except Exception:
                logger.error("Forecast tracking failed — skipping", exc_info=True)

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

            # Invoke Claude
            result = await self._agent_runner.invoke(
                agent_type="weekly_analysis",
                prompt_package=package,
                run_id=run_id,
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

                # Validate and annotate response
                final_report, validation = self._validate_and_annotate(parsed, week_start)

                # Fallback: if validation failed but we have suggestions, record them unvalidated
                if validation is None and parsed.suggestions:
                    from analysis.response_validator import ValidationResult
                    validation = ValidationResult(
                        approved_suggestions=parsed.suggestions,
                        approved_predictions=parsed.predictions,
                    )
                    logger.warning("Validation failed for %s — recording unvalidated suggestions", run_id)

                # Record approved agent suggestions to tracker
                weekly_agent_ids = self._record_agent_suggestions(validation, run_id, parsed)

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
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        run_id = f"wfo-{bot_id}-{date}"

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
            data_end = details.get("data_end", date)
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

            # Assemble prompt for Claude review
            assembler = WFOPromptAssembler(
                bot_id=bot_id,
                memory_dir=self._memory_dir,
                wfo_output_dir=output_dir,
            )
            package = assembler.assemble()

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "claude_invocation", "handler": "wfo",
            })

            # Invoke Claude
            result = await self._agent_runner.invoke(
                agent_type="wfo",
                prompt_package=package,
                run_id=run_id,
                max_turns=3,
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

        except Exception as exc:
            logger.exception("WFO handler failed for %s", run_id)
            self._event_stream.broadcast("wfo_error", {
                "bot_id": bot_id,
                "error": str(exc),
            })

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

            from schemas.bug_triage import ErrorEvent, TriageOutcome
            from skills.run_bug_triage import TriageRunner
            from skills.triage_context_builder import TriageContextBuilder
            from analysis.triage_prompt_assembler import TriagePromptAssembler

            # Build ErrorEvent from action details
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

            # Run triage pipeline
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

            # Only invoke Claude for actionable outcomes
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
                )

                result = await self._agent_runner.invoke(
                    agent_type="triage",
                    prompt_package=package,
                    run_id=run_id,
                    allowed_tools=["Read", "Bash", "Grep", "Glob"],
                )

                if result.success:
                    await self._notify(
                        notification_type="triage_result",
                        priority=self._severity_to_priority(triage_result.severity),
                        title=f"Triage [{triage_result.severity.value.upper()}] {bot_id}",
                        body=result.response[:2000],
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
                self._suggestion_tracker.implement(correction.target_id)
                self._event_stream.broadcast("suggestion_accepted", {
                    "suggestion_id": correction.target_id,
                })
                # Link hypothesis lifecycle
                self._update_hypothesis_from_feedback(
                    correction.target_id, accepted=True,
                )
            elif correction.correction_type == CorrectionType.SUGGESTION_REJECT:
                self._suggestion_tracker.reject(correction.target_id, text[:200])
                self._event_stream.broadcast("suggestion_rejected", {
                    "suggestion_id": correction.target_id,
                })
                # Link hypothesis lifecycle
                self._update_hypothesis_from_feedback(
                    correction.target_id, accepted=False,
                )

        await self._notify(
            notification_type="feedback_received",
            priority=NotificationPriority.LOW,
            title="Feedback recorded",
            body=f"Correction type: {correction.correction_type.value}",
        )

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

                # Find linked suggestion_id for this bot from suggestion_ids mapping
                linked_sid = ""
                if suggestion_ids:
                    for sid, stitle in suggestion_ids.items():
                        # Link if the suggestion title is similar to the proposal
                        if stitle and proposal.title and stitle.lower() in proposal.title.lower():
                            linked_sid = sid
                            break

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

    def _validate_and_annotate(self, parsed, date_or_week: str):
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

            validator = ResponseValidator(
                rejected_suggestions=rejected,
                forecast_meta=forecast_meta,
                category_scorecard=scorecard,
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
                    }
                    for b in validation.blocked_suggestions
                ]
                entry = {
                    "date": date_or_week,
                    "approved_count": len(validation.approved_suggestions),
                    "blocked_count": len(validation.blocked_suggestions),
                    "blocked_details": blocked_details,
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

    def _record_suggestions(
        self, suggestions: list, run_id: str,
    ) -> dict[str, str]:
        """Convert StrategySuggestions to SuggestionRecords and persist via tracker.

        Returns a mapping of suggestion_id → title for metadata injection.
        """
        if not self._suggestion_tracker or not suggestions:
            return {}

        import hashlib
        from schemas.suggestion_tracking import SuggestionRecord

        id_map: dict[str, str] = {}
        for idx, suggestion in enumerate(suggestions):
            title = getattr(suggestion, "title", "") or ""
            bot_id = getattr(suggestion, "bot_id", "") or ""
            tier = getattr(suggestion, "tier", "parameter")
            description = getattr(suggestion, "description", "") or ""

            # Deterministic ID: SHA256(run_id + index + title)[:12]
            raw = f"{run_id}:{idx}:{title}"
            suggestion_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

            category_str = str(getattr(suggestion, "category", "") or "")
            confidence = float(getattr(suggestion, "confidence", 0.0) or 0.0)
            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                title=title,
                tier=str(tier.value) if hasattr(tier, "value") else str(tier),
                category=category_str,
                source_report_id=run_id,
                description=description,
                confidence=confidence,
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
    ) -> dict[str, str]:
        """Record Claude's approved suggestions from validation into SuggestionTracker.

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

        # Build hypothesis_id map from structural proposals
        hypothesis_map: dict[str, str] = {}
        if parsed and parsed.structural_proposals:
            for proposal in parsed.structural_proposals:
                if proposal.hypothesis_id and proposal.bot_id:
                    hypothesis_map[proposal.bot_id] = proposal.hypothesis_id

        id_map: dict[str, str] = {}
        for idx, suggestion in enumerate(validation_result.approved_suggestions):
            title = suggestion.title or ""
            bot_id = suggestion.bot_id or ""
            category = suggestion.category or "parameter"

            # Deterministic ID: SHA256(run_id + index + title)[:12]
            raw = f"{run_id}:agent:{idx}:{title}"
            suggestion_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

            # Map category to tier using shared mapping
            from schemas.agent_response import CATEGORY_TO_TIER

            tier = CATEGORY_TO_TIER.get(category, "parameter")

            confidence = float(getattr(suggestion, "confidence", 0.0) or 0.0)
            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                title=title,
                tier=tier,
                category=category,
                source_report_id=run_id,
                description=suggestion.evidence_summary or "",
                confidence=confidence,
                hypothesis_id=hypothesis_map.get(bot_id),
            )

            recorded = self._suggestion_tracker.record(record)
            if recorded is not False:
                id_map[suggestion_id] = title
                logger.info("Recorded agent suggestion %s: %s", suggestion_id, title)

        if id_map:
            self._event_stream.broadcast("agent_suggestions_recorded", {
                "run_id": run_id, "count": len(id_map),
            })

        return id_map

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
            await self._dispatcher.dispatch(payload, self._notification_prefs, hour_utc)
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
                    await self._deployment_monitor.check_merge_status(deployment.deployment_id)
                elif deployment.status == DeploymentStatus.MERGED:
                    snapshot = self._deployment_monitor.collect_metrics_snapshot(deployment.bot_id)
                    if snapshot:
                        self._deployment_monitor.record_pre_deploy_metrics(
                            deployment.deployment_id, snapshot,
                        )
                    self._deployment_monitor.mark_deployed(deployment.deployment_id)
                    logger.info(
                        "Deployment %s marked as DEPLOYED, monitoring started",
                        deployment.deployment_id,
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
