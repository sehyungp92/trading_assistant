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
                    body=result.response[:2000],
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

            # Run strategy engine
            from analysis.strategy_engine import StrategyEngine

            engine = StrategyEngine(week_start=week_start, week_end=week_end)
            refinement_report = engine.build_report(portfolio_summary.bot_summaries)

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

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "weekly_analysis",
            })

            # Invoke Claude
            result = await self._agent_runner.invoke(
                agent_type="weekly_analysis",
                prompt_package=package,
                run_id=run_id,
            )

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
                    body=result.response[:2000],
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

        await self._notify(
            notification_type="feedback_received",
            priority=NotificationPriority.LOW,
            title="Feedback recorded",
            body=f"Correction type: {correction.correction_type.value}",
        )

    # --- Private helpers ---

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
        """Run simulation skills based on strategy engine findings.

        Runs FilterSensitivityAnalyzer and CounterfactualSimulator for bots
        with detected issues. Runs ExitStrategySimulator when exit timing
        issues are flagged.
        """
        results: dict = {}

        try:
            from skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer
            from skills.counterfactual_simulator import CounterfactualSimulator
            from skills.exit_strategy_simulator import ExitStrategySimulator
            from schemas.exit_simulation import ExitStrategyConfig, ExitStrategyType

            suggestions = getattr(refinement_report, "suggestions", [])
            counterfactual = CounterfactualSimulator()
            exit_sim = ExitStrategySimulator()

            # Collect bots that have issues
            bots_with_filter_issues: set[str] = set()
            bots_with_regime_issues: dict[str, str] = {}  # bot_id -> regime
            bots_with_exit_issues: set[str] = set()

            for suggestion in suggestions:
                bot_id = getattr(suggestion, "bot_id", None) or ""
                title = getattr(suggestion, "title", "") or ""
                title_lower = title.lower()
                if "filter" in title_lower:
                    bots_with_filter_issues.add(bot_id)
                if "regime" in title_lower:
                    # Extract regime name from suggestion if available
                    regime = getattr(suggestion, "regime", None) or "ranging"
                    bots_with_regime_issues[bot_id] = regime
                if "exit" in title_lower or "stop" in title_lower:
                    bots_with_exit_issues.add(bot_id)

            # Run FilterSensitivity for bots with filter issues
            for bot_id in bots_with_filter_issues:
                if not bot_id:
                    continue
                try:
                    _, missed = self._load_trades_for_week(bot_id, week_start, week_end)
                    if missed:
                        analyzer = FilterSensitivityAnalyzer(bot_id=bot_id, date=week_start)
                        report = analyzer.analyze(missed)
                        results[f"filter_sensitivity_{bot_id}"] = report.model_dump(mode="json")
                except Exception:
                    logger.warning("FilterSensitivity failed for %s", bot_id)

            # Run Counterfactual for bots with regime issues
            for bot_id, regime in bots_with_regime_issues.items():
                if not bot_id:
                    continue
                try:
                    trades, missed = self._load_trades_for_week(bot_id, week_start, week_end)
                    if trades or missed:
                        result = counterfactual.simulate_regime_gate(trades, missed, regime)
                        results[f"counterfactual_{bot_id}"] = result.model_dump(mode="json")
                except Exception:
                    logger.warning("Counterfactual failed for %s", bot_id)

            # Run ExitStrategy for bots with exit issues
            for bot_id in bots_with_exit_issues:
                if not bot_id:
                    continue
                try:
                    trades, _ = self._load_trades_for_week(bot_id, week_start, week_end)
                    if trades:
                        config = ExitStrategyConfig(
                            strategy_type=ExitStrategyType.TRAILING_STOP,
                            params={"trail_pct": 2.0},
                        )
                        result = exit_sim.simulate(trades, config)
                        results[f"exit_simulation_{bot_id}"] = result.model_dump(mode="json")
                except Exception:
                    logger.warning("ExitStrategy failed for %s", bot_id)

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

            # 1. Portfolio allocation (cross-bot)
            allocator = PortfolioAllocator(week_start, week_end)
            n_bots = len(bot_summaries)
            current = {bid: 100.0 / n_bots for bid in bot_summaries} if n_bots > 0 else {}
            alloc_report = allocator.compute(bot_summaries, current)
            results["portfolio_allocation"] = alloc_report.model_dump(mode="json")

            # 2. Synergy analysis (cross-strategy)
            per_strat = {
                bid: s.per_strategy_summary
                for bid, s in bot_summaries.items()
            }
            synergy = SynergyAnalyzer(week_start, week_end)
            synergy_report = synergy.compute(per_strat)
            results["synergy_analysis"] = synergy_report.model_dump(mode="json")

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

            # 5. Regime-conditional metrics
            from analysis.strategy_engine import StrategyEngine as _SE

            engine = _SE(week_start=week_start, week_end=week_end)
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
                # Also check for raw coordination JSONL
                coord_jsonl = self._curated_dir.parent / "data" / "coordination" / f"coordination_{date_str}.jsonl"
                if coord_jsonl.exists():
                    for line in coord_jsonl.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                events.append(CoordinatorAction(**json.loads(line)))
                            except Exception:
                                pass
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
