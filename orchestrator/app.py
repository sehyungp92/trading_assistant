"""Orchestrator FastAPI entry point — wires all components together.

Run with: uvicorn orchestrator.app:app --reload
For production, use create_app() factory to configure paths.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

from comms.dispatcher import NotificationDispatcher
from orchestrator.adapters.vps_receiver import VPSReceiver
from orchestrator.agent_runner import AgentRunner
from orchestrator.latency_tracker import LatencyTracker
from orchestrator.config import AppConfig
from orchestrator.conversation_tracker import ConversationTracker
from orchestrator.db.queue import EventQueue
from orchestrator.event_stream import AuditTrailConsumer, EventStream
from orchestrator.handlers import Handlers
from orchestrator.memory_consolidator import MemoryConsolidator
from orchestrator.monitoring import MonitoringCheck, MonitoringLoop
from orchestrator.orchestrator_brain import OrchestratorBrain
from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs
from orchestrator.session_store import SessionStore
from orchestrator.skills_registry import SkillsRegistry
from orchestrator.subagent import SubagentManager
from orchestrator.task_registry import TaskRegistry
from orchestrator.worker import Worker
from schemas.notifications import NotificationChannel, NotificationPreferences

logger = logging.getLogger(__name__)


def _load_notification_prefs(prefs_path: Path) -> NotificationPreferences:
    """Load notification preferences from disk, or return defaults."""
    if prefs_path.exists():
        try:
            return NotificationPreferences(**json.loads(prefs_path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Could not load notification prefs from %s, using defaults", prefs_path)
    return NotificationPreferences()


def _save_notification_prefs(prefs: NotificationPreferences, prefs_path: Path) -> None:
    """Persist notification preferences to disk."""
    prefs_path.parent.mkdir(parents=True, exist_ok=True)
    prefs_path.write_text(prefs.model_dump_json(indent=2), encoding="utf-8")


def _register_channel_adapters(
    config: AppConfig,
    dispatcher: NotificationDispatcher,
) -> list:
    """Register communication channel adapters based on config. Returns adapters for lifecycle."""
    adapters = []

    if config.telegram_bot_token:
        from comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig

        telegram = TelegramBotAdapter(config=TelegramBotConfig(
            token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
        ))
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, telegram)
        adapters.append(telegram)
        logger.info("Telegram adapter registered (chat_id=%s)", config.telegram_chat_id)

    if config.discord_bot_token:
        from comms.discord_bot import DiscordBotAdapter, DiscordBotConfig

        discord_adapter = DiscordBotAdapter(config=DiscordBotConfig(
            token=config.discord_bot_token,
            channel_id=int(config.discord_channel_id) if config.discord_channel_id else 0,
        ))
        dispatcher.register_adapter(NotificationChannel.DISCORD, discord_adapter)
        adapters.append(discord_adapter)
        logger.info("Discord adapter registered (channel_id=%s)", config.discord_channel_id)

    if config.smtp_host and config.smtp_user:
        from comms.email_adapter import EmailAdapter, EmailConfig

        email = EmailAdapter(config=EmailConfig(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            username=config.smtp_user,
            password=config.smtp_pass,
            from_address=config.email_from,
        ))
        dispatcher.register_adapter(NotificationChannel.EMAIL, email)
        adapters.append(email)
        logger.info("Email adapter registered (smtp=%s)", config.smtp_host)

    return adapters


async def _expire_approvals_with_notification(
    approval_tracker,
    telegram_bot,
    dispatcher,
    notification_prefs,
) -> None:
    """Expire old approvals and send notification for each expired request."""
    expired_ids = approval_tracker.expire_old(max_age_days=7)
    if not expired_ids and telegram_bot is None:
        return
    for rid in expired_ids:
        logger.info("Approval request %s expired", rid)
        if telegram_bot is not None:
            try:
                req = approval_tracker.get_by_id(rid)
                params = ", ".join(
                    pc.get("param_name", "?") for pc in (req.param_changes if req else [])
                )
                text = (
                    f"\u23f0 Approval Expired: {rid}\n"
                    f"Bot: {req.bot_id if req else '?'}\n"
                    f"Params: {params}\n"
                    f"Request was pending for >7 days."
                )
                await telegram_bot.send_message(text)
                # Also edit the original card if message_id exists
                if req and req.message_id:
                    try:
                        await telegram_bot.edit_message(
                            req.message_id,
                            f"Suggestion {rid}\nBot: {req.bot_id}\n\u23f0 EXPIRED — pending >7 days",
                        )
                    except Exception:
                        pass
            except Exception:
                logger.warning("Failed to send expiry notification for %s", rid)


def create_app(db_dir: str | None = None, config: AppConfig | None = None) -> FastAPI:
    """Factory function. Tests inject a temp directory for DB files."""
    if config is None:
        config = AppConfig.from_env()
    if db_dir is None:
        db_dir = config.data_dir

    db_path = Path(db_dir)
    queue = EventQueue(db_path=str(db_path / "events.db"))
    registry = TaskRegistry(db_path=str(db_path / "tasks.db"))
    brain = OrchestratorBrain()
    event_stream = EventStream()
    conversation_tracker = ConversationTracker()
    worker = Worker(
        queue=queue, registry=registry, brain=brain,
        event_stream=event_stream, conversation_tracker=conversation_tracker,
    )
    session_store = SessionStore(base_dir=str(db_path / ".assistant" / "sessions"))
    subagent_mgr = SubagentManager()

    # Latency tracker — shared across VPSReceiver, monitoring, and /metrics
    latency_tracker = LatencyTracker()

    # Integration layer components
    dispatcher = NotificationDispatcher()
    skills_registry = SkillsRegistry()
    agent_runner = AgentRunner(
        runs_dir=db_path / "runs",
        session_store=session_store,
        skills_registry=skills_registry,
    )
    monitoring_loop = MonitoringLoop(
        checks=[MonitoringCheck(
            registry=registry,
            heartbeat_dir=str(db_path / "heartbeats"),
            queue=queue,
            brain=brain,
            heartbeat_md_path=str(db_path / "memory" / "heartbeat.md"),
            relay_url=config.relay_url,
            latency_tracker=latency_tracker,
        )],
        event_stream=event_stream,
    )

    # Register channel adapters from config
    channel_adapters = _register_channel_adapters(config, dispatcher)

    # Extract Telegram adapter reference (needed for autonomous pipeline)
    telegram_adapter = None
    for adapter in channel_adapters:
        from comms.telegram_bot import TelegramBotAdapter as _TBA
        if isinstance(adapter, _TBA):
            telegram_adapter = adapter
            break

    # SuggestionTracker — shared path with ContextBuilder (memory/findings/)
    from skills.suggestion_tracker import SuggestionTracker

    suggestion_tracker = SuggestionTracker(store_dir=db_path / "memory" / "findings")

    # Load persisted notification preferences
    prefs_path = db_path / "data" / "notification_prefs.json"
    notification_prefs = _load_notification_prefs(prefs_path)

    # Autonomous pipeline (feature-flagged)
    autonomous_pipeline = None
    approval_tracker = None
    approval_handler = None
    if config.autonomous_enabled:
        from skills.approval_handler import ApprovalHandler
        from skills.approval_tracker import ApprovalTracker
        from skills.autonomous_pipeline import AutonomousPipeline
        from skills.config_registry import ConfigRegistry as _ConfigRegistry
        from skills.file_change_generator import FileChangeGenerator
        from skills.github_pr import PRBuilder
        from skills.suggestion_backtester import SuggestionBacktester

        config_registry = _ConfigRegistry(Path(config.bot_config_dir))
        backtester = SuggestionBacktester(config_registry, db_path)
        approval_tracker = ApprovalTracker(db_path / "memory" / "findings" / "approvals.jsonl")
        file_change_gen = FileChangeGenerator()
        pr_builder = PRBuilder(dry_run=False)

        # Telegram bot/renderer (may be None if not configured)
        telegram_bot = telegram_adapter  # Extracted from channel_adapters above
        telegram_renderer = None
        if config.telegram_bot_token:
            from comms.telegram_renderer import TelegramRenderer
            telegram_renderer = TelegramRenderer()

        approval_handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=suggestion_tracker,
            file_change_generator=file_change_gen,
            pr_builder=pr_builder,
            config_registry=config_registry,
            event_stream=event_stream,
            telegram_bot=telegram_bot,
        )
        autonomous_pipeline = AutonomousPipeline(
            config_registry=config_registry,
            backtester=backtester,
            approval_tracker=approval_tracker,
            suggestion_tracker=suggestion_tracker,
            telegram_bot=telegram_bot,
            telegram_renderer=telegram_renderer,
            event_stream=event_stream,
        )
        logger.info("Autonomous pipeline enabled")

    # PR review check function (needs approval_tracker + pr_builder from autonomous block)
    async def _check_pr_reviews() -> None:
        """Check PR reviews for approved requests and notify on attention needed."""
        if approval_tracker is None:
            return
        try:
            from skills.github_pr import PRBuilder as _PRB
            _pr_builder = pr_builder if config.autonomous_enabled else _PRB(dry_run=True)
            approved_with_prs = approval_tracker.get_approved_with_prs()
            for req in approved_with_prs:
                if not req.pr_url:
                    continue
                profile = None
                if config.autonomous_enabled:
                    profile = config_registry.get_profile(req.bot_id)
                repo_dir = Path(profile.repo_dir) if profile else db_path
                status = await _pr_builder.check_pr_reviews(req.pr_url, repo_dir)
                if status and status.needs_attention:
                    msg = (
                        f"\U0001f50d PR Review Needs Attention\n"
                        f"PR: {status.pr_url}\n"
                        f"State: {status.review_state.value}\n"
                        f"Reviewers: {', '.join(status.reviewers) or 'none'}\n"
                    )
                    if status.actionable_comments:
                        msg += f"Comments: {len(status.actionable_comments)}\n"
                    if telegram_adapter is not None:
                        try:
                            await telegram_adapter.send_message(msg)
                        except Exception:
                            logger.warning("Failed to send PR review notification")
                    if event_stream:
                        event_stream.broadcast("pr_review_needs_attention", {
                            "request_id": req.request_id,
                            "pr_url": req.pr_url,
                            "review_state": status.review_state.value,
                        })
        except Exception:
            logger.exception("PR review check failed")

    # ThresholdLearner — adaptive threshold learning (feature-flagged)
    threshold_learner = None
    if config.adaptive_thresholds_enabled:
        from skills.threshold_learner import ThresholdLearner

        threshold_learner = ThresholdLearner(findings_dir=db_path / "memory" / "findings")
        logger.info("Adaptive threshold learning enabled")

    # Deployment monitor (feature-flagged)
    deployment_monitor = None
    if config.deployment_monitoring_enabled:
        from skills.deployment_monitor import DeploymentMonitor

        _dm_pr_builder = pr_builder if config.autonomous_enabled else None
        _dm_config_registry = config_registry if config.autonomous_enabled else None
        _dm_file_change_gen = file_change_gen if config.autonomous_enabled else None
        deployment_monitor = DeploymentMonitor(
            findings_dir=db_path / "memory" / "findings",
            curated_dir=db_path / "data" / "curated",
            pr_builder=_dm_pr_builder,
            config_registry=_dm_config_registry,
            event_stream=event_stream,
            file_change_generator=_dm_file_change_gen,
        )
        logger.info("Deployment monitoring enabled")

    handlers = Handlers(
        agent_runner=agent_runner,
        event_stream=event_stream,
        dispatcher=dispatcher,
        notification_prefs=notification_prefs,
        curated_dir=db_path / "data" / "curated",
        memory_dir=db_path / "memory",
        runs_dir=db_path / "runs",
        source_root=db_path,
        bots=config.bot_ids,
        heartbeat_dir=db_path / "heartbeats",
        failure_log_path=db_path / "data" / "failure_log.jsonl",
        worker=worker,
        brain=brain,
        suggestion_tracker=suggestion_tracker,
        autonomous_pipeline=autonomous_pipeline,
        deployment_monitor=deployment_monitor,
        threshold_learner=threshold_learner,
    )

    # Wire handler slots on worker
    worker.on_alert = handlers.handle_alert
    worker.on_heartbeat = handlers.handle_heartbeat
    worker.on_notification = handlers.handle_notification
    # Long-running handlers are spawned via SubagentManager
    worker.on_triage = lambda action: subagent_mgr.spawn(
        "triage", lambda a=action: handlers.handle_triage(a))
    worker.on_daily_analysis = lambda action: subagent_mgr.spawn(
        "daily_analysis", lambda a=action: handlers.handle_daily_analysis(a))
    worker.on_weekly_analysis = lambda action: subagent_mgr.spawn(
        "weekly_analysis", lambda a=action: handlers.handle_weekly_analysis(a))
    worker.on_wfo = lambda action: subagent_mgr.spawn(
        "wfo", lambda a=action: handlers.handle_wfo(a))
    worker.on_feedback = handlers.handle_feedback

    # Register Telegram approval callbacks when autonomous pipeline is enabled
    if approval_handler is not None:
        from comms.telegram_handlers import TelegramCallbackRouter

        # Get or create callback router
        callback_router = TelegramCallbackRouter()

        async def _on_approve_callback(request_id: str) -> str:
            response = await approval_handler.handle_approve(request_id)
            # Auto-create deployment record if PR was created and monitoring is enabled
            if deployment_monitor and "PR created:" in response:
                req = approval_tracker.get_by_id(request_id)
                if req and req.pr_url:
                    try:
                        import hashlib
                        dep_id = hashlib.sha256(
                            f"dep-{request_id}".encode(),
                        ).hexdigest()[:16]
                        deployment_monitor.create_deployment(
                            deployment_id=dep_id,
                            approval_request_id=request_id,
                            pr_url=req.pr_url,
                            bot_id=req.bot_id,
                            param_changes=req.param_changes,
                            pr_number=0,
                        )
                        logger.info(
                            "Created deployment record %s for PR %s",
                            dep_id, req.pr_url,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to create deployment record for %s",
                            request_id,
                        )
            return response

        async def _on_reject_callback(request_id: str) -> str:
            return await approval_handler.handle_reject(request_id)

        async def _on_detail_callback(request_id: str) -> str:
            return await approval_handler.handle_detail(request_id)

        # Register prefix-based callbacks
        callback_router.register("approve_suggestion_", _on_approve_callback)
        callback_router.register("reject_suggestion_", _on_reject_callback)
        callback_router.register("detail_suggestion_", _on_detail_callback)

        # Register /pending command
        async def _pending_command(**kwargs) -> str:
            pending = approval_tracker.get_pending()
            if not pending:
                return "No pending approval requests"
            lines = ["Pending approval requests:"]
            for r in pending:
                lines.append(f"  [{r.request_id}] {r.bot_id}: {', '.join(pc.get('param_name', '?') for pc in r.param_changes)}")
            return "\n".join(lines)

        callback_router.register("cmd_pending", _pending_command)

        # Connect callback router to Telegram adapter for incoming updates
        if telegram_adapter is not None:
            telegram_adapter.set_callback_router(callback_router)

        logger.info("Telegram approval callbacks registered")

    # Relay polling — only active when RELAY_URL is configured
    vps_receiver: VPSReceiver | None = None
    if config.relay_url:
        vps_receiver = VPSReceiver(
            relay_url=config.relay_url,
            local_queue=queue,
            api_key=config.relay_api_key,
            latency_tracker=latency_tracker,
        )

    # ProactiveScanner for morning/evening notifications
    from skills.proactive_scanner import ProactiveScanner

    scanner = ProactiveScanner()
    curated_dir = db_path / "data" / "curated"

    async def _morning_scan() -> None:
        """Gather overnight data and dispatch morning scan notifications."""
        from datetime import timedelta

        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        errors: list[dict] = []
        unusual_losses: list[dict] = []

        # Check dead-letter queue for errors
        try:
            dead_letters = await queue.get_dead_letters(limit=20)
            for dl in dead_letters:
                if isinstance(dl, dict):
                    errors.append(dl)
        except Exception:
            logger.warning("Morning scan: could not read dead-letter queue")

        # Check yesterday's curated data for unusual losses
        for bot_id in config.bot_ids:
            summary_path = curated_dir / yesterday / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    pnl = summary.get("net_pnl", 0)
                    if pnl < 0:
                        unusual_losses.append({
                            "bot_id": bot_id,
                            "date": yesterday,
                            "pnl": pnl,
                            "reason": f"Loss on {yesterday}",
                        })
                except Exception:
                    logger.warning("Morning scan: could not read summary for %s/%s", bot_id, yesterday)

        result = scanner.morning_scan(events=[], errors=errors, unusual_losses=unusual_losses)
        hour_utc = datetime.now(timezone.utc).hour
        for payload in result.payloads:
            try:
                await dispatcher.dispatch(payload, notification_prefs, hour_utc)
            except Exception:
                logger.exception("Morning scan dispatch failed")

    async def _evening_report() -> None:
        """Check if daily report is ready and send evening notification."""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_report_ready = any(
            (curated_dir / date / bot_id / "summary.json").exists()
            for bot_id in config.bot_ids
        ) if config.bot_ids else False

        result = scanner.evening_scan(date=date, daily_report_ready=daily_report_ready)
        hour_utc = datetime.now(timezone.utc).hour
        for payload in result.payloads:
            try:
                await dispatcher.dispatch(payload, notification_prefs, hour_utc)
            except Exception:
                logger.exception("Evening report dispatch failed")

    # AutoOutcomeMeasurer — runs weekly to measure suggestion outcomes
    from skills.auto_outcome_measurer import AutoOutcomeMeasurer
    from schemas.suggestion_tracking import SuggestionStatus

    outcome_measurer = AutoOutcomeMeasurer(curated_dir=curated_dir)

    # HypothesisLibrary — shared for outcome linking
    from skills.hypothesis_library import HypothesisLibrary

    hypothesis_library = HypothesisLibrary(db_path / "memory" / "findings")

    async def _measure_outcomes() -> None:
        """Measure outcomes of implemented suggestions and link to hypotheses."""
        suggestions = suggestion_tracker.load_all()
        implemented = [
            s for s in suggestions
            if s.get("status") == SuggestionStatus.IMPLEMENTED.value
        ]
        # Check which suggestions already have outcomes
        existing_outcomes = suggestion_tracker.load_outcomes()
        measured_ids = {o.get("suggestion_id") for o in existing_outcomes}

        for s in implemented:
            sid = s.get("suggestion_id", "")
            if sid in measured_ids:
                continue
            try:
                result = outcome_measurer.measure(
                    suggestion_id=sid,
                    bot_id=s.get("bot_id", ""),
                    implemented_date=s.get("resolved_at", "")[:10],
                )
                if result:
                    from schemas.suggestion_tracking import SuggestionOutcome
                    outcome = SuggestionOutcome(
                        suggestion_id=sid,
                        implemented_date=s.get("resolved_at", "")[:10],
                        pnl_delta_7d=result.pnl_delta,
                        win_rate_delta_7d=result.win_rate_after - result.win_rate_before,
                        drawdown_delta_7d=result.drawdown_after - result.drawdown_before,
                    )
                    suggestion_tracker.record_outcome(outcome)

                    # Link hypothesis outcome if suggestion has hypothesis_id
                    hyp_id = s.get("hypothesis_id")
                    if hyp_id:
                        try:
                            hypothesis_library.record_outcome(hyp_id, positive=outcome.net_positive_7d)
                        except Exception:
                            logger.warning("Failed to record hypothesis outcome for %s", hyp_id)

                    # Promote linked patterns on positive outcome
                    if outcome.net_positive_7d:
                        try:
                            from skills.pattern_library import PatternLibrary
                            from schemas.pattern_library import PatternStatus as _PS
                            pat_lib = PatternLibrary(db_path / "memory" / "findings")
                            for p in pat_lib.load_all():
                                if p.linked_suggestion_id == sid and p.status == _PS.PROPOSED:
                                    pat_lib.validate_pattern(p.pattern_id)
                                    logger.info("Promoted pattern %s to VALIDATED", p.pattern_id)
                        except Exception:
                            logger.warning("Failed to promote pattern for suggestion %s", sid)
            except Exception:
                logger.warning("Outcome measurement failed for %s", sid)

        # Evaluate predictions against actual curated data
        try:
            from skills.prediction_tracker import PredictionTracker

            pred_tracker = PredictionTracker(db_path / "memory" / "findings")
            predictions = pred_tracker.load_predictions()
            if predictions:
                # Get unique weeks and evaluate un-evaluated ones
                weeks = {p.week for p in predictions}
                for week in sorted(weeks):
                    evaluation = pred_tracker.evaluate_predictions(week, curated_dir)
                    if evaluation.total > 0:
                        logger.info(
                            "Prediction evaluation for %s: %d/%d correct (%.0f%%)",
                            week, evaluation.correct, evaluation.total, evaluation.accuracy * 100,
                        )
        except Exception:
            logger.warning("Prediction evaluation failed during outcome measurement")

    # TransferOutcomeMeasurer — runs weekly to measure cross-bot transfer outcomes
    async def _measure_transfer_outcomes() -> None:
        """Measure outcomes of cross-bot pattern transfers."""
        try:
            from skills.pattern_library import PatternLibrary
            from skills.transfer_proposal_builder import TransferProposalBuilder

            lib = PatternLibrary(db_path / "memory" / "findings")
            builder = TransferProposalBuilder(
                pattern_library=lib,
                curated_dir=curated_dir,
                bots=config.bot_ids,
                findings_dir=db_path / "memory" / "findings",
            )
            outcomes = builder.measure_transfer_outcomes()
            if outcomes:
                logger.info("Measured %d transfer outcomes", len(outcomes))
        except Exception:
            logger.exception("Transfer outcome measurement failed")

    # MemoryConsolidator — rebuilds index.json and consolidates old findings
    consolidator = MemoryConsolidator(
        findings_dir=db_path / "memory" / "findings",
        base_dir=db_path,
    )

    async def _consolidate_memory() -> None:
        """Rebuild memory index and conditionally consolidate findings."""
        try:
            consolidator.rebuild_index()
            if consolidator.needs_consolidation("corrections.jsonl"):
                consolidator.consolidate("corrections.jsonl")
            if consolidator.needs_consolidation("failure-log.jsonl"):
                consolidator.consolidate("failure-log.jsonl")
            # Promote candidate hypotheses that have been proposed multiple times
            promoted = hypothesis_library.promote_candidates()
            if promoted:
                logger.info("Promoted %d candidate hypotheses to active", promoted)
        except Exception:
            logger.exception("Memory consolidation failed")

    # AuditTrailConsumer — persists SSE events to JSONL
    audit_consumer = AuditTrailConsumer(log_dir=db_path / "logs")

    # Build scheduler jobs
    scheduler_config = SchedulerConfig()
    scheduler_jobs = create_scheduler_jobs(
        config=scheduler_config,
        worker_fn=lambda: worker.process_batch(),
        monitoring_fn=lambda: monitoring_loop.run_all(),
        relay_fn=vps_receiver.poll if vps_receiver else _noop_relay,
        daily_analysis_fn=lambda: queue.enqueue({
            "event_type": "daily_analysis_trigger",
            "bot_id": "system",
            "event_id": f"daily-trigger-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "payload": "{}",
        }),
        weekly_analysis_fn=lambda: queue.enqueue({
            "event_type": "weekly_summary_trigger",
            "bot_id": "system",
            "event_id": f"weekly-trigger-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "payload": "{}",
        }),
        wfo_fn=lambda: queue.enqueue({
            "event_type": "wfo_trigger",
            "bot_id": "system",
            "event_id": f"wfo-trigger-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "payload": "{}",
        }),
        stale_event_recovery_fn=lambda: queue.recover_stale(),
        morning_scan_fn=_morning_scan,
        evening_report_fn=_evening_report,
        outcome_measurement_fn=_measure_outcomes,
        memory_consolidation_fn=_consolidate_memory,
        transfer_outcome_fn=_measure_transfer_outcomes,
        approval_expiry_fn=(lambda: _expire_approvals_with_notification(
            approval_tracker, telegram_adapter, dispatcher, notification_prefs,
        )) if approval_tracker else None,
        pr_review_check_fn=_check_pr_reviews if approval_tracker else None,
        deployment_check_fn=handlers._check_deployments if deployment_monitor else None,
        threshold_learning_fn=threshold_learner.learn_thresholds if threshold_learner else None,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await queue.initialize()
        await registry.initialize()

        # Start channel adapters
        for adapter in channel_adapters:
            try:
                await adapter.start()
            except Exception:
                logger.warning("Failed to start %s adapter", type(adapter).__name__)

        # Start Telegram update polling (for callback queries and slash commands)
        if telegram_adapter is not None and telegram_adapter._callback_router is not None:
            try:
                await telegram_adapter.start_polling()
            except Exception:
                logger.warning("Failed to start Telegram polling")

        # Start audit trail consumer
        audit_consumer.start(event_stream)

        # Catch up on events buffered while PC was off
        if vps_receiver:
            try:
                await vps_receiver.drain()
            except Exception:
                logger.warning("Startup drain failed — will retry via scheduler")

        # Start scheduler
        scheduler = _create_scheduler(scheduler_jobs)
        app.state.scheduler = scheduler

        yield

        # Shutdown
        await audit_consumer.stop()
        if hasattr(app.state, "scheduler") and app.state.scheduler:
            app.state.scheduler.shutdown(wait=False)
        await subagent_mgr.cancel_all()

        # Stop channel adapters
        for adapter in channel_adapters:
            try:
                await adapter.stop()
            except Exception:
                logger.warning("Failed to stop %s adapter", type(adapter).__name__)

        await queue.close()
        await registry.close()

    app = FastAPI(title="Trading Assistant Orchestrator", lifespan=lifespan)
    app.state.start_time = datetime.now(timezone.utc)

    # Expose on app.state so test fixtures can manually initialize/close
    # (httpx ASGITransport does not trigger lifespan events)
    app.state.queue = queue
    app.state.registry = registry
    app.state.worker = worker
    app.state.event_stream = event_stream
    app.state.session_store = session_store
    app.state.subagent_mgr = subagent_mgr
    app.state.dispatcher = dispatcher
    app.state.agent_runner = agent_runner
    app.state.brain = brain
    app.state.handlers = handlers
    app.state.monitoring_loop = monitoring_loop
    app.state.notification_preferences = notification_prefs
    app.state.vps_receiver = vps_receiver
    app.state.latency_tracker = latency_tracker
    app.state.config = config
    app.state.consolidator = consolidator
    app.state.audit_consumer = audit_consumer
    app.state.conversation_tracker = conversation_tracker
    app.state.skills_registry = skills_registry
    app.state.autonomous_pipeline = autonomous_pipeline
    app.state.approval_tracker = approval_tracker
    app.state.approval_handler = approval_handler
    app.state.deployment_monitor = deployment_monitor
    app.state.threshold_learner = threshold_learner

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.get("/metrics")
    async def metrics():
        """Operational metrics for monitoring the orchestrator itself."""
        from schemas.orchestrator_metrics import BotLatencyStats, OrchestratorMetrics

        pending = await queue.count_pending()
        dead_count = await queue.count_dead_letters()
        running = subagent_mgr.get_running()
        uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()
        error_rate = brain.get_error_rate_1h()

        agg = latency_tracker.get_aggregate_stats()
        per_bot = [
            BotLatencyStats(bot_id=bid, p50=s.p50, p95=s.p95, max=s.max, sample_count=s.sample_count)
            for bid, s in latency_tracker.get_all_stats().items()
        ]

        return OrchestratorMetrics(
            queue_depth=pending,
            dead_letter_count=dead_count,
            active_agents=len(running),
            error_rate_1h=error_rate,
            uptime_seconds=uptime,
            last_daily_analysis=brain.last_daily_analysis,
            last_weekly_analysis=brain.last_weekly_analysis,
            delivery_latency_p50=agg.p50,
            delivery_latency_p95=agg.p95,
            delivery_latency_max=agg.max,
            per_bot_latency=per_bot,
        ).model_dump(mode="json")

    @app.post("/ingest")
    async def ingest_event(event: dict):
        """Direct event ingest — bypasses relay, useful for testing."""
        event.setdefault("received_at", datetime.now(timezone.utc).isoformat())
        # Record latency for directly ingested events too
        ex_ts = event.get("exchange_timestamp", "")
        rx_ts = event.get("received_at", "")
        if ex_ts and rx_ts:
            latency_tracker.record(event.get("bot_id", "unknown"), ex_ts, rx_ts)
        inserted = await queue.enqueue(event)
        return {"inserted": inserted, "event_id": event.get("event_id")}

    @app.get("/events/pending")
    async def pending_events(limit: int = 20):
        return await queue.peek(limit=limit)

    @app.get("/tasks")
    async def list_tasks(status: str | None = None):
        if status:
            from schemas.tasks import TaskStatus
            return [t.model_dump(mode="json") for t in await registry.list_by_status(TaskStatus(status))]
        return []

    @app.post("/process")
    async def trigger_processing(limit: int = 10):
        """Manually trigger event processing (for testing, normally done by scheduler)."""
        processed = await worker.process_batch(limit=limit)
        return {"processed": processed}

    @app.get("/events/stream")
    async def sse_stream(last_sequence: int = 0):
        from starlette.responses import StreamingResponse

        async def generate():
            # Send catch-up events
            for event in event_stream.get_recent(since_sequence=last_sequence):
                yield f"id: {event.sequence}\nevent: {event.event_type}\ndata: {event.model_dump_json()}\n\n"

            # Subscribe for new events
            q = event_stream.subscribe()
            try:
                while True:
                    event = await q.get()
                    yield f"id: {event.sequence}\nevent: {event.event_type}\ndata: {event.model_dump_json()}\n\n"
            finally:
                event_stream.unsubscribe(q)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/events/dead-letter")
    async def dead_letter_events(limit: int = 50):
        return await queue.get_dead_letters(limit=limit)

    @app.post("/events/dead-letter/{event_id}/reprocess")
    async def reprocess_dead_letter(event_id: str):
        success = await queue.reprocess_dead_letter(event_id)
        if not success:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Event not found in dead-letter queue")
        return {"status": "requeued", "event_id": event_id}

    @app.get("/notifications/preferences")
    async def get_notification_preferences():
        return app.state.notification_preferences.model_dump()

    @app.put("/notifications/preferences")
    async def update_notification_preferences(prefs: NotificationPreferences):
        app.state.notification_preferences = prefs
        # Also update the handlers' reference
        handlers._notification_prefs = prefs
        # Persist to disk
        _save_notification_prefs(prefs, prefs_path)
        return prefs.model_dump()

    @app.post("/feedback")
    async def submit_feedback(body: dict):
        """Submit user feedback (approve/reject suggestions, corrections).

        Enqueues a user_feedback event that flows through brain → worker → handle_feedback.
        """
        text = body.get("text", "")
        if not text:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="'text' field is required")
        import secrets
        feedback_event = {
            "event_type": "user_feedback",
            "bot_id": body.get("bot_id", "user"),
            "event_id": f"feedback-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}",
            "payload": json.dumps({
                "text": text,
                "report_id": body.get("report_id", "unknown"),
            }),
        }
        inserted = await queue.enqueue(feedback_event)
        return {"inserted": inserted, "event_id": feedback_event["event_id"]}

    @app.get("/sessions")
    async def list_sessions(agent_type: str | None = None, date: str | None = None):
        return session_store.list_sessions(agent_type=agent_type, date=date)

    @app.get("/subagents")
    async def list_subagents():
        running = subagent_mgr.get_running()
        return [{"id": a.id, "agent_type": a.agent_type, "started_at": a.started_at.isoformat()} for a in running]

    @app.post("/subagents/{agent_id}/cancel")
    async def cancel_subagent(agent_id: str):
        success = await subagent_mgr.cancel(agent_id)
        if not success:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Agent not found or not running")
        return {"status": "cancelled", "agent_id": agent_id}

    return app


def _create_scheduler(jobs: list[dict]) -> object | None:
    """Create and start an AsyncIOScheduler from job definitions.

    Returns None if APScheduler is not installed (e.g., during testing).
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("APScheduler not installed — scheduler disabled")
        return None

    scheduler = AsyncIOScheduler()
    for job in jobs:
        job = dict(job)  # shallow copy to avoid mutating the original
        trigger_type = job.pop("trigger")
        func = job.pop("func")
        name = job.pop("name")

        if trigger_type == "interval":
            trigger = IntervalTrigger(seconds=job.pop("seconds"))
        elif trigger_type == "cron":
            trigger = CronTrigger(**job)
            job = {}
        else:
            logger.warning("Unknown trigger type %s for job %s", trigger_type, name)
            continue

        scheduler.add_job(func, trigger, id=name, name=name, **job)

    scheduler.start()
    return scheduler


async def _noop_relay() -> None:
    """Placeholder relay poll — replace with actual relay client."""
    pass


# Default app instance for `uvicorn orchestrator.app:app`
app = create_app()
