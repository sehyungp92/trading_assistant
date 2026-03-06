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

    # Load persisted notification preferences
    prefs_path = db_path / "data" / "notification_prefs.json"
    notification_prefs = _load_notification_prefs(prefs_path)

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

    # Relay polling — only active when RELAY_URL is configured
    vps_receiver: VPSReceiver | None = None
    if config.relay_url:
        vps_receiver = VPSReceiver(
            relay_url=config.relay_url,
            local_queue=queue,
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
                    pnl = summary.get("total_pnl", 0)
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
    from skills.suggestion_tracker import SuggestionTracker
    from schemas.suggestion_tracking import SuggestionStatus

    outcome_measurer = AutoOutcomeMeasurer(curated_dir=curated_dir)
    suggestion_tracker = SuggestionTracker(store_dir=db_path / "data" / "findings")

    async def _measure_outcomes() -> None:
        """Measure outcomes of implemented suggestions."""
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
            except Exception:
                logger.warning("Outcome measurement failed for %s", sid)

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
