"""Orchestrator FastAPI entry point — wires all components together.

Run with: uvicorn orchestrator.app:app --reload
For production, use create_app() factory to configure paths.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from comms.dispatcher import NotificationDispatcher
from comms.telegram_handlers import TelegramCallbackResponse, TelegramCallbackRouter
from comms.telegram_renderer import TelegramRenderer
from orchestrator.adapters.vps_receiver import VPSReceiver
from orchestrator.agent_runner import AgentRunner
from orchestrator.cost_tracker import CostTracker
from orchestrator.catchup import StartupCatchup, bootstrap_run_store_from_history
from orchestrator.latency_tracker import LatencyTracker
from orchestrator.config import AppConfig
from orchestrator.conversation_tracker import ConversationTracker
from orchestrator.db.queue import EventQueue
from orchestrator.event_stream import AuditTrailConsumer, EventStream
from orchestrator.handlers import Handlers
from orchestrator.memory_consolidator import MemoryConsolidator
from orchestrator.monitoring import MonitoringCheck, MonitoringLoop
from orchestrator.orchestrator_brain import OrchestratorBrain
from orchestrator.scheduler import (
    SchedulerConfig,
    ScheduledJobRunner,
    ScheduledJobSpec,
    build_scheduled_job_specs,
    job_specs_to_scheduler_jobs,
)
from orchestrator.scheduled_runs import ScheduledRunStore
from orchestrator.session_store import SessionStore
from orchestrator.skills_registry import SkillsRegistry
from orchestrator.subagent import SubagentManager
from orchestrator.task_registry import TaskRegistry
from orchestrator.worker import Worker
from schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
)
from schemas.notifications import (
    ChannelConfig,
    NotificationChannel,
    NotificationPayload,
    NotificationPreferences,
    NotificationPriority,
)

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _scope_token(scope_key: str) -> str:
    return hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:10]


def _build_scheduled_event(
    *,
    job_key: str,
    scope_key: str,
    scheduled_for: datetime,
    event_type: str,
    bot_id: str,
    payload: dict,
) -> dict:
    scheduled_for = scheduled_for.astimezone(timezone.utc).replace(microsecond=0)
    received_at = _utc_now()
    event_id = (
        f"{job_key}-{_scope_token(scope_key)}-"
        f"{scheduled_for.strftime('%Y%m%dT%H%M')}"
    )
    return {
        "event_id": event_id,
        "bot_id": bot_id,
        "event_type": event_type,
        "payload": json.dumps(payload),
        "exchange_timestamp": scheduled_for.isoformat(),
        "received_at": received_at.isoformat(),
    }


def _normalize_queue_event(event: dict) -> dict:
    normalized = dict(event)
    missing = [
        field for field in ("event_id", "bot_id", "event_type", "payload")
        if field not in normalized
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required event field(s): {', '.join(missing)}",
        )

    payload = normalized["payload"]
    if isinstance(payload, (dict, list)):
        normalized["payload"] = json.dumps(payload)
    elif not isinstance(payload, str):
        raise HTTPException(
            status_code=400,
            detail="'payload' must be a JSON string, object, or array",
        )

    received_at = normalized.setdefault("received_at", _utc_now().isoformat())
    normalized.setdefault("exchange_timestamp", received_at)
    return normalized


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


def _selection_from_env(provider_raw: str, model_raw: str = "") -> AgentSelection | None:
    provider_name = provider_raw.strip().lower()
    if not provider_name:
        return None
    try:
        provider = AgentProvider(provider_name)
    except ValueError:
        logger.warning("Ignoring invalid AGENT provider value: %r", provider_raw)
        return None
    model = model_raw.strip() or None
    return AgentSelection(provider=provider, model=model)


def _seed_agent_preferences(config: AppConfig) -> AgentPreferences:
    default_selection = _selection_from_env(
        config.agent_default_provider,
        config.agent_default_model,
    ) or AgentSelection(provider=AgentProvider.CLAUDE_MAX)

    overrides: dict[AgentWorkflow, AgentSelection | None] = {}
    workflow_env_map = (
        (AgentWorkflow.DAILY_ANALYSIS, config.daily_agent_provider, config.daily_agent_model),
        (AgentWorkflow.WEEKLY_ANALYSIS, config.weekly_agent_provider, config.weekly_agent_model),
        (AgentWorkflow.WFO, config.wfo_agent_provider, config.wfo_agent_model),
        (AgentWorkflow.TRIAGE, config.triage_agent_provider, config.triage_agent_model),
    )
    for workflow, provider_raw, model_raw in workflow_env_map:
        selection = _selection_from_env(provider_raw, model_raw)
        if selection is not None:
            overrides[workflow] = selection

    return AgentPreferences(default=default_selection, overrides=overrides)


def _load_agent_preferences(
    prefs_path: Path,
    config: AppConfig,
) -> AgentPreferences:
    """Load agent preferences from disk or build initial values from env."""
    if prefs_path.exists():
        try:
            return AgentPreferences(**json.loads(prefs_path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Could not load agent prefs from %s, using defaults", prefs_path)
    return _seed_agent_preferences(config)


def _save_agent_preferences(prefs: AgentPreferences, prefs_path: Path) -> None:
    """Persist agent preferences to disk."""
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
        from comms.telegram_renderer import TelegramRenderer

        telegram = TelegramBotAdapter(config=TelegramBotConfig(
            token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
        ))
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, telegram)
        dispatcher.register_renderer(NotificationChannel.TELEGRAM, TelegramRenderer())
        adapters.append(telegram)
        logger.info("Telegram adapter registered (chat_id=%s)", config.telegram_chat_id)

    if config.discord_bot_token:
        from comms.discord_bot import DiscordBotAdapter, DiscordBotConfig
        from comms.renderer import PlainTextRenderer

        discord_adapter = DiscordBotAdapter(config=DiscordBotConfig(
            token=config.discord_bot_token,
            channel_id=int(config.discord_channel_id) if config.discord_channel_id else 0,
        ))
        dispatcher.register_adapter(NotificationChannel.DISCORD, discord_adapter)
        dispatcher.register_renderer(NotificationChannel.DISCORD, PlainTextRenderer())
        adapters.append(discord_adapter)
        logger.info("Discord adapter registered (channel_id=%s)", config.discord_channel_id)

    if config.smtp_host and config.smtp_user:
        from comms.email_adapter import EmailAdapter, EmailConfig
        from comms.renderer import PlainTextRenderer

        email = EmailAdapter(config=EmailConfig(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            username=config.smtp_user,
            password=config.smtp_pass,
            from_address=config.email_from,
        ))
        dispatcher.register_adapter(NotificationChannel.EMAIL, email)
        dispatcher.register_renderer(NotificationChannel.EMAIL, PlainTextRenderer())
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
    raw_data_dir = db_path / "raw"
    curated_dir = db_path / "data" / "curated"
    queue = EventQueue(db_path=str(db_path / "events.db"))
    registry = TaskRegistry(db_path=str(db_path / "tasks.db"))
    scheduled_run_store = ScheduledRunStore(db_path=str(db_path / "scheduled_runs.db"))
    brain = OrchestratorBrain()
    event_stream = EventStream()
    conversation_tracker = ConversationTracker()
    worker = Worker(
        queue=queue, registry=registry, brain=brain,
        event_stream=event_stream, conversation_tracker=conversation_tracker,
        raw_data_dir=raw_data_dir,
        bot_configs=config.bot_configs,
    )
    session_store = SessionStore(base_dir=str(db_path / ".assistant" / "sessions"))
    subagent_mgr = SubagentManager()

    # Latency tracker — shared across VPSReceiver, monitoring, and /metrics
    latency_tracker = LatencyTracker()
    prefs_path = db_path / "data" / "notification_prefs.json"
    notification_prefs = _load_notification_prefs(prefs_path)
    agent_prefs_path = db_path / "data" / "agent_preferences.json"
    agent_preferences = _load_agent_preferences(agent_prefs_path, config)

    # Integration layer components
    dispatcher = NotificationDispatcher()
    skills_registry = SkillsRegistry()
    cost_tracker = CostTracker(db_path / "data" / "cost_log.jsonl")
    agent_runner = AgentRunner(
        runs_dir=db_path / "runs",
        session_store=session_store,
        claude_command=config.claude_command,
        claude_command_args=config.claude_command_args,
        codex_command=config.codex_command,
        codex_command_args=config.codex_command_args,
        skills_registry=skills_registry,
        preferences=agent_preferences,
        zai_api_key=config.zai_api_key,
        openrouter_api_key=config.openrouter_api_key,
        event_stream=event_stream,
        cost_tracker=cost_tracker,
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

    # Auto-seed notification preferences from registered adapters if empty
    if not notification_prefs.channels and dispatcher.adapters:
        seeded_channels = []
        for channel in dispatcher.adapters:
            chat_id = ""
            if channel == NotificationChannel.TELEGRAM:
                chat_id = config.telegram_chat_id or ""
            elif channel == NotificationChannel.EMAIL:
                chat_id = config.email_to or ""
            seeded_channels.append(ChannelConfig(channel=channel, enabled=True, chat_id=chat_id))
        notification_prefs = NotificationPreferences(channels=seeded_channels)
        _save_notification_prefs(notification_prefs, prefs_path)
        logger.info("Auto-seeded notification preferences from %d registered adapters", len(seeded_channels))

    # Extract Telegram adapter reference (needed for autonomous pipeline)
    telegram_adapter = None
    for adapter in channel_adapters:
        from comms.telegram_bot import TelegramBotAdapter as _TBA
        if isinstance(adapter, _TBA):
            telegram_adapter = adapter
            break

    callback_router = TelegramCallbackRouter() if telegram_adapter is not None else None
    telegram_renderer = TelegramRenderer() if telegram_adapter is not None else None

    # SuggestionTracker — shared path with ContextBuilder (memory/findings/)
    from skills.suggestion_tracker import SuggestionTracker

    suggestion_tracker = SuggestionTracker(store_dir=db_path / "memory" / "findings")

    # Autonomous pipeline (feature-flagged)
    autonomous_pipeline = None
    approval_tracker = None
    approval_handler = None
    config_registry = None
    file_change_gen = None
    pr_builder = None
    repo_workspace_manager = None
    repo_task_runner = None
    calibration_tracker = None
    if config.autonomous_enabled:
        from orchestrator.repo_task_runner import RepoTaskRunner
        from skills.approval_handler import ApprovalHandler
        from skills.approval_tracker import ApprovalTracker
        from skills.autonomous_pipeline import AutonomousPipeline
        from skills.config_registry import ConfigRegistry as _ConfigRegistry
        from skills.file_change_generator import FileChangeGenerator
        from skills.github_pr import PRBuilder
        from skills.repo_workspace import RepoWorkspaceManager
        from skills.suggestion_backtester import SuggestionBacktester

        config_registry = _ConfigRegistry(Path(config.bot_config_dir))
        backtester = SuggestionBacktester(config_registry, db_path)
        approval_tracker = ApprovalTracker(db_path / "memory" / "findings" / "approvals.jsonl")
        file_change_gen = FileChangeGenerator()
        pr_builder = PRBuilder(dry_run=False, github_token=config.github_token)
        repo_workspace_manager = RepoWorkspaceManager(
            cache_root=Path(config.bot_repo_cache_dir),
            task_root=db_path / "runs" / "repo_tasks",
        )
        repo_task_runner = RepoTaskRunner(agent_runner)

        # Telegram bot/renderer (may be None if not configured)
        telegram_bot = telegram_adapter  # Extracted from channel_adapters above

        approval_handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=suggestion_tracker,
            file_change_generator=file_change_gen,
            pr_builder=pr_builder,
            config_registry=config_registry,
            event_stream=event_stream,
            telegram_bot=telegram_bot,
            repo_workspace_manager=repo_workspace_manager,
            repo_task_runner=repo_task_runner,
            # structural_experiment_tracker wired after creation below
            # hypothesis_library wired after creation below
        )
        # Parameter searcher for iterative neighborhood exploration
        from skills.parameter_searcher import ParameterSearcher
        from skills.backtest_calibration_tracker import BacktestCalibrationTracker
        from skills.cost_model import CostModel
        from skills.backtest_simulator import BacktestSimulator as _BacktestSim

        _cost_cfg = getattr(config, "cost_model_config", None)
        if _cost_cfg:
            _cost_model = CostModel(_cost_cfg)
            _simulator = _BacktestSim(_cost_model)
            parameter_searcher = ParameterSearcher(
                config_registry=config_registry,
                simulator=_simulator,
                cost_model=_cost_model,
            )
        else:
            parameter_searcher = None

        calibration_tracker = BacktestCalibrationTracker(
            store_dir=db_path / "memory" / "findings",
        )

        autonomous_pipeline = AutonomousPipeline(
            config_registry=config_registry,
            backtester=backtester,
            approval_tracker=approval_tracker,
            suggestion_tracker=suggestion_tracker,
            telegram_bot=telegram_bot,
            telegram_renderer=telegram_renderer,
            event_stream=event_stream,
            parameter_searcher=parameter_searcher,
            experiment_config_generator=None,
            experiment_tracker=None,
            calibration_tracker=calibration_tracker,
            search_log_dir=db_path / "memory" / "findings",
            curated_dir=curated_dir,
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

        if not config.autonomous_enabled:
            logger.warning(
                "deployment_monitoring_enabled=True but autonomous_enabled=False: "
                "rollback PRs will not work (pr_builder is None)"
            )
        _dm_pr_builder = pr_builder if config.autonomous_enabled else None
        _dm_config_registry = config_registry if config.autonomous_enabled else None
        _dm_file_change_gen = file_change_gen if config.autonomous_enabled else None
        deployment_monitor = DeploymentMonitor(
            findings_dir=db_path / "memory" / "findings",
            curated_dir=curated_dir,
            pr_builder=_dm_pr_builder,
            config_registry=_dm_config_registry,
            event_stream=event_stream,
            file_change_generator=_dm_file_change_gen,
        )
        logger.info("Deployment monitoring enabled")

    # Experiment manager (feature-flagged)
    experiment_manager = None
    experiment_config_gen = None
    if config.ab_testing_enabled:
        from skills.experiment_manager import ExperimentManager
        from skills.experiment_config_generator import ExperimentConfigGenerator

        experiment_manager = ExperimentManager(
            findings_dir=db_path / "memory" / "findings",
        )
        _ecg_registry = config_registry if config.autonomous_enabled else None
        experiment_config_gen = ExperimentConfigGenerator(
            config_registry=_ecg_registry,
        )
        logger.info("A/B testing enabled")

    # ReliabilityTracker — tracks bug fix interventions and recurrence
    from skills.reliability_tracker import ReliabilityTracker

    reliability_tracker = ReliabilityTracker(store_dir=db_path / "memory" / "findings")

    # StructuralExperimentTracker — tracks structural experiment lifecycle
    from skills.structural_experiment_tracker import StructuralExperimentTracker

    structural_experiment_tracker = StructuralExperimentTracker(
        store_dir=db_path / "memory" / "findings",
    )

    # Wire structural_experiment_tracker into approval_handler (created before tracker)
    if approval_handler is not None:
        approval_handler._structural_experiment_tracker = structural_experiment_tracker

    handlers = Handlers(
        agent_runner=agent_runner,
        event_stream=event_stream,
        dispatcher=dispatcher,
        notification_prefs=notification_prefs,
        curated_dir=curated_dir,
        raw_data_dir=raw_data_dir,
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
        approval_handler=approval_handler,
        approval_tracker=approval_tracker,
        pr_builder=pr_builder,
        config_registry=config_registry,
        repo_workspace_manager=repo_workspace_manager,
        deployment_monitor=deployment_monitor,
        threshold_learner=threshold_learner,
        experiment_manager=experiment_manager,
        experiment_config_gen=experiment_config_gen,
        bot_configs=config.bot_configs,
        reliability_tracker=reliability_tracker,
        structural_experiment_tracker=structural_experiment_tracker,
        strategy_registry=config.strategy_registry,
    )

    # Late-wire experiment components into autonomous pipeline (defined after pipeline creation)
    if config.autonomous_enabled and autonomous_pipeline is not None:
        autonomous_pipeline._experiment_tracker = structural_experiment_tracker
        if experiment_config_gen is not None:
            autonomous_pipeline._experiment_config_generator = experiment_config_gen
        if experiment_manager is not None:
            autonomous_pipeline._experiment_manager = experiment_manager

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

    def _store_agent_preferences(preferences: AgentPreferences) -> None:
        agent_runner.update_preferences(preferences)
        _save_agent_preferences(preferences, agent_prefs_path)
        try:
            app.state.agent_preferences = agent_runner.get_preferences()
        except NameError:
            pass

    def _render_agent_settings_response(
        scope: AgentWorkflow | str | None,
        *,
        edit_message: bool,
        answer: str,
    ) -> TelegramCallbackResponse:
        if telegram_renderer is None:
            return TelegramCallbackResponse(text="Telegram settings are unavailable.", answer=answer)
        text, keyboard = telegram_renderer.render_agent_settings(
            agent_runner.get_preferences_view(),
            scope=scope,
        )
        return TelegramCallbackResponse(
            text=text,
            keyboard=keyboard,
            answer=answer,
            edit_message=edit_message,
        )

    async def _settings_command(**kwargs) -> TelegramCallbackResponse:
        return _render_agent_settings_response(
            None,
            edit_message=False,
            answer="Agent settings",
        )

    async def _settings_home(**kwargs) -> TelegramCallbackResponse:
        return _render_agent_settings_response(
            None,
            edit_message=True,
            answer="Agent settings",
        )

    async def _settings_scope(scope_name: str) -> TelegramCallbackResponse:
        if scope_name == "global":
            return _render_agent_settings_response(
                "global",
                edit_message=True,
                answer="Global provider",
            )
        try:
            workflow = AgentWorkflow(scope_name)
        except ValueError as exc:
            return TelegramCallbackResponse(text=f"Unknown settings scope: {exc}", answer="Invalid scope")
        return _render_agent_settings_response(
            workflow,
            edit_message=True,
            answer=f"{scope_name} settings",
        )

    async def _settings_set(payload: str) -> TelegramCallbackResponse:
        scope_name, provider_name = (payload.split("|", 1) + [""])[:2]
        try:
            provider = AgentProvider(provider_name)
        except ValueError:
            return TelegramCallbackResponse(text="Unknown provider.", answer="Invalid provider")

        prefs = agent_runner.get_preferences()
        if scope_name == "global":
            prefs.default = AgentSelection(provider=provider)
            render_scope: AgentWorkflow | str | None = "global"
        else:
            try:
                workflow = AgentWorkflow(scope_name)
            except ValueError:
                return TelegramCallbackResponse(text="Unknown workflow.", answer="Invalid workflow")
            prefs.overrides[workflow] = AgentSelection(provider=provider)
            render_scope = workflow

        try:
            _store_agent_preferences(prefs)
        except ValueError as exc:
            return TelegramCallbackResponse(text=str(exc), answer="Provider unavailable")

        return _render_agent_settings_response(
            render_scope,
            edit_message=True,
            answer="Provider updated",
        )

    async def _settings_clear(scope_name: str) -> TelegramCallbackResponse:
        try:
            workflow = AgentWorkflow(scope_name)
        except ValueError:
            return TelegramCallbackResponse(text="Unknown workflow.", answer="Invalid workflow")

        prefs = agent_runner.get_preferences()
        prefs.overrides.pop(workflow, None)
        _store_agent_preferences(prefs)
        return _render_agent_settings_response(
            workflow,
            edit_message=True,
            answer="Override cleared",
        )

    if callback_router is not None:
        callback_router.register("cmd_settings", _settings_command)
        callback_router.register("agent_settings_home", _settings_home)
        callback_router.register("agent_settings_scope_", _settings_scope)
        callback_router.register("agent_settings_set_", _settings_set)
        callback_router.register("agent_settings_clear_", _settings_clear)
        telegram_adapter.set_callback_router(callback_router)

    # Register Telegram approval callbacks when autonomous pipeline is enabled
    if approval_handler is not None:
        callback_router = callback_router or TelegramCallbackRouter()

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
                            suggestion_id=req.suggestion_id,
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

        logger.info("Telegram approval callbacks registered")

    # Register experiment Telegram callbacks (independent of autonomous pipeline)
    if experiment_manager is not None and telegram_adapter is not None:
        callback_router = callback_router or TelegramCallbackRouter()

        async def _on_start_experiment(experiment_id: str) -> str:
            try:
                experiment_manager.activate_experiment(experiment_id)
                exp = experiment_manager.get_by_id(experiment_id)
                return f"\u25b6\ufe0f Experiment started: {exp.title if exp else experiment_id}"
            except Exception as e:
                return f"Failed to start experiment: {e}"

        async def _on_cancel_experiment(experiment_id: str) -> str:
            try:
                experiment_manager.cancel_experiment(experiment_id)
                return f"\u274c Experiment cancelled: {experiment_id}"
            except Exception as e:
                return f"Failed to cancel experiment: {e}"

        callback_router.register("start_experiment_", _on_start_experiment)
        callback_router.register("cancel_experiment_", _on_cancel_experiment)

        logger.info("Experiment Telegram callbacks registered")

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

    async def _morning_scan(
        bot_ids: list[str] | None = None,
        scheduled_for: datetime | None = None,
    ) -> None:
        """Gather overnight data and dispatch morning scan notifications."""
        from orchestrator.tz_utils import bot_trading_date

        reference_time = (scheduled_for or _utc_now()).astimezone(timezone.utc)
        scan_bots = bot_ids or config.bot_ids
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

        # Check each bot's most recent completed trading day (per-bot timezone)
        for bot_id in scan_bots:
            if config.bot_configs and bot_id in config.bot_configs:
                tz = config.bot_configs[bot_id].timezone
            else:
                tz = "UTC"
            # "yesterday" in the bot's local timezone
            yesterday = bot_trading_date(tz, reference_time - timedelta(days=1))
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
        hour_utc = reference_time.hour
        for payload in result.payloads:
            try:
                await dispatcher.dispatch(payload, notification_prefs, hour_utc)
            except Exception:
                logger.exception("Morning scan dispatch failed")

    async def _evening_report(
        bot_ids: list[str] | None = None,
        scheduled_for: datetime | None = None,
    ) -> None:
        """Check if daily report is ready and send evening notification."""
        from orchestrator.tz_utils import bot_trading_date

        reference_time = (scheduled_for or _utc_now()).astimezone(timezone.utc)
        scan_bots = bot_ids or config.bot_ids
        daily_report_ready = False
        date = reference_time.strftime("%Y-%m-%d")
        if scan_bots:
            for bot_id in scan_bots:
                if config.bot_configs and bot_id in config.bot_configs:
                    tz = config.bot_configs[bot_id].timezone
                else:
                    tz = "UTC"
                bot_date = bot_trading_date(tz, reference_time)
                if (curated_dir / bot_date / bot_id / "summary.json").exists():
                    daily_report_ready = True
                    break

        result = scanner.evening_scan(date=date, daily_report_ready=daily_report_ready)
        hour_utc = reference_time.hour
        for payload in result.payloads:
            try:
                await dispatcher.dispatch(payload, notification_prefs, hour_utc)
            except Exception:
                logger.exception("Evening report dispatch failed")

    # AutoOutcomeMeasurer — runs weekly to measure suggestion outcomes
    from skills.auto_outcome_measurer import AutoOutcomeMeasurer
    from schemas.suggestion_tracking import SuggestionStatus

    outcome_measurer = AutoOutcomeMeasurer(
        curated_dir=curated_dir,
        findings_dir=db_path / "memory" / "findings",
        calibration_tracker=calibration_tracker,
    )

    # HypothesisLibrary — shared for outcome linking
    from skills.hypothesis_library import HypothesisLibrary

    hypothesis_library = HypothesisLibrary(db_path / "memory" / "findings")

    # Wire hypothesis_library into approval_handler (created before library)
    if approval_handler is not None:
        approval_handler._hypothesis_library = hypothesis_library

    async def _measure_outcomes(scheduled_for: datetime | None = None) -> None:
        """Measure outcomes of deployed suggestions and link to hypotheses."""
        suggestions = suggestion_tracker.load_all()
        deployed = [
            s for s in suggestions
            if s.get("status") == SuggestionStatus.DEPLOYED.value
        ]
        # Check which suggestions already have outcomes
        existing_outcomes = suggestion_tracker.load_outcomes()
        measured_ids = {o.get("suggestion_id") for o in existing_outcomes}

        for s in deployed:
            sid = s.get("suggestion_id", "")
            if sid in measured_ids:
                continue
            # Portfolio suggestions are measured separately below
            if s.get("bot_id") == "PORTFOLIO":
                continue
            anchor_date = (s.get("deployed_at") or "")[:10]
            if not anchor_date:
                continue
            try:
                result = outcome_measurer.measure_progressive(
                    suggestion_id=sid,
                    bot_id=s.get("bot_id", ""),
                    implemented_date=anchor_date,
                )
                if result:
                    # Persist the full enhanced measurement to outcomes.jsonl
                    enhanced_path = db_path / "memory" / "findings" / "outcomes.jsonl"
                    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(enhanced_path, "a", encoding="utf-8") as f:
                        f.write(result.model_dump_json() + "\n")

                    suggestion_tracker.mark_measured(sid)

                    # Only update hypothesis for high/medium quality verdicts
                    verdict_str = result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict)
                    is_decisive = verdict_str in ("positive", "negative")

                    # Link hypothesis outcome if suggestion has hypothesis_id
                    hyp_id = s.get("hypothesis_id")
                    if hyp_id and is_decisive:
                        try:
                            hypothesis_library.record_outcome(
                                hyp_id, positive=(verdict_str == "positive"),
                            )
                        except Exception:
                            logger.warning("Failed to record hypothesis outcome for %s", hyp_id)

                    # Promote linked patterns on positive outcome
                    if verdict_str == "positive":
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

        # --- Portfolio-level outcome measurement ---
        try:
            from skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

            portfolio_measurer = PortfolioOutcomeMeasurer(
                findings_dir=db_path / "memory" / "findings",
                curated_dir=curated_dir,
            )
            portfolio_outcomes = portfolio_measurer.measure_deployed()
            for po in portfolio_outcomes:
                po_sid = po.get("suggestion_id", "")
                if po_sid:
                    suggestion_tracker.mark_measured(po_sid)
                    # Link hypothesis outcome if suggestion has hypothesis_id
                    po_verdict = po.get("verdict", "")
                    po_hyp_id = None
                    for s in deployed:
                        if s.get("suggestion_id") == po_sid:
                            po_hyp_id = s.get("hypothesis_id")
                            break
                    if po_hyp_id and po_verdict in ("positive", "negative"):
                        try:
                            hypothesis_library.record_outcome(
                                po_hyp_id, positive=(po_verdict == "positive"),
                            )
                        except Exception:
                            logger.warning("Failed to record hypothesis outcome for portfolio %s", po_hyp_id)
            if portfolio_outcomes:
                logger.info("Measured %d portfolio outcomes", len(portfolio_outcomes))
        except Exception:
            logger.warning("Portfolio outcome measurement failed")

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

        # Causal outcome reasoning — invoke Claude to analyze WHY outcomes happened
        try:
            from analysis.outcome_reasoning_prompt import OutcomeReasoningAssembler

            # Load recent outcomes that haven't been reasoned about yet
            outcomes_path = db_path / "memory" / "findings" / "outcomes.jsonl"
            reasoning_path = db_path / "memory" / "findings" / "outcome_reasonings.jsonl"
            if outcomes_path.exists():
                import json as _json
                recent_outcomes = []
                reasoned_ids: set[str] = set()
                if reasoning_path.exists():
                    for line in reasoning_path.read_text(encoding="utf-8").strip().splitlines():
                        if line.strip():
                            try:
                                reasoned_ids.add(_json.loads(line).get("suggestion_id", ""))
                            except _json.JSONDecodeError:
                                pass
                for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
                    if line.strip():
                        try:
                            o = _json.loads(line)
                            if o.get("suggestion_id", "") not in reasoned_ids:
                                recent_outcomes.append(o)
                        except _json.JSONDecodeError:
                            pass

                if recent_outcomes:
                    assembler = OutcomeReasoningAssembler(
                        memory_dir=db_path / "memory",
                        curated_dir=curated_dir,
                        bot_configs=config.bot_configs,
                    )
                    reasoning_pkg = assembler.assemble(recent_outcomes)
                    reasoning_result = await agent_runner.invoke(
                        agent_type="outcome_reasoning",
                        prompt_package=reasoning_pkg,
                        run_id=f"reasoning-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                        allowed_tools=["Read"],
                        max_turns=5,
                    )
                    if reasoning_result.success:
                        from analysis.response_parser import parse_response as _parse

                        parsed_reasoning = _parse(reasoning_result.response)
                        if parsed_reasoning.raw_structured and "reasonings" in parsed_reasoning.raw_structured:
                            reasoning_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(reasoning_path, "a", encoding="utf-8") as f:
                                for r in parsed_reasoning.raw_structured["reasonings"]:
                                    r["reasoned_at"] = datetime.now(timezone.utc).isoformat()
                                    f.write(_json.dumps(r) + "\n")
                            logger.info(
                                "Recorded %d outcome reasonings",
                                len(parsed_reasoning.raw_structured["reasonings"]),
                            )

                            # ACT on reasoning results (autoresearch keep/discard)
                            findings = db_path / "memory" / "findings"
                            spurious_path = findings / "spurious_outcomes.jsonl"
                            recalib_path = findings / "recalibrations.jsonl"

                            # Build suggestion lookup for enriching recalibrations
                            suggestion_lookup = {s.get("suggestion_id", ""): s for s in deployed}

                            # Hoist builder outside loop (reused for all transferable reasonings)
                            _tpb = None
                            for r in parsed_reasoning.raw_structured["reasonings"]:
                                sid = r.get("suggestion_id", "")
                                # Transferable patterns → cross-bot transfer proposals
                                if r.get("transferable") and sid:
                                    try:
                                        if _tpb is None:
                                            from skills.pattern_library import PatternLibrary
                                            from skills.transfer_proposal_builder import TransferProposalBuilder
                                            _tpb = TransferProposalBuilder(
                                                pattern_library=PatternLibrary(findings),
                                                curated_dir=curated_dir,
                                                bots=config.bot_ids,
                                                findings_dir=findings,
                                            )
                                        _tpb.create_from_reasoning(r, source_bot=r.get("bot_id", ""))
                                    except Exception:
                                        logger.warning("Transfer proposal from reasoning failed for %s", sid)
                                # Spurious effects → log for context
                                if r.get("genuine_effect") is False and sid:
                                    try:
                                        spurious_path.parent.mkdir(parents=True, exist_ok=True)
                                        with open(spurious_path, "a", encoding="utf-8") as _sf:
                                            _sf.write(_json.dumps({
                                                "suggestion_id": sid,
                                                "mechanism": r.get("mechanism", ""),
                                                "confounders": r.get("confounders", []),
                                                "recorded_at": datetime.now(timezone.utc).isoformat(),
                                            }) + "\n")
                                    except Exception:
                                        logger.warning("Failed to record spurious outcome for %s", sid)
                                # Revised confidence → recalibration log
                                revised = r.get("revised_confidence")
                                if revised is not None and sid:
                                    try:
                                        recalib_path.parent.mkdir(parents=True, exist_ok=True)
                                        sugg_rec = suggestion_lookup.get(sid, {})
                                        with open(recalib_path, "a", encoding="utf-8") as _rf:
                                            _rf.write(_json.dumps({
                                                "suggestion_id": sid,
                                                "bot_id": sugg_rec.get("bot_id", ""),
                                                "category": sugg_rec.get("category", ""),
                                                "revised_confidence": revised,
                                                "lessons_learned": r.get("lessons_learned", []),
                                                "recorded_at": datetime.now(timezone.utc).isoformat(),
                                            }) + "\n")
                                    except Exception:
                                        logger.warning("Failed to record recalibration for %s", sid)
        except Exception:
            logger.warning("Outcome reasoning failed — skipping", exc_info=True)

    # TransferOutcomeMeasurer — runs weekly to measure cross-bot transfer outcomes
    async def _measure_transfer_outcomes(scheduled_for: datetime | None = None) -> None:
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

    async def _verify_reliability(scheduled_for: datetime | None = None) -> None:
        """Auto-verify reliability interventions past observation window."""
        try:
            verified = reliability_tracker.verify_completed()
            if verified:
                logger.info("Auto-verified %d reliability interventions", len(verified))
                # Create hypothesis candidates from chronic bug classes
                summary = reliability_tracker.compute_summary()
                if summary.chronic_bug_classes:
                    created = hypothesis_library.create_from_reliability(summary)
                    if created:
                        logger.info(
                            "Created %d reliability hypothesis candidates", len(created),
                        )
        except Exception:
            logger.exception("Reliability verification failed")

    # Discovery analysis — runs Saturday to find novel patterns in raw data
    async def _discovery_analysis(scheduled_for: datetime | None = None) -> None:
        """Run discovery agent to find novel patterns not covered by detectors."""
        try:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            action = Action(
                action_type="discovery_analysis",
                bot_id="",
                details={"date": date, "bots": config.bot_ids},
            )
            await handlers.handle_discovery_analysis(action)
        except Exception:
            logger.exception("Discovery analysis failed")

    # Learning cycle — runs Sunday 11:00 UTC after outcome measurement
    async def _run_learning_cycle(scheduled_for: datetime | None = None) -> None:
        """Run the autonomous weekly learning cycle."""
        try:
            from skills.learning_cycle import LearningCycle
            from datetime import timedelta

            now = datetime.now(timezone.utc)
            week_end = now.strftime("%Y-%m-%d")
            week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")

            cycle = LearningCycle(
                curated_dir=curated_dir,
                memory_dir=db_path / "memory",
                runs_dir=runs_dir,
                bots=config.bot_ids,
                suggestion_tracker=suggestion_tracker,
                hypothesis_library=hypothesis_library,
                experiment_tracker=structural_experiment_tracker,
                prediction_tracker=None,  # not needed for cycle
                calibration_tracker=calibration_tracker,
            )
            entry = await cycle.run(week_start, week_end)

            event_stream.broadcast("learning_cycle_completed", {
                "week_start": week_start,
                "week_end": week_end,
                "net_improvement": entry.net_improvement,
                "composite_delta": entry.composite_delta,
                "lessons": entry.lessons_for_next_week[:3],
            })

            logger.info(
                "Learning cycle complete: net_improvement=%s",
                entry.net_improvement,
            )
        except Exception:
            logger.exception("Learning cycle failed")

    # MemoryConsolidator — rebuilds index.json and consolidates old findings
    consolidator = MemoryConsolidator(
        findings_dir=db_path / "memory" / "findings",
        base_dir=db_path,
    )

    async def _consolidate_memory(scheduled_for: datetime | None = None) -> None:
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

    # Experiment check function (for scheduler)
    async def _check_experiments() -> None:
        """Check active experiments for auto-conclusion."""
        if experiment_manager is None:
            return
        try:
            active = experiment_manager.get_active()
            for exp in active:
                if experiment_manager.check_auto_conclusion(exp.experiment_id):
                    result = experiment_manager.analyze_experiment(exp.experiment_id)
                    experiment_manager.conclude_experiment(exp.experiment_id, result)
                    logger.info(
                        "Auto-concluded experiment %s: %s",
                        exp.experiment_id, result.recommendation,
                    )
                    event_stream.broadcast("experiment_concluded", {
                        "experiment_id": exp.experiment_id,
                        "recommendation": result.recommendation,
                        "winner": result.winner,
                        "p_value": result.p_value,
                    })
                    # Escalate: link back to suggestion and hypothesis
                    if result.recommendation == "adopt_treatment":
                        if getattr(exp, "suggestion_id", None) and suggestion_tracker:
                            try:
                                suggestion_tracker.accept(exp.suggestion_id)
                                logger.info(
                                    "Auto-accepted suggestion %s (experiment %s passed)",
                                    exp.suggestion_id, exp.experiment_id,
                                )
                            except Exception:
                                logger.warning("Failed to auto-accept suggestion %s", exp.suggestion_id)
                        if getattr(exp, "hypothesis_id", None):
                            try:
                                hypothesis_library.record_outcome(exp.hypothesis_id, positive=True)
                                logger.info(
                                    "Recorded positive outcome for hypothesis %s",
                                    exp.hypothesis_id,
                                )
                            except Exception:
                                logger.warning("Failed to record hypothesis outcome %s", exp.hypothesis_id)
                    elif result.recommendation == "keep_control":
                        if getattr(exp, "hypothesis_id", None):
                            try:
                                hypothesis_library.record_outcome(exp.hypothesis_id, positive=False)
                            except Exception:
                                pass
                    if telegram_adapter is not None:
                        try:
                            from comms.telegram_renderer import TelegramRenderer
                            renderer = TelegramRenderer()
                            text = renderer.render_experiment_result(exp, result)
                            await telegram_adapter.send_message(text)
                        except Exception:
                            logger.warning("Failed to send experiment result notification")
        except Exception:
            logger.exception("Experiment check failed")

        # Evaluate mature structural experiments
        try:
            from skills.ground_truth_computer import GroundTruthComputer

            # Metric name mapping: common names → GroundTruthSnapshot field names
            _METRIC_FIELD_MAP = {
                "sharpe": "sharpe_ratio_30d",
                "sharpe_ratio": "sharpe_ratio_30d",
                "sharpe_ratio_30d": "sharpe_ratio_30d",
                "win_rate": "win_rate",
                "drawdown": "max_drawdown_pct",
                "max_drawdown": "max_drawdown_pct",
                "max_drawdown_pct": "max_drawdown_pct",
                "process_quality": "avg_process_quality",
                "avg_process_quality": "avg_process_quality",
                "composite": "composite_score",
                "composite_score": "composite_score",
                "pnl": "pnl_total",
                "pnl_total": "pnl_total",
                "net_pnl": "pnl_total",
                "calmar": "calmar_ratio_30d",
                "calmar_ratio": "calmar_ratio_30d",
                "calmar_ratio_30d": "calmar_ratio_30d",
                "profit_factor": "profit_factor",
                "expectancy": "expectancy",
                "expected_r": "expected_total_r",
                "expected_total_r": "expected_total_r",
                "trade_count": "trade_count",
            }

            gt = GroundTruthComputer(curated_dir)
            evaluable = structural_experiment_tracker.get_evaluable_experiments()
            for exp in evaluable:
                try:
                    activated_date = exp.activated_at.strftime("%Y-%m-%d") if exp.activated_at else ""
                    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if not activated_date:
                        continue
                    gt_before = gt.compute_snapshot(exp.bot_id, activated_date)
                    gt_after = gt.compute_snapshot(exp.bot_id, now_date)

                    # Check trade sufficiency before evaluating
                    max_min_trades = max(
                        (getattr(c, "minimum_trade_count", 20) or 20)
                        for c in exp.acceptance_criteria
                    ) if exp.acceptance_criteria else 20
                    actual_trades = getattr(gt_after, "trade_count", None)
                    if actual_trades is not None and actual_trades < max_min_trades:
                        logger.info(
                            "Experiment %s: insufficient trades (%d < %d), skipping evaluation",
                            exp.experiment_id, actual_trades, max_min_trades,
                        )
                        continue

                    # Evaluate each acceptance criterion
                    criteria_met: list[bool] = []
                    actual_values: list[float] = []
                    for criterion in exp.acceptance_criteria:
                        field = _METRIC_FIELD_MAP.get(criterion.metric, criterion.metric)
                        before_val = getattr(gt_before, field, None)
                        after_val = getattr(gt_after, field, None)
                        if before_val is None or after_val is None:
                            criteria_met.append(False)
                            actual_values.append(0.0)
                            continue
                        delta = after_val - before_val
                        actual_values.append(round(delta, 4))
                        if criterion.direction == "improve":
                            met = delta >= criterion.minimum_change
                        else:  # not_degrade
                            met = delta >= -criterion.minimum_change
                        if met and getattr(criterion, "baseline_value", None) is not None:
                            met = after_val is not None and after_val >= criterion.baseline_value
                        criteria_met.append(met)

                    structural_experiment_tracker.resolve(
                        exp.experiment_id, criteria_met, actual_values,
                    )
                    passed = all(criteria_met) if criteria_met else False
                    logger.info(
                        "Resolved structural experiment %s: %s",
                        exp.experiment_id, "PASSED" if passed else "FAILED",
                    )

                    # Update hypothesis lifecycle
                    if exp.hypothesis_id:
                        try:
                            hypothesis_library.record_outcome(exp.hypothesis_id, positive=passed)
                        except Exception:
                            logger.warning("Failed to record hypothesis outcome for %s", exp.hypothesis_id)
                except Exception:
                    logger.exception("Failed to evaluate structural experiment %s", exp.experiment_id)
        except Exception:
            logger.exception("Structural experiment check failed")

    # AuditTrailConsumer — persists SSE events to JSONL
    audit_consumer = AuditTrailConsumer(log_dir=db_path / "logs")


    def _bot_scope_key(bot_ids: list[str]) -> str:
        if not bot_ids:
            return "global"
        return f"bots:{','.join(sorted(bot_ids))}"

    async def _worker_job(scheduled_for: datetime | None = None) -> None:
        await worker.process_batch()

    async def _monitoring_job(scheduled_for: datetime | None = None) -> None:
        await monitoring_loop.run_all()

    async def _relay_job(scheduled_for: datetime | None = None) -> None:
        if vps_receiver is None:
            await _noop_relay()
            return
        await vps_receiver.poll()

    async def _stale_recovery_job(scheduled_for: datetime | None = None) -> None:
        await queue.recover_stale()

    async def _weekly_summary_trigger(scheduled_for: datetime | None = None) -> None:
        run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
        week_end = run_at.date()
        week_start = week_end - timedelta(days=6)
        await queue.enqueue(_build_scheduled_event(
            job_key="weekly_summary",
            scope_key="global",
            scheduled_for=run_at,
            event_type="weekly_summary_trigger",
            bot_id="system",
            payload={
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
            },
        ))

    async def _approval_expiry_job(scheduled_for: datetime | None = None) -> None:
        if approval_tracker is None:
            return
        await _expire_approvals_with_notification(
            approval_tracker,
            telegram_adapter,
            dispatcher,
            notification_prefs,
        )

    async def _pr_review_job(scheduled_for: datetime | None = None) -> None:
        await _check_pr_reviews()

    async def _deployment_check_job(scheduled_for: datetime | None = None) -> None:
        if deployment_monitor is None:
            return
        await handlers._check_deployments()

    async def _threshold_learning_job(scheduled_for: datetime | None = None) -> None:
        if threshold_learner is None:
            return
        await asyncio.to_thread(threshold_learner.learn_thresholds)

    async def _experiment_check_job(scheduled_for: datetime | None = None) -> None:
        await _check_experiments()

    scheduler_config = SchedulerConfig()
    scheduled_job_runner = ScheduledJobRunner(scheduled_run_store)
    scheduled_job_specs: list[ScheduledJobSpec]
    tracked_daily_fns: list[dict] | None = None
    tracked_morning_fns: list[dict] | None = None
    tracked_evening_fns: list[dict] | None = None
    tracked_wfo_fns: list[dict] | None = None
    tracked_daily_fn = None
    tracked_morning_fn = None
    tracked_evening_fn = None

    if config.bot_configs:
        from orchestrator.tz_utils import (
            bot_trading_date as _bot_trading_date,
            group_bots_by_analysis_time as _group_bots_by_analysis_time,
            market_close_utc as _market_close_utc,
        )

        tracked_daily_fns = []
        tracked_morning_fns = []
        tracked_evening_fns = []
        tracked_groups = _group_bots_by_analysis_time(config.bot_configs)

        for time_key, bot_list in tracked_groups.items():
            trigger_hour, trigger_minute = (int(value) for value in time_key.split(":"))
            scope_key = _bot_scope_key(bot_list)
            suffix = time_key.replace(":", "")
            tz_name = config.bot_configs[bot_list[0]].timezone
            cfg0 = config.bot_configs[bot_list[0]]
            close_utc = _market_close_utc(cfg0.timezone, cfg0.market_close_local)
            morning_utc = close_utc - timedelta(hours=9)
            evening_utc = close_utc + timedelta(hours=1)

            def _make_daily_trigger(bots: list[str], tz_name: str, scope_key: str):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
                    await queue.enqueue(_build_scheduled_event(
                        job_key="daily_analysis",
                        scope_key=scope_key,
                        scheduled_for=run_at,
                        event_type="daily_analysis_trigger",
                        bot_id="system",
                        payload={
                            "bots": bots,
                            "date": _bot_trading_date(tz_name, run_at),
                            "run_scope": scope_key,
                        },
                    ))

                return _trigger

            def _make_morning_trigger(bots: list[str]):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    await _morning_scan(bot_ids=bots, scheduled_for=scheduled_for)

                return _trigger

            def _make_evening_trigger(bots: list[str]):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    await _evening_report(bot_ids=bots, scheduled_for=scheduled_for)

                return _trigger

            tracked_daily_fns.append({
                "fn": _make_daily_trigger(bot_list, tz_name, scope_key),
                "hour": trigger_hour,
                "minute": trigger_minute,
                "name_suffix": suffix,
                "scope_key": scope_key,
            })
            tracked_morning_fns.append({
                "fn": _make_morning_trigger(bot_list),
                "hour": morning_utc.hour,
                "minute": morning_utc.minute,
                "name_suffix": suffix,
                "scope_key": scope_key,
            })
            tracked_evening_fns.append({
                "fn": _make_evening_trigger(bot_list),
                "hour": evening_utc.hour,
                "minute": evening_utc.minute,
                "name_suffix": suffix,
                "scope_key": scope_key,
            })
    else:
        scope_key = _bot_scope_key(config.bot_ids)

        async def _global_daily_trigger(scheduled_for: datetime | None = None) -> None:
            run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
            await queue.enqueue(_build_scheduled_event(
                job_key="daily_analysis",
                scope_key=scope_key,
                scheduled_for=run_at,
                event_type="daily_analysis_trigger",
                bot_id="system",
                payload={
                    "bots": config.bot_ids,
                    "date": run_at.strftime("%Y-%m-%d"),
                    "run_scope": scope_key,
                },
            ))

        async def _global_morning_trigger(scheduled_for: datetime | None = None) -> None:
            await _morning_scan(bot_ids=config.bot_ids or None, scheduled_for=scheduled_for)

        async def _global_evening_trigger(scheduled_for: datetime | None = None) -> None:
            await _evening_report(bot_ids=config.bot_ids or None, scheduled_for=scheduled_for)

        tracked_daily_fn = _global_daily_trigger
        tracked_morning_fn = _global_morning_trigger
        tracked_evening_fn = _global_evening_trigger

    if config.bot_ids:
        tracked_wfo_fns = []
        for bot_id in config.bot_ids:
            scope_key = f"bot:{bot_id}"

            def _make_wfo_trigger(bot_id: str, scope_key: str):
                async def _trigger(scheduled_for: datetime | None = None) -> None:
                    run_at = (scheduled_for or _utc_now()).astimezone(timezone.utc).replace(microsecond=0)
                    await queue.enqueue(_build_scheduled_event(
                        job_key="wfo",
                        scope_key=scope_key,
                        scheduled_for=run_at,
                        event_type="wfo_trigger",
                        bot_id=bot_id,
                        payload={
                            "bot_id": bot_id,
                            "data_end": run_at.date().isoformat(),
                        },
                    ))

                return _trigger

            tracked_wfo_fns.append({
                "fn": _make_wfo_trigger(bot_id, scope_key),
                "name_suffix": bot_id,
                "scope_key": scope_key,
            })

    scheduled_job_specs = build_scheduled_job_specs(
        config=scheduler_config,
        worker_fn=_worker_job,
        monitoring_fn=_monitoring_job,
        relay_fn=_relay_job,
        daily_analysis_fn=tracked_daily_fn,
        daily_analysis_fns=tracked_daily_fns,
        weekly_analysis_fn=_weekly_summary_trigger,
        wfo_fns=tracked_wfo_fns,
        stale_event_recovery_fn=_stale_recovery_job,
        morning_scan_fn=tracked_morning_fn,
        evening_report_fn=tracked_evening_fn,
        morning_scan_fns=tracked_morning_fns,
        evening_report_fns=tracked_evening_fns,
        outcome_measurement_fn=_measure_outcomes,
        memory_consolidation_fn=_consolidate_memory,
        transfer_outcome_fn=_measure_transfer_outcomes,
        approval_expiry_fn=_approval_expiry_job if approval_tracker else None,
        pr_review_check_fn=_pr_review_job if approval_tracker else None,
        deployment_check_fn=_deployment_check_job if deployment_monitor else None,
        threshold_learning_fn=_threshold_learning_job if threshold_learner else None,
        experiment_check_fn=_experiment_check_job if experiment_manager else None,
        reliability_verification_fn=_verify_reliability,
        discovery_fn=_discovery_analysis,
        learning_cycle_fn=_run_learning_cycle,
    )
    scheduler_jobs = job_specs_to_scheduler_jobs(scheduled_job_specs, scheduled_job_runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await queue.initialize()
            await registry.initialize()
            await scheduled_run_store.initialize()

            # Validate bot timezones (fail fast on invalid IANA zones)
            for bid, cfg in (config.bot_configs or {}).items():
                try:
                    from zoneinfo import ZoneInfo
                    ZoneInfo(cfg.timezone)
                except Exception:
                    raise ValueError(f"Invalid timezone '{cfg.timezone}' for bot '{bid}'")

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

            # Catch up on missed scheduled jobs (e.g., laptop was off)
            run_history_path = db_path / "data" / "run_history.jsonl"
            try:
                seeded_baseline = None
                store_was_empty = await scheduled_run_store.is_empty()
                if store_was_empty:
                    seeded_baseline = await bootstrap_run_store_from_history(
                        scheduled_run_store,
                        scheduled_job_specs,
                        run_history_path,
                    )
                if store_was_empty and await scheduled_run_store.get_baseline() is None:
                    await scheduled_run_store.set_baseline(seeded_baseline or _utc_now())

                catchup = StartupCatchup(
                    job_specs=scheduled_job_specs,
                    run_store=scheduled_run_store,
                )
                for occurrence in await catchup.build_plan(now=_utc_now()):
                    logger.info(
                        "Startup catch-up: running %s (%s @ %s)",
                        occurrence.spec.job_key,
                        occurrence.spec.scope_key,
                        occurrence.scheduled_for.isoformat(),
                    )
                    try:
                        await scheduled_job_runner.run(
                            occurrence.spec,
                            scheduled_for=occurrence.scheduled_for,
                        )
                    except Exception:
                        logger.warning(
                            "Scheduled catch-up failed for %s (%s)",
                            occurrence.spec.job_key,
                            occurrence.spec.scope_key,
                            exc_info=True,
                        )
            except Exception:
                logger.warning("Startup catch-up failed", exc_info=True)

            # Start scheduler (graceful degradation if it fails)
            try:
                scheduler = _create_scheduler(scheduler_jobs)
            except Exception:
                logger.error("Failed to create scheduler", exc_info=True)
                scheduler = None
            app.state.scheduler = scheduler

        except Exception as exc:
            logger.critical("Startup failed: %s", exc, exc_info=True)
            # Best-effort crash notification
            try:
                payload = NotificationPayload(
                    notification_type="system_alert",
                    priority=NotificationPriority.CRITICAL,
                    title="Trading Assistant Startup Failed",
                    body=f"Startup failure: {exc}",
                )
                await dispatcher.dispatch(payload, notification_prefs, 0)
            except Exception:
                pass
            raise

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
        await scheduled_run_store.close()

    app = FastAPI(title="Trading Assistant Orchestrator", lifespan=lifespan)
    app.state.start_time = datetime.now(timezone.utc)

    @app.middleware("http")
    async def _require_api_key(request: Request, call_next):
        """Protect the control plane when ORCHESTRATOR_API_KEY is configured."""
        required_key = config.orchestrator_api_key
        if not required_key or request.url.path == "/health":
            return await call_next(request)

        provided_key = request.headers.get("X-Api-Key", "")
        if provided_key != required_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        return await call_next(request)

    # Expose on app.state so test fixtures can manually initialize/close
    # (httpx ASGITransport does not trigger lifespan events)
    app.state.queue = queue
    app.state.registry = registry
    app.state.scheduled_run_store = scheduled_run_store
    app.state.scheduled_job_specs = scheduled_job_specs
    app.state.worker = worker
    app.state.event_stream = event_stream
    app.state.session_store = session_store
    app.state.subagent_mgr = subagent_mgr
    app.state.dispatcher = dispatcher
    app.state.agent_runner = agent_runner
    app.state.agent_preferences = agent_runner.get_preferences()
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
    app.state.repo_task_runner = repo_task_runner
    app.state.deployment_monitor = deployment_monitor
    app.state.threshold_learner = threshold_learner
    app.state.experiment_manager = experiment_manager
    app.state.experiment_config_gen = experiment_config_gen
    app.state.telegram_callback_router = callback_router
    app.state.telegram_renderer = telegram_renderer

    @app.get("/health")
    async def health():
        scheduler_ok = hasattr(app.state, "scheduler") and app.state.scheduler is not None
        return {
            "status": "ok" if scheduler_ok else "degraded",
            "scheduler": "running" if scheduler_ok else "unavailable",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

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
        event = _normalize_queue_event(event)
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

    @app.get("/agent/preferences")
    async def get_agent_preferences():
        return agent_runner.get_preferences_view().model_dump(mode="json")

    @app.put("/agent/preferences")
    async def update_agent_preferences(prefs: AgentPreferences):
        try:
            agent_runner.update_preferences(prefs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        app.state.agent_preferences = agent_runner.get_preferences()
        _save_agent_preferences(app.state.agent_preferences, agent_prefs_path)
        return agent_runner.get_preferences_view().model_dump(mode="json")

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
            raise HTTPException(status_code=400, detail="'text' field is required")
        import secrets

        event_time = _utc_now()
        feedback_event = {
            "event_type": "user_feedback",
            "bot_id": body.get("bot_id", "user"),
            "event_id": f"feedback-{event_time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(6)}",
            "payload": json.dumps({
                "text": text,
                "report_id": body.get("report_id", "unknown"),
            }),
            "exchange_timestamp": event_time.isoformat(),
            "received_at": event_time.isoformat(),
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

    @app.get("/learning/dashboard")
    async def learning_dashboard():
        """Learning system dashboard — ground truth, lessons, experiments, scorecard."""
        from skills.learning_ledger import LearningLedger
        from skills.suggestion_scorer import SuggestionScorer

        findings_dir = db_path / "memory" / "findings"
        ledger = LearningLedger(findings_dir)
        scorer = SuggestionScorer(findings_dir)

        trend = ledger.get_trend(weeks=12)
        lessons = ledger.get_lessons(weeks=4)
        latest = ledger.get_latest()
        scorecard = scorer.compute_scorecard()

        active_hypotheses = []
        try:
            records = hypothesis_library.get_active()
            active_hypotheses = [
                {
                    "hypothesis_id": getattr(h, "hypothesis_id", ""),
                    "title": getattr(h, "title", ""),
                    "category": getattr(h, "category", ""),
                    "effectiveness": getattr(h, "effectiveness", 0.0),
                }
                for h in records
            ]
        except Exception:
            pass

        active_experiments = []
        try:
            exps = structural_experiment_tracker.get_active_experiments()
            active_experiments = [
                {
                    "experiment_id": getattr(e, "experiment_id", ""),
                    "title": getattr(e, "title", ""),
                    "bot_id": getattr(e, "bot_id", ""),
                    "status": getattr(e, "status", ""),
                }
                for e in exps
            ]
        except Exception:
            pass

        prediction_accuracy = {}
        try:
            from analysis.context_builder import ContextBuilder
            ctx = ContextBuilder(db_path / "memory")
            pkg = ctx.base_package()
            prediction_accuracy = pkg.data.get("prediction_accuracy_by_metric", {})
        except Exception:
            pass

        return {
            "ground_truth_trend": trend,
            "recent_lessons": lessons,
            "active_hypotheses": active_hypotheses,
            "active_experiments": active_experiments,
            "category_scorecard": scorecard.model_dump(mode="json"),
            "prediction_accuracy": prediction_accuracy,
            "net_improvement": latest.net_improvement if latest else None,
            "latest_entry": latest.model_dump(mode="json") if latest else None,
        }

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

        # Extract APScheduler add_job kwargs before building trigger
        misfire_grace_time = job.pop("misfire_grace_time", None)
        coalesce = job.pop("coalesce", None)

        if trigger_type == "interval":
            trigger = IntervalTrigger(seconds=job.pop("seconds"))
        elif trigger_type == "cron":
            trigger = CronTrigger(**job)
            job = {}
        else:
            logger.warning("Unknown trigger type %s for job %s", trigger_type, name)
            continue

        add_kwargs: dict = {}
        if misfire_grace_time is not None:
            add_kwargs["misfire_grace_time"] = misfire_grace_time
        if coalesce is not None:
            add_kwargs["coalesce"] = coalesce

        scheduler.add_job(func, trigger, id=name, name=name, **add_kwargs, **job)

    scheduler.start()
    return scheduler


async def _noop_relay() -> None:
    """Placeholder relay poll — replace with actual relay client."""
    pass


# Default app instance for `uvicorn orchestrator.app:app`
app = create_app()
