# tests/test_app_wiring.py
"""Tests for app.py wiring — channel adapters, config, scanner, prefs persistence."""
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.app import (
    _load_agent_preferences,
    _load_notification_prefs,
    _register_channel_adapters,
    _save_agent_preferences,
    _save_notification_prefs,
    create_app,
)
from orchestrator.config import AppConfig
from schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
)
from schemas.notifications import (
    ChannelConfig,
    NotificationChannel,
    NotificationPreferences,
)
from comms.dispatcher import NotificationDispatcher


class TestAppConfigFromEnv:
    def test_command_args_parse_from_env(self):
        env = {
            "CLAUDE_COMMAND_ARGS": '["--","claude"]',
            "CODEX_COMMAND_ARGS": '["--","codex"]',
        }

        with patch.dict(os.environ, env, clear=False):
            config = AppConfig.from_env()

        assert config.claude_command_args == ["--", "claude"]
        assert config.codex_command_args == ["--", "codex"]

    def test_invalid_command_args_env_falls_back_to_empty_list(self):
        env = {
            "CLAUDE_COMMAND_ARGS": '{"not":"a-list"}',
            "CODEX_COMMAND_ARGS": "not-json",
        }

        with patch.dict(os.environ, env, clear=False):
            config = AppConfig.from_env()

        assert config.claude_command_args == []
        assert config.codex_command_args == []

    def test_loads_dotenv_defaults_when_process_env_missing(self, tmp_path):
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(
            "BOT_IDS=dotenv_bot\nCLAUDE_COMMAND=from-dotenv\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env(dotenv_path=dotenv_path)

        assert config.bot_ids == ["dotenv_bot"]
        assert config.claude_command == "from-dotenv"

    def test_process_env_overrides_dotenv(self, tmp_path):
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text("BOT_IDS=dotenv_bot\n", encoding="utf-8")

        with patch.dict(os.environ, {"BOT_IDS": "env_bot"}, clear=True):
            config = AppConfig.from_env(dotenv_path=dotenv_path)

        assert config.bot_ids == ["env_bot"]


class TestChannelAdapterRegistration:
    def test_telegram_registered_when_token_set(self):
        config = AppConfig(telegram_bot_token="fake-token", telegram_chat_id="12345")
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert NotificationChannel.TELEGRAM in dispatcher.adapters
        assert len(adapters) == 1

    def test_discord_registered_when_token_set(self):
        config = AppConfig(discord_bot_token="fake-token", discord_channel_id="99999")
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert NotificationChannel.DISCORD in dispatcher.adapters
        assert len(adapters) == 1

    def test_email_registered_when_smtp_set(self):
        config = AppConfig(
            smtp_host="smtp.test.com", smtp_user="user", smtp_pass="pass",
            email_from="from@test.com",
        )
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert NotificationChannel.EMAIL in dispatcher.adapters
        assert len(adapters) == 1

    def test_no_adapters_when_no_config(self):
        config = AppConfig()
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert len(dispatcher.adapters) == 0
        assert len(adapters) == 0

    def test_all_adapters_registered(self):
        config = AppConfig(
            telegram_bot_token="tg-token", telegram_chat_id="123",
            discord_bot_token="dc-token", discord_channel_id="456",
            smtp_host="smtp.test.com", smtp_user="user", smtp_pass="pass",
            email_from="from@test.com",
        )
        dispatcher = NotificationDispatcher()
        adapters = _register_channel_adapters(config, dispatcher)
        assert len(adapters) == 3
        assert NotificationChannel.TELEGRAM in dispatcher.adapters
        assert NotificationChannel.DISCORD in dispatcher.adapters
        assert NotificationChannel.EMAIL in dispatcher.adapters


class TestNotificationPreferencesPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        prefs_path = tmp_path / "data" / "notification_prefs.json"
        prefs = NotificationPreferences(channels=[
            ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True, chat_id="123"),
        ])
        _save_notification_prefs(prefs, prefs_path)
        assert prefs_path.exists()
        loaded = _load_notification_prefs(prefs_path)
        assert len(loaded.channels) == 1
        assert loaded.channels[0].channel == NotificationChannel.TELEGRAM
        assert loaded.channels[0].chat_id == "123"

    def test_load_returns_defaults_when_file_missing(self, tmp_path):
        prefs_path = tmp_path / "nonexistent.json"
        loaded = _load_notification_prefs(prefs_path)
        assert loaded.channels == []

    def test_load_returns_defaults_on_corrupt_file(self, tmp_path):
        prefs_path = tmp_path / "prefs.json"
        prefs_path.write_text("not valid json{{{", encoding="utf-8")
        loaded = _load_notification_prefs(prefs_path)
        assert loaded.channels == []

    def test_save_creates_parent_dirs(self, tmp_path):
        prefs_path = tmp_path / "deep" / "nested" / "prefs.json"
        prefs = NotificationPreferences()
        _save_notification_prefs(prefs, prefs_path)
        assert prefs_path.exists()


class TestAgentPreferencesPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        prefs_path = tmp_path / "data" / "agent_preferences.json"
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4"),
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(
                    provider=AgentProvider.OPENROUTER,
                    model="minimax/minimax-m2.5",
                ),
            },
        )

        _save_agent_preferences(prefs, prefs_path)
        loaded = _load_agent_preferences(prefs_path, AppConfig())

        assert loaded.default.provider == AgentProvider.CODEX_PRO
        assert loaded.default.model == "gpt-5.4"
        assert loaded.overrides[AgentWorkflow.TRIAGE].provider == AgentProvider.OPENROUTER

    def test_load_seeds_from_env_when_file_missing(self, tmp_path):
        prefs_path = tmp_path / "missing.json"
        config = AppConfig(
            agent_default_provider="codex_pro",
            daily_agent_provider="openrouter",
            daily_agent_model="minimax/minimax-m2.5",
        )

        loaded = _load_agent_preferences(prefs_path, config)

        assert loaded.default.provider == AgentProvider.CODEX_PRO
        assert loaded.overrides[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.OPENROUTER

    def test_load_falls_back_to_seeded_defaults_on_corrupt_file(self, tmp_path):
        prefs_path = tmp_path / "prefs.json"
        prefs_path.write_text("not valid json", encoding="utf-8")
        config = AppConfig(agent_default_provider="claude_max", weekly_agent_provider="codex_pro")

        loaded = _load_agent_preferences(prefs_path, config)

        assert loaded.default.provider == AgentProvider.CLAUDE_MAX
        assert loaded.overrides[AgentWorkflow.WEEKLY_ANALYSIS].provider == AgentProvider.CODEX_PRO


class TestCreateAppWithConfig:
    def test_bot_ids_passed_to_handlers(self, tmp_path):
        config = AppConfig(bot_ids=["bot_x", "bot_y"])
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.handlers._bots == ["bot_x", "bot_y"]

    def test_empty_config_still_creates_app(self, tmp_path):
        config = AppConfig()
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.handlers._bots == []
        assert len(app.state.dispatcher.adapters) == 0

    def test_config_exposed_on_app_state(self, tmp_path):
        config = AppConfig(bot_ids=["bot_z"])
        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.config.bot_ids == ["bot_z"]

    def test_agent_runner_shares_event_stream(self, tmp_path):
        app = create_app(db_dir=str(tmp_path), config=AppConfig())
        assert app.state.agent_runner._event_stream is app.state.event_stream

    def test_agent_runner_receives_launcher_args(self, tmp_path):
        config = AppConfig(
            claude_command="wsl.exe",
            claude_command_args=["--", "claude"],
            codex_command="wsl.exe",
            codex_command_args=["--", "codex"],
        )

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.agent_runner._claude_command == "wsl.exe"
        assert app.state.agent_runner._claude_command_args == ["--", "claude"]
        assert app.state.agent_runner._codex_command == "wsl.exe"
        assert app.state.agent_runner._codex_command_args == ["--", "codex"]

    def test_telegram_adapter_wired(self, tmp_path):
        config = AppConfig(telegram_bot_token="token", telegram_chat_id="123")
        app = create_app(db_dir=str(tmp_path), config=config)
        assert NotificationChannel.TELEGRAM in app.state.dispatcher.adapters

    def test_prefs_loaded_from_disk(self, tmp_path):
        # Pre-create a prefs file
        prefs_path = tmp_path / "data" / "notification_prefs.json"
        prefs_path.parent.mkdir(parents=True)
        prefs_path.write_text(json.dumps({
            "channels": [{"channel": "telegram", "enabled": True, "chat_id": "saved-123"}]
        }), encoding="utf-8")

        config = AppConfig()
        app = create_app(db_dir=str(tmp_path), config=config)
        assert len(app.state.notification_preferences.channels) == 1
        assert app.state.notification_preferences.channels[0].chat_id == "saved-123"

    def test_agent_prefs_loaded_from_disk(self, tmp_path):
        prefs_path = tmp_path / "data" / "agent_preferences.json"
        prefs_path.parent.mkdir(parents=True)
        prefs_path.write_text(json.dumps({
            "default": {"provider": "openrouter", "model": "minimax/minimax-m2.5"},
            "overrides": {
                "triage": {"provider": "codex_pro", "model": "gpt-5.4"},
            },
        }), encoding="utf-8")

        app = create_app(db_dir=str(tmp_path), config=AppConfig())

        assert app.state.agent_preferences.default.provider == AgentProvider.OPENROUTER
        assert app.state.agent_preferences.overrides[AgentWorkflow.TRIAGE].provider == AgentProvider.CODEX_PRO

    def test_agent_prefs_seed_from_env_when_missing(self, tmp_path):
        config = AppConfig(
            agent_default_provider="zai_coding_plan",
            agent_default_model="glm-5",
            wfo_agent_provider="claude_max",
            wfo_agent_model="opus",
        )

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.agent_preferences.default.provider == AgentProvider.ZAI_CODING_PLAN
        assert app.state.agent_preferences.default.model == "glm-5"
        assert app.state.agent_preferences.overrides[AgentWorkflow.WFO].provider == AgentProvider.CLAUDE_MAX
        assert app.state.agent_preferences.overrides[AgentWorkflow.WFO].model == "opus"

    def test_telegram_settings_router_registered_when_telegram_enabled(self, tmp_path):
        config = AppConfig(telegram_bot_token="token", telegram_chat_id="123")

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.telegram_callback_router is not None
        assert "cmd_settings" in app.state.telegram_callback_router.handlers
        assert "agent_settings_scope_" in app.state.telegram_callback_router.handlers

    def test_autonomous_app_exposes_repo_task_runner(self, tmp_path):
        config_dir = tmp_path / "bot_configs"
        config_dir.mkdir()
        (config_dir / "bot1.yaml").write_text(
            "bot_id: bot1\nrepo_dir: .\nparameters: []\n",
            encoding="utf-8",
        )
        config = AppConfig(
            autonomous_enabled=True,
            bot_config_dir=str(config_dir),
        )

        app = create_app(db_dir=str(tmp_path), config=config)

        assert app.state.repo_task_runner is not None
        assert app.state.approval_handler._repo_task_runner is app.state.repo_task_runner

    @pytest.mark.asyncio
    async def test_lifespan_drains_relay_and_runs_startup_catchup(self, tmp_path):
        old_date = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        run_history_path = tmp_path / "data" / "run_history.jsonl"
        run_history_path.parent.mkdir(parents=True, exist_ok=True)
        run_history_path.write_text(json.dumps({
            "run_id": f"daily-{old_date}",
            "agent_type": "daily_analysis",
            "status": "completed",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }) + "\n", encoding="utf-8")

        config = AppConfig(bot_ids=["bot1"], relay_url="https://relay.example")

        with (
            patch("orchestrator.app.VPSReceiver") as MockReceiver,
            patch("orchestrator.app._create_scheduler", return_value=None),
        ):
            receiver = MockReceiver.return_value
            receiver.drain = AsyncMock()
            receiver.poll = AsyncMock()

            app = create_app(db_dir=str(tmp_path), config=config)

            async with app.router.lifespan_context(app):
                receiver.drain.assert_awaited_once()
                pending = await app.state.queue.peek(limit=20)

            assert any(event["event_type"] == "daily_analysis_trigger" for event in pending)
