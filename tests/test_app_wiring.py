# tests/test_app_wiring.py
"""Tests for app.py wiring — channel adapters, config, scanner, prefs persistence."""
import json

import pytest

from orchestrator.app import (
    _load_notification_prefs,
    _register_channel_adapters,
    _save_notification_prefs,
    create_app,
)
from orchestrator.config import AppConfig
from schemas.notifications import (
    ChannelConfig,
    NotificationChannel,
    NotificationPreferences,
)
from comms.dispatcher import NotificationDispatcher


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
