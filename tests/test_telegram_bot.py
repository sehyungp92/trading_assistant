# tests/test_telegram_bot.py
"""Tests for Telegram bot adapter."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from comms.telegram_bot import TelegramBotAdapter, TelegramBotConfig
from comms.telegram_handlers import TelegramCallbackResponse


class TestTelegramBotConfig:
    def test_defaults(self):
        cfg = TelegramBotConfig(token="fake-token", chat_id="12345")
        assert cfg.token == "fake-token"
        assert cfg.chat_id == "12345"
        assert cfg.parse_mode == "MarkdownV2"


class TestTelegramBotAdapter:
    @pytest.fixture
    def mock_bot(self):
        bot = AsyncMock()
        bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
        bot.edit_message_text = AsyncMock(return_value=MagicMock(message_id=42))
        bot.pin_chat_message = AsyncMock()
        return bot

    @pytest.fixture
    def adapter(self, mock_bot):
        config = TelegramBotConfig(token="fake-token", chat_id="12345")
        a = TelegramBotAdapter(config)
        a._bot = mock_bot
        return a

    @pytest.mark.asyncio
    async def test_send_message(self, adapter, mock_bot):
        msg_id = await adapter.send_message("Hello, world!")
        assert msg_id == 42
        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == "12345"
        assert call_kwargs["text"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_send_message_with_keyboard(self, adapter, mock_bot):
        keyboard = [[{"text": "Daily", "callback_data": "cmd_daily"}]]
        msg_id = await adapter.send_message("Panel", keyboard=keyboard)
        assert msg_id == 42
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert "reply_markup" in call_kwargs

    @pytest.mark.asyncio
    async def test_edit_message(self, adapter, mock_bot):
        await adapter.edit_message(42, "Updated text")
        mock_bot.edit_message_text.assert_called_once()
        call_kwargs = mock_bot.edit_message_text.call_args.kwargs
        assert call_kwargs["message_id"] == 42
        assert call_kwargs["text"] == "Updated text"

    @pytest.mark.asyncio
    async def test_edit_message_with_keyboard(self, adapter, mock_bot):
        keyboard = [[{"text": "Weekly", "callback_data": "cmd_weekly"}]]
        await adapter.edit_message(42, "Updated", keyboard=keyboard)
        call_kwargs = mock_bot.edit_message_text.call_args.kwargs
        assert "reply_markup" in call_kwargs

    @pytest.mark.asyncio
    async def test_pin_message(self, adapter, mock_bot):
        await adapter.pin_message(42)
        mock_bot.pin_chat_message.assert_called_once_with(
            chat_id="12345", message_id=42, disable_notification=True
        )

    @pytest.mark.asyncio
    async def test_send_and_pin(self, adapter, mock_bot):
        msg_id = await adapter.send_and_pin("Important message")
        assert msg_id == 42
        mock_bot.send_message.assert_called_once()
        mock_bot.pin_chat_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_response_can_edit_existing_message(self, adapter):
        router = AsyncMock()
        router.dispatch = AsyncMock(return_value=TelegramCallbackResponse(
            text="Updated settings",
            keyboard=[[{"text": "Back", "callback_data": "agent_settings_home"}]],
            answer="Updated",
            edit_message=True,
        ))
        adapter._callback_router = router
        adapter.edit_message = AsyncMock()

        query = AsyncMock()
        query.data = "agent_settings_home"
        query.answer = AsyncMock()
        query.message = MagicMock(message_id=77)
        update = MagicMock(callback_query=query, message=None)

        await adapter._handle_update(update)

        query.answer.assert_called_once_with(text="Updated")
        adapter.edit_message.assert_called_once_with(
            77,
            "Updated settings",
            keyboard=[[{"text": "Back", "callback_data": "agent_settings_home"}]],
        )

    @pytest.mark.asyncio
    async def test_slash_command_sends_keyboard_when_provided(self, adapter):
        router = AsyncMock()
        router.dispatch_slash = AsyncMock(return_value=TelegramCallbackResponse(
            text="Settings",
            keyboard=[[{"text": "Global", "callback_data": "agent_settings_scope_global"}]],
        ))
        adapter._callback_router = router
        adapter.send_message = AsyncMock(return_value=42)

        message = MagicMock()
        message.text = "/settings"
        update = MagicMock(callback_query=None, message=message)

        await adapter._handle_update(update)

        router.dispatch_slash.assert_called_once_with("/settings")
        adapter.send_message.assert_called_once_with(
            "Settings",
            keyboard=[[{"text": "Global", "callback_data": "agent_settings_scope_global"}]],
        )
