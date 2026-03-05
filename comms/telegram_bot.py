# comms/telegram_bot.py
"""Telegram bot adapter — wraps send/edit/pin operations with lifecycle and retry."""
from __future__ import annotations

from dataclasses import dataclass

from comms.base_channel import BaseChannel


@dataclass
class TelegramBotConfig:
    token: str
    chat_id: str
    parse_mode: str = "MarkdownV2"


def _build_inline_keyboard(keyboard: list[list[dict]]) -> dict:
    return {
        "inline_keyboard": [
            [{"text": btn["text"], "callback_data": btn["callback_data"]} for btn in row]
            for row in keyboard
        ]
    }


class TelegramBotAdapter(BaseChannel):
    """Async adapter for Telegram Bot API operations."""

    def __init__(self, config: TelegramBotConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._bot: object | None = None

    async def _start(self) -> None:
        from telegram import Bot
        self._bot = Bot(token=self._config.token)

    async def _stop(self) -> None:
        self._bot = None

    async def _send(self, text: str, keyboard: list[list[dict]] | None = None) -> int:
        return await self.send_message(text, keyboard=keyboard)

    async def start(self) -> None:
        await super().start()

    async def send_message(self, text: str, keyboard: list[list[dict]] | None = None) -> int:
        kwargs: dict = {"chat_id": self._config.chat_id, "text": text}
        if keyboard:
            kwargs["reply_markup"] = _build_inline_keyboard(keyboard)
        result = await self._bot.send_message(**kwargs)
        return result.message_id

    async def edit_message(self, message_id: int, text: str, keyboard: list[list[dict]] | None = None) -> None:
        kwargs: dict = {"chat_id": self._config.chat_id, "message_id": message_id, "text": text}
        if keyboard:
            kwargs["reply_markup"] = _build_inline_keyboard(keyboard)
        await self._bot.edit_message_text(**kwargs)

    async def pin_message(self, message_id: int) -> None:
        await self._bot.pin_chat_message(
            chat_id=self._config.chat_id, message_id=message_id, disable_notification=True
        )

    async def send_and_pin(self, text: str, keyboard: list[list[dict]] | None = None) -> int:
        msg_id = await self.send_message(text, keyboard=keyboard)
        await self.pin_message(msg_id)
        return msg_id
