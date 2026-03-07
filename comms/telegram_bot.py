# comms/telegram_bot.py
"""Telegram bot adapter — wraps send/edit/pin operations with lifecycle and retry."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from comms.base_channel import BaseChannel

logger = logging.getLogger(__name__)


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
        self._callback_router = None
        self._polling_task: asyncio.Task | None = None

    def set_callback_router(self, router) -> None:
        """Connect a TelegramCallbackRouter to handle incoming callback queries."""
        self._callback_router = router

    async def _start(self) -> None:
        from telegram import Bot
        self._bot = Bot(token=self._config.token)

    async def _stop(self) -> None:
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        self._bot = None

    async def start_polling(self) -> None:
        """Start polling for incoming updates (callback queries and slash commands).

        Requires a callback router to be set via set_callback_router().
        """
        if self._callback_router is None:
            logger.warning("No callback router set — polling not started")
            return
        self._polling_task = asyncio.create_task(self._poll_loop())
        logger.info("Telegram update polling started")

    async def _poll_loop(self) -> None:
        """Long-poll for Telegram updates and dispatch to callback router."""
        offset = 0
        while True:
            try:
                updates = await self._bot.get_updates(
                    offset=offset, timeout=30,
                    allowed_updates=["callback_query", "message"],
                )
                for update in updates:
                    offset = update.update_id + 1
                    await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Telegram polling error — retrying in 5s")
                await asyncio.sleep(5)

    async def _handle_update(self, update) -> None:
        """Process a single Telegram update."""
        if update.callback_query is not None:
            query = update.callback_query
            callback_data = query.data or ""
            try:
                result = await self._callback_router.dispatch(callback_data)
                answer = result or "Done"
                await query.answer(text=answer[:200])
                # Send result as a reply if non-trivial
                if result:
                    try:
                        await self.send_message(result)
                    except Exception:
                        logger.warning("Failed to send callback result message")
            except Exception:
                logger.exception("Callback dispatch error for %s", callback_data)
                await query.answer(text="Error processing request")

        elif update.message is not None and update.message.text:
            text = update.message.text.strip()
            if text.startswith("/"):
                command = text.split()[0].lower()
                try:
                    result = await self._callback_router.dispatch_slash(command)
                    if result:
                        await self.send_message(result)
                except Exception:
                    logger.exception("Slash command error for %s", command)

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
