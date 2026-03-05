# comms/telegram_handlers.py
"""Telegram callback query router and slash command fallback."""
from __future__ import annotations

from typing import Callable, Awaitable

_SLASH_MAP: dict[str, str] = {
    "/daily": "cmd_daily",
    "/weekly": "cmd_weekly",
    "/botstatus": "cmd_bot_status",
    "/bot_status": "cmd_bot_status",
    "/topmissed": "cmd_top_missed",
    "/top_missed": "cmd_top_missed",
    "/openprs": "cmd_open_prs",
    "/open_prs": "cmd_open_prs",
    "/approve": "cmd_approve_all",
    "/settings": "cmd_settings",
}


class TelegramCallbackRouter:
    """Routes callback queries and slash commands to handler functions."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Awaitable[str | None]]] = {}

    @property
    def handlers(self) -> dict[str, Callable[..., Awaitable[str | None]]]:
        return self._handlers

    def register(self, callback_data: str, handler: Callable[..., Awaitable[str | None]]) -> None:
        self._handlers[callback_data] = handler

    async def dispatch(self, callback_data: str, context: dict | None = None) -> str | None:
        handler = self._handlers.get(callback_data)
        if handler is None:
            return None
        if context is not None:
            return await handler(context=context)
        return await handler()

    async def dispatch_slash(self, command: str, context: dict | None = None) -> str | None:
        if command == "/help":
            return self._build_help_text()
        callback_data = _SLASH_MAP.get(command)
        if callback_data is None:
            return None
        return await self.dispatch(callback_data, context=context)

    def _build_help_text(self) -> str:
        lines = ["Available commands:"]
        for slash, cb in sorted(_SLASH_MAP.items()):
            if cb in self._handlers:
                lines.append(f"  {slash}")
        return "\n".join(lines)
