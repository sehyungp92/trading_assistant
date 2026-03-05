# comms/discord_bot.py
"""Discord bot adapter — wraps send/pin/thread operations with lifecycle and retry."""
from __future__ import annotations

from dataclasses import dataclass

from comms.base_channel import BaseChannel


@dataclass
class DiscordBotConfig:
    token: str
    channel_id: int


class DiscordBotAdapter(BaseChannel):
    """Async adapter for Discord bot operations."""

    def __init__(self, config: DiscordBotConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._channel: object | None = None
        self._threads: dict[int, object] = {}

    async def _start(self) -> None:
        import discord
        client = discord.Client(intents=discord.Intents.default())

    async def _stop(self) -> None:
        self._channel = None
        self._threads.clear()

    async def _send(self, content: str) -> int:
        return await self.send_message(content)

    async def start(self) -> None:
        await super().start()

    async def send_message(self, content: str) -> int:
        msg = await self._channel.send(content=content)
        return msg.id

    async def send_embed(self, embed_dict: dict) -> int:
        try:
            import discord
            embed = discord.Embed.from_dict(embed_dict)
        except ImportError:
            embed = embed_dict
        msg = await self._channel.send(embed=embed)
        return msg.id

    async def pin_message(self, message_id: int) -> None:
        msg = await self._channel.fetch_message(message_id)
        await msg.pin()

    async def create_thread(self, message_id: int, name: str) -> int:
        msg = await self._channel.fetch_message(message_id)
        thread = await msg.create_thread(name=name)
        self._threads[thread.id] = thread
        return thread.id

    async def send_to_thread(self, thread_id: int, content: str) -> int:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise ValueError(f"Thread {thread_id} not found. Call create_thread first.")
        msg = await thread.send(content=content)
        return msg.id
