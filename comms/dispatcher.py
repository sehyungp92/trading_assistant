# comms/dispatcher.py
"""Notification dispatcher — routes payloads to channel adapters with retry support."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from comms.base_channel import BaseChannel
from schemas.notifications import (
    NotificationChannel,
    NotificationPayload,
    NotificationPreferences,
    ChannelConfig,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ChannelAdapter(Protocol):
    async def send(self, payload: NotificationPayload, channel_config: ChannelConfig) -> None: ...


@dataclass
class DeliveryResult:
    channel: NotificationChannel
    success: bool
    error: str = ""


@dataclass
class ChannelHealth:
    channel: NotificationChannel
    is_running: bool
    consecutive_failures: int


class NotificationDispatcher:
    """Routes notifications to channel adapters based on user preferences."""

    def __init__(self) -> None:
        self._adapters: dict[NotificationChannel, ChannelAdapter] = {}

    @property
    def adapters(self) -> dict[NotificationChannel, ChannelAdapter]:
        return self._adapters

    def register_adapter(self, channel: NotificationChannel, adapter: ChannelAdapter) -> None:
        self._adapters[channel] = adapter

    def get_channel_health(self) -> list[ChannelHealth]:
        """Report health status for all registered channels."""
        results: list[ChannelHealth] = []
        for channel, adapter in self._adapters.items():
            if isinstance(adapter, BaseChannel):
                results.append(ChannelHealth(
                    channel=channel,
                    is_running=adapter.is_running,
                    consecutive_failures=adapter.consecutive_failures,
                ))
            else:
                results.append(ChannelHealth(
                    channel=channel,
                    is_running=True,
                    consecutive_failures=0,
                ))
        return results

    async def dispatch(
        self,
        payload: NotificationPayload,
        prefs: NotificationPreferences,
        current_hour_utc: int,
    ) -> list[DeliveryResult]:
        eligible = prefs.get_channels_for_priority(payload.priority, current_hour_utc)
        results: list[DeliveryResult] = []
        for cfg in eligible:
            adapter = self._adapters.get(cfg.channel)
            if adapter is None:
                logger.warning("No adapter registered for %s", cfg.channel.value)
                results.append(DeliveryResult(
                    channel=cfg.channel, success=False, error="No adapter registered"
                ))
                continue
            try:
                if isinstance(adapter, BaseChannel):
                    await adapter.send_with_retry(payload, cfg)
                else:
                    await adapter.send(payload, cfg)
                results.append(DeliveryResult(channel=cfg.channel, success=True))
            except Exception as e:
                logger.exception("Failed to send via %s", cfg.channel.value)
                results.append(DeliveryResult(
                    channel=cfg.channel, success=False, error=str(e)
                ))
        return results
