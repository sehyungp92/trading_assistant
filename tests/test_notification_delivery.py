"""End-to-end notification delivery tests — verifies rendered text arrives at BaseChannel adapters."""
from __future__ import annotations

import pytest

from comms.base_channel import BaseChannel
from comms.dispatcher import NotificationDispatcher
from comms.renderer import PlainTextRenderer
from schemas.notifications import (
    ChannelConfig,
    NotificationChannel,
    NotificationPayload,
    NotificationPreferences,
    NotificationPriority,
)


class RecordingChannel(BaseChannel):
    """BaseChannel subclass that records what was sent."""

    def __init__(self) -> None:
        super().__init__()
        self.sent_args: list[tuple] = []

    async def _start(self) -> None:
        pass

    async def _stop(self) -> None:
        pass

    async def _send(self, *args, **kwargs) -> None:
        self.sent_args.append(args)


class TestNotificationDelivery:
    async def test_basechannel_receives_rendered_text(self):
        """BaseChannel adapter should receive rendered text, not raw NotificationPayload."""
        channel = RecordingChannel()
        await channel.start()

        dispatcher = NotificationDispatcher()
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, channel)
        dispatcher.register_renderer(NotificationChannel.TELEGRAM, PlainTextRenderer())

        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Test Alert",
            body="Something happened",
        )
        prefs = NotificationPreferences(
            channels=[ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True, chat_id="123")],
        )
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=12)

        assert len(results) == 1
        assert results[0].success is True
        # The channel should have received a rendered string, not a NotificationPayload
        assert len(channel.sent_args) == 1
        sent_text = channel.sent_args[0][0]
        assert isinstance(sent_text, str)
        assert "Test Alert" in sent_text
        assert "Something happened" in sent_text

    async def test_email_channel_receives_three_args(self):
        """Email channel should receive (chat_id, title, rendered_body)."""
        channel = RecordingChannel()
        await channel.start()

        dispatcher = NotificationDispatcher()
        dispatcher.register_adapter(NotificationChannel.EMAIL, channel)
        dispatcher.register_renderer(NotificationChannel.EMAIL, PlainTextRenderer())

        payload = NotificationPayload(
            notification_type="daily_report",
            priority=NotificationPriority.NORMAL,
            title="Daily Report",
            body="Report content here",
        )
        prefs = NotificationPreferences(
            channels=[ChannelConfig(channel=NotificationChannel.EMAIL, enabled=True, chat_id="user@example.com")],
        )
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=12)

        assert len(results) == 1
        assert results[0].success is True
        assert len(channel.sent_args) == 1
        args = channel.sent_args[0]
        assert args[0] == "user@example.com"  # chat_id as to-address
        assert args[1] == "Daily Report"  # title as subject
        assert "Report content here" in args[2]  # rendered body

    async def test_fallback_renderer_used_when_none_registered(self):
        """Without a registered renderer, dispatcher should use PlainTextRenderer fallback."""
        channel = RecordingChannel()
        await channel.start()

        dispatcher = NotificationDispatcher()
        dispatcher.register_adapter(NotificationChannel.DISCORD, channel)
        # No renderer registered — should use fallback

        payload = NotificationPayload(
            notification_type="weekly_summary",
            priority=NotificationPriority.NORMAL,
            title="Weekly Summary",
            body="Summary content",
        )
        prefs = NotificationPreferences(
            channels=[ChannelConfig(channel=NotificationChannel.DISCORD, enabled=True, chat_id="")],
        )
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=12)

        assert results[0].success is True
        sent_text = channel.sent_args[0][0]
        assert isinstance(sent_text, str)
        assert "Weekly Summary" in sent_text

    async def test_delivery_failure_logged(self):
        """Failed delivery should return DeliveryResult with success=False."""
        class FailingChannel(BaseChannel):
            async def _start(self) -> None:
                pass
            async def _stop(self) -> None:
                pass
            async def _send(self, *args, **kwargs) -> None:
                raise ConnectionError("Connection lost")

        channel = FailingChannel(max_retries=1)
        await channel.start()

        dispatcher = NotificationDispatcher()
        dispatcher.register_adapter(NotificationChannel.TELEGRAM, channel)
        dispatcher.register_renderer(NotificationChannel.TELEGRAM, PlainTextRenderer())

        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.HIGH,
            title="Test",
            body="Test body",
        )
        prefs = NotificationPreferences(
            channels=[ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True, chat_id="123")],
        )
        results = await dispatcher.dispatch(payload, prefs, current_hour_utc=12)

        assert len(results) == 1
        assert results[0].success is False
        assert "Connection lost" in results[0].error
