# tests/test_notification_schemas.py
"""Tests for notification schemas."""
from datetime import datetime, timezone

from schemas.notifications import (
    NotificationChannel,
    NotificationPriority,
    NotificationPreferences,
    ChannelConfig,
    ControlPanelState,
    NotificationPayload,
    BotStatusLine,
)


class TestNotificationChannel:
    def test_all_channels_exist(self):
        assert NotificationChannel.TELEGRAM == "telegram"
        assert NotificationChannel.DISCORD == "discord"
        assert NotificationChannel.EMAIL == "email"


class TestNotificationPriority:
    def test_all_priorities_exist(self):
        assert NotificationPriority.CRITICAL == "critical"
        assert NotificationPriority.HIGH == "high"
        assert NotificationPriority.NORMAL == "normal"
        assert NotificationPriority.LOW == "low"

    def test_ordering(self):
        ordered = sorted(NotificationPriority, key=lambda p: p.rank, reverse=True)
        assert ordered[0] == NotificationPriority.CRITICAL
        assert ordered[-1] == NotificationPriority.LOW

    def test_critical_bypasses_quiet_hours(self):
        assert NotificationPriority.CRITICAL.bypasses_quiet_hours is True
        assert NotificationPriority.HIGH.bypasses_quiet_hours is False
        assert NotificationPriority.NORMAL.bypasses_quiet_hours is False
        assert NotificationPriority.LOW.bypasses_quiet_hours is False


class TestChannelConfig:
    def test_defaults(self):
        cfg = ChannelConfig(channel=NotificationChannel.TELEGRAM)
        assert cfg.enabled is True
        assert cfg.chat_id == ""
        assert cfg.quiet_hours_start is None
        assert cfg.quiet_hours_end is None

    def test_is_quiet_during_quiet_hours(self):
        cfg = ChannelConfig(
            channel=NotificationChannel.TELEGRAM,
            quiet_hours_start=22,
            quiet_hours_end=8,
        )
        assert cfg.is_quiet_at(23) is True
        assert cfg.is_quiet_at(3) is True
        assert cfg.is_quiet_at(12) is False
        assert cfg.is_quiet_at(8) is False

    def test_no_quiet_hours(self):
        cfg = ChannelConfig(channel=NotificationChannel.TELEGRAM)
        assert cfg.is_quiet_at(3) is False


class TestNotificationPreferences:
    def test_default_channels(self):
        prefs = NotificationPreferences()
        assert len(prefs.channels) == 0

    def test_add_channel(self):
        prefs = NotificationPreferences(
            channels=[
                ChannelConfig(channel=NotificationChannel.TELEGRAM, chat_id="12345"),
            ]
        )
        assert len(prefs.channels) == 1
        assert prefs.channels[0].chat_id == "12345"

    def test_get_active_channels(self):
        prefs = NotificationPreferences(
            channels=[
                ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True),
                ChannelConfig(channel=NotificationChannel.DISCORD, enabled=False),
                ChannelConfig(channel=NotificationChannel.EMAIL, enabled=True),
            ]
        )
        active = prefs.get_active_channels()
        assert len(active) == 2
        assert active[0].channel == NotificationChannel.TELEGRAM
        assert active[1].channel == NotificationChannel.EMAIL

    def test_get_channels_for_priority_respects_quiet_hours(self):
        prefs = NotificationPreferences(
            channels=[
                ChannelConfig(
                    channel=NotificationChannel.TELEGRAM,
                    quiet_hours_start=22,
                    quiet_hours_end=8,
                ),
            ]
        )
        channels = prefs.get_channels_for_priority(NotificationPriority.NORMAL, current_hour_utc=3)
        assert len(channels) == 0
        channels = prefs.get_channels_for_priority(NotificationPriority.CRITICAL, current_hour_utc=3)
        assert len(channels) == 1


class TestBotStatusLine:
    def test_status_emoji(self):
        green = BotStatusLine(bot_id="bot1", status="green", pnl=210.0, wins=4, losses=1, summary="Strong trend")
        assert green.status_emoji == "🟢"
        yellow = BotStatusLine(bot_id="bot2", status="yellow", pnl=82.0, wins=3, losses=2, summary="Normal losses")
        assert yellow.status_emoji == "🟡"
        red = BotStatusLine(bot_id="bot3", status="red", pnl=50.0, wins=2, losses=3, summary="Filter issues")
        assert red.status_emoji == "🔴"


class TestControlPanelState:
    def test_full_panel(self):
        panel = ControlPanelState(
            date="2026-03-01",
            portfolio_pnl=342.0,
            portfolio_pnl_pct=1.2,
            drawdown_pct=-0.3,
            exposure_pct=47.0,
            daily_report_ready=True,
            alert_count=1,
            alert_summary="Bot3 volume filter",
            wfo_status="Bot2 running (est. 45min)",
            pending_pr_count=0,
            risk_status="OK",
            risk_detail="concentration: 35/100",
            bot_statuses=[
                BotStatusLine(bot_id="bot1", status="green", pnl=210.0, wins=4, losses=1, summary="Strong trend"),
            ],
        )
        assert panel.date == "2026-03-01"
        assert panel.portfolio_pnl == 342.0
        assert panel.daily_report_ready is True
        assert panel.alert_count == 1
        assert len(panel.bot_statuses) == 1


class TestNotificationPayload:
    def test_minimal_payload(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            priority=NotificationPriority.NORMAL,
            title="Daily Report — March 1, 2026",
            body="Portfolio: +$342 (+1.2%)",
        )
        assert payload.notification_type == "daily_report"
        assert payload.priority == NotificationPriority.NORMAL
        assert payload.attachments == []

    def test_payload_with_data(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Bot3 crash",
            body="RuntimeError in signal handler",
            data={"bot_id": "bot3", "error_type": "RuntimeError"},
        )
        assert payload.data["bot_id"] == "bot3"
        assert payload.priority == NotificationPriority.CRITICAL

    def test_payload_with_attachments(self):
        payload = NotificationPayload(
            notification_type="weekly_digest",
            priority=NotificationPriority.LOW,
            title="Weekly Digest",
            body="Summary",
            attachments=["reports/weekly_2026-03-01.md", "reports/wfo_bot2.md"],
        )
        assert len(payload.attachments) == 2
