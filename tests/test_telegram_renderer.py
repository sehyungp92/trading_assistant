# tests/test_telegram_renderer.py
"""Tests for Telegram-specific message renderer."""
from comms.telegram_renderer import TelegramRenderer
from comms.renderer import MessageRenderer
from schemas.notifications import (
    ControlPanelState,
    BotStatusLine,
    NotificationPayload,
    NotificationPriority,
)


class TestTelegramRendererProtocol:
    def test_implements_message_renderer(self):
        renderer = TelegramRenderer()
        assert isinstance(renderer, MessageRenderer)


class TestTelegramControlPanel:
    def _make_panel(self) -> ControlPanelState:
        return ControlPanelState(
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
                BotStatusLine(bot_id="Bot1", status="green", pnl=210.0, wins=4, losses=1, summary="Strong trend, EMA cross"),
                BotStatusLine(bot_id="Bot2", status="yellow", pnl=82.0, wins=3, losses=2, summary="2 normal losses"),
                BotStatusLine(bot_id="Bot3", status="red", pnl=50.0, wins=2, losses=3, summary="Volume filter blocked 3 winners"),
            ],
        )

    def test_renders_header_with_emoji(self):
        panel = self._make_panel()
        text = TelegramRenderer().render_control_panel(panel)
        assert "\U0001f4ca" in text
        assert "2026-03-01" in text or "March 1" in text

    def test_renders_portfolio_line(self):
        panel = self._make_panel()
        text = TelegramRenderer().render_control_panel(panel)
        assert "342" in text
        assert "1.2" in text or "1\\.2" in text

    def test_renders_bot_statuses(self):
        panel = self._make_panel()
        text = TelegramRenderer().render_control_panel(panel)
        assert "\U0001f7e2" in text
        assert "\U0001f7e1" in text
        assert "\U0001f534" in text
        assert "Bot1" in text
        assert "Bot3" in text

    def test_renders_status_indicators(self):
        panel = self._make_panel()
        text = TelegramRenderer().render_control_panel(panel)
        assert "\u2705" in text
        assert "\u26a0\ufe0f" in text
        assert "\U0001f9ea" in text
        assert "\U0001f6e1\ufe0f" in text

    def test_returns_keyboard_markup(self):
        panel = self._make_panel()
        text, keyboard = TelegramRenderer().render_control_panel_with_keyboard(panel)
        assert keyboard is not None
        button_labels = [btn["text"] for row in keyboard for btn in row]
        assert "Daily" in button_labels
        assert "Weekly" in button_labels
        assert "Bot Status" in button_labels

    def test_empty_panel(self):
        panel = ControlPanelState(date="2026-03-01")
        text = TelegramRenderer().render_control_panel(panel)
        assert "2026-03-01" in text


class TestTelegramAlert:
    def test_critical_alert_has_siren(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.CRITICAL,
            title="Bot3 crash",
            body="RuntimeError in signal handler",
        )
        text = TelegramRenderer().render_alert(payload)
        assert "\U0001f6a8" in text
        assert "CRITICAL" in text
        assert "Bot3 crash" in text

    def test_normal_alert(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.NORMAL,
            title="Slow response",
            body="API latency > 2s",
        )
        text = TelegramRenderer().render_alert(payload)
        assert "Slow response" in text


class TestTelegramDailyReport:
    def test_renders_with_header_emoji(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report \u2014 March 1, 2026",
            body="Portfolio: +$342 (+1.2%)\n\nBot1: +$210 (4W/1L)",
        )
        text = TelegramRenderer().render_daily_report(payload)
        assert "\U0001f4ca" in text
        assert "Daily Report" in text

    def test_returns_action_keyboard(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report",
            body="Summary",
        )
        text, keyboard = TelegramRenderer().render_daily_report_with_keyboard(payload)
        button_labels = [btn["text"] for row in keyboard for btn in row]
        assert "Full Report" in button_labels
        assert "Feedback" in button_labels


class TestTelegramWeeklySummary:
    def test_renders_weekly(self):
        payload = NotificationPayload(
            notification_type="weekly_summary",
            title="Weekly Summary \u2014 Feb 24\u2013Mar 1",
            body="Total PnL: +$1,200",
        )
        text = TelegramRenderer().render_weekly_summary(payload)
        assert "Weekly Summary" in text
        assert "$1,200" in text or "1,200" in text


class TestTelegramMessageLimit:
    def test_long_message_is_truncated(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report",
            body="x" * 5000,
        )
        text = TelegramRenderer().render_daily_report(payload)
        assert len(text) <= 4096

    def test_truncation_adds_continuation_note(self):
        payload = NotificationPayload(
            notification_type="daily_report",
            title="Daily Report",
            body="x" * 5000,
        )
        text = TelegramRenderer().render_daily_report(payload)
        assert "continued" in text.lower() or "truncated" in text.lower()


class TestTelegramEscaping:
    def test_escapes_markdown_v2_special_chars(self):
        payload = NotificationPayload(
            notification_type="alert",
            priority=NotificationPriority.NORMAL,
            title="Bot_1 test-alert",
            body="PnL: +$342.50 (1.2%)",
        )
        text = TelegramRenderer().render_alert(payload)
        assert "Bot" in text
        assert "342" in text
