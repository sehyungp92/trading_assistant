# comms/telegram_renderer.py
"""Telegram message renderer — MarkdownV2 formatting with inline keyboards."""
from __future__ import annotations

import re

from schemas.notifications import (
    ControlPanelState,
    NotificationPayload,
    NotificationPriority,
)

_TELEGRAM_MAX_LENGTH = 4096
_TRUNCATION_NOTE = "\n\n... (truncated, use /full for complete report)"


def _escape_md2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters."""
    return re.sub(r"([_*\[\]()~`>#+\-=|{}.!])", r"\\\1", text)


def _truncate(text: str, max_length: int = _TELEGRAM_MAX_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - len(_TRUNCATION_NOTE)] + _TRUNCATION_NOTE


_CONTROL_PANEL_KEYBOARD = [
    [
        {"text": "Daily", "callback_data": "cmd_daily"},
        {"text": "Weekly", "callback_data": "cmd_weekly"},
        {"text": "Bot Status", "callback_data": "cmd_bot_status"},
    ],
    [
        {"text": "Top Missed", "callback_data": "cmd_top_missed"},
        {"text": "Open PRs", "callback_data": "cmd_open_prs"},
        {"text": "Approve All", "callback_data": "cmd_approve_all"},
    ],
    [
        {"text": "Settings", "callback_data": "cmd_settings"},
    ],
]

_DAILY_REPORT_KEYBOARD = [
    [
        {"text": "Full Report", "callback_data": "cmd_full_report"},
        {"text": "Feedback", "callback_data": "cmd_feedback"},
    ],
    [
        {"text": "Bot Detail", "callback_data": "cmd_bot_detail"},
        {"text": "Approve Change", "callback_data": "cmd_approve_change"},
    ],
]


class TelegramRenderer:
    """Renders notifications for Telegram with MarkdownV2 formatting."""

    def render(self, payload: NotificationPayload) -> str:
        dispatch = {
            "alert": self.render_alert,
            "daily_report": self.render_daily_report,
            "weekly_summary": self.render_weekly_summary,
        }
        handler = dispatch.get(payload.notification_type)
        if handler:
            return handler(payload)
        return self._render_generic(payload)

    def render_control_panel(self, panel: ControlPanelState) -> str:
        text, _ = self.render_control_panel_with_keyboard(panel)
        return text

    def render_control_panel_with_keyboard(
        self, panel: ControlPanelState
    ) -> tuple[str, list[list[dict]]]:
        lines: list[str] = []
        lines.append(f"\U0001f4ca {panel.date} \u2014 Control Panel")
        lines.append("")
        lines.append(
            f"Portfolio: +${panel.portfolio_pnl:.0f} "
            f"({panel.portfolio_pnl_pct:+.1f}%) "
            f"| DD: {panel.drawdown_pct:.1f}% "
            f"| Exposure: {panel.exposure_pct:.0f}%"
        )
        lines.append("")
        if panel.daily_report_ready:
            lines.append("\u2705 Daily report ready")
        if panel.alert_count > 0:
            lines.append(f"\u26a0\ufe0f {panel.alert_count} alert(s) ({panel.alert_summary})")
        if panel.wfo_status:
            lines.append(f"\U0001f9ea WFO: {panel.wfo_status}")
        lines.append(f"\U0001f9f0 {panel.pending_pr_count} PRs pending")
        lines.append(f"\U0001f6e1\ufe0f Risk: {panel.risk_status} ({panel.risk_detail})")
        lines.append("")
        for bot in panel.bot_statuses:
            lines.append(
                f"{bot.status_emoji} {bot.bot_id}: +${bot.pnl:.0f} "
                f"({bot.wins}W/{bot.losses}L) \u2014 {bot.summary}"
            )
        text = "\n".join(lines)
        return text, _CONTROL_PANEL_KEYBOARD

    def render_alert(self, payload: NotificationPayload) -> str:
        priority = payload.priority
        if priority == NotificationPriority.CRITICAL:
            header = f"\U0001f6a8 CRITICAL \u2014 {payload.title}"
        elif priority == NotificationPriority.HIGH:
            header = f"\u26a0\ufe0f HIGH \u2014 {payload.title}"
        else:
            header = f"\u2139\ufe0f {payload.title}"
        text = f"{header}\n\n{payload.body}"
        return _truncate(text)

    def render_daily_report(self, payload: NotificationPayload) -> str:
        text, _ = self.render_daily_report_with_keyboard(payload)
        return text

    def render_daily_report_with_keyboard(
        self, payload: NotificationPayload
    ) -> tuple[str, list[list[dict]]]:
        lines: list[str] = []
        lines.append(f"\U0001f4ca {payload.title}")
        lines.append("")
        lines.append(payload.body)
        text = _truncate("\n".join(lines))
        return text, _DAILY_REPORT_KEYBOARD

    def render_weekly_summary(self, payload: NotificationPayload) -> str:
        lines: list[str] = []
        lines.append(f"\U0001f4c8 {payload.title}")
        lines.append("")
        lines.append(payload.body)
        return _truncate("\n".join(lines))

    def _render_generic(self, payload: NotificationPayload) -> str:
        text = f"{payload.title}\n\n{payload.body}"
        return _truncate(text)
