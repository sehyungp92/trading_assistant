# comms/discord_renderer.py
"""Discord embed renderer — rich formatting for Discord messages."""
from __future__ import annotations

from schemas.notifications import (
    ControlPanelState,
    NotificationPayload,
    NotificationPriority,
)

_COLORS = {
    NotificationPriority.CRITICAL: 0xFF0000,
    NotificationPriority.HIGH: 0xE67E22,
    NotificationPriority.NORMAL: 0x3498DB,
    NotificationPriority.LOW: 0x95A5A6,
}


class DiscordRenderer:
    """Renders notifications as Discord embeds."""

    def render(self, payload: NotificationPayload) -> str:
        dispatch = {
            "alert": self.render_alert,
            "daily_report": self.render_daily_report,
            "weekly_summary": self.render_weekly_summary,
        }
        handler = dispatch.get(payload.notification_type)
        if handler:
            return handler(payload)
        return f"{payload.title}\n\n{payload.body}"

    def render_control_panel(self, panel: ControlPanelState) -> str:
        lines: list[str] = []
        lines.append(f"📊 {panel.date} — Control Panel")
        lines.append(
            f"Portfolio: +${panel.portfolio_pnl:.0f} ({panel.portfolio_pnl_pct:+.1f}%) "
            f"| DD: {panel.drawdown_pct:.1f}% | Exposure: {panel.exposure_pct:.0f}%"
        )
        for bot in panel.bot_statuses:
            lines.append(f"{bot.status_emoji} {bot.bot_id}: +${bot.pnl:.0f} ({bot.wins}W/{bot.losses}L) — {bot.summary}")
        return "\n".join(lines)

    def render_alert(self, payload: NotificationPayload) -> str:
        embed = self.render_alert_embed(payload)
        return f"{embed['title']}\n\n{embed['description']}"

    def render_alert_embed(self, payload: NotificationPayload) -> dict:
        priority = payload.priority
        if priority == NotificationPriority.CRITICAL:
            title = f"🚨 CRITICAL — {payload.title}"
        elif priority == NotificationPriority.HIGH:
            title = f"⚠️ HIGH — {payload.title}"
        else:
            title = f"ℹ️ {payload.title}"
        return {
            "title": title,
            "description": payload.body,
            "color": _COLORS.get(priority, 0x3498DB),
        }

    def render_daily_report(self, payload: NotificationPayload) -> str:
        embed = self.render_daily_report_embed(payload)
        return f"{embed['title']}\n\n{embed['description']}"

    def render_daily_report_embed(self, payload: NotificationPayload) -> dict:
        embed: dict = {
            "title": f"📊 {payload.title}",
            "description": payload.body,
            "color": 0x3498DB,
            "fields": [],
        }
        bot_statuses = payload.data.get("bot_statuses", [])
        for bot in bot_statuses:
            bot_id = bot.get("bot_id", "?")
            status = bot.get("status", "")
            emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(status, "⚪")
            pnl = bot.get("pnl", 0)
            wins = bot.get("wins", 0)
            losses = bot.get("losses", 0)
            summary = bot.get("summary", "")
            embed["fields"].append({
                "name": f"{emoji} {bot_id}",
                "value": f"+${pnl:.0f} ({wins}W/{losses}L) — {summary}",
                "inline": True,
            })
        return embed

    def render_weekly_summary(self, payload: NotificationPayload) -> str:
        embed = self.render_weekly_summary_embed(payload)
        return f"{embed['title']}\n\n{embed['description']}"

    def render_weekly_summary_embed(self, payload: NotificationPayload) -> dict:
        return {
            "title": f"📈 {payload.title}",
            "description": payload.body,
            "color": 0x2ECC71,
        }
