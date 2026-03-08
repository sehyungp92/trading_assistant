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

    def render_approval_request(
        self, request,
    ) -> tuple[str, list[list[dict]]]:
        """Render an approval request card with inline keyboard buttons.

        Args:
            request: ApprovalRequest with backtest_summary and param_changes.

        Returns:
            (message_text, inline_keyboard) tuple.
        """
        lines: list[str] = []
        lines.append("\U0001f514 Suggestion Approval Request")
        lines.append(f"Bot: {_escape_md2(request.bot_id)}")
        lines.append("")

        # Parameter changes
        if request.param_changes:
            lines.append("Parameter Changes:")
            for pc in request.param_changes:
                name = pc.get("param_name", "?")
                current = pc.get("current", "?")
                proposed = pc.get("proposed", "?")
                lines.append(f"  {_escape_md2(name)}: {current} -> {proposed}")
            lines.append("")

        # Backtest results
        bs = request.backtest_summary
        if bs is not None:
            lines.append(f"Backtest Results ({bs.context.data_days}d, {bs.context.trade_count} trades):")
            lines.append("Metric     | Baseline | Proposed | Change")
            lines.append(
                f"Sharpe     | {bs.baseline.sharpe_ratio:.2f}     "
                f"| {bs.proposed.sharpe_ratio:.2f}     "
                f"| {bs.sharpe_change_pct:+.1f}%"
            )
            lines.append(
                f"MaxDD      | {bs.baseline.max_drawdown_pct:.1f}%    "
                f"| {bs.proposed.max_drawdown_pct:.1f}%    "
                f"| {bs.max_dd_change_pct:+.1f}%"
            )
            lines.append(
                f"ProfitFact | {bs.baseline.profit_factor:.2f}     "
                f"| {bs.proposed.profit_factor:.2f}     "
                f"| {bs.profit_factor_change_pct:+.1f}%"
            )
            lines.append(
                f"WinRate    | {bs.baseline.win_rate:.1%}    "
                f"| {bs.proposed.win_rate:.1%}    "
                f"| {bs.win_rate_change_pct:+.1f}%"
            )
            lines.append("")
            safety = "\u2705 PASS" if bs.passes_safety else "\u274c FAIL"
            lines.append(f"Safety: {safety}")
            if bs.safety_notes:
                for note in bs.safety_notes:
                    lines.append(f"  - {_escape_md2(note)}")

        text = _truncate("\n".join(lines))

        keyboard = [
            [
                {"text": "\u2705 Approve", "callback_data": f"approve_suggestion_{request.request_id}"},
                {"text": "\u274c Reject", "callback_data": f"reject_suggestion_{request.request_id}"},
            ],
            [
                {"text": "\U0001f4ca Details", "callback_data": f"detail_suggestion_{request.request_id}"},
            ],
        ]

        return text, keyboard

    def render_experiment_proposal(
        self,
        config,  # ExperimentConfig
    ) -> tuple[str, list[list[dict]]]:
        """Render an experiment proposal card with start/cancel buttons."""
        lines: list[str] = []
        lines.append("\U0001f9ea A/B Experiment Proposal")
        lines.append(f"ID: {_escape_md2(config.experiment_id[:8])}")
        lines.append(f"Bot: {_escape_md2(config.bot_id)}")
        lines.append(f"Type: {config.experiment_type.value}")
        lines.append("")
        lines.append(f"Title: {_escape_md2(config.title)}")
        if config.description:
            lines.append(f"Description: {_escape_md2(config.description)}")
        lines.append("")

        lines.append("Variants:")
        for v in config.variants:
            params_str = ", ".join(f"{k}={v_val}" for k, v_val in v.params.items())
            lines.append(f"  {_escape_md2(v.name)} ({v.allocation_pct:.0f}%): {_escape_md2(params_str)}")
        lines.append("")

        lines.append(f"Metric: {config.success_metric}")
        lines.append(f"Duration: {config.max_duration_days}d")
        lines.append(f"Min trades/variant: {config.min_trades_per_variant}")
        lines.append(f"Significance: {config.significance_level}")

        text = _truncate("\n".join(lines))

        keyboard = [
            [
                {"text": "\u25b6\ufe0f Start", "callback_data": f"start_experiment_{config.experiment_id}"},
                {"text": "\u274c Cancel", "callback_data": f"cancel_experiment_{config.experiment_id}"},
            ],
        ]

        return text, keyboard

    def render_experiment_result(
        self,
        config,   # ExperimentConfig
        result,   # ExperimentResult
    ) -> str:
        """Render experiment conclusion summary."""
        lines: list[str] = []

        icon = {
            "adopt_treatment": "\u2705",
            "keep_control": "\U0001f6d1",
            "inconclusive": "\u2753",
            "extend": "\u23f3",
        }.get(result.recommendation, "\u2753")

        lines.append(f"\U0001f9ea Experiment Concluded: {_escape_md2(config.title)}")
        lines.append(f"Bot: {_escape_md2(config.bot_id)}")
        lines.append(f"Result: {icon} {result.recommendation.replace('_', ' ').upper()}")
        lines.append("")

        lines.append("Variant Results:")
        for vm in result.variant_metrics:
            lines.append(
                f"  {_escape_md2(vm.variant_name)}: "
                f"{vm.trade_count} trades, "
                f"PnL ${vm.total_pnl:.0f}, "
                f"WR {vm.win_rate:.0%}, "
                f"Sharpe {vm.sharpe:.2f}"
            )
        lines.append("")

        if result.p_value is not None:
            lines.append(f"p\\-value: {result.p_value:.4f}")
        if result.effect_size is not None:
            lines.append(f"Effect size \\(Cohen's d\\): {result.effect_size:.3f}")
        if result.confidence_interval_95 is not None:
            lo, hi = result.confidence_interval_95
            lines.append(f"95% CI: \\[{lo:.4f}, {hi:.4f}\\]")
        if result.winner:
            lines.append(f"Winner: {_escape_md2(result.winner)}")

        return _truncate("\n".join(lines))

    def _render_generic(self, payload: NotificationPayload) -> str:
        text = f"{payload.title}\n\n{payload.body}"
        return _truncate(text)
