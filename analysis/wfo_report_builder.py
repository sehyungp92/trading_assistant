# analysis/wfo_report_builder.py
"""WFO report builder — generates human-readable markdown from WFO results.

Produces a structured report with:
- Parameter comparison (current vs suggested)
- Fold-by-fold OOS performance
- Cost sensitivity table
- Robustness score and regime breakdown
- Safety flags and "what could go wrong"
- Final recommendation (ADOPT / TEST_FURTHER / REJECT)
"""
from __future__ import annotations

from schemas.wfo_results import WFOReport, SafetyFlag


class WFOReportBuilder:
    """Generates markdown reports from WFOReport data."""

    def build_markdown(self, report: WFOReport) -> str:
        """Build a complete markdown report."""
        sections: list[str] = []
        sections.append(self._header(report))
        sections.append(self._param_comparison(report))
        sections.append(self._fold_summary(report))
        sections.append(self._cost_sensitivity(report))
        sections.append(self._robustness(report))
        sections.append(self._safety_flags(report))
        sections.append(self._what_could_go_wrong(report))
        sections.append(self._recommendation(report))
        return "\n\n".join(s for s in sections if s)

    def _header(self, r: WFOReport) -> str:
        method = r.config_summary.get("method", "unknown")
        return (
            f"# WFO Report — {r.bot_id}\n\n"
            f"**Method:** {method}  \n"
            f"**Config:** {r.config_summary}"
        )

    def _param_comparison(self, r: WFOReport) -> str:
        if not r.current_params and not r.suggested_params:
            return ""
        lines = ["## Parameter Comparison\n", "| Parameter | Current | Suggested |", "|---|---|---|"]
        all_keys = set(r.current_params.keys()) | set(r.suggested_params.keys())
        for key in sorted(all_keys):
            cur = r.current_params.get(key, "—")
            sug = r.suggested_params.get(key, "—")
            lines.append(f"| {key} | {cur} | {sug} |")
        return "\n".join(lines)

    def _fold_summary(self, r: WFOReport) -> str:
        if not r.fold_results:
            return "## Fold Results\n\nNo folds completed."
        lines = [
            "## Fold Results\n",
            "| Fold | IS Sharpe | OOS Sharpe | OOS PnL | OOS Trades | Degradation |",
            "|---|---|---|---|---|---|",
        ]
        for fr in r.fold_results:
            deg = f"{fr.oos_degradation_pct:.0%}"
            lines.append(
                f"| {fr.fold.fold_number} | {fr.is_metrics.sharpe_ratio:.2f} | "
                f"{fr.oos_metrics.sharpe_ratio:.2f} | ${fr.oos_metrics.net_pnl:.0f} | "
                f"{fr.oos_metrics.total_trades} | {deg} |"
            )
        return "\n".join(lines)

    def _cost_sensitivity(self, r: WFOReport) -> str:
        if not r.cost_sensitivity:
            return ""
        lines = [
            "## Cost Sensitivity\n",
            "| Cost Multiplier | Sharpe | Net PnL |",
            "|---|---|---|",
        ]
        for cs in r.cost_sensitivity:
            mult = f"{cs.cost_multiplier}x"
            lines.append(f"| {mult} | {cs.metrics.sharpe_ratio:.2f} | ${cs.metrics.net_pnl:.0f} |")
        return "\n".join(lines)

    def _robustness(self, r: WFOReport) -> str:
        rob = r.robustness
        return (
            f"## Robustness\n\n"
            f"**Score:** {rob.robustness_score:.0f}/100  \n"
            f"**Neighborhood stable:** {rob.neighborhood_stable}  \n"
            f"**Regime stable:** {rob.regime_stable} "
            f"({rob.profitable_regime_count} profitable regimes)"
        )

    def _safety_flags(self, r: WFOReport) -> str:
        if not r.safety_flags:
            return "## Safety Flags\n\nNone."
        lines = ["## Safety Flags\n"]
        for f in r.safety_flags:
            icon = "🔴" if f.severity == "high" else "🟡" if f.severity == "medium" else "🟢"
            lines.append(f"- {icon} **{f.flag_type}** ({f.severity}): {f.description}")
        return "\n".join(lines)

    def _what_could_go_wrong(self, r: WFOReport) -> str:
        risks: list[str] = []
        if r.safety_flags:
            for f in r.safety_flags:
                risks.append(f"- {f.description}")
        if any(fr.oos_degradation_pct > 0.3 for fr in r.fold_results):
            risks.append("- Significant IS→OOS degradation in some folds (>30%)")
        if r.robustness.robustness_score < 60:
            risks.append("- Low robustness score — results may not generalize")
        if not risks:
            risks.append("- No major risks identified")
        return "## What Could Go Wrong\n\n" + "\n".join(risks)

    def _recommendation(self, r: WFOReport) -> str:
        return (
            f"## Recommendation: **{r.recommendation.value.upper()}**\n\n"
            f"{r.recommendation_reasoning}"
        )
