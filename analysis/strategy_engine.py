# analysis/strategy_engine.py
"""Strategy refinement engine — deterministic 4-tier suggestion generator.

Analyzes weekly metrics and produces strategy suggestions. All rules-based,
no LLM calls. Claude interprets these in the weekly report prompt.

Tier 1 (Parameter): e.g. stop-loss too tight, threshold misaligned
Tier 2 (Filter): filter cost exceeds benefit over the week
Tier 3 (Strategy Variant): regime mismatch → suggest regime gate
Tier 4 (Hypothesis): reserved for Claude to synthesize in the weekly report
"""
from __future__ import annotations

from schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
    RefinementReport,
)
from schemas.weekly_metrics import (
    BotWeeklySummary,
    FilterWeeklySummary,
    RegimePerformanceTrend,
)


class StrategyEngine:
    """Deterministic strategy suggestion generator."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        tight_stop_ratio: float = 0.3,
        filter_cost_threshold: float = 0.0,
        regime_loss_threshold: float = 0.0,
        regime_min_weeks: int = 3,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.tight_stop_ratio = tight_stop_ratio
        self.filter_cost_threshold = filter_cost_threshold
        self.regime_loss_threshold = regime_loss_threshold
        self.regime_min_weeks = regime_min_weeks

    def analyze_parameters(
        self, summary: BotWeeklySummary
    ) -> list[StrategySuggestion]:
        """Tier 1: Detect parameter misalignment from weekly stats."""
        suggestions: list[StrategySuggestion] = []

        # Tight stop detection: avg_loss is small relative to avg_win
        if summary.avg_win > 0 and summary.avg_loss != 0:
            loss_win_ratio = abs(summary.avg_loss) / summary.avg_win
            if loss_win_ratio < self.tight_stop_ratio:
                suggestions.append(
                    StrategySuggestion(
                        tier=SuggestionTier.PARAMETER,
                        bot_id=summary.bot_id,
                        title=f"Stop loss may be too tight on {summary.bot_id}",
                        description=(
                            f"Avg loss (${abs(summary.avg_loss):.0f}) is only "
                            f"{loss_win_ratio:.0%} of avg win (${summary.avg_win:.0f}). "
                            f"Stops may be clipping winners too early. "
                            f"Consider widening stop by 0.5× ATR."
                        ),
                        current_value=f"loss/win_ratio={loss_win_ratio:.2f}",
                        suggested_value="loss/win_ratio>=0.3",
                        evidence_days=7,
                        confidence=0.7,
                    )
                )

        return suggestions

    def analyze_filters(
        self, bot_id: str, filter_summaries: list[FilterWeeklySummary]
    ) -> list[StrategySuggestion]:
        """Tier 2: Detect filters that cost more than they save."""
        suggestions: list[StrategySuggestion] = []

        for f in filter_summaries:
            if f.net_impact_pnl < self.filter_cost_threshold:
                suggestions.append(
                    StrategySuggestion(
                        tier=SuggestionTier.FILTER,
                        bot_id=bot_id,
                        title=f"Relax {f.filter_name} on {bot_id}",
                        description=(
                            f"{f.filter_name} blocked {f.total_blocks} entries this week. "
                            f"Net impact: ${f.net_impact_pnl:.0f} (cost exceeds benefit). "
                            f"Consider relaxing the threshold."
                        ),
                        evidence_days=7,
                        estimated_impact_pnl=abs(f.net_impact_pnl),
                        confidence=max(0.0, min(1.0, f.confidence)),
                    )
                )

        return suggestions

    def analyze_regime_fit(
        self, bot_id: str, regime_trends: list[RegimePerformanceTrend]
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect consistent losses in a specific regime."""
        suggestions: list[StrategySuggestion] = []

        for trend in regime_trends:
            if len(trend.weekly_pnl) < self.regime_min_weeks:
                continue

            # All weeks negative in this regime
            losing_weeks = sum(1 for pnl in trend.weekly_pnl if pnl < self.regime_loss_threshold)
            if losing_weeks >= self.regime_min_weeks:
                total_loss = sum(pnl for pnl in trend.weekly_pnl if pnl < 0)
                suggestions.append(
                    StrategySuggestion(
                        tier=SuggestionTier.STRATEGY_VARIANT,
                        bot_id=bot_id,
                        title=f"Add regime gate for {trend.regime} on {bot_id}",
                        description=(
                            f"{bot_id} lost in {trend.regime} regime for "
                            f"{losing_weeks}/{len(trend.weekly_pnl)} weeks "
                            f"(total: ${total_loss:.0f}). "
                            f"Consider adding a regime gate to disable trading "
                            f"in {trend.regime} conditions."
                        ),
                        requires_human_judgment=True,
                        evidence_days=len(trend.weekly_pnl) * 7,
                        confidence=0.5,
                    )
                )

        return suggestions

    def build_report(
        self,
        bot_summaries: dict[str, BotWeeklySummary],
        filter_summaries: dict[str, list[FilterWeeklySummary]] | None = None,
        regime_trends: dict[str, list[RegimePerformanceTrend]] | None = None,
    ) -> RefinementReport:
        """Build the complete refinement report across all bots."""
        all_suggestions: list[StrategySuggestion] = []

        for bot_id, summary in bot_summaries.items():
            all_suggestions.extend(self.analyze_parameters(summary))

            if filter_summaries and bot_id in filter_summaries:
                all_suggestions.extend(
                    self.analyze_filters(bot_id, filter_summaries[bot_id])
                )

            if regime_trends and bot_id in regime_trends:
                all_suggestions.extend(
                    self.analyze_regime_fit(bot_id, regime_trends[bot_id])
                )

        return RefinementReport(
            week_start=self.week_start,
            week_end=self.week_end,
            suggestions=all_suggestions,
        )
