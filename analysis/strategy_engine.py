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

from schemas.regime_conditional import (
    RegimeAllocation,
    RegimeConditionalReport,
    RegimeDistribution,
    RegimeStrategyMetrics,
)
from schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
    RefinementReport,
)
from schemas.weekly_metrics import (
    BotWeeklySummary,
    FilterWeeklySummary,
    RegimePerformanceTrend,
    StrategyWeeklySummary,
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
        self, bot_id: str, regime_trends: list[RegimePerformanceTrend],
        trades: list | None = None,
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect consistent losses in a specific regime.

        If trades are provided, includes quantified exclusion impact in the description.
        """
        return self.analyze_regime_fit_quantified(bot_id, regime_trends, trades)

    def compute_regime_exclusion_impact(
        self, bot_id: str, trades: list, regime_to_exclude: str
    ) -> dict:
        """Compute P&L impact of excluding all trades in a specific regime."""
        baseline_pnl = sum(t.pnl for t in trades)
        kept = [t for t in trades if (t.market_regime or "unknown") != regime_to_exclude]
        excluded_pnl = sum(t.pnl for t in kept)
        excluded_count = len(trades) - len(kept)
        return {
            "regime": regime_to_exclude,
            "baseline_pnl": baseline_pnl,
            "excluded_pnl": excluded_pnl,
            "delta_pnl": excluded_pnl - baseline_pnl,
            "excluded_trade_count": excluded_count,
            "total_trade_count": len(trades),
        }

    def analyze_regime_fit_quantified(
        self, bot_id: str, regime_trends: list["RegimePerformanceTrend"],
        trades: list | None = None,
    ) -> list["StrategySuggestion"]:
        """Tier 3: Regime fit analysis with quantified exclusion impact."""
        suggestions: list[StrategySuggestion] = []

        for trend in regime_trends:
            if len(trend.weekly_pnl) < self.regime_min_weeks:
                continue
            losing_weeks = sum(1 for pnl in trend.weekly_pnl if pnl < self.regime_loss_threshold)
            if losing_weeks < self.regime_min_weeks:
                continue

            total_loss = sum(pnl for pnl in trend.weekly_pnl if pnl < 0)
            desc = (
                f"{bot_id} lost in {trend.regime} regime for "
                f"{losing_weeks}/{len(trend.weekly_pnl)} weeks "
                f"(total: ${total_loss:.0f}). "
            )

            if trades:
                impact = self.compute_regime_exclusion_impact(bot_id, trades, trend.regime)
                desc += (
                    f"Excluding {trend.regime} trades would change PnL from "
                    f"${impact['baseline_pnl']:.0f} to ${impact['excluded_pnl']:.0f} "
                    f"(+${impact['delta_pnl']:.0f}, removing {impact['excluded_trade_count']} trades). "
                )

            desc += f"Consider adding a regime gate to disable trading in {trend.regime} conditions."

            suggestions.append(
                StrategySuggestion(
                    tier=SuggestionTier.STRATEGY_VARIANT,
                    bot_id=bot_id,
                    title=f"Add regime gate for {trend.regime} on {bot_id}",
                    description=desc,
                    requires_human_judgment=True,
                    evidence_days=len(trend.weekly_pnl) * 7,
                    confidence=0.5,
                    estimated_impact_pnl=abs(total_loss),
                )
            )

        return suggestions

    def detect_alpha_decay(
        self,
        bot_id: str,
        rolling_sharpe_30d: float,
        rolling_sharpe_60d: float,
        rolling_sharpe_90d: float,
        decay_threshold: float = 0.3,
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect declining Sharpe ratio over 30/60/90 day windows."""
        if rolling_sharpe_90d <= 0:
            return []
        # Check if 30d Sharpe is significantly below 90d Sharpe
        decay_ratio = (rolling_sharpe_90d - rolling_sharpe_30d) / rolling_sharpe_90d
        if decay_ratio < decay_threshold:
            return []
        return [StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id=bot_id,
            title=f"Alpha decay detected — {bot_id}",
            description=(
                f"30d Sharpe ({rolling_sharpe_30d:.2f}) is {decay_ratio:.0%} below "
                f"90d Sharpe ({rolling_sharpe_90d:.2f}). The strategy may be losing edge. "
                f"Review signal quality and market regime alignment."
            ),
            evidence_days=90,
            confidence=min(0.9, 0.5 + decay_ratio),
            requires_human_judgment=True,
        )]

    def detect_signal_decay(
        self,
        bot_id: str,
        signal_outcome_correlation_30d: float,
        signal_outcome_correlation_90d: float,
        decay_threshold: float = 0.2,
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect declining signal-to-outcome correlation."""
        drop = signal_outcome_correlation_90d - signal_outcome_correlation_30d
        if drop < decay_threshold:
            return []
        return [StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id=bot_id,
            title=f"Signal quality decay — {bot_id}",
            description=(
                f"Signal→outcome correlation dropped from {signal_outcome_correlation_90d:.2f} "
                f"(90d) to {signal_outcome_correlation_30d:.2f} (30d). "
                f"Signal may need recalibration or replacement."
            ),
            evidence_days=90,
            confidence=min(0.9, 0.5 + drop),
            requires_human_judgment=True,
        )]

    def detect_exit_timing_issues(
        self,
        bot_id: str,
        avg_exit_efficiency: float,
        premature_exit_pct: float,
        efficiency_threshold: float = 0.5,
        premature_threshold: float = 0.4,
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect systematic premature exits."""
        if avg_exit_efficiency >= efficiency_threshold and premature_exit_pct <= premature_threshold:
            return []
        suggestions: list[StrategySuggestion] = []
        if avg_exit_efficiency < efficiency_threshold:
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.STRATEGY_VARIANT,
                bot_id=bot_id,
                title=f"Premature exits — {bot_id}",
                description=(
                    f"Average exit efficiency is {avg_exit_efficiency:.0%} (captures "
                    f"{avg_exit_efficiency:.0%} of available move). "
                    f"{premature_exit_pct:.0%} of exits are premature. "
                    f"Consider trailing stop or wider take-profit."
                ),
                evidence_days=30,
                confidence=0.6,
                requires_human_judgment=True,
            ))
        return suggestions

    def detect_correlation_breakdown(
        self,
        correlations: list,  # list[CorrelationSummary]
        threshold: float = 0.7,
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect rising cross-bot return correlation (systemic risk)."""
        suggestions: list[StrategySuggestion] = []
        for corr in correlations:
            if corr.rolling_30d_correlation >= threshold:
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.STRATEGY_VARIANT,
                    bot_id=f"{corr.bot_a}+{corr.bot_b}",
                    title=f"High correlation — {corr.bot_a} / {corr.bot_b}",
                    description=(
                        f"30d return correlation is {corr.rolling_30d_correlation:.2f}. "
                        f"Same-direction trading {corr.same_direction_pct:.0%} of the time. "
                        f"This increases systemic risk during adverse moves. "
                        f"Consider diversifying signal sources or staggering entry timing."
                    ),
                    evidence_days=30,
                    confidence=min(0.9, corr.rolling_30d_correlation),
                    requires_human_judgment=True,
                ))
        return suggestions

    def detect_time_of_day_patterns(
        self,
        bot_id: str,
        hourly_buckets: list,  # list[HourlyBucket]
        min_trades: int = 10,
        loss_threshold: float = 0.35,
    ) -> list[StrategySuggestion]:
        """Tier 2: Detect hours with consistently poor performance."""
        suggestions: list[StrategySuggestion] = []
        for bucket in hourly_buckets:
            if bucket.trade_count < min_trades:
                continue
            if bucket.pnl < 0 and bucket.win_rate < loss_threshold:
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.FILTER,
                    bot_id=bot_id,
                    title=f"Poor hour {bucket.hour:02d}:00 — {bot_id}",
                    description=(
                        f"Hour {bucket.hour:02d}:00 UTC: {bucket.trade_count} trades, "
                        f"PnL ${bucket.pnl:.0f}, win rate {bucket.win_rate:.0%}. "
                        f"Consider adding a time-of-day gate to avoid this hour."
                    ),
                    evidence_days=7,
                    confidence=0.6,
                ))
        return suggestions

    def detect_drawdown_patterns(
        self,
        bot_id: str,
        largest_single_loss_pct: float,
        max_drawdown_pct: float,
        avg_loss_pct: float,
        concentration_threshold: float = 3.0,
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect concentrated drawdown (single loss dominates)."""
        if avg_loss_pct <= 0:
            return []
        concentration = largest_single_loss_pct / avg_loss_pct
        if concentration < concentration_threshold:
            return []
        return [StrategySuggestion(
            tier=SuggestionTier.STRATEGY_VARIANT,
            bot_id=bot_id,
            title=f"Concentrated drawdown risk — {bot_id}",
            description=(
                f"Largest single loss ({largest_single_loss_pct:.1f}%) is "
                f"{concentration:.1f}x the average loss ({avg_loss_pct:.1f}%). "
                f"Max drawdown: {max_drawdown_pct:.1f}%. "
                f"Consider tighter per-trade risk limits or position sizing adjustments."
            ),
            evidence_days=30,
            confidence=0.65,
            requires_human_judgment=True,
        )]

    def detect_position_sizing_issues(
        self,
        bot_id: str,
        avg_win_pct: float,
        avg_loss_pct: float,
        win_rate: float,
        loss_win_ratio_threshold: float = 1.5,
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect asymmetric position sizing (losses > wins despite positive win rate)."""
        if avg_win_pct <= 0 or win_rate < 0.5:
            return []
        loss_win_ratio = avg_loss_pct / avg_win_pct
        if loss_win_ratio < loss_win_ratio_threshold:
            return []
        return [StrategySuggestion(
            tier=SuggestionTier.STRATEGY_VARIANT,
            bot_id=bot_id,
            title=f"Position sizing imbalance — {bot_id}",
            description=(
                f"Average loss ({avg_loss_pct:.1f}%) is {loss_win_ratio:.1f}x "
                f"average win ({avg_win_pct:.1f}%) despite {win_rate:.0%} win rate. "
                f"Risk/reward is asymmetric — consider reducing position size on "
                f"lower-confidence signals or tightening stop placement."
            ),
            evidence_days=30,
            confidence=0.6,
            requires_human_judgment=True,
        )]

    def detect_component_signal_decay(
        self,
        bot_id: str,
        signal_health_data: dict,
        stability_threshold: float = 0.3,
        correlation_threshold: float = 0.05,
        min_trades: int = 5,
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect degraded signal components from signal_health data."""
        components = signal_health_data.get("components", [])
        degraded: list[str] = []

        for comp in components:
            trade_count = comp.get("trade_count", 0)
            if trade_count < min_trades:
                continue
            stability = comp.get("stability", 1.0)
            win_corr = abs(comp.get("win_correlation", 1.0))
            if stability < stability_threshold or win_corr < correlation_threshold:
                degraded.append(comp.get("component_name", "unknown"))

        if not degraded:
            return []

        return [StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id=bot_id,
            title=f"Signal component decay — {bot_id}",
            description=(
                f"Degraded signal components detected: {', '.join(degraded)}. "
                f"These components show low stability (<{stability_threshold}) or "
                f"near-zero win correlation (<{correlation_threshold}). "
                f"Review whether these signals still carry predictive value."
            ),
            evidence_days=7,
            confidence=0.5,
            requires_human_judgment=True,
        )]

    def detect_filter_interactions(
        self,
        bot_id: str,
        filter_interactions: list,
    ) -> list[StrategySuggestion]:
        """Tier 2: Generate suggestions from filter interaction analysis.

        Args:
            bot_id: Bot identifier.
            filter_interactions: List of FilterPairInteraction dicts or objects.
        """
        suggestions: list[StrategySuggestion] = []

        for pair in filter_interactions:
            itype = pair.get("interaction_type", "") if isinstance(pair, dict) else getattr(pair, "interaction_type", "")
            if itype == "independent":
                continue

            if isinstance(pair, dict):
                fa = pair.get("filter_a", "")
                fb = pair.get("filter_b", "")
                rec = pair.get("recommendation", "")
                redundancy = pair.get("redundancy_score", 0.0)
            else:
                fa = getattr(pair, "filter_a", "")
                fb = getattr(pair, "filter_b", "")
                rec = getattr(pair, "recommendation", "")
                redundancy = getattr(pair, "redundancy_score", 0.0)

            if itype == "redundant":
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.FILTER,
                    bot_id=bot_id,
                    title=f"Redundant filters: {fa} + {fb} on {bot_id}",
                    description=(
                        f"Filters {fa} and {fb} are redundant "
                        f"(overlap score: {redundancy:.0%}). {rec}"
                    ),
                    evidence_days=7,
                    confidence=min(0.9, redundancy),
                    requires_human_judgment=True,
                ))
            elif itype == "complementary":
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.FILTER,
                    bot_id=bot_id,
                    title=f"Complementary filters: {fa} + {fb} on {bot_id}",
                    description=(
                        f"Filters {fa} and {fb} are complementary. {rec}"
                    ),
                    evidence_days=7,
                    confidence=0.5,
                ))

        return suggestions

    def detect_factor_correlation_decay(
        self,
        bot_id: str,
        factor_rolling_data: list[dict],
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect degrading signal factors from rolling 30d analysis.

        Produces HYPOTHESIS suggestions for factors with degrading trend + below_threshold.
        """
        suggestions: list[StrategySuggestion] = []

        for factor in factor_rolling_data:
            trend = factor.get("win_rate_trend", "stable")
            below = factor.get("below_threshold", False)
            if trend != "degrading" and not below:
                continue

            name = factor.get("factor_name", "unknown")
            wr = factor.get("rolling_30d_win_rate", 0)
            days = factor.get("days_of_data", 0)

            parts = [f"Factor '{name}' on {bot_id}"]
            if trend == "degrading":
                parts.append("shows degrading win rate trend over 30d window")
            if below:
                parts.append(f"(rolling win rate {wr:.0%} is below threshold)")
            parts.append(f"Based on {days} days of data.")
            parts.append("Consider recalibrating or replacing this signal factor.")

            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.HYPOTHESIS,
                bot_id=bot_id,
                title=f"Factor decay — {name} on {bot_id}",
                description=" ".join(parts),
                evidence_days=days,
                confidence=0.5,
                requires_human_judgment=True,
            ))

        return suggestions

    def compute_regime_conditional_metrics(
        self,
        per_strategy_summaries: dict[str, dict[str, StrategyWeeklySummary]],
        trades_by_bot: dict[str, list],
    ) -> RegimeConditionalReport:
        """Compute regime-conditional performance metrics and allocation suggestions.

        Args:
            per_strategy_summaries: outer key=bot_id, inner key=strategy_id
            trades_by_bot: bot_id → list of TradeEvent
        """
        import math
        import statistics
        from collections import defaultdict

        # Group trades by (bot_id, strategy_id, regime)
        grouped: dict[tuple[str, str, str], list] = defaultdict(list)
        regime_counts: dict[str, int] = defaultdict(int)
        total_trades = 0

        for bot_id, trades in trades_by_bot.items():
            for t in trades:
                regime = getattr(t, "market_regime", None) or "unknown"
                strat = getattr(t, "entry_signal", "") or "default"
                grouped[(bot_id, strat, regime)].append(t)
                regime_counts[regime] += 1
                total_trades += 1

        # Compute per-group metrics
        metrics: list[RegimeStrategyMetrics] = []
        regime_strategy_sharpes: dict[str, dict[str, float]] = defaultdict(dict)

        for (bot_id, strat_id, regime), trades in grouped.items():
            if not trades:
                continue
            pnls = [t.pnl for t in trades]
            wins = [p for p in pnls if p > 0]
            win_rate = len(wins) / len(pnls) if pnls else 0.0
            expectancy = statistics.mean(pnls) if pnls else 0.0
            sharpe = 0.0
            if len(pnls) >= 2:
                std = statistics.stdev(pnls)
                if std > 0:
                    sharpe = (statistics.mean(pnls) / std) * math.sqrt(252)

            # Max drawdown from cumulative PnL
            cumsum = 0.0
            peak = 0.0
            max_dd = 0.0
            for p in pnls:
                cumsum += p
                if cumsum > peak:
                    peak = cumsum
                dd = (peak - cumsum) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

            key = f"{bot_id}:{strat_id}"
            regime_strategy_sharpes[regime][key] = sharpe

            metrics.append(RegimeStrategyMetrics(
                bot_id=bot_id,
                strategy_id=strat_id,
                regime=regime,
                trade_count=len(trades),
                win_rate=round(win_rate, 4),
                expectancy=round(expectancy, 2),
                sharpe=round(sharpe, 4),
                max_drawdown_pct=round(max_dd * 100, 2),
            ))

        # Regime distribution
        regime_dist: list[RegimeDistribution] = []
        for regime, count in regime_counts.items():
            pct = (count / total_trades * 100.0) if total_trades > 0 else 0.0
            regime_dist.append(RegimeDistribution(
                regime=regime,
                pct_of_time=round(pct, 1),
                trade_count=count,
            ))

        # Optimal allocations per regime (inverse-volatility weighted)
        allocations: list[RegimeAllocation] = []
        for regime, strat_sharpes in regime_strategy_sharpes.items():
            if not strat_sharpes:
                continue
            # Use max(sharpe, 0.01) to avoid division by zero; zero/negative → minimum alloc
            inv_vol = {}
            for key, s in strat_sharpes.items():
                inv_vol[key] = max(s, 0.01)
            total_inv = sum(inv_vol.values())
            alloc = {k: round(v / total_inv * 100.0, 1) for k, v in inv_vol.items()} if total_inv > 0 else {}
            allocations.append(RegimeAllocation(
                regime=regime,
                allocations=alloc,
                rationale=f"Inverse-volatility allocation across {len(alloc)} strategies in {regime} regime",
            ))

        # Generate suggestions for underperforming strategy-regime combos
        suggestions: list[dict] = []
        for m in metrics:
            if m.trade_count >= 10 and m.win_rate < 0.35 and m.expectancy < 0:
                suggestions.append({
                    "regime": m.regime,
                    "strategy": f"{m.bot_id}:{m.strategy_id}",
                    "current_alloc": "equal",
                    "suggested_alloc": "reduce",
                    "reason": (
                        f"In {m.regime}, {m.bot_id}:{m.strategy_id} has "
                        f"{m.win_rate:.0%} win rate and ${m.expectancy:.0f} expectancy "
                        f"over {m.trade_count} trades. Consider scaling down."
                    ),
                })

        return RegimeConditionalReport(
            week_start=self.week_start,
            week_end=self.week_end,
            metrics=metrics,
            optimal_allocations=allocations,
            regime_distribution=regime_dist,
            suggestions=suggestions,
        )

    def build_report(
        self,
        bot_summaries: dict[str, BotWeeklySummary],
        filter_summaries: dict[str, list[FilterWeeklySummary]] | None = None,
        regime_trends: dict[str, list[RegimePerformanceTrend]] | None = None,
        rolling_sharpe: dict[str, dict[str, float]] | None = None,
        signal_correlations: dict[str, dict[str, float]] | None = None,
        hourly_buckets: dict[str, list] | None = None,
        correlation_summaries: list | None = None,
        drawdown_data: dict[str, dict] | None = None,
        signal_health: dict[str, dict] | None = None,
        factor_rolling: dict[str, list[dict]] | None = None,
        filter_interactions: dict[str, list] | None = None,
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

        # New detectors
        if rolling_sharpe:
            for bot_id, sharpe in rolling_sharpe.items():
                all_suggestions.extend(self.detect_alpha_decay(
                    bot_id, sharpe.get("30d", 0), sharpe.get("60d", 0), sharpe.get("90d", 0),
                ))

        if signal_correlations:
            for bot_id, corr in signal_correlations.items():
                all_suggestions.extend(self.detect_signal_decay(
                    bot_id, corr.get("30d", 0), corr.get("90d", 0),
                ))

        if hourly_buckets:
            for bot_id, buckets in hourly_buckets.items():
                all_suggestions.extend(self.detect_time_of_day_patterns(bot_id, buckets))

        if correlation_summaries:
            all_suggestions.extend(self.detect_correlation_breakdown(correlation_summaries))

        if drawdown_data:
            for bot_id, dd in drawdown_data.items():
                all_suggestions.extend(self.detect_drawdown_patterns(
                    bot_id,
                    dd.get("largest_single_loss_pct", 0),
                    dd.get("max_drawdown_pct", 0),
                    dd.get("avg_loss_pct", 0),
                ))

        for bot_id, summary in bot_summaries.items():
            if summary.avg_win > 0 and abs(summary.avg_loss) > 0:
                all_suggestions.extend(self.detect_position_sizing_issues(
                    bot_id,
                    avg_win_pct=summary.avg_win,
                    avg_loss_pct=abs(summary.avg_loss),
                    win_rate=summary.win_rate,
                ))

        if signal_health:
            for bot_id, sh_data in signal_health.items():
                all_suggestions.extend(
                    self.detect_component_signal_decay(bot_id, sh_data)
                )

        if factor_rolling:
            for bot_id, factors in factor_rolling.items():
                all_suggestions.extend(
                    self.detect_factor_correlation_decay(bot_id, factors)
                )

        if filter_interactions:
            for bot_id, interactions in filter_interactions.items():
                all_suggestions.extend(
                    self.detect_filter_interactions(bot_id, interactions)
                )

        return RefinementReport(
            week_start=self.week_start,
            week_end=self.week_end,
            suggestions=all_suggestions,
        )
