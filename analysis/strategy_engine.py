# analysis/strategy_engine.py
"""Strategy refinement engine — deterministic 4-tier suggestion generator.

Analyzes weekly metrics and produces strategy suggestions. All rules-based,
no LLM calls. The configured analysis provider interprets these in the weekly report prompt.

Tier 1 (Parameter): e.g. stop-loss too tight, threshold misaligned
Tier 2 (Filter): filter cost exceeds benefit over the week
Tier 3 (Strategy Variant): regime mismatch → suggest regime gate
Tier 4 (Hypothesis): reserved for the analysis runtime to synthesize in the weekly report
"""
from __future__ import annotations

from schemas.detection_context import DetectionContext
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

    # Archetype-specific default overrides per detector.
    _ARCHETYPE_DEFAULTS: dict[str, dict[str, dict[str, float]]] = {
        "alpha_decay": {
            "trend_follow": {"decay_threshold": 0.55},
            "divergence_swing": {"decay_threshold": 0.55},
            "breakout": {"decay_threshold": 0.55},
            "box_breakout": {"decay_threshold": 0.55},
            "multi_tf_momentum": {"decay_threshold": 0.55},
            "pullback": {"decay_threshold": 0.80},
            "intraday_momentum": {"decay_threshold": 0.80},
            "opening_range_breakout": {"decay_threshold": 0.80},
            "vwap_pullback": {"decay_threshold": 0.80},
            "flow_following": {"decay_threshold": 0.80},
        },
        "exit_timing": {
            "trend_follow": {"efficiency_threshold": 0.20},
            "divergence_swing": {"efficiency_threshold": 0.25},
            "multi_tf_momentum": {"efficiency_threshold": 0.35},
            "intraday_momentum": {"efficiency_threshold": 0.45},
            "opening_range_breakout": {"efficiency_threshold": 0.45},
            "vwap_pullback": {"efficiency_threshold": 0.45},
            "flow_following": {"efficiency_threshold": 0.40},
        },
    }

    # Map detector_name → suggestion category for value-map lookups.
    _DETECTOR_TO_CATEGORY: dict[str, str] = {
        "tight_stop": "stop_loss",
        "wide_stop": "stop_loss",
        "filter_cost": "filter_threshold",
        "regime_loss": "regime_gate",
        "alpha_decay": "signal",
        "signal_decay": "signal",
        "component_signal_decay": "signal",
        "factor_decay": "signal",
        "exit_timing": "exit_timing",
        "correlation": "signal",
        "time_of_day": "signal",
        "drawdown_concentration": "stop_loss",
        "position_sizing": "position_sizing",
        "filter_interactions": "filter_threshold",
        "microstructure": "signal",
    }

    # Keywords indicating direction of change
    _DECREASE_KEYWORDS = frozenset({
        "tighten", "reduce", "lower", "decrease", "narrow", "less", "smaller",
        "cut", "shrink", "restrict", "shorten",
    })
    _INCREASE_KEYWORDS = frozenset({
        "widen", "increase", "raise", "expand", "more", "larger", "bigger",
        "extend", "loosen", "relax", "lengthen",
    })

    def __init__(
        self,
        week_start: str,
        week_end: str,
        tight_stop_ratio: float = 0.3,
        filter_cost_threshold: float = 0.0,
        regime_loss_threshold: float = 0.0,
        regime_min_weeks: int = 3,
        threshold_learner: object | None = None,
        strategy_registry: object | None = None,
        category_scorecard: object | None = None,
        detector_confidence: dict[str, float] | None = None,
        recent_suggestions: list[dict] | None = None,
        convergence_report: dict | None = None,
        category_value_map: dict | None = None,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.tight_stop_ratio = tight_stop_ratio
        self.filter_cost_threshold = filter_cost_threshold
        self.regime_loss_threshold = regime_loss_threshold
        self.regime_min_weeks = regime_min_weeks
        self._threshold_learner = threshold_learner
        self._strategy_registry = strategy_registry
        self._category_scorecard = category_scorecard
        self._detector_confidence = detector_confidence or {}
        self._recent_suggestions = recent_suggestions or []
        self._convergence_report = convergence_report or {}
        self._category_value_map = category_value_map or {}

    def _get_threshold(
        self,
        detector_name: str,
        threshold_name: str,
        bot_id: str,
        default: float,
    ) -> float:
        """Return learned threshold if available, else default."""
        if self._threshold_learner is None:
            return default
        return self._threshold_learner.get_threshold(
            detector_name, threshold_name, bot_id, default,
        )

    def _archetype_default(self, strategy_id: str, detector: str, param: str) -> float | None:
        """Return archetype-specific default for a detector param, or None."""
        if not self._strategy_registry or not strategy_id:
            return None
        arch = self._strategy_registry.archetype_for_strategy(strategy_id)
        if not arch:
            return None
        arch_str = arch.value if hasattr(arch, "value") else str(arch)
        return self._ARCHETYPE_DEFAULTS.get(detector, {}).get(arch_str, {}).get(param)

    def _resolve_strategy_id(self, bot_id: str) -> str:
        """Resolve the primary strategy_id for a bot_id from registry."""
        if not self._strategy_registry:
            return ""
        strats = self._strategy_registry.strategies_for_bot(bot_id)
        return next(iter(strats)) if len(strats) == 1 else ""

    def _archetype_str(self, strategy_id: str) -> str:
        """Return archetype string for a strategy_id."""
        if not self._strategy_registry or not strategy_id:
            return ""
        arch = self._strategy_registry.archetype_for_strategy(strategy_id)
        return arch.value if arch and hasattr(arch, "value") else str(arch) if arch else ""

    def analyze_parameters(
        self, summary: BotWeeklySummary
    ) -> list[StrategySuggestion]:
        """Tier 1: Detect parameter misalignment from weekly stats."""
        suggestions: list[StrategySuggestion] = []

        # Tight stop detection: avg_loss is small relative to avg_win
        if summary.avg_win > 0 and summary.avg_loss != 0:
            loss_win_ratio = abs(summary.avg_loss) / summary.avg_win
            threshold = self._get_threshold(
                "tight_stop", "tight_stop_ratio", summary.bot_id,
                self.tight_stop_ratio,
            )
            if loss_win_ratio < threshold:
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
                        detection_context=DetectionContext(
                            detector_name="tight_stop",
                            bot_id=summary.bot_id,
                            threshold_name="tight_stop_ratio",
                            threshold_value=threshold,
                            observed_value=loss_win_ratio,
                        ),
                    )
                )

        return suggestions

    def analyze_filters(
        self, bot_id: str, filter_summaries: list[FilterWeeklySummary]
    ) -> list[StrategySuggestion]:
        """Tier 2: Detect filters that cost more than they save."""
        suggestions: list[StrategySuggestion] = []

        threshold = self._get_threshold(
            "filter_cost", "filter_cost_threshold", bot_id,
            self.filter_cost_threshold,
        )
        for f in filter_summaries:
            if f.net_impact_pnl < threshold:
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
                        detection_context=DetectionContext(
                            detector_name="filter_cost",
                            bot_id=bot_id,
                            threshold_name="filter_cost_threshold",
                            threshold_value=threshold,
                            observed_value=f.net_impact_pnl,
                        ),
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

        effective_min_weeks = int(self._get_threshold(
            "regime_loss", "regime_min_weeks", bot_id,
            float(self.regime_min_weeks),
        ))
        effective_loss_threshold = self._get_threshold(
            "regime_loss", "regime_loss_threshold", bot_id,
            self.regime_loss_threshold,
        )

        for trend in regime_trends:
            if len(trend.weekly_pnl) < effective_min_weeks:
                continue
            losing_weeks = sum(1 for pnl in trend.weekly_pnl if pnl < effective_loss_threshold)
            if losing_weeks < effective_min_weeks:
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
                    detection_context=DetectionContext(
                        detector_name="regime_loss",
                        bot_id=bot_id,
                        threshold_name="regime_min_weeks",
                        threshold_value=float(effective_min_weeks),
                        observed_value=float(losing_weeks),
                    ),
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
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        """Tier 4: Detect declining Sharpe ratio over 30/60/90 day windows."""
        if rolling_sharpe_90d <= 0:
            return []
        sid = strategy_id or self._resolve_strategy_id(bot_id)
        arch_default = self._archetype_default(sid, "alpha_decay", "decay_threshold")
        base_threshold = arch_default if arch_default is not None else decay_threshold
        # Check if 30d Sharpe is significantly below 90d Sharpe
        decay_ratio = (rolling_sharpe_90d - rolling_sharpe_30d) / rolling_sharpe_90d
        effective_threshold = self._get_threshold(
            "alpha_decay", "decay_threshold", bot_id, base_threshold,
        )
        if decay_ratio < effective_threshold:
            return []
        arch_str = self._archetype_str(sid)
        return [StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id=bot_id,
            strategy_id=sid,
            strategy_archetype=arch_str,
            title=f"Alpha decay detected — {bot_id}",
            description=(
                f"30d Sharpe ({rolling_sharpe_30d:.2f}) is {decay_ratio:.0%} below "
                f"90d Sharpe ({rolling_sharpe_90d:.2f}). The strategy may be losing edge. "
                f"Review signal quality and market regime alignment."
            ),
            evidence_days=90,
            confidence=min(0.9, 0.5 + decay_ratio),
            requires_human_judgment=True,
            detection_context=DetectionContext(
                detector_name="alpha_decay",
                bot_id=bot_id,
                threshold_name="decay_threshold",
                threshold_value=effective_threshold,
                observed_value=decay_ratio,
            ),
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
        effective_threshold = self._get_threshold(
            "signal_decay", "decay_threshold", bot_id, decay_threshold,
        )
        if drop < effective_threshold:
            return []
        return [StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id=bot_id,
            title=f"Signal quality decay — {bot_id}",
            description=(
                f"Signal->outcome correlation dropped from {signal_outcome_correlation_90d:.2f} "
                f"(90d) to {signal_outcome_correlation_30d:.2f} (30d). "
                f"Signal may need recalibration or replacement."
            ),
            evidence_days=90,
            confidence=min(0.9, 0.5 + drop),
            requires_human_judgment=True,
            detection_context=DetectionContext(
                detector_name="signal_decay",
                bot_id=bot_id,
                threshold_name="decay_threshold",
                threshold_value=effective_threshold,
                observed_value=drop,
            ),
        )]

    def detect_exit_timing_issues(
        self,
        bot_id: str,
        avg_exit_efficiency: float,
        premature_exit_pct: float,
        efficiency_threshold: float = 0.5,
        premature_threshold: float = 0.4,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        """Tier 3: Detect systematic premature exits."""
        sid = strategy_id or self._resolve_strategy_id(bot_id)
        arch_default = self._archetype_default(sid, "exit_timing", "efficiency_threshold")
        base_efficiency = arch_default if arch_default is not None else efficiency_threshold
        effective_efficiency = self._get_threshold(
            "exit_timing", "efficiency_threshold", bot_id, base_efficiency,
        )
        effective_premature = self._get_threshold(
            "exit_timing", "premature_threshold", bot_id, premature_threshold,
        )
        if avg_exit_efficiency >= effective_efficiency and premature_exit_pct <= effective_premature:
            return []
        suggestions: list[StrategySuggestion] = []
        arch_str = self._archetype_str(sid)
        if avg_exit_efficiency < effective_efficiency:
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.STRATEGY_VARIANT,
                bot_id=bot_id,
                strategy_id=sid,
                strategy_archetype=arch_str,
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
                detection_context=DetectionContext(
                    detector_name="exit_timing",
                    bot_id=bot_id,
                    threshold_name="efficiency_threshold",
                    threshold_value=effective_efficiency,
                    observed_value=avg_exit_efficiency,
                ),
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
            pair_id = f"{corr.bot_a}+{corr.bot_b}"
            effective_threshold = self._get_threshold(
                "correlation", "threshold", pair_id, threshold,
            )
            if corr.rolling_30d_correlation >= effective_threshold:
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.STRATEGY_VARIANT,
                    bot_id=pair_id,
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
                    detection_context=DetectionContext(
                        detector_name="correlation",
                        bot_id=pair_id,
                        threshold_name="threshold",
                        threshold_value=effective_threshold,
                        observed_value=corr.rolling_30d_correlation,
                    ),
                ))
        return suggestions

    def detect_time_of_day_patterns(
        self,
        bot_id: str,
        hourly_buckets: list,  # list[HourlyBucket]
        min_trades: int = 10,
        loss_threshold: float = 0.35,
        strategy_id: str = "",
    ) -> list[StrategySuggestion]:
        """Tier 2: Detect hours with consistently poor performance."""
        sid = strategy_id or self._resolve_strategy_id(bot_id)
        arch_str = self._archetype_str(sid)
        # Determine archetype relevance note
        high_relevance_archetypes = {
            "intraday_momentum", "opening_range_breakout", "vwap_pullback",
            "flow_following",
        }
        low_relevance_archetypes = {
            "trend_follow", "divergence_swing", "pullback",
        }
        if arch_str in high_relevance_archetypes:
            archetype_note = "HIGH RELEVANCE — intraday strategy, time-of-day is a primary performance lever"
        elif arch_str in low_relevance_archetypes:
            archetype_note = "LOW RELEVANCE — multi-day/swing strategy, time-of-day impact is secondary"
        else:
            archetype_note = ""

        effective_threshold = self._get_threshold(
            "time_of_day", "loss_threshold", bot_id, loss_threshold,
        )
        suggestions: list[StrategySuggestion] = []
        for bucket in hourly_buckets:
            if bucket.trade_count < min_trades:
                continue
            if bucket.pnl < 0 and bucket.win_rate < effective_threshold:
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.FILTER,
                    bot_id=bot_id,
                    strategy_id=sid,
                    strategy_archetype=arch_str,
                    archetype_note=archetype_note,
                    title=f"Poor hour {bucket.hour:02d}:00 — {bot_id}",
                    description=(
                        f"Hour {bucket.hour:02d}:00 UTC: {bucket.trade_count} trades, "
                        f"PnL ${bucket.pnl:.0f}, win rate {bucket.win_rate:.0%}. "
                        f"Consider adding a time-of-day gate to avoid this hour."
                    ),
                    evidence_days=7,
                    confidence=0.6,
                    detection_context=DetectionContext(
                        detector_name="time_of_day",
                        bot_id=bot_id,
                        threshold_name="loss_threshold",
                        threshold_value=effective_threshold,
                        observed_value=bucket.win_rate,
                    ),
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
        effective_threshold = self._get_threshold(
            "drawdown_concentration", "concentration_threshold", bot_id,
            concentration_threshold,
        )
        if concentration < effective_threshold:
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
            detection_context=DetectionContext(
                detector_name="drawdown_concentration",
                bot_id=bot_id,
                threshold_name="concentration_threshold",
                threshold_value=effective_threshold,
                observed_value=concentration,
            ),
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
        effective_threshold = self._get_threshold(
            "position_sizing", "loss_win_ratio_threshold", bot_id,
            loss_win_ratio_threshold,
        )
        if loss_win_ratio < effective_threshold:
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
            detection_context=DetectionContext(
                detector_name="position_sizing",
                bot_id=bot_id,
                threshold_name="loss_win_ratio_threshold",
                threshold_value=effective_threshold,
                observed_value=loss_win_ratio,
            ),
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
        effective_stability = self._get_threshold(
            "component_signal_decay", "stability_threshold", bot_id,
            stability_threshold,
        )
        effective_correlation = self._get_threshold(
            "component_signal_decay", "correlation_threshold", bot_id,
            correlation_threshold,
        )
        components = signal_health_data.get("components", [])
        degraded: list[str] = []

        for comp in components:
            trade_count = comp.get("trade_count", 0)
            if trade_count < min_trades:
                continue
            stability = comp.get("stability", 1.0)
            win_corr = abs(comp.get("win_correlation", 1.0))
            if stability < effective_stability or win_corr < effective_correlation:
                degraded.append(comp.get("component_name", "unknown"))

        if not degraded:
            return []

        min_stability = min(
            (comp.get("stability", 1.0) for comp in components
             if comp.get("component_name", "unknown") in degraded),
            default=0.0,
        )

        return [StrategySuggestion(
            tier=SuggestionTier.HYPOTHESIS,
            bot_id=bot_id,
            title=f"Signal component decay — {bot_id}",
            description=(
                f"Degraded signal components detected: {', '.join(degraded)}. "
                f"These components show low stability (<{effective_stability}) or "
                f"near-zero win correlation (<{effective_correlation}). "
                f"Review whether these signals still carry predictive value."
            ),
            evidence_days=7,
            confidence=0.5,
            requires_human_judgment=True,
            detection_context=DetectionContext(
                detector_name="component_signal_decay",
                bot_id=bot_id,
                threshold_name="stability_threshold",
                threshold_value=effective_stability,
                observed_value=min_stability,
            ),
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
        effective_redundancy = self._get_threshold(
            "filter_interactions", "redundancy_threshold", bot_id, 0.5,
        )
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
                    detection_context=DetectionContext(
                        detector_name="filter_interactions",
                        bot_id=bot_id,
                        threshold_name="redundancy_threshold",
                        threshold_value=effective_redundancy,
                        observed_value=redundancy,
                    ),
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

            effective_threshold = 1.0 if below else 0.0

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
                detection_context=DetectionContext(
                    detector_name="factor_decay",
                    bot_id=bot_id,
                    threshold_name="below_threshold",
                    threshold_value=effective_threshold,
                    observed_value=wr,
                ),
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
            trades_by_bot: bot_id -> list of TradeEvent
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
            # Use max(sharpe, 0.01) to avoid division by zero; zero/negative -> minimum alloc
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

    def detect_microstructure_issues(
        self,
        bot_id: str,
        orderbook_stats: dict,
        spread_threshold_bps: float = 5.0,
        imbalance_threshold: float = 2.0,
    ) -> list[StrategySuggestion]:
        """Tier 2: Detect adverse microstructure conditions at entry/exit."""
        effective_spread = self._get_threshold(
            "microstructure", "spread_threshold_bps", bot_id, spread_threshold_bps,
        )
        effective_imbalance = self._get_threshold(
            "microstructure", "imbalance_threshold", bot_id, imbalance_threshold,
        )
        suggestions: list[StrategySuggestion] = []

        by_context = orderbook_stats.get("by_context", {})
        entry_data = by_context.get("entry", {})
        if not entry_data:
            return []

        entry_spread = entry_data.get("spread_stats", {}).get("mean", 0)
        entry_imbalance = entry_data.get("imbalance_stats", {}).get("mean", 1.0)
        entry_count = entry_data.get("count", 0)

        if entry_count < 5:
            return []

        if entry_spread > effective_spread:
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.FILTER,
                bot_id=bot_id,
                title=f"Wide spreads at entry — {bot_id}",
                description=(
                    f"Average spread at entry is {entry_spread:.1f} bps "
                    f"(threshold: {effective_spread:.1f} bps) across {entry_count} entries. "
                    f"Consider adding a spread-width gate or preferring limit orders."
                ),
                evidence_days=7,
                confidence=0.6,
                detection_context=DetectionContext(
                    detector_name="microstructure",
                    bot_id=bot_id,
                    threshold_name="spread_threshold_bps",
                    threshold_value=effective_spread,
                    observed_value=entry_spread,
                ),
            ))

        if entry_imbalance > effective_imbalance or (entry_imbalance > 0 and entry_imbalance < 1.0 / effective_imbalance):
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.FILTER,
                bot_id=bot_id,
                title=f"Order book imbalance at entry — {bot_id}",
                description=(
                    f"Average bid/ask imbalance at entry is {entry_imbalance:.2f} "
                    f"across {entry_count} entries. Values far from 1.0 suggest "
                    f"positioning against order flow. Review trade direction vs "
                    f"book pressure for adverse selection."
                ),
                evidence_days=7,
                confidence=0.5,
                requires_human_judgment=True,
                detection_context=DetectionContext(
                    detector_name="microstructure",
                    bot_id=bot_id,
                    threshold_name="imbalance_threshold",
                    threshold_value=effective_imbalance,
                    observed_value=entry_imbalance,
                ),
            ))

        return suggestions

    def _infer_direction(self, suggestion: StrategySuggestion) -> int:
        """Infer change direction: +1 (increase), -1 (decrease), 0 (unknown).

        Tries numeric comparison first, falls back to keyword analysis.
        """
        # Try numeric comparison: suggested_value vs current_value
        try:
            sv = suggestion.suggested_value
            cv = suggestion.current_value
            if sv and cv:
                sv_f = float(str(sv).split("=")[-1].split(">")[0].split("<")[0].strip())
                cv_f = float(str(cv).split("=")[-1].split(">")[0].split("<")[0].strip())
                if sv_f > cv_f:
                    return 1
                elif sv_f < cv_f:
                    return -1
        except (ValueError, TypeError, IndexError, AttributeError):
            pass

        # Fall back to keyword analysis on title + description
        text = (suggestion.title + " " + suggestion.description).lower()
        for kw in self._INCREASE_KEYWORDS:
            if kw in text:
                return 1
        for kw in self._DECREASE_KEYWORDS:
            if kw in text:
                return -1
        return 0

    def _infer_direction_from_dict(self, rec: dict) -> int:
        """Infer direction from a persisted suggestion dict."""
        # Try proposed_value vs detection_context.threshold_value
        pv = rec.get("proposed_value")
        ctx = rec.get("detection_context") or {}
        cv = ctx.get("threshold_value") or ctx.get("observed_value")
        if pv is not None and cv is not None:
            try:
                if float(pv) > float(cv):
                    return 1
                elif float(pv) < float(cv):
                    return -1
            except (ValueError, TypeError):
                pass

        text = (rec.get("title", "") + " " + rec.get("description", "")).lower()
        for kw in self._INCREASE_KEYWORDS:
            if kw in text:
                return 1
        for kw in self._DECREASE_KEYWORDS:
            if kw in text:
                return -1
        return 0

    def _contradicts_recent(
        self, bot_id: str, detector_name: str, direction: int,
    ) -> bool:
        """Check if a recent suggestion from same detector+bot had opposite direction."""
        if direction == 0 or not self._recent_suggestions:
            return False
        for rec in self._recent_suggestions:
            if rec.get("bot_id") != bot_id:
                continue
            ctx = rec.get("detection_context") or {}
            rec_detector = ctx.get("detector_name", "")
            if rec_detector != detector_name:
                continue
            rec_direction = self._infer_direction_from_dict(rec)
            if rec_direction != 0 and rec_direction != direction:
                return True
        return False

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
        orderbook_stats: dict[str, dict] | None = None,
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

        if orderbook_stats:
            for bot_id, ob_data in orderbook_stats.items():
                all_suggestions.extend(
                    self.detect_microstructure_issues(bot_id, ob_data)
                )

        # Apply per-detector confidence calibration from outcome data
        if self._detector_confidence:
            calibrated: list[StrategySuggestion] = []
            for s in all_suggestions:
                det_name = ""
                if s.detection_context:
                    det_name = s.detection_context.detector_name
                multiplier = self._detector_confidence.get(det_name, 1.0)
                if multiplier != 1.0 and det_name:
                    adjusted_conf = round(s.confidence * multiplier, 3)
                    s = s.model_copy(update={"confidence": adjusted_conf})
                calibrated.append(s)
            all_suggestions = calibrated

        # Anti-oscillation: filter out suggestions that contradict recent ones
        if self._recent_suggestions:
            filtered: list[StrategySuggestion] = []
            for s in all_suggestions:
                det_name = ""
                if s.detection_context:
                    det_name = s.detection_context.detector_name
                direction = self._infer_direction(s)
                if det_name and direction != 0 and self._contradicts_recent(
                    s.bot_id, det_name, direction,
                ):
                    continue  # Skip contradictory suggestion
                filtered.append(s)
            all_suggestions = filtered

        # If convergence report shows oscillation, dampen all confidence
        if self._convergence_report.get("oscillation_detected"):
            all_suggestions = [
                s.model_copy(update={"confidence": round(s.confidence * 0.7, 3)})
                for s in all_suggestions
            ]

        # Optimization allocation: adjust confidence based on category value
        if self._category_value_map:
            adjusted: list[StrategySuggestion] = []
            for s in all_suggestions:
                det_name_for_cat = ""
                if s.detection_context:
                    det_name_for_cat = s.detection_context.detector_name
                cat = self._DETECTOR_TO_CATEGORY.get(det_name_for_cat, "")
                key = f"{s.bot_id}:{cat}" if cat else ""
                entry = self._category_value_map.get(key, {}) if key else {}
                vps = entry.get("value_per_suggestion") if entry else None
                if entry.get("unexplored"):
                    pass  # neutral treatment — don't penalize unexplored categories
                elif vps is not None and vps != 0:
                    # Scale factor proportionally, clamped to +-10%
                    raw_adj = max(-0.1, min(0.1, vps * 0.5))
                    factor = 1.0 + raw_adj
                    s = s.model_copy(update={
                        "confidence": round(s.confidence * factor, 3),
                    })
                adjusted.append(s)
            all_suggestions = adjusted

        # Suppress suggestions for categories with proven poor track records
        if self._category_scorecard:
            all_suggestions = [
                s for s in all_suggestions
                if not self._should_suppress(s.bot_id, s.tier.value)
            ]

        return RefinementReport(
            week_start=self.week_start,
            week_end=self.week_end,
            suggestions=all_suggestions,
        )

    def _should_suppress(self, bot_id: str, tier_value: str) -> bool:
        """Check if a (bot_id, tier) pair should be suppressed due to poor track record.

        Maps scorecard categories back to suggestion tiers via CATEGORY_TO_TIER,
        then suppresses when sample_size >= 5 AND win_rate < 0.3 AND avg_pnl_delta < 0.
        """
        if not self._category_scorecard:
            return False
        scores = getattr(self._category_scorecard, "scores", None)
        if not scores:
            return False
        from schemas.agent_response import CATEGORY_TO_TIER
        for score in scores:
            if score.bot_id != bot_id:
                continue
            mapped_tier = CATEGORY_TO_TIER.get(score.category, score.category)
            if mapped_tier != tier_value:
                continue
            if score.sample_size >= 5 and score.win_rate < 0.3 and score.avg_pnl_delta < 0:
                return True
        return False

    # ── Portfolio-level detectors (Phase 2) ───────────────────────────

    def detect_family_imbalance(
        self,
        family_summaries: dict[str, dict],
        family_allocations: dict[str, float],
        min_days: int = 30,
    ) -> list[StrategySuggestion]:
        """Detect families consistently underperforming their allocation weight (2A).

        Args:
            family_summaries: family → {total_net_pnl, trade_count, days, ...}
            family_allocations: family → allocation weight (0-1)
        """
        suggestions: list[StrategySuggestion] = []
        if not family_summaries or not family_allocations:
            return suggestions

        total_pnl = sum(s.get("total_net_pnl", 0.0) for s in family_summaries.values())
        if total_pnl == 0:
            return suggestions

        for family, summary in family_summaries.items():
            alloc_weight = family_allocations.get(family, 0.0)
            if alloc_weight <= 0:
                continue

            days = summary.get("days", 0)
            if days < min_days:
                continue

            family_pnl = summary.get("total_net_pnl", 0.0)
            pnl_share = family_pnl / total_pnl  # safe: total_pnl != 0 guarded above

            # Family PnL share is significantly below its allocation weight
            if alloc_weight > 0.1 and pnl_share < alloc_weight * 0.5:
                # Suggest rebalancing — max 15% shift
                current = alloc_weight
                suggested = max(0.05, current - min(0.15, current * 0.3))
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.PORTFOLIO,
                    bot_id="PORTFOLIO",
                    title=f"Reduce {family} family allocation",
                    description=(
                        f"{family} family contributes {pnl_share:.1%} of PnL but holds "
                        f"{alloc_weight:.1%} allocation over {days} days. "
                        f"Consider reducing from {current:.1%} to {suggested:.1%}."
                    ),
                    current_value=f"{current:.4f}",
                    suggested_value=f"{suggested:.4f}",
                    evidence_days=days,
                    confidence=min(0.7, 0.4 + (days - min_days) / 100),
                    detection_context=DetectionContext(
                        detector_name="detect_family_imbalance",
                        bot_id="PORTFOLIO",
                        threshold_name="alloc_weight",
                        threshold_value=alloc_weight,
                        observed_value=round(pnl_share, 4),
                    ),
                ))

        return suggestions

    def detect_correlation_concentration(
        self,
        correlation_matrix: dict[str, float],
        current_allocations: dict[str, float],
        threshold: float = 0.7,
        weight_threshold: float = 0.4,
    ) -> list[StrategySuggestion]:
        """Detect pairs with high correlation holding excessive combined weight (2B).

        Args:
            correlation_matrix: "botA_botB" → correlation coefficient
            current_allocations: bot_id → allocation weight (0-1)
        """
        suggestions: list[StrategySuggestion] = []
        if not correlation_matrix or not current_allocations:
            return suggestions

        for pair_key, corr_val in correlation_matrix.items():
            if corr_val <= threshold:
                continue

            parts = pair_key.split("_", 1)
            if len(parts) != 2:
                continue
            bot_a, bot_b = parts

            weight_a = current_allocations.get(bot_a, 0.0)
            weight_b = current_allocations.get(bot_b, 0.0)
            combined = weight_a + weight_b

            if combined > weight_threshold:
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.PORTFOLIO,
                    bot_id="PORTFOLIO",
                    title=f"Reduce correlated pair {bot_a}/{bot_b} combined weight",
                    description=(
                        f"{bot_a} and {bot_b} have correlation {corr_val:.2f} "
                        f"with combined allocation {combined:.1%} (>{weight_threshold:.0%}). "
                        f"High correlation with high combined weight creates concentration risk."
                    ),
                    confidence=min(0.8, 0.5 + (corr_val - threshold) * 2),
                    detection_context=DetectionContext(
                        detector_name="detect_correlation_concentration",
                        bot_id="PORTFOLIO",
                        threshold_name="correlation_threshold",
                        threshold_value=threshold,
                        observed_value=corr_val,
                    ),
                ))

        return suggestions

    def detect_drawdown_tier_miscalibration(
        self,
        historical_drawdowns: list[float],
        current_tiers: list[list[float]],
        min_days: int = 90,
    ) -> list[StrategySuggestion]:
        """Detect drawdown tiers that never trigger or trigger too often (2C).

        Safety: only suggests narrowing, never removing or loosening.

        Args:
            historical_drawdowns: list of daily drawdown percentages (0-100)
            current_tiers: list of [threshold_pct, multiplier] pairs
        """
        suggestions: list[StrategySuggestion] = []
        if len(historical_drawdowns) < min_days or not current_tiers:
            return suggestions

        for tier_idx, tier in enumerate(current_tiers):
            if len(tier) < 2:
                continue
            threshold = tier[0]

            # Count how many days breached this tier
            breaches = sum(1 for dd in historical_drawdowns if dd >= threshold)
            breach_rate = breaches / len(historical_drawdowns)

            if breach_rate == 0.0 and tier_idx > 0:
                # Tier never triggers — may be too loose. Suggest narrowing.
                prev_threshold = current_tiers[tier_idx - 1][0] if tier_idx > 0 else 0
                midpoint = (prev_threshold + threshold) / 2
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.PORTFOLIO,
                    bot_id="PORTFOLIO",
                    title=f"Tighten drawdown tier {tier_idx + 1} threshold",
                    description=(
                        f"Drawdown tier at {threshold}% never triggered in "
                        f"{len(historical_drawdowns)} days. Consider narrowing from "
                        f"{threshold}% to {midpoint:.1f}% (midpoint with tier {tier_idx})."
                    ),
                    current_value=f"{threshold}",
                    suggested_value=f"{midpoint:.1f}",
                    evidence_days=len(historical_drawdowns),
                    confidence=0.5,
                    detection_context=DetectionContext(
                        detector_name="detect_drawdown_tier_miscalibration",
                        bot_id="PORTFOLIO",
                        threshold_name=f"drawdown_tier_{tier_idx}",
                        threshold_value=threshold,
                        observed_value=0.0,
                    ),
                ))
            elif breach_rate > 0.3:
                # Tier triggers too often — suggests threshold is too tight
                # But we only narrow, never loosen — so no suggestion here
                pass

        return suggestions

    def detect_coordination_gaps(
        self,
        concurrent_positions: dict,
        existing_coordination: dict | None = None,
        min_co_occurrences: int = 50,
    ) -> list[StrategySuggestion]:
        """Detect strategies that frequently collide without coordination rules (2D).

        Args:
            concurrent_positions: from build_concurrent_position_analysis()
            existing_coordination: CoordinationConfig dict (signals, cooldown_pairs)
        """
        suggestions: list[StrategySuggestion] = []
        if not concurrent_positions:
            return suggestions

        pairs = concurrent_positions.get("pairs", {})
        existing_cooldowns: set[str] = set()
        if existing_coordination:
            for cp in existing_coordination.get("cooldown_pairs", []):
                strats = cp.get("strategies", [])
                if len(strats) >= 2:
                    existing_cooldowns.add("_".join(sorted(strats[:2])))

        # observation_days from top-level metadata, or estimate from co-occurrence counts
        obs_days = concurrent_positions.get("observation_days", 0)

        for pair_key, data in pairs.items():
            co_occ = data.get("co_occurrences", 0)
            if co_occ < min_co_occurrences:
                continue

            # Skip if already coordinated
            if pair_key in existing_cooldowns:
                continue

            same_dir = data.get("same_direction_count", 0)
            same_dir_pct = same_dir / co_occ if co_occ > 0 else 0.0

            # Use observation_days from data, or estimate conservatively from co-occurrences
            pair_days = data.get("observation_days", obs_days) or max(co_occ // 2, 0)

            if same_dir_pct > 0.6:
                suggestions.append(StrategySuggestion(
                    tier=SuggestionTier.PORTFOLIO,
                    bot_id="PORTFOLIO",
                    title=f"Add coordination for {pair_key}",
                    description=(
                        f"Strategies {pair_key} have {co_occ} co-occurrences with "
                        f"{same_dir_pct:.0%} same-direction over ~{pair_days} days. "
                        f"No coordination rule exists. "
                        f"Consider adding a cooldown or direction filter."
                    ),
                    evidence_days=pair_days,
                    confidence=min(0.7, 0.4 + co_occ / 200),
                    detection_context=DetectionContext(
                        detector_name="detect_coordination_gaps",
                        bot_id="PORTFOLIO",
                        threshold_name="same_direction_pct",
                        threshold_value=0.6,
                        observed_value=round(same_dir_pct, 3),
                    ),
                ))

        return suggestions

    def detect_heat_cap_utilization(
        self,
        daily_heat_series: list[float],
        heat_cap_R: float,
        min_days: int = 30,
    ) -> list[StrategySuggestion]:
        """Detect heat cap consistently too tight or too loose (2E).

        Args:
            daily_heat_series: list of daily peak heat values (in R)
            heat_cap_R: current heat cap setting
        """
        suggestions: list[StrategySuggestion] = []
        if len(daily_heat_series) < min_days or heat_cap_R <= 0:
            return suggestions

        utilization_ratios = [h / heat_cap_R for h in daily_heat_series if heat_cap_R > 0]
        if not utilization_ratios:
            return suggestions

        avg_util = sum(utilization_ratios) / len(utilization_ratios)
        high_util_days = sum(1 for u in utilization_ratios if u > 0.9)
        high_util_pct = high_util_days / len(utilization_ratios)

        if high_util_pct > 0.3:
            # Consistently near cap — opportunity cost
            # Max +10% adjustment
            suggested = round(heat_cap_R * 1.10, 1)
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.PORTFOLIO,
                bot_id="PORTFOLIO",
                title="Increase heat_cap_R — consistently at capacity",
                description=(
                    f"Heat utilization exceeds 90% of {heat_cap_R}R on "
                    f"{high_util_pct:.0%} of days ({high_util_days}/{len(utilization_ratios)}). "
                    f"Avg utilization: {avg_util:.0%}. "
                    f"Consider increasing from {heat_cap_R}R to {suggested}R (+10%)."
                ),
                current_value=str(heat_cap_R),
                suggested_value=str(suggested),
                evidence_days=len(daily_heat_series),
                confidence=min(0.7, 0.4 + high_util_pct),
                detection_context=DetectionContext(
                    detector_name="detect_heat_cap_utilization",
                    bot_id="PORTFOLIO",
                    threshold_name="heat_cap_R",
                    threshold_value=heat_cap_R,
                    observed_value=round(avg_util * heat_cap_R, 2),
                ),
            ))
        elif avg_util < 0.3:
            # Very underutilized — overly conservative
            suggested = round(heat_cap_R * 0.90, 1)
            suggestions.append(StrategySuggestion(
                tier=SuggestionTier.PORTFOLIO,
                bot_id="PORTFOLIO",
                title="Reduce heat_cap_R — significantly underutilized",
                description=(
                    f"Heat utilization averages only {avg_util:.0%} of {heat_cap_R}R "
                    f"over {len(utilization_ratios)} days. Cap may be overly conservative. "
                    f"Consider reducing from {heat_cap_R}R to {suggested}R (-10%)."
                ),
                current_value=str(heat_cap_R),
                suggested_value=str(suggested),
                evidence_days=len(daily_heat_series),
                confidence=0.5,
                detection_context=DetectionContext(
                    detector_name="detect_heat_cap_utilization",
                    bot_id="PORTFOLIO",
                    threshold_name="heat_cap_R",
                    threshold_value=heat_cap_R,
                    observed_value=round(avg_util * heat_cap_R, 2),
                ),
            ))

        return suggestions
