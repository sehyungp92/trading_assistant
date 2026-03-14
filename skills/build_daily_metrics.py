"""Data reduction pipeline — events -> per-bot curated daily files.

Claude sees reduced, classified, pre-scored data. This deterministic pipeline
does the heavy lifting of classification. Claude does interpretation and synthesis.

Output directory: data/curated/<date>/<bot_id>/
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from schemas.events import TradeEvent, MissedOpportunityEvent
from schemas.exit_efficiency import ExitEfficiencyStats
from schemas.factor_attribution import FactorAttribution, FactorStats
from schemas.daily_metrics import (
    BotDailySummary,
    WinnerLoserRecord,
    ProcessFailureRecord,
    NotableMissedRecord,
    RegimeAnalysis,
    FilterAnalysis,
    RootCauseSummary,
)


class DailyMetricsBuilder:
    """Builds curated daily metrics for a single bot on a single day."""

    def __init__(self, date: str, bot_id: str, bot_timezone: str = "UTC") -> None:
        self.date = date
        self.bot_id = bot_id
        self.bot_timezone = bot_timezone

    def build_summary(
        self,
        trades: list[TradeEvent],
        daily_snapshot: dict | None = None,
        missed: list[MissedOpportunityEvent] | None = None,
    ) -> BotDailySummary:
        """Aggregate trade list into daily summary stats."""
        if not trades:
            summary = BotDailySummary(date=self.date, bot_id=self.bot_id)
            self._merge_snapshot_into_summary(summary, daily_snapshot, trades=[], missed=missed or [])
            return summary

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_pnl = sum(t.pnl for t in trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        avg_pq = sum(t.process_quality_score for t in trades) / len(trades)

        summary = BotDailySummary(
            date=self.date,
            bot_id=self.bot_id,
            total_trades=len(trades),
            win_count=len(wins),
            loss_count=len(losses),
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,  # fees not separately tracked yet
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_process_quality=avg_pq,
        )
        self._merge_snapshot_into_summary(summary, daily_snapshot, trades=trades, missed=missed or [])
        return summary

    @staticmethod
    def _count_missed_winners(missed: list[MissedOpportunityEvent]) -> int:
        winners = 0
        for event in missed:
            if event.would_have_hit_tp is True:
                winners += 1
            elif event.would_have_hit_tp is None and (event.outcome_24h or 0.0) > 0:
                winners += 1
        return winners

    def _merge_snapshot_into_summary(
        self,
        summary: BotDailySummary,
        daily_snapshot: dict | None,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
    ) -> None:
        if not daily_snapshot:
            if missed:
                summary.missed_count = len(missed)
                summary.missed_would_have_won = self._count_missed_winners(missed)
            return

        # Preserve trade-derived metrics when real trade events exist.
        if not trades:
            summary.total_trades = int(daily_snapshot.get("total_trades", summary.total_trades) or 0)
            summary.win_count = int(daily_snapshot.get("win_count", summary.win_count) or 0)
            summary.loss_count = int(daily_snapshot.get("loss_count", summary.loss_count) or 0)
            summary.gross_pnl = float(daily_snapshot.get("gross_pnl", summary.gross_pnl) or 0.0)
            summary.net_pnl = float(daily_snapshot.get("net_pnl", summary.net_pnl) or 0.0)
            summary.avg_win = float(daily_snapshot.get("avg_win", summary.avg_win) or 0.0)
            summary.avg_loss = float(daily_snapshot.get("avg_loss", summary.avg_loss) or 0.0)
            summary.avg_process_quality = float(
                daily_snapshot.get("avg_process_quality", summary.avg_process_quality) or 0.0
            )

        summary.max_drawdown_pct = float(
            daily_snapshot.get("max_drawdown_pct", summary.max_drawdown_pct) or 0.0
        )
        summary.sharpe_rolling_30d = float(
            daily_snapshot.get("sharpe_rolling_30d", summary.sharpe_rolling_30d) or 0.0
        )
        summary.sortino_rolling_30d = float(
            daily_snapshot.get("sortino_rolling_30d", summary.sortino_rolling_30d) or 0.0
        )
        summary.exposure_pct = float(daily_snapshot.get("exposure_pct", summary.exposure_pct) or 0.0)
        summary.error_count = int(daily_snapshot.get("error_count", summary.error_count) or 0)
        summary.uptime_pct = float(daily_snapshot.get("uptime_pct", summary.uptime_pct) or 0.0)

        if missed:
            summary.missed_count = len(missed)
            summary.missed_would_have_won = self._count_missed_winners(missed)
        else:
            summary.missed_count = int(daily_snapshot.get("missed_count", summary.missed_count) or 0)
            summary.missed_would_have_won = int(
                daily_snapshot.get("missed_would_have_won", summary.missed_would_have_won) or 0
            )

    def top_winners(self, trades: list[TradeEvent], n: int = 5) -> list[WinnerLoserRecord]:
        """Top N winning trades by PnL, descending."""
        wins = sorted([t for t in trades if t.pnl > 0], key=lambda t: t.pnl, reverse=True)
        return [self._to_winner_loser(t) for t in wins[:n]]

    def top_losers(self, trades: list[TradeEvent], n: int = 5) -> list[WinnerLoserRecord]:
        """Top N losing trades by PnL, ascending (most negative first)."""
        losses = sorted([t for t in trades if t.pnl <= 0], key=lambda t: t.pnl)
        return [self._to_winner_loser(t) for t in losses[:n]]

    def process_failures(
        self, trades: list[TradeEvent], threshold: int = 60
    ) -> list[ProcessFailureRecord]:
        """Trades with process_quality_score below threshold."""
        return [
            ProcessFailureRecord(
                trade_id=t.trade_id,
                bot_id=t.bot_id,
                pair=t.pair,
                process_quality_score=t.process_quality_score,
                root_causes=t.root_causes,
                pnl=t.pnl,
                entry_signal=t.entry_signal,
                market_regime=t.market_regime,
            )
            for t in trades
            if t.process_quality_score < threshold
        ]

    def notable_missed(
        self,
        missed: list[MissedOpportunityEvent],
        trades: list[TradeEvent],
    ) -> list[NotableMissedRecord]:
        """Missed opportunities where outcome > 2x average win."""
        wins = [t for t in trades if t.pnl > 0]
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        threshold = 2.0 * avg_win if avg_win > 0 else float("inf")

        return [
            NotableMissedRecord(
                bot_id=m.bot_id,
                pair=m.pair,
                signal=m.signal,
                blocked_by=m.blocked_by,
                hypothetical_entry=m.hypothetical_entry,
                outcome_24h=m.outcome_24h or 0.0,
                confidence=m.confidence,
                assumption_tags=m.assumption_tags,
            )
            for m in missed
            if (m.outcome_24h or 0.0) > threshold
        ]

    def regime_analysis(self, trades: list[TradeEvent]) -> RegimeAnalysis:
        """PnL breakdown by market regime."""
        regime_pnl: dict[str, float] = defaultdict(float)
        regime_count: dict[str, int] = defaultdict(int)
        regime_wins: dict[str, int] = defaultdict(int)

        for t in trades:
            regime = t.market_regime or "unknown"
            regime_pnl[regime] += t.pnl
            regime_count[regime] += 1
            if t.pnl > 0:
                regime_wins[regime] += 1

        regime_win_rate = {
            r: regime_wins.get(r, 0) / c for r, c in regime_count.items()
        }

        return RegimeAnalysis(
            bot_id=self.bot_id,
            date=self.date,
            regime_pnl=dict(regime_pnl),
            regime_trade_count=dict(regime_count),
            regime_win_rate=regime_win_rate,
        )

    def filter_analysis(self, missed: list[MissedOpportunityEvent]) -> FilterAnalysis:
        """Impact analysis for each filter."""
        block_counts: dict[str, int] = defaultdict(int)
        saved_pnl: dict[str, float] = defaultdict(float)
        missed_pnl: dict[str, float] = defaultdict(float)

        for m in missed:
            filt = m.blocked_by or "unknown"
            block_counts[filt] += 1
            outcome = m.outcome_24h or 0.0
            if outcome <= 0:
                saved_pnl[filt] += abs(outcome)
            else:
                missed_pnl[filt] += outcome

        return FilterAnalysis(
            bot_id=self.bot_id,
            date=self.date,
            filter_block_counts=dict(block_counts),
            filter_saved_pnl=dict(saved_pnl),
            filter_missed_pnl=dict(missed_pnl),
        )

    def root_cause_summary(self, trades: list[TradeEvent]) -> RootCauseSummary:
        """Distribution of root causes across all trades."""
        dist: dict[str, int] = defaultdict(int)
        for t in trades:
            for cause in t.root_causes:
                dist[cause] += 1

        return RootCauseSummary(
            bot_id=self.bot_id,
            date=self.date,
            distribution=dict(dist),
            total_trades=len(trades),
        )

    def hourly_performance(self, trades: list[TradeEvent]) -> dict:
        """Compute time-of-day performance buckets via HourlyAnalyzer."""
        from skills.hourly_analyzer import HourlyAnalyzer
        analyzer = HourlyAnalyzer(bot_id=self.bot_id, date=self.date, bot_timezone=self.bot_timezone)
        return analyzer.compute(trades).model_dump(mode="json")

    def slippage_stats(self, trades: list[TradeEvent]) -> dict:
        """Compute per-symbol, per-hour slippage distributions via SlippageAnalyzer."""
        from skills.slippage_analyzer import SlippageAnalyzer
        analyzer = SlippageAnalyzer(bot_id=self.bot_id, date=self.date, bot_timezone=self.bot_timezone)
        return analyzer.compute(trades).model_dump(mode="json")

    def factor_attribution(self, trades: list[TradeEvent]) -> FactorAttribution:
        """Per-factor performance attribution across daily trades."""
        factor_data: dict[str, dict] = {}

        for t in trades:
            if not t.signal_factors:
                continue
            for fd in t.signal_factors:
                name = fd["factor_name"]
                if name not in factor_data:
                    factor_data[name] = {
                        "trade_count": 0,
                        "win_count": 0,
                        "total_pnl": 0.0,
                        "contributions": [],
                    }
                bucket = factor_data[name]
                bucket["trade_count"] += 1
                if t.pnl > 0:
                    bucket["win_count"] += 1
                bucket["total_pnl"] += t.pnl
                bucket["contributions"].append(fd.get("contribution", 0.0))

        factors = sorted(
            [
                FactorStats(
                    factor_name=name,
                    trade_count=d["trade_count"],
                    win_count=d["win_count"],
                    total_pnl=d["total_pnl"],
                    avg_contribution=(
                        sum(d["contributions"]) / len(d["contributions"])
                        if d["contributions"]
                        else 0.0
                    ),
                )
                for name, d in factor_data.items()
            ],
            key=lambda f: f.factor_name,
        )

        return FactorAttribution(
            bot_id=self.bot_id,
            date=self.date,
            factors=factors,
        )

    def exit_efficiency(self, trades: list[TradeEvent]) -> ExitEfficiencyStats:
        """Measure how well exits capture available price moves.

        For each trade with post-exit price data, compute:
        - continuation: how much price moved after exit
        - exit_efficiency: actual_capture / max_available_move
        - premature exit detection (left money on the table)
        """
        efficiencies: list[float] = []
        premature_count = 0
        reason_effs: dict[str, list[float]] = defaultdict(list)
        regime_effs: dict[str, list[float]] = defaultdict(list)

        for t in trades:
            if t.post_exit_1h_price is None and t.post_exit_4h_price is None:
                continue

            # Compute continuation amounts
            cont_1h = (t.post_exit_1h_price - t.exit_price) if t.post_exit_1h_price is not None else None
            cont_4h = (t.post_exit_4h_price - t.exit_price) if t.post_exit_4h_price is not None else None

            # For SHORT trades, flip sign: price going down after exit = left money
            if t.side == "SHORT":
                if cont_1h is not None:
                    cont_1h = -cont_1h
                if cont_4h is not None:
                    cont_4h = -cont_4h

            # Detect premature exit: positive continuation means price kept going
            # in the trade's favor after exit
            best_continuation = max(
                c for c in [cont_1h, cont_4h] if c is not None
            )
            if best_continuation > 0:
                premature_count += 1

            # Compute exit efficiency = actual_capture / max_available_move
            # actual_capture is the trade's PnL (unsigned magnitude)
            actual_capture = abs(t.pnl)
            # max_available = actual capture + best continuation (if positive)
            max_available = actual_capture + max(best_continuation, 0.0)

            if max_available > 0:
                eff = actual_capture / max_available
            else:
                eff = 1.0  # no further move available, exit was optimal

            efficiencies.append(eff)

            reason = t.exit_reason or "unknown"
            regime = t.market_regime or "unknown"
            reason_effs[reason].append(eff)
            regime_effs[regime].append(eff)

        total_with_data = len(efficiencies)
        avg_eff = sum(efficiencies) / total_with_data if total_with_data else 0.0
        premature_pct = premature_count / total_with_data if total_with_data else 0.0

        by_exit_reason = {
            r: sum(vals) / len(vals) for r, vals in reason_effs.items()
        }
        by_regime = {
            r: sum(vals) / len(vals) for r, vals in regime_effs.items()
        }

        return ExitEfficiencyStats(
            bot_id=self.bot_id,
            date=self.date,
            avg_efficiency=avg_eff,
            premature_exit_pct=premature_pct,
            by_exit_reason=by_exit_reason,
            by_regime=by_regime,
            total_trades_with_data=total_with_data,
        )

    def build_per_strategy_from_snapshot(
        self, daily_snapshot: dict,
    ) -> dict[str, "PerStrategySummary"]:
        """Map DailySnapshot.per_strategy_summary (loose dict) into typed objects."""
        from schemas.daily_metrics import PerStrategySummary

        result = {}
        raw = daily_snapshot.get("per_strategy_summary", {})

        for strategy_id, data in raw.items():
            if not isinstance(data, dict):
                continue

            # Normalize win_rate: bots may emit 0-100 or 0-1
            win_rate = data.get("win_rate", 0.0)
            if win_rate > 1.0:
                win_rate = win_rate / 100.0

            result[strategy_id] = PerStrategySummary(
                strategy_id=strategy_id,
                trades=data.get("trades", 0),
                win_count=data.get("win_count", 0),
                loss_count=data.get("loss_count", 0),
                gross_pnl=data.get("gross_pnl", 0.0),
                net_pnl=data.get("net_pnl", 0.0),
                win_rate=win_rate,
                avg_win=data.get("avg_win", 0.0),
                avg_loss=data.get("avg_loss", 0.0),
                best_trade_pnl=data.get("best_trade_pnl", 0.0),
                worst_trade_pnl=data.get("worst_trade_pnl", 0.0),
                avg_entry_slippage_bps=data.get("avg_entry_slippage_bps"),
            )

        return result

    def build_excursion_stats(self, trades: list[TradeEvent]) -> dict:
        """Build aggregate excursion statistics from trades with MFE/MAE data."""
        excursion_trades = [t for t in trades if t.mfe_pct is not None and t.mae_pct is not None]

        if not excursion_trades:
            return {"coverage": 0, "total_trades": len(trades), "stats": {}}

        mfe_pcts = [t.mfe_pct for t in excursion_trades]
        mae_pcts = [t.mae_pct for t in excursion_trades]
        efficiencies = [t.exit_efficiency for t in excursion_trades if t.exit_efficiency is not None]
        mfe_rs = [t.mfe_r for t in excursion_trades if t.mfe_r is not None]
        mae_rs = [t.mae_r for t in excursion_trades if t.mae_r is not None]

        def _stats(values: list[float]) -> dict:
            if not values:
                return {}
            s = sorted(values)
            n = len(s)
            return {
                "mean": sum(s) / n,
                "median": s[n // 2],
                "p25": s[max(0, n // 4)],
                "p75": s[max(0, 3 * n // 4)],
                "min": s[0],
                "max": s[-1],
            }

        # Winners vs losers breakdown
        winners = [t for t in excursion_trades if t.pnl > 0]
        losers = [t for t in excursion_trades if t.pnl <= 0]

        return {
            "coverage": len(excursion_trades),
            "total_trades": len(trades),
            "coverage_pct": len(excursion_trades) / len(trades) * 100 if trades else 0,
            "stats": {
                "mfe_pct": _stats(mfe_pcts),
                "mae_pct": _stats(mae_pcts),
                "exit_efficiency": _stats(efficiencies),
                "mfe_r": _stats(mfe_rs),
                "mae_r": _stats(mae_rs),
            },
            "winners": {
                "count": len(winners),
                "avg_mfe_pct": sum(t.mfe_pct for t in winners) / len(winners) if winners else 0,
                "avg_mae_pct": sum(t.mae_pct for t in winners) / len(winners) if winners else 0,
                "avg_exit_efficiency": (
                    sum(t.exit_efficiency for t in winners if t.exit_efficiency is not None)
                    / sum(1 for t in winners if t.exit_efficiency is not None)
                    if any(t.exit_efficiency is not None for t in winners) else 0
                ),
            },
            "losers": {
                "count": len(losers),
                "avg_mfe_pct": sum(t.mfe_pct for t in losers) / len(losers) if losers else 0,
                "avg_mae_pct": sum(t.mae_pct for t in losers) / len(losers) if losers else 0,
            },
        }

    def build_experiment_breakdown(self, snapshot_data: dict) -> dict:
        """Extract experiment breakdown from DailySnapshot data.

        Args:
            snapshot_data: Dict containing DailySnapshot fields including
                optional 'experiment_breakdown' dict.

        Returns:
            Dict with per-experiment, per-variant daily metrics.
        """
        breakdown = snapshot_data.get("experiment_breakdown", {})
        if not breakdown:
            return {}
        return breakdown

    def coordinator_impact(self, coordination_events: list[dict]) -> dict:
        """Aggregate coordinator events into a daily summary.

        Args:
            coordination_events: list of raw coordinator event dicts
                (parsed from coordination_YYYY-MM-DD.jsonl)

        Returns:
            dict with event counts, action breakdown, and rule summary.
        """
        if not coordination_events:
            return {
                "bot_id": self.bot_id,
                "date": self.date,
                "total_events": 0,
                "by_action": {},
                "by_rule": {},
                "symbols_affected": [],
                "events": [],
            }

        by_action: dict[str, int] = defaultdict(int)
        by_rule: dict[str, int] = defaultdict(int)
        symbols: set[str] = set()

        for evt in coordination_events:
            action = evt.get("action", "unknown")
            rule = evt.get("rule", "unknown")
            symbol = evt.get("symbol", "")
            by_action[action] += 1
            by_rule[rule] += 1
            if symbol:
                symbols.add(symbol)

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "total_events": len(coordination_events),
            "by_action": dict(by_action),
            "by_rule": dict(by_rule),
            "symbols_affected": sorted(symbols),
            "events": coordination_events,
        }

    def build_filter_decision_summary(self, filter_events: list[dict]) -> dict:
        """Summarize per-filter pass/block decisions from FilterDecisionEvent data."""
        from collections import defaultdict

        per_filter: dict[str, dict] = defaultdict(lambda: {
            "pass_count": 0,
            "block_count": 0,
            "margins": [],
            "near_misses": 0,
        })

        for evt in filter_events:
            payload = evt.get("payload", evt)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue

            name = payload.get("filter_name", "unknown")
            passed = payload.get("passed", True)
            threshold = payload.get("threshold", 0)
            actual = payload.get("actual_value", 0)

            bucket = per_filter[name]
            if passed:
                bucket["pass_count"] += 1
            else:
                bucket["block_count"] += 1

            margin_pct = 0.0
            if threshold != 0:
                margin_pct = ((actual - threshold) / abs(threshold)) * 100
            bucket["margins"].append(margin_pct)

            if abs(margin_pct) < 10:
                bucket["near_misses"] += 1

        summary = {}
        for name, data in per_filter.items():
            margins = data["margins"]
            total = data["pass_count"] + data["block_count"]
            summary[name] = {
                "pass_count": data["pass_count"],
                "block_count": data["block_count"],
                "total": total,
                "block_rate": data["block_count"] / total if total > 0 else 0,
                "avg_margin_pct": sum(margins) / len(margins) if margins else 0,
                "near_miss_count": data["near_misses"],
                "near_miss_rate": data["near_misses"] / len(margins) if margins else 0,
            }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "filters": summary,
            "total_evaluations": sum(d["pass_count"] + d["block_count"] for d in per_filter.values()),
        }

    def build_indicator_snapshot_summary(self, indicator_events: list[dict]) -> dict:
        """Summarize indicator values at signal evaluation points."""
        from collections import defaultdict

        indicator_values: dict[str, list[float]] = defaultdict(list)
        decision_counts: dict[str, int] = defaultdict(int)
        signal_strengths: list[float] = []

        for evt in indicator_events:
            payload = evt.get("payload", evt)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue

            indicators = payload.get("indicators", {})
            for name, value in indicators.items():
                if isinstance(value, (int, float)):
                    indicator_values[name].append(value)

            decision = payload.get("decision", "skip")
            decision_counts[decision] += 1

            strength = payload.get("signal_strength", 0)
            if isinstance(strength, (int, float)):
                signal_strengths.append(strength)

        indicator_stats = {}
        for name, values in indicator_values.items():
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            indicator_stats[name] = {
                "count": n,
                "mean": sum(values) / n if n else 0,
                "min": sorted_vals[0] if n else 0,
                "max": sorted_vals[-1] if n else 0,
                "median": sorted_vals[n // 2] if n else 0,
            }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "indicator_stats": indicator_stats,
            "decision_counts": dict(decision_counts),
            "total_snapshots": sum(decision_counts.values()),
            "avg_signal_strength": sum(signal_strengths) / len(signal_strengths) if signal_strengths else 0,
        }

    def build_orderbook_summary(self, orderbook_events: list[dict]) -> dict:
        """Summarize order book state at trading decision points."""
        from collections import defaultdict

        spreads: list[float] = []
        imbalances: list[float] = []
        by_context: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "spreads": [], "imbalances": [],
        })

        for evt in orderbook_events:
            payload = evt.get("payload", evt)
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue

            spread = payload.get("spread_bps", 0)
            spreads.append(spread)

            bid_depth = payload.get("bid_depth_10bps", 0)
            ask_depth = payload.get("ask_depth_10bps", 0)
            imbalance = bid_depth / ask_depth if ask_depth > 0 else 0
            imbalances.append(imbalance)

            context = payload.get("trade_context", "signal")
            by_context[context]["count"] += 1
            by_context[context]["spreads"].append(spread)
            by_context[context]["imbalances"].append(imbalance)

        def _stats(values: list[float]) -> dict:
            if not values:
                return {"count": 0, "mean": 0, "min": 0, "max": 0, "median": 0}
            s = sorted(values)
            n = len(s)
            return {
                "count": n,
                "mean": sum(s) / n,
                "min": s[0],
                "max": s[-1],
                "median": s[n // 2],
            }

        context_summary = {}
        for ctx, data in by_context.items():
            context_summary[ctx] = {
                "count": data["count"],
                "spread_stats": _stats(data["spreads"]),
                "imbalance_stats": _stats(data["imbalances"]),
            }

        return {
            "bot_id": self.bot_id,
            "date": self.date,
            "spread_stats": _stats(spreads),
            "imbalance_stats": _stats(imbalances),
            "by_context": context_summary,
            "total_snapshots": len(spreads),
        }

    def write_curated(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        base_dir: Path,
        coordination_events: list[dict] | None = None,
        daily_snapshot: dict | None = None,
        findings_dir: Path | None = None,
        filter_decision_events: list[dict] | None = None,
        indicator_snapshot_events: list[dict] | None = None,
        orderbook_context_events: list[dict] | None = None,
    ) -> Path:
        """Write all curated files to base_dir/<date>/<bot_id>/. Returns output dir."""
        output_dir = base_dir / self.date / self.bot_id
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.build_summary(trades, daily_snapshot=daily_snapshot, missed=missed)

        # Enrich summary with per-strategy data from snapshot
        if daily_snapshot:
            per_strategy = self.build_per_strategy_from_snapshot(daily_snapshot)
            if per_strategy:
                summary.per_strategy_summary = per_strategy

        winners = self.top_winners(trades)
        losers = self.top_losers(trades)
        failures = self.process_failures(trades)
        notable = self.notable_missed(missed, trades)
        regime = self.regime_analysis(trades)
        filters = self.filter_analysis(missed)
        root_causes = self.root_cause_summary(trades)

        self._write_jsonl(output_dir / "trades.jsonl", trades)
        self._write_jsonl(output_dir / "missed.jsonl", missed)
        self._write_json(output_dir / "summary.json", summary.model_dump(mode="json"))
        self._write_json(output_dir / "winners.json", [w.model_dump(mode="json") for w in winners])
        self._write_json(output_dir / "losers.json", [l.model_dump(mode="json") for l in losers])
        self._write_json(output_dir / "process_failures.json", [f.model_dump(mode="json") for f in failures])
        self._write_json(output_dir / "notable_missed.json", [n.model_dump(mode="json") for n in notable])
        self._write_json(output_dir / "regime_analysis.json", regime.model_dump(mode="json"))
        self._write_json(output_dir / "filter_analysis.json", filters.model_dump(mode="json"))
        self._write_json(output_dir / "root_cause_summary.json", root_causes.model_dump(mode="json"))
        self._write_json(output_dir / "hourly_performance.json", self.hourly_performance(trades))
        self._write_json(output_dir / "slippage_stats.json", self.slippage_stats(trades))
        factor_attr = self.factor_attribution(trades)
        self._write_json(output_dir / "factor_attribution.json", factor_attr.model_dump(mode="json"))
        self._write_json(output_dir / "exit_efficiency.json", self.exit_efficiency(trades).model_dump(mode="json"))

        # Record factor history for rolling analysis
        if findings_dir is not None:
            try:
                from schemas.signal_factor_history import DailyFactorSnapshot, FactorDayStats
                from skills.signal_factor_tracker import SignalFactorTracker

                snapshot = DailyFactorSnapshot(
                    date=self.date,
                    bot_id=self.bot_id,
                    factors=[
                        FactorDayStats(
                            factor_name=f.factor_name,
                            trade_count=f.trade_count,
                            win_rate=f.win_rate,
                            avg_pnl=f.avg_pnl,
                            total_pnl=f.total_pnl,
                            avg_contribution=f.avg_contribution,
                        )
                        for f in factor_attr.factors
                    ],
                )
                SignalFactorTracker(findings_dir).record_daily(snapshot)
            except Exception:
                pass  # Don't block pipeline on history recording failure

        # Write excursion stats if any trades carry MFE/MAE data
        excursion = self.build_excursion_stats(trades)
        if excursion["coverage"] > 0:
            self._write_json(output_dir / "excursion_stats.json", excursion)

        # Write regime→bps mapping for WFO cost model
        from skills.slippage_analyzer import SlippageAnalyzer as _SlipAnalyzer
        _sa = _SlipAnalyzer(bot_id=self.bot_id, date=self.date)
        self._write_json(output_dir / "regime_bps.json", _sa.export_regime_bps(trades))

        # Write coordinator impact if events provided
        if coordination_events:
            self._write_json(
                output_dir / "coordinator_impact.json",
                self.coordinator_impact(coordination_events),
            )

        # Write overlay state summary if present in snapshot
        if daily_snapshot:
            overlay_summary = daily_snapshot.get("overlay_state_summary")
            if overlay_summary:
                self._write_json(output_dir / "overlay_state_summary.json", overlay_summary)

        # 1.4: Write experiment data if present in snapshot
        if daily_snapshot:
            experiment_data = self.build_experiment_breakdown(daily_snapshot)
            if experiment_data:
                self._write_json(output_dir / "experiment_data.json", experiment_data)

        # 1.5: Write signal health if any trades carry signal_evolution data
        if any(t.signal_evolution for t in trades):
            from skills.signal_health_analyzer import SignalHealthAnalyzer
            analyzer = SignalHealthAnalyzer(bot_id=self.bot_id, date=self.date)
            report = analyzer.compute(trades)
            self._write_json(output_dir / "signal_health.json", report.model_dump(mode="json"))

        # 2.6: Write fill quality if any trades carry fill detail data
        if any(t.entry_fill_details or t.exit_fill_details for t in trades):
            from skills.fill_quality_analyzer import FillQualityAnalyzer
            analyzer = FillQualityAnalyzer(bot_id=self.bot_id, date=self.date)
            report = analyzer.compute(trades)
            self._write_json(output_dir / "fill_quality.json", report.model_dump(mode="json"))

        # 2B: Write enriched event summaries if events provided
        if filter_decision_events:
            self._write_json(
                output_dir / "filter_decisions.json",
                self.build_filter_decision_summary(filter_decision_events),
            )

        if indicator_snapshot_events:
            self._write_json(
                output_dir / "indicator_snapshots.json",
                self.build_indicator_snapshot_summary(indicator_snapshot_events),
            )

        if orderbook_context_events:
            self._write_json(
                output_dir / "orderbook_stats.json",
                self.build_orderbook_summary(orderbook_context_events),
            )

        return output_dir

    def _write_json(self, path: Path, data: dict | list) -> None:
        path.write_text(json.dumps(data, indent=2, default=str))

    def _write_jsonl(self, path: Path, records: list[BaseModel]) -> None:
        if not records:
            path.write_text("", encoding="utf-8")
            return
        path.write_text(
            "\n".join(record.model_dump_json() for record in records) + "\n",
            encoding="utf-8",
        )

    def _to_winner_loser(self, t: TradeEvent) -> WinnerLoserRecord:
        return WinnerLoserRecord(
            trade_id=t.trade_id,
            bot_id=t.bot_id,
            pair=t.pair,
            side=t.side,
            pnl=t.pnl,
            pnl_pct=t.pnl_pct,
            entry_signal=t.entry_signal,
            exit_reason=t.exit_reason,
            market_regime=t.market_regime,
            process_quality_score=t.process_quality_score,
            root_causes=t.root_causes,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            atr_at_entry=t.atr_at_entry,
        )
