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

    def __init__(self, date: str, bot_id: str) -> None:
        self.date = date
        self.bot_id = bot_id

    def build_summary(self, trades: list[TradeEvent]) -> BotDailySummary:
        """Aggregate trade list into daily summary stats."""
        if not trades:
            return BotDailySummary(date=self.date, bot_id=self.bot_id)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_pnl = sum(t.pnl for t in trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0.0
        avg_pq = sum(t.process_quality_score for t in trades) / len(trades)

        return BotDailySummary(
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
        analyzer = HourlyAnalyzer(bot_id=self.bot_id, date=self.date)
        return analyzer.compute(trades).model_dump(mode="json")

    def slippage_stats(self, trades: list[TradeEvent]) -> dict:
        """Compute per-symbol, per-hour slippage distributions via SlippageAnalyzer."""
        from skills.slippage_analyzer import SlippageAnalyzer
        analyzer = SlippageAnalyzer(bot_id=self.bot_id, date=self.date)
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

    def write_curated(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        base_dir: Path,
    ) -> Path:
        """Write all curated files to base_dir/<date>/<bot_id>/. Returns output dir."""
        output_dir = base_dir / self.date / self.bot_id
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = self.build_summary(trades)
        winners = self.top_winners(trades)
        losers = self.top_losers(trades)
        failures = self.process_failures(trades)
        notable = self.notable_missed(missed, trades)
        regime = self.regime_analysis(trades)
        filters = self.filter_analysis(missed)
        root_causes = self.root_cause_summary(trades)

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
        self._write_json(output_dir / "factor_attribution.json", self.factor_attribution(trades).model_dump(mode="json"))
        self._write_json(output_dir / "exit_efficiency.json", self.exit_efficiency(trades).model_dump(mode="json"))

        # Write regime→bps mapping for WFO cost model
        from skills.slippage_analyzer import SlippageAnalyzer as _SlipAnalyzer
        _sa = _SlipAnalyzer(bot_id=self.bot_id, date=self.date)
        self._write_json(output_dir / "regime_bps.json", _sa.export_regime_bps(trades))

        return output_dir

    def _write_json(self, path: Path, data: dict | list) -> None:
        path.write_text(json.dumps(data, indent=2, default=str))

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
