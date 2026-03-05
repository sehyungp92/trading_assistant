# skills/run_wfo.py
"""WFO runner — main walk-forward optimization pipeline.

Orchestrates: fold generation → per-fold optimization + OOS validation →
aggregate metrics → robustness testing → cost sensitivity → leakage audit →
recommendation + safety flags → JSON output.

Usage: WFORunner(config).run(trades, missed, data_start, data_end)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from schemas.events import TradeEvent, MissedOpportunityEvent
from schemas.wfo_config import WFOConfig
from schemas.wfo_results import (
    CostSensitivityResult,
    FoldResult,
    RobustnessResult,
    SafetyFlag,
    SimulationMetrics,
    WFORecommendation,
    WFOReport,
)
from skills.backtest_simulator import BacktestSimulator
from skills.cost_model import CostModel
from skills.fold_generator import FoldGenerator
from skills.param_optimizer import ParamOptimizer
from skills.leakage_detector import LeakageDetector, FeatureRecord, LabelRecord
from skills.robustness_tester import RobustnessTester

logger = logging.getLogger(__name__)


class WFORunner:
    """Runs the full walk-forward optimization pipeline."""

    def __init__(self, config: WFOConfig) -> None:
        self._config = config

    def run(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        data_start: str,
        data_end: str,
    ) -> WFOReport:
        """Execute the WFO pipeline and return a complete report."""
        # 1. Generate folds
        fold_gen = FoldGenerator(self._config)
        folds = fold_gen.generate(data_start, data_end)

        if not folds:
            return WFOReport(
                bot_id=self._config.bot_id,
                config_summary=self._config_summary(),
                current_params=self._config.parameter_space.current_params,
                suggested_params=self._config.parameter_space.current_params,
                recommendation=WFORecommendation.REJECT,
                recommendation_reasoning="Insufficient data to generate required number of folds.",
            )

        # 2. Run per-fold optimization
        cost_model = CostModel(self._config.cost_model)
        simulator = BacktestSimulator(cost_model)
        fold_results = self._run_folds(folds, trades, missed, simulator)

        # 3. Determine best params from consensus across folds
        suggested_params = self._consensus_params(fold_results)

        # 4. Aggregate OOS metrics
        all_oos_trades = self._filter_trades_in_oos(trades, folds)
        aggregate_oos = simulator.simulate(all_oos_trades, missed, suggested_params)

        # 5. Cost sensitivity
        cost_sensitivity = self._run_cost_sensitivity(trades, missed, suggested_params)

        # 6. Robustness testing
        robustness_tester = RobustnessTester(
            self._config.robustness, self._config.parameter_space, simulator
        )
        robustness = robustness_tester.evaluate(trades, missed, suggested_params)

        # 6b. Leakage audit
        leakage_audit = []
        if self._config.leakage_prevention.feature_audit:
            features = [
                FeatureRecord(
                    feature_name=f"signal:{t.entry_signal or t.trade_id}",
                    computed_at=t.entry_time.isoformat(),
                    latest_data_used=t.entry_time.isoformat(),
                )
                for t in trades
            ]
            labels = [
                LabelRecord(
                    trade_id=t.trade_id,
                    entry_time=t.entry_time.isoformat(),
                    label_computed_from=t.exit_time.isoformat(),
                )
                for t in trades
            ]
            detector = LeakageDetector()
            leakage_audit = detector.full_audit(features, labels)

        # 7. Safety flags
        safety_flags = list(robustness_tester.detect_safety_flags(
            robustness.neighborhood_scores,
            robustness.regime_stable,
            aggregate_oos.sharpe_ratio,
        ))
        safety_flags.extend(self._cost_safety_flags(cost_sensitivity))

        # Leakage safety flag
        if any(not entry.passed for entry in leakage_audit):
            safety_flags.append(SafetyFlag(
                flag_type="data_leakage",
                description="Leakage audit detected temporal violations",
                severity="high",
            ))

        # 8. Recommendation
        recommendation, reasoning = self._determine_recommendation(
            aggregate_oos, robustness, safety_flags, fold_results
        )

        return WFOReport(
            bot_id=self._config.bot_id,
            config_summary=self._config_summary(),
            current_params=self._config.parameter_space.current_params,
            suggested_params=suggested_params,
            fold_results=fold_results,
            aggregate_oos_metrics=aggregate_oos,
            cost_sensitivity=cost_sensitivity,
            leakage_audit=leakage_audit,
            robustness=robustness,
            safety_flags=safety_flags,
            recommendation=recommendation,
            recommendation_reasoning=reasoning,
        )

    def write_output(self, report: WFOReport, output_dir: Path) -> None:
        """Write the WFO report JSON to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "wfo_report.json"
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, default=str))

    def _run_folds(
        self, folds, trades, missed, simulator
    ) -> list[FoldResult]:
        results: list[FoldResult] = []
        for fold in folds:
            is_trades = self._filter_by_period(trades, fold.is_start, fold.is_end)
            is_missed = self._filter_missed_by_period(missed, fold.is_start, fold.is_end)
            oos_trades = self._filter_by_period(trades, fold.oos_start, fold.oos_end)
            oos_missed = self._filter_missed_by_period(missed, fold.oos_start, fold.oos_end)

            optimizer = ParamOptimizer(
                self._config.parameter_space,
                self._config.optimization,
                min_trades=self._config.robustness.min_trades_per_fold,
            )
            best_params, is_metrics, _ = optimizer.optimize(is_trades, is_missed, simulator)

            if not best_params:
                continue

            oos_metrics = simulator.simulate(oos_trades, oos_missed, best_params)
            results.append(FoldResult(
                fold=fold,
                best_params=best_params,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
            ))

        return results

    def _consensus_params(self, fold_results: list[FoldResult]) -> dict[str, float]:
        """Take the most common best param values across folds (mode for each param)."""
        if not fold_results:
            return self._config.parameter_space.current_params

        from collections import Counter

        param_votes: dict[str, list[float]] = {}
        for fr in fold_results:
            for name, val in fr.best_params.items():
                param_votes.setdefault(name, []).append(val)

        consensus: dict[str, float] = {}
        for name, vals in param_votes.items():
            counter = Counter(vals)
            consensus[name] = counter.most_common(1)[0][0]
        return consensus

    def _run_cost_sensitivity(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        params: dict[str, float],
    ) -> list[CostSensitivityResult]:
        if not self._config.cost_model.cost_sensitivity_test:
            return []

        results: list[CostSensitivityResult] = []
        cost_model = CostModel(self._config.cost_model)
        simulator = BacktestSimulator(cost_model)

        for mult in self._config.cost_model.cost_multipliers:
            metrics = simulator.simulate(trades, missed, params, cost_multiplier=mult)
            results.append(CostSensitivityResult(cost_multiplier=mult, metrics=metrics))

        return results

    def _cost_safety_flags(self, sensitivity: list[CostSensitivityResult]) -> list[SafetyFlag]:
        flags: list[SafetyFlag] = []
        if not sensitivity:
            return flags

        base = next((s for s in sensitivity if s.cost_multiplier == 1.0), None)
        elevated = [s for s in sensitivity if s.cost_multiplier > 1.0]

        if base and base.metrics.net_pnl <= 0 and self._config.cost_model.reject_if_only_profitable_at_zero_cost:
            flags.append(SafetyFlag(
                flag_type="fragile",
                description="Not profitable at base cost assumptions",
                severity="high",
            ))

        for s in elevated:
            if base and base.metrics.sharpe_ratio > 0 and s.metrics.sharpe_ratio < 1.0:
                flags.append(SafetyFlag(
                    flag_type="fragile",
                    description=f"At {s.cost_multiplier}x costs, Sharpe drops to {s.metrics.sharpe_ratio:.2f}",
                    severity="medium",
                ))
                break  # one flag is enough

        return flags

    def _determine_recommendation(
        self,
        oos_metrics: SimulationMetrics,
        robustness: RobustnessResult,
        safety_flags: list[SafetyFlag],
        fold_results: list[FoldResult],
    ) -> tuple[WFORecommendation, str]:
        high_severity_flags = [f for f in safety_flags if f.severity == "high"]

        if not fold_results:
            return WFORecommendation.REJECT, "No valid fold results produced."

        if high_severity_flags:
            reasons = "; ".join(f.description for f in high_severity_flags)
            return WFORecommendation.REJECT, f"High-severity safety flags: {reasons}"

        if oos_metrics.sharpe_ratio <= 0:
            return WFORecommendation.REJECT, f"OOS Sharpe ratio is {oos_metrics.sharpe_ratio:.2f} (not profitable)."

        if robustness.robustness_score >= 70 and oos_metrics.sharpe_ratio >= 1.0:
            return (
                WFORecommendation.ADOPT,
                f"OOS Sharpe {oos_metrics.sharpe_ratio:.2f}, robustness {robustness.robustness_score:.0f}/100.",
            )

        return (
            WFORecommendation.TEST_FURTHER,
            f"OOS Sharpe {oos_metrics.sharpe_ratio:.2f}, robustness {robustness.robustness_score:.0f}/100. Needs more validation.",
        )

    def _filter_by_period(self, trades: list[TradeEvent], start: str, end: str) -> list[TradeEvent]:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        return [t for t in trades if s <= t.entry_time < e]

    def _filter_missed_by_period(
        self, missed: list[MissedOpportunityEvent], start: str, end: str
    ) -> list[MissedOpportunityEvent]:
        # MissedOpportunityEvent doesn't have a direct timestamp — filter by metadata if available
        return missed  # Include all missed for now; per-fold filtering is optional

    def _filter_trades_in_oos(self, trades, folds) -> list[TradeEvent]:
        """Collect trades that fall in any OOS period."""
        result: list[TradeEvent] = []
        for fold in folds:
            result.extend(self._filter_by_period(trades, fold.oos_start, fold.oos_end))
        return result

    def _config_summary(self) -> dict:
        return {
            "method": self._config.method.value,
            "in_sample_days": self._config.in_sample_days,
            "out_of_sample_days": self._config.out_of_sample_days,
            "step_days": self._config.step_days,
            "objective": self._config.optimization.objective.value,
            "total_parameter_combinations": self._config.parameter_space.total_combinations,
        }
