# tests/test_wfo_report_builder.py
"""Tests for the WFO report builder — markdown generation."""
from schemas.wfo_results import (
    FoldDefinition,
    FoldResult,
    SimulationMetrics,
    CostSensitivityResult,
    RobustnessResult,
    SafetyFlag,
    WFORecommendation,
    WFOReport,
)
from analysis.wfo_report_builder import WFOReportBuilder


def _sample_report() -> WFOReport:
    fold = FoldDefinition(
        fold_number=0,
        is_start="2025-07-01", is_end="2025-12-28",
        oos_start="2025-12-28", oos_end="2026-01-27",
    )
    return WFOReport(
        bot_id="bot2",
        config_summary={"method": "anchored", "in_sample_days": 180},
        current_params={"rsi": 30.0, "stop_atr": 1.5},
        suggested_params={"rsi": 35.0, "stop_atr": 2.0},
        fold_results=[
            FoldResult(
                fold=fold,
                best_params={"rsi": 35.0, "stop_atr": 2.0},
                is_metrics=SimulationMetrics(sharpe_ratio=2.0, net_pnl=5000.0, total_trades=100),
                oos_metrics=SimulationMetrics(sharpe_ratio=1.5, net_pnl=800.0, total_trades=25),
            ),
        ],
        aggregate_oos_metrics=SimulationMetrics(sharpe_ratio=1.5, net_pnl=800.0, max_drawdown_pct=0.08),
        cost_sensitivity=[
            CostSensitivityResult(cost_multiplier=1.0, metrics=SimulationMetrics(sharpe_ratio=1.5)),
            CostSensitivityResult(cost_multiplier=1.5, metrics=SimulationMetrics(sharpe_ratio=1.1)),
            CostSensitivityResult(cost_multiplier=2.0, metrics=SimulationMetrics(sharpe_ratio=0.7)),
        ],
        robustness=RobustnessResult(
            robustness_score=82.0,
            neighborhood_stable=True,
            regime_stable=True,
            profitable_regime_count=3,
        ),
        safety_flags=[
            SafetyFlag(flag_type="fragile", description="At 2x costs, Sharpe drops to 0.70", severity="medium"),
        ],
        recommendation=WFORecommendation.ADOPT,
        recommendation_reasoning="OOS Sharpe 1.5, robustness 82/100.",
    )


class TestReportBuilder:
    def test_generates_markdown(self):
        report = _sample_report()
        builder = WFOReportBuilder()
        md = builder.build_markdown(report)
        assert isinstance(md, str)
        assert "bot2" in md
        assert "ADOPT" in md.upper()

    def test_contains_param_comparison(self):
        report = _sample_report()
        builder = WFOReportBuilder()
        md = builder.build_markdown(report)
        assert "rsi" in md
        assert "30" in md  # current
        assert "35" in md  # suggested

    def test_contains_cost_sensitivity(self):
        report = _sample_report()
        builder = WFOReportBuilder()
        md = builder.build_markdown(report)
        assert "1.5x" in md or "1.5x" in md
        assert "2.0x" in md or "2.0x" in md

    def test_contains_safety_flags(self):
        report = _sample_report()
        builder = WFOReportBuilder()
        md = builder.build_markdown(report)
        assert "fragile" in md.lower()

    def test_contains_what_could_go_wrong(self):
        report = _sample_report()
        builder = WFOReportBuilder()
        md = builder.build_markdown(report)
        assert "could go wrong" in md.lower() or "risk" in md.lower()

    def test_empty_report(self):
        report = WFOReport(
            bot_id="bot1",
            config_summary={},
            recommendation=WFORecommendation.REJECT,
            recommendation_reasoning="No data.",
        )
        builder = WFOReportBuilder()
        md = builder.build_markdown(report)
        assert "REJECT" in md.upper()
