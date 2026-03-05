# tests/test_wfo_results.py
"""Tests for WFO result schemas."""
from schemas.wfo_results import (
    FoldDefinition,
    SimulationMetrics,
    FoldResult,
    CostSensitivityResult,
    LeakageAuditEntry,
    RobustnessResult,
    WFORecommendation,
    SafetyFlag,
    WFOReport,
)


class TestFoldDefinition:
    def test_creates_fold(self):
        f = FoldDefinition(
            fold_number=0,
            is_start="2025-07-01",
            is_end="2025-12-28",
            oos_start="2025-12-28",
            oos_end="2026-01-27",
        )
        assert f.fold_number == 0
        assert f.is_start == "2025-07-01"
        assert f.oos_end == "2026-01-27"

    def test_fold_numbering(self):
        f = FoldDefinition(
            fold_number=3,
            is_start="2025-07-01",
            is_end="2026-01-27",
            oos_start="2026-01-27",
            oos_end="2026-02-26",
        )
        assert f.fold_number == 3


class TestSimulationMetrics:
    def test_creates_metrics(self):
        m = SimulationMetrics(
            total_trades=100,
            win_count=60,
            loss_count=40,
            gross_pnl=5000.0,
            net_pnl=4500.0,
            max_drawdown_pct=0.08,
            sharpe_ratio=1.8,
            sortino_ratio=2.5,
            calmar_ratio=3.0,
            profit_factor=1.9,
            total_fees=300.0,
            total_slippage=200.0,
        )
        assert m.win_rate == 0.6
        assert m.net_pnl == 4500.0

    def test_win_rate_zero_trades(self):
        m = SimulationMetrics()
        assert m.win_rate == 0.0

    def test_default_metrics_all_zero(self):
        m = SimulationMetrics()
        assert m.total_trades == 0
        assert m.gross_pnl == 0.0
        assert m.sharpe_ratio == 0.0


class TestFoldResult:
    def test_creates_result(self):
        fold = FoldDefinition(
            fold_number=0,
            is_start="2025-07-01", is_end="2025-12-28",
            oos_start="2025-12-28", oos_end="2026-01-27",
        )
        r = FoldResult(
            fold=fold,
            best_params={"rsi": 35.0, "stop_atr": 2.0},
            is_metrics=SimulationMetrics(total_trades=120, net_pnl=6000.0, sharpe_ratio=2.0),
            oos_metrics=SimulationMetrics(total_trades=25, net_pnl=800.0, sharpe_ratio=1.2),
        )
        assert r.best_params["rsi"] == 35.0
        assert r.is_metrics.net_pnl == 6000.0
        assert r.oos_metrics.net_pnl == 800.0

    def test_oos_degradation(self):
        fold = FoldDefinition(
            fold_number=0,
            is_start="2025-07-01", is_end="2025-12-28",
            oos_start="2025-12-28", oos_end="2026-01-27",
        )
        r = FoldResult(
            fold=fold,
            best_params={"rsi": 30.0},
            is_metrics=SimulationMetrics(sharpe_ratio=2.0),
            oos_metrics=SimulationMetrics(sharpe_ratio=0.5),
        )
        assert r.oos_degradation_pct == 0.75  # (2.0 - 0.5) / 2.0


class TestCostSensitivityResult:
    def test_creates_result(self):
        c = CostSensitivityResult(
            cost_multiplier=1.5,
            metrics=SimulationMetrics(net_pnl=3000.0, sharpe_ratio=1.1),
        )
        assert c.cost_multiplier == 1.5
        assert c.metrics.sharpe_ratio == 1.1


class TestLeakageAuditEntry:
    def test_creates_entry(self):
        e = LeakageAuditEntry(
            feature_name="rsi_14",
            computed_at="2026-01-15T12:00:00",
            latest_data_used="2026-01-15T11:00:00",
            passed=True,
        )
        assert e.passed is True

    def test_failed_entry(self):
        e = LeakageAuditEntry(
            feature_name="regime_label",
            computed_at="2026-01-15T12:00:00",
            latest_data_used="2026-01-16T00:00:00",
            passed=False,
            violation="Used future data: latest_data_used > computed_at",
        )
        assert e.passed is False
        assert "future data" in e.violation


class TestRobustnessResult:
    def test_creates_result(self):
        r = RobustnessResult(
            neighborhood_scores={"rsi=30": 1.5, "rsi=35": 1.8, "rsi=25": 1.4},
            neighborhood_stable=True,
            regime_pnl={"trending_up": 500.0, "trending_down": -50.0, "ranging": 100.0, "volatile": 200.0},
            profitable_regime_count=3,
            regime_stable=True,
            robustness_score=82.0,
        )
        assert r.neighborhood_stable is True
        assert r.regime_stable is True
        assert r.robustness_score == 82.0


class TestWFORecommendation:
    def test_all_recommendations_exist(self):
        assert WFORecommendation.ADOPT == "adopt"
        assert WFORecommendation.TEST_FURTHER == "test_further"
        assert WFORecommendation.REJECT == "reject"


class TestSafetyFlag:
    def test_creates_flag(self):
        f = SafetyFlag(
            flag_type="likely_overfit",
            description="Spiky optimization surface: ±10% params reduce Sharpe by >50%",
            severity="high",
        )
        assert f.flag_type == "likely_overfit"
        assert f.severity == "high"


class TestWFOReport:
    def test_creates_minimal_report(self):
        r = WFOReport(
            bot_id="bot2",
            config_summary={"method": "anchored", "in_sample_days": 180},
            current_params={"rsi": 30.0},
            suggested_params={"rsi": 35.0},
            recommendation=WFORecommendation.ADOPT,
        )
        assert r.bot_id == "bot2"
        assert r.recommendation == WFORecommendation.ADOPT

    def test_full_report(self):
        fold = FoldDefinition(
            fold_number=0,
            is_start="2025-07-01", is_end="2025-12-28",
            oos_start="2025-12-28", oos_end="2026-01-27",
        )
        r = WFOReport(
            bot_id="bot2",
            config_summary={"method": "anchored"},
            current_params={"rsi": 30.0},
            suggested_params={"rsi": 35.0},
            fold_results=[
                FoldResult(
                    fold=fold,
                    best_params={"rsi": 35.0},
                    is_metrics=SimulationMetrics(sharpe_ratio=2.0),
                    oos_metrics=SimulationMetrics(sharpe_ratio=1.5),
                ),
            ],
            aggregate_oos_metrics=SimulationMetrics(sharpe_ratio=1.5),
            cost_sensitivity=[
                CostSensitivityResult(cost_multiplier=1.0, metrics=SimulationMetrics(sharpe_ratio=1.5)),
                CostSensitivityResult(cost_multiplier=1.5, metrics=SimulationMetrics(sharpe_ratio=1.1)),
                CostSensitivityResult(cost_multiplier=2.0, metrics=SimulationMetrics(sharpe_ratio=0.7)),
            ],
            robustness=RobustnessResult(
                robustness_score=75.0,
                neighborhood_stable=True,
                regime_stable=True,
                profitable_regime_count=3,
            ),
            safety_flags=[
                SafetyFlag(flag_type="fragile", description="At 2x costs, Sharpe < 1.0", severity="medium"),
            ],
            recommendation=WFORecommendation.TEST_FURTHER,
            recommendation_reasoning="OOS Sharpe 1.5 is good but fragile at higher costs.",
        )
        assert len(r.fold_results) == 1
        assert len(r.cost_sensitivity) == 3
        assert len(r.safety_flags) == 1
        assert r.recommendation == WFORecommendation.TEST_FURTHER

    def test_rejected_report(self):
        r = WFOReport(
            bot_id="bot1",
            config_summary={"method": "anchored"},
            current_params={"ema": 20.0},
            suggested_params={"ema": 20.0},
            recommendation=WFORecommendation.REJECT,
            recommendation_reasoning="Current params already optimal. No improvement found.",
            safety_flags=[
                SafetyFlag(flag_type="low_conviction", description="Flat optimization surface", severity="high"),
            ],
        )
        assert r.recommendation == WFORecommendation.REJECT
        assert r.suggested_params == r.current_params
