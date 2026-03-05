# tests/test_wfo_leakage_integration.py
"""Integration tests for leakage detector wired into WFO runner pipeline."""
from datetime import datetime, timedelta

from schemas.events import TradeEvent, MissedOpportunityEvent
from schemas.wfo_config import (
    WFOConfig,
    WFOMethod,
    ParameterDef,
    ParameterSpace,
    CostModelConfig,
    OptimizationConfig,
    OptimizationObjective,
    RobustnessConfig,
    LeakagePreventionConfig,
)
from schemas.wfo_results import WFOReport, WFORecommendation
from skills.run_wfo import WFORunner


def _trade(
    trade_id: str,
    pnl: float,
    date: str = "2025-08-15",
    signal_strength: float = 0.8,
    regime: str = "trending_up",
) -> TradeEvent:
    dt = datetime.strptime(date, "%Y-%m-%d")
    return TradeEvent(
        trade_id=trade_id,
        bot_id="bot1",
        pair="BTCUSDT",
        side="LONG",
        entry_time=dt,
        exit_time=dt,
        entry_price=40000.0,
        exit_price=40000.0 + pnl / 0.1,
        position_size=0.1,
        pnl=pnl,
        pnl_pct=pnl / 4000 * 100,
        entry_signal_strength=signal_strength,
        market_regime=regime,
    )


def _make_config(min_folds: int = 1, is_days: int = 60, oos_days: int = 30) -> WFOConfig:
    return WFOConfig(
        bot_id="bot1",
        method=WFOMethod.ANCHORED,
        in_sample_days=is_days,
        out_of_sample_days=oos_days,
        step_days=30,
        min_folds=min_folds,
        parameter_space=ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="signal_strength_min", min_value=0.3, max_value=0.9, step=0.3, current_value=0.6),
            ],
        ),
        optimization=OptimizationConfig(
            objective=OptimizationObjective.SHARPE,
            max_drawdown_constraint=0.50,
        ),
        cost_model=CostModelConfig(fees_per_trade_bps=0.0, fixed_slippage_bps=0.0),
        robustness=RobustnessConfig(
            min_trades_per_fold=1,
            min_profitable_regimes=1,
            total_regime_types=4,
        ),
    )


def _generate_trades(count: int = 50) -> list[TradeEvent]:
    """Generate trades spread across 365 days with varied regimes."""
    import random

    random.seed(42)
    regimes = ["trending_up", "trending_down", "ranging", "volatile"]
    trades = []
    for i in range(count):
        day_offset = int(i * 365 / count)
        dt = datetime(2025, 1, 1) + timedelta(days=day_offset)
        pnl = random.uniform(-100, 200)
        trades.append(TradeEvent(
            trade_id=f"t{i}",
            bot_id="bot1",
            pair="BTCUSDT",
            side="LONG",
            entry_time=dt,
            exit_time=dt,
            entry_price=40000.0,
            exit_price=40000.0 + pnl / 0.1,
            position_size=0.1,
            pnl=pnl,
            pnl_pct=pnl / 4000 * 100,
            entry_signal_strength=random.uniform(0.2, 1.0),
            market_regime=regimes[i % len(regimes)],
        ))
    return trades


class TestWFOLeakageIntegration:
    def test_leakage_audit_populated_when_enabled(self):
        """Run WFO with default config (feature_audit=True). Check leakage_audit is non-empty."""
        cfg = _make_config(min_folds=1, is_days=60, oos_days=30)
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(
            trades=trades,
            missed=[],
            data_start="2025-01-01",
            data_end="2026-01-01",
        )
        assert isinstance(report, WFOReport)
        assert len(report.leakage_audit) > 0

    def test_leakage_audit_all_pass_for_clean_data(self):
        """Normal trades (entry_time == exit_time or exit_time > entry_time) should all pass."""
        cfg = _make_config(min_folds=1, is_days=60, oos_days=30)
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(
            trades=trades,
            missed=[],
            data_start="2025-01-01",
            data_end="2026-01-01",
        )
        assert len(report.leakage_audit) > 0
        assert all(entry.passed for entry in report.leakage_audit)

    def test_leakage_audit_empty_when_disabled(self):
        """Set feature_audit=False in leakage_prevention config. Check leakage_audit == []."""
        cfg = _make_config(min_folds=1, is_days=60, oos_days=30)
        cfg.leakage_prevention = LeakagePreventionConfig(feature_audit=False)
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(
            trades=trades,
            missed=[],
            data_start="2025-01-01",
            data_end="2026-01-01",
        )
        assert report.leakage_audit == []

    def test_leakage_violation_adds_high_safety_flag(self):
        """Create a trade where exit_time < entry_time (temporal violation for label).
        Check a SafetyFlag with flag_type='data_leakage' and severity='high' appears."""
        cfg = _make_config(min_folds=1, is_days=60, oos_days=30)
        trades = _generate_trades(50)

        # Inject a trade with exit_time BEFORE entry_time (temporal violation)
        bad_entry = datetime(2025, 6, 15)
        bad_exit = datetime(2025, 6, 14)  # exit before entry = label leakage
        trades.append(TradeEvent(
            trade_id="t_bad",
            bot_id="bot1",
            pair="BTCUSDT",
            side="LONG",
            entry_time=bad_entry,
            exit_time=bad_exit,
            entry_price=40000.0,
            exit_price=40100.0,
            position_size=0.1,
            pnl=10.0,
            pnl_pct=0.025,
            entry_signal_strength=0.8,
            market_regime="trending_up",
        ))

        runner = WFORunner(cfg)
        report = runner.run(
            trades=trades,
            missed=[],
            data_start="2025-01-01",
            data_end="2026-01-01",
        )

        leakage_flags = [f for f in report.safety_flags if f.flag_type == "data_leakage"]
        assert len(leakage_flags) >= 1
        assert leakage_flags[0].severity == "high"

    def test_leakage_flag_causes_reject(self):
        """When data_leakage high-severity flag exists, the recommendation should be REJECT."""
        cfg = _make_config(min_folds=1, is_days=60, oos_days=30)
        trades = _generate_trades(50)

        # Inject a trade with exit_time BEFORE entry_time (temporal violation)
        bad_entry = datetime(2025, 6, 15)
        bad_exit = datetime(2025, 6, 14)  # exit before entry = label leakage
        trades.append(TradeEvent(
            trade_id="t_bad",
            bot_id="bot1",
            pair="BTCUSDT",
            side="LONG",
            entry_time=bad_entry,
            exit_time=bad_exit,
            entry_price=40000.0,
            exit_price=40100.0,
            position_size=0.1,
            pnl=10.0,
            pnl_pct=0.025,
            entry_signal_strength=0.8,
            market_regime="trending_up",
        ))

        runner = WFORunner(cfg)
        report = runner.run(
            trades=trades,
            missed=[],
            data_start="2025-01-01",
            data_end="2026-01-01",
        )

        assert report.recommendation == WFORecommendation.REJECT
        assert "data_leakage" in report.recommendation_reasoning.lower() or \
               "leakage" in report.recommendation_reasoning.lower() or \
               "temporal" in report.recommendation_reasoning.lower()

    def test_existing_wfo_tests_still_pass(self):
        """Run the same test as test_produces_report from existing tests to verify no regression."""
        cfg = _make_config(min_folds=1, is_days=60, oos_days=30)
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(
            trades=trades,
            missed=[],
            data_start="2025-01-01",
            data_end="2026-01-01",
        )
        assert isinstance(report, WFOReport)
        assert report.bot_id == "bot1"
        assert len(report.fold_results) >= 1
        assert report.recommendation in [
            WFORecommendation.ADOPT,
            WFORecommendation.TEST_FURTHER,
            WFORecommendation.REJECT,
        ]
