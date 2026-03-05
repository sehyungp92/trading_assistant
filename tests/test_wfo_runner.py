# tests/test_wfo_runner.py
"""Tests for the main WFO runner pipeline."""
import json
from datetime import datetime
from pathlib import Path

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
        dt = datetime(2025, 1, 1) + __import__("datetime").timedelta(days=day_offset)
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


class TestWFORunnerExecutes:
    def test_produces_report(self):
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

    def test_includes_cost_sensitivity(self):
        cfg = _make_config(min_folds=1)
        cfg.cost_model = CostModelConfig(
            fees_per_trade_bps=7.0,
            fixed_slippage_bps=5.0,
            cost_multipliers=[1.0, 1.5, 2.0],
        )
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(trades=trades, missed=[], data_start="2025-01-01", data_end="2026-01-01")
        assert len(report.cost_sensitivity) == 3

    def test_includes_robustness(self):
        cfg = _make_config(min_folds=1)
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(trades=trades, missed=[], data_start="2025-01-01", data_end="2026-01-01")
        assert 0 <= report.robustness.robustness_score <= 100

    def test_returns_reject_on_insufficient_data(self):
        cfg = _make_config(min_folds=100)  # impossible to achieve
        trades = _generate_trades(10)
        runner = WFORunner(cfg)
        report = runner.run(trades=trades, missed=[], data_start="2025-01-01", data_end="2025-06-01")
        assert report.recommendation == WFORecommendation.REJECT
        assert "insufficient" in report.recommendation_reasoning.lower()


class TestWFORunnerOutputs:
    def test_writes_report_json(self, tmp_path: Path):
        cfg = _make_config(min_folds=1)
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(trades=trades, missed=[], data_start="2025-01-01", data_end="2026-01-01")
        runner.write_output(report, tmp_path)
        assert (tmp_path / "wfo_report.json").exists()
        data = json.loads((tmp_path / "wfo_report.json").read_text())
        assert data["bot_id"] == "bot1"
        assert "recommendation" in data

    def test_cost_sensitivity_flag(self):
        cfg = _make_config(min_folds=1)
        cfg.cost_model = CostModelConfig(
            fees_per_trade_bps=100.0,  # absurdly high fees
            fixed_slippage_bps=100.0,
            cost_multipliers=[1.0, 1.5, 2.0],
            reject_if_only_profitable_at_zero_cost=True,
        )
        trades = _generate_trades(50)
        runner = WFORunner(cfg)
        report = runner.run(trades=trades, missed=[], data_start="2025-01-01", data_end="2026-01-01")
        flag_types = [f.flag_type for f in report.safety_flags]
        # With absurd costs, should flag as fragile or reject
        assert "fragile" in flag_types or report.recommendation == WFORecommendation.REJECT
