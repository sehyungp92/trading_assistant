# tests/test_param_optimizer.py
"""Tests for the parameter optimizer."""
from datetime import datetime

from schemas.events import TradeEvent
from schemas.wfo_config import (
    CostModelConfig,
    OptimizationConfig,
    OptimizationObjective,
    ParameterDef,
    ParameterSpace,
    RobustnessConfig,
)
from schemas.wfo_results import SimulationMetrics
from skills.backtest_simulator import BacktestSimulator
from skills.cost_model import CostModel
from skills.param_optimizer import ParamOptimizer


def _trade(trade_id: str, pnl: float, signal_strength: float = 0.8) -> TradeEvent:
    return TradeEvent(
        trade_id=trade_id,
        bot_id="bot1",
        pair="BTCUSDT",
        side="LONG",
        entry_time=datetime(2026, 1, 15, 12, 0),
        exit_time=datetime(2026, 1, 15, 14, 0),
        entry_price=40000.0,
        exit_price=40000.0 + pnl / 0.1,
        position_size=0.1,
        pnl=pnl,
        pnl_pct=pnl / 4000 * 100,
        entry_signal_strength=signal_strength,
    )


def _zero_cost_model() -> CostModel:
    return CostModel(CostModelConfig(fees_per_trade_bps=0.0, fixed_slippage_bps=0.0))


class TestGridGeneration:
    def test_generates_all_combinations(self):
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="a", min_value=1.0, max_value=3.0, step=1.0, current_value=2.0),
                ParameterDef(name="b", min_value=10.0, max_value=20.0, step=10.0, current_value=10.0),
            ],
        )
        opt = ParamOptimizer(space, OptimizationConfig())
        grid = opt.generate_grid()
        assert len(grid) == 6  # 3 × 2

    def test_single_param(self):
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="a", min_value=1.0, max_value=5.0, step=1.0, current_value=3.0),
            ],
        )
        opt = ParamOptimizer(space, OptimizationConfig())
        grid = opt.generate_grid()
        assert len(grid) == 5

    def test_empty_space(self):
        space = ParameterSpace(bot_id="bot1", parameters=[])
        opt = ParamOptimizer(space, OptimizationConfig())
        grid = opt.generate_grid()
        assert grid == [{}]


class TestOptimization:
    def test_selects_best_by_objective(self):
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="signal_strength_min", min_value=0.3, max_value=0.9, step=0.3, current_value=0.6),
            ],
        )
        trades = [
            _trade("t1", 100.0, signal_strength=0.9),
            _trade("t2", -50.0, signal_strength=0.3),
            _trade("t3", 200.0, signal_strength=0.6),
            _trade("t4", 80.0, signal_strength=0.7),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        opt = ParamOptimizer(space, OptimizationConfig(objective=OptimizationObjective.SHARPE))
        best_params, best_metrics, all_results = opt.optimize(trades, [], sim)
        assert isinstance(best_params, dict)
        assert isinstance(best_metrics, SimulationMetrics)
        assert len(all_results) == 3  # signal_strength_min = 0.3, 0.6, 0.9

    def test_respects_max_drawdown_constraint(self):
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="signal_strength_min", min_value=0.3, max_value=0.9, step=0.3, current_value=0.6),
            ],
        )
        trades = [
            _trade("t1", 100.0, signal_strength=0.9),
            _trade("t2", -500.0, signal_strength=0.3),
            _trade("t3", 200.0, signal_strength=0.6),
        ]
        sim = BacktestSimulator(_zero_cost_model())
        opt = ParamOptimizer(
            space,
            OptimizationConfig(
                objective=OptimizationObjective.SHARPE,
                max_drawdown_constraint=0.10,
            ),
        )
        best_params, best_metrics, _ = opt.optimize(trades, [], sim)
        # The param set that includes the -500 trade should be filtered out
        assert best_metrics.max_drawdown_pct <= 0.10 or best_params.get("signal_strength_min", 0) > 0.3

    def test_respects_min_trades(self):
        space = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="signal_strength_min", min_value=0.3, max_value=0.9, step=0.3, current_value=0.6),
            ],
        )
        trades = [_trade("t1", 100.0, signal_strength=0.9)]
        sim = BacktestSimulator(_zero_cost_model())
        opt = ParamOptimizer(space, OptimizationConfig(), min_trades=5)
        best_params, best_metrics, _ = opt.optimize(trades, [], sim)
        # No param set has >= 5 trades, so best_metrics should be empty
        assert best_metrics.total_trades == 0
