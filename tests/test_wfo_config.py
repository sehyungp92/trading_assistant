# tests/test_wfo_config.py
"""Tests for WFO configuration schemas."""
from schemas.wfo_config import (
    WFOMethod,
    OptimizationObjective,
    SlippageModel,
    CostModelConfig,
    LeakagePreventionConfig,
    RobustnessConfig,
    OutputConfig,
    ParameterDef,
    ParameterSpace,
    WFOConfig,
)


class TestWFOMethod:
    def test_all_methods_exist(self):
        assert WFOMethod.ANCHORED == "anchored"
        assert WFOMethod.ROLLING == "rolling"


class TestOptimizationObjective:
    def test_all_objectives_exist(self):
        assert OptimizationObjective.SHARPE == "sharpe"
        assert OptimizationObjective.SORTINO == "sortino"
        assert OptimizationObjective.CALMAR == "calmar"
        assert OptimizationObjective.PROFIT_FACTOR == "profit_factor"


class TestSlippageModel:
    def test_all_models_exist(self):
        assert SlippageModel.FIXED == "fixed"
        assert SlippageModel.SPREAD_PROPORTIONAL == "spread_proportional"
        assert SlippageModel.EMPIRICAL == "empirical"


class TestCostModelConfig:
    def test_creates_with_defaults(self):
        c = CostModelConfig()
        assert c.fees_per_trade_bps == 7.0
        assert c.slippage_model == SlippageModel.FIXED
        assert c.fixed_slippage_bps == 5.0
        assert c.spread_impact is True
        assert c.reject_if_only_profitable_at_zero_cost is True
        assert c.cost_sensitivity_test is True
        assert c.cost_multipliers == [1.0, 1.5, 2.0]

    def test_creates_empirical(self):
        c = CostModelConfig(
            slippage_model=SlippageModel.EMPIRICAL,
            slippage_source="data/curated/slippage_stats.csv",
        )
        assert c.slippage_model == SlippageModel.EMPIRICAL
        assert c.slippage_source == "data/curated/slippage_stats.csv"


class TestLeakagePreventionConfig:
    def test_creates_with_defaults(self):
        lp = LeakagePreventionConfig()
        assert lp.strict_temporal_split is True
        assert lp.no_forward_fill_labels is True
        assert lp.feature_audit is True


class TestRobustnessConfig:
    def test_creates_with_defaults(self):
        r = RobustnessConfig()
        assert r.neighborhood_test is True
        assert r.regime_stability is True
        assert r.min_trades_per_fold == 30
        assert r.neighborhood_pct == 0.1
        assert r.min_profitable_regimes == 3
        assert r.total_regime_types == 4


class TestParameterDef:
    def test_creates_parameter(self):
        p = ParameterDef(
            name="rsi_threshold",
            min_value=20.0,
            max_value=40.0,
            step=5.0,
            current_value=30.0,
        )
        assert p.name == "rsi_threshold"
        assert p.grid_values == [20.0, 25.0, 30.0, 35.0, 40.0]

    def test_grid_values_single_step(self):
        p = ParameterDef(name="stop_atr", min_value=1.0, max_value=1.0, step=0.5, current_value=1.0)
        assert p.grid_values == [1.0]

    def test_grid_values_fractional_step(self):
        p = ParameterDef(name="trail", min_value=1.0, max_value=3.0, step=0.5, current_value=1.5)
        assert p.grid_values == [1.0, 1.5, 2.0, 2.5, 3.0]


class TestParameterSpace:
    def test_creates_space(self):
        ps = ParameterSpace(
            bot_id="bot2",
            parameters=[
                ParameterDef(name="rsi_threshold", min_value=20.0, max_value=40.0, step=5.0, current_value=30.0),
                ParameterDef(name="stop_atr", min_value=1.0, max_value=3.0, step=0.5, current_value=1.5),
            ],
        )
        assert ps.bot_id == "bot2"
        assert len(ps.parameters) == 2
        assert ps.total_combinations == 25  # 5 × 5

    def test_current_params(self):
        ps = ParameterSpace(
            bot_id="bot1",
            parameters=[
                ParameterDef(name="rsi", min_value=20.0, max_value=40.0, step=5.0, current_value=30.0),
            ],
        )
        assert ps.current_params == {"rsi": 30.0}

    def test_empty_space(self):
        ps = ParameterSpace(bot_id="bot1", parameters=[])
        assert ps.total_combinations == 0
        assert ps.current_params == {}


class TestWFOConfig:
    def test_creates_full_config(self):
        cfg = WFOConfig(
            bot_id="bot2",
            method=WFOMethod.ANCHORED,
            in_sample_days=180,
            out_of_sample_days=30,
            step_days=30,
            min_folds=6,
            parameter_space=ParameterSpace(
                bot_id="bot2",
                parameters=[
                    ParameterDef(name="rsi", min_value=20.0, max_value=40.0, step=5.0, current_value=30.0),
                ],
            ),
        )
        assert cfg.bot_id == "bot2"
        assert cfg.method == WFOMethod.ANCHORED
        assert cfg.in_sample_days == 180
        assert cfg.optimization.objective == OptimizationObjective.CALMAR

    def test_defaults(self):
        cfg = WFOConfig(
            bot_id="bot1",
            parameter_space=ParameterSpace(bot_id="bot1", parameters=[]),
        )
        assert cfg.method == WFOMethod.ANCHORED
        assert cfg.in_sample_days == 180
        assert cfg.out_of_sample_days == 30
        assert cfg.step_days == 30
        assert cfg.min_folds == 6
        assert cfg.optimization.objective == OptimizationObjective.CALMAR
        assert cfg.cost_model.fees_per_trade_bps == 7.0
        assert cfg.leakage_prevention.strict_temporal_split is True
        assert cfg.robustness.neighborhood_test is True
        assert cfg.output.param_recommendations is True

    def test_from_yaml_dict(self):
        """Config can be built from a dict (as if loaded from YAML)."""
        raw = {
            "bot_id": "bot3",
            "method": "rolling",
            "in_sample_days": 90,
            "out_of_sample_days": 14,
            "step_days": 14,
            "min_folds": 4,
            "parameter_space": {
                "bot_id": "bot3",
                "parameters": [
                    {"name": "ema_period", "min_value": 10, "max_value": 30, "step": 5, "current_value": 20},
                ],
            },
            "optimization": {"objective": "sharpe", "max_drawdown_constraint": 0.20},
        }
        cfg = WFOConfig(**raw)
        assert cfg.method == WFOMethod.ROLLING
        assert cfg.optimization.objective == OptimizationObjective.SHARPE
        assert cfg.optimization.max_drawdown_constraint == 0.20
