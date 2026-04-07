# tests/test_wfo_integration.py
"""Integration test — full WFO pipeline from config to report to prompt package."""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from schemas.events import TradeEvent
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
from schemas.wfo_results import WFORecommendation, WFOReport
from skills.run_wfo import WFORunner
from analysis.wfo_report_builder import WFOReportBuilder
from analysis.wfo_prompt_assembler import WFOPromptAssembler


def _generate_realistic_trades(seed: int = 42) -> list[TradeEvent]:
    """Generate 200 trades spread over 400 days with regime variation."""
    random.seed(seed)
    regimes = ["trending_up", "trending_down", "ranging", "volatile"]
    trades: list[TradeEvent] = []
    base_date = datetime(2025, 1, 1)

    for i in range(200):
        offset = int(i * 400 / 200)
        dt = base_date + timedelta(days=offset)
        regime = regimes[i % len(regimes)]

        # Trending regimes are more profitable, ranging less so
        if regime in ("trending_up", "trending_down"):
            pnl = random.gauss(50, 80)
        elif regime == "ranging":
            pnl = random.gauss(-10, 60)
        else:
            pnl = random.gauss(20, 120)

        signal_strength = random.uniform(0.2, 1.0)

        trades.append(TradeEvent(
            trade_id=f"int-t{i}",
            bot_id="bot2",
            pair="BTCUSDT",
            side="LONG",
            entry_time=dt,
            exit_time=dt + timedelta(hours=2),
            entry_price=40000.0,
            exit_price=40000.0 + pnl / 0.1,
            position_size=0.1,
            pnl=pnl,
            pnl_pct=pnl / 4000 * 100,
            entry_signal_strength=signal_strength,
            market_regime=regime,
            entry_signal="ema_cross",
        ))
    return trades


class TestFullWFOPipeline:
    def test_end_to_end(self, tmp_path: Path):
        """Config → WFO run → report builder → prompt assembler → all outputs valid."""
        # 1. Build config
        config = WFOConfig(
            bot_id="bot2",
            method=WFOMethod.ANCHORED,
            in_sample_days=120,
            out_of_sample_days=30,
            step_days=30,
            min_folds=2,
            parameter_space=ParameterSpace(
                bot_id="bot2",
                parameters=[
                    ParameterDef(
                        name="signal_strength_min",
                        min_value=0.3,
                        max_value=0.9,
                        step=0.3,
                        current_value=0.6,
                    ),
                ],
            ),
            optimization=OptimizationConfig(
                objective=OptimizationObjective.SHARPE,
                max_drawdown_constraint=0.50,
            ),
            cost_model=CostModelConfig(
                fees_per_trade_bps=7.0,
                fixed_slippage_bps=5.0,
                cost_multipliers=[1.0, 1.5, 2.0],
            ),
            robustness=RobustnessConfig(
                min_trades_per_fold=5,
                min_profitable_regimes=2,
                total_regime_types=4,
            ),
        )

        # 2. Generate trades
        trades = _generate_realistic_trades()

        # 3. Run WFO
        runner = WFORunner(config)
        report = runner.run(
            trades=trades,
            missed=[],
            data_start="2025-01-01",
            data_end="2026-03-01",
        )

        # 4. Validate report structure
        assert isinstance(report, WFOReport)
        assert report.bot_id == "bot2"
        assert len(report.fold_results) >= 2
        assert len(report.cost_sensitivity) == 3
        assert report.robustness.robustness_score >= 0
        assert report.recommendation in list(WFORecommendation)

        # 5. Write output
        output_dir = tmp_path / "runs" / "wfo" / "bot2"
        runner.write_output(report, output_dir)
        assert (output_dir / "wfo_report.json").exists()
        loaded = json.loads((output_dir / "wfo_report.json").read_text())
        assert loaded["bot_id"] == "bot2"

        # 6. Build markdown report
        builder = WFOReportBuilder()
        md = builder.build_markdown(report)
        assert len(md) > 100  # non-trivial report
        assert "bot2" in md
        assert "Parameter Comparison" in md

        # 7. Assemble prompt package
        memory_dir = tmp_path / "memory"
        policy_dir = memory_dir / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("You analyze WFO results.")
        (policy_dir / "trading_rules.md").write_text("Max 15% drawdown.")
        (policy_dir / "soul.md").write_text("Conservative risk tolerance.")

        assembler = WFOPromptAssembler(
            bot_id="bot2",
            memory_dir=memory_dir,
            wfo_output_dir=output_dir,
        )
        package = assembler.assemble()
        assert package.system_prompt
        assert package.task_prompt
        assert package.data["wfo_report"]["bot_id"] == "bot2"

    def test_yaml_config_loading(self, tmp_path: Path):
        """Config can be loaded from YAML dict (simulating wfo_config.yaml)."""
        import yaml

        yaml_content = {
            "bot_id": "bot3",
            "method": "rolling",
            "in_sample_days": 90,
            "out_of_sample_days": 14,
            "step_days": 14,
            "min_folds": 2,
            "parameter_space": {
                "bot_id": "bot3",
                "parameters": [
                    {"name": "signal_strength_min", "min_value": 0.2, "max_value": 0.8, "step": 0.2, "current_value": 0.5},
                ],
            },
            "optimization": {"objective": "calmar", "max_drawdown_constraint": 0.20},
            "cost_model": {"fees_per_trade_bps": 7, "fixed_slippage_bps": 5},
            "robustness": {"min_trades_per_fold": 3, "min_profitable_regimes": 2, "total_regime_types": 4},
        }

        yaml_path = tmp_path / "wfo_config.yaml"
        yaml_path.write_text(yaml.dump(yaml_content))

        loaded = yaml.safe_load(yaml_path.read_text())
        config = WFOConfig(**loaded)
        assert config.bot_id == "bot3"
        assert config.method.value == "rolling"

        # Run pipeline with loaded config
        trades = _generate_realistic_trades(seed=99)
        runner = WFORunner(config)
        report = runner.run(trades=trades, missed=[], data_start="2025-01-01", data_end="2026-03-01")
        assert isinstance(report, WFOReport)

    def test_insufficient_data_produces_reject(self):
        config = WFOConfig(
            bot_id="bot1",
            in_sample_days=180,
            out_of_sample_days=30,
            min_folds=10,
            parameter_space=ParameterSpace(bot_id="bot1", parameters=[]),
        )
        runner = WFORunner(config)
        report = runner.run(
            trades=[],
            missed=[],
            data_start="2025-01-01",
            data_end="2025-06-01",
        )
        assert report.recommendation == WFORecommendation.REJECT
