"""Tests for experiment approval UX — Telegram rendering of experiments."""
from __future__ import annotations

import pytest

from comms.telegram_renderer import TelegramRenderer
from schemas.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentType,
    ExperimentVariant,
    VariantMetrics,
)


@pytest.fixture
def renderer():
    return TelegramRenderer()


@pytest.fixture
def sample_config():
    return ExperimentConfig(
        experiment_id="exp_abcd1234abcd1234",
        bot_id="bot_alpha",
        experiment_type=ExperimentType.PARAMETER_AB,
        title="RSI threshold test",
        description="Testing RSI from 30 to 25",
        variants=[
            ExperimentVariant(name="control", params={"rsi": 30}, allocation_pct=50.0),
            ExperimentVariant(name="treatment", params={"rsi": 25}, allocation_pct=50.0),
        ],
        max_duration_days=14,
        min_trades_per_variant=30,
        significance_level=0.05,
        source_suggestion_id="sug_001",
    )


@pytest.fixture
def sample_result():
    return ExperimentResult(
        experiment_id="exp_abcd1234abcd1234",
        variant_metrics=[
            VariantMetrics(
                variant_name="control",
                trade_count=50,
                total_pnl=1200.0,
                avg_pnl=24.0,
                win_rate=0.58,
                sharpe=1.45,
                profit_factor=1.8,
                max_drawdown_pct=5.2,
            ),
            VariantMetrics(
                variant_name="treatment",
                trade_count=48,
                total_pnl=1800.0,
                avg_pnl=37.5,
                win_rate=0.65,
                sharpe=1.92,
                profit_factor=2.1,
                max_drawdown_pct=4.1,
            ),
        ],
        p_value=0.032,
        effect_size=0.45,
        winner="treatment",
        recommendation="adopt_treatment",
    )


class TestRenderExperimentProposal:
    def test_returns_text_and_keyboard(self, renderer, sample_config):
        """render_experiment_proposal returns text and keyboard."""
        text, keyboard = renderer.render_experiment_proposal(sample_config)
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(keyboard, list)
        assert len(keyboard) >= 1

    def test_keyboard_has_correct_callback_data(self, renderer, sample_config):
        """Keyboard has correct callback data format."""
        _, keyboard = renderer.render_experiment_proposal(sample_config)
        buttons = keyboard[0]
        assert len(buttons) == 2
        start_btn = buttons[0]
        cancel_btn = buttons[1]
        assert start_btn["callback_data"] == f"start_experiment_{sample_config.experiment_id}"
        assert cancel_btn["callback_data"] == f"cancel_experiment_{sample_config.experiment_id}"

    def test_handles_missing_optional_fields(self, renderer):
        """render_experiment_proposal handles missing optional fields."""
        config = ExperimentConfig(
            experiment_id="exp_minimal",
            bot_id="bot_x",
            title="Minimal test",
            variants=[
                ExperimentVariant(name="control", params={}, allocation_pct=50.0),
                ExperimentVariant(name="treatment", params={"p": 1}, allocation_pct=50.0),
            ],
        )
        text, keyboard = renderer.render_experiment_proposal(config)
        assert "Minimal test" in text
        assert len(keyboard) >= 1

    def test_text_contains_variant_info(self, renderer, sample_config):
        """Text includes variant names and allocations."""
        text, _ = renderer.render_experiment_proposal(sample_config)
        assert "control" in text
        assert "treatment" in text
        assert "50%" in text


class TestRenderExperimentResult:
    def test_contains_variant_metrics(self, renderer, sample_result):
        """render_experiment_result contains variant metrics."""
        text = renderer.render_experiment_result(sample_result)
        assert "control" in text
        assert "treatment" in text
        assert "50 trades" in text
        assert "48 trades" in text
        assert "1200" in text
        assert "1800" in text

    def test_truncates_at_4096_chars(self, renderer):
        """render_experiment_result truncates at 4096 chars."""
        # Create result with many variants to push length
        metrics = [
            VariantMetrics(
                variant_name=f"variant_{i}" * 20,
                trade_count=100,
                total_pnl=5000.0,
                avg_pnl=50.0,
                win_rate=0.6,
                sharpe=1.5,
                profit_factor=2.0,
                max_drawdown_pct=3.0,
            )
            for i in range(100)
        ]
        result = ExperimentResult(
            experiment_id="exp_long",
            variant_metrics=metrics,
            recommendation="inconclusive",
        )
        text = renderer.render_experiment_result(result)
        assert len(text) <= 4096

    def test_recommendation_displayed_correctly(self, renderer, sample_result):
        """render_experiment_result recommendation displayed correctly."""
        text = renderer.render_experiment_result(sample_result)
        assert "adopt_treatment" in text.replace("\\", "")
        assert "Winner" in text

    def test_recommendation_inconclusive(self, renderer):
        """Inconclusive recommendation renders correctly."""
        result = ExperimentResult(
            experiment_id="exp_inc",
            variant_metrics=[
                VariantMetrics(variant_name="control", trade_count=10, sharpe=1.0),
                VariantMetrics(variant_name="treatment", trade_count=10, sharpe=1.1),
            ],
            recommendation="inconclusive",
        )
        text = renderer.render_experiment_result(result)
        assert "inconclusive" in text.replace("\\", "")
        assert "Winner" not in text  # winner is None

    def test_p_value_and_effect_size_displayed(self, renderer, sample_result):
        """p-value and effect size are displayed when present."""
        text = renderer.render_experiment_result(sample_result)
        assert "0.0320" in text
        assert "0.450" in text
