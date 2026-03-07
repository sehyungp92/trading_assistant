# tests/test_experiment_schemas.py
"""Tests for A/B experiment schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentType,
    ExperimentVariant,
    VariantMetrics,
)


def _make_variants(
    alloc: tuple[float, float] = (50.0, 50.0),
) -> list[ExperimentVariant]:
    """Helper: build a control + treatment variant pair."""
    return [
        ExperimentVariant(
            name="control", params={}, allocation_pct=alloc[0]
        ),
        ExperimentVariant(
            name="treatment",
            params={"take_profit": 1.5},
            allocation_pct=alloc[1],
        ),
    ]


def _make_config(**overrides) -> ExperimentConfig:
    """Helper: build a valid ExperimentConfig with optional overrides."""
    defaults = dict(
        experiment_id="exp-001",
        bot_id="bot_alpha",
        title="TP multiplier test",
        variants=_make_variants(),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


# ------------------------------------------------------------------
# 1. Valid 2-variant config (50/50) creates successfully
# ------------------------------------------------------------------
class TestExperimentConfigValid:
    def test_valid_two_variants(self):
        cfg = _make_config()
        assert cfg.experiment_id == "exp-001"
        assert cfg.bot_id == "bot_alpha"
        assert len(cfg.variants) == 2
        assert cfg.status == ExperimentStatus.DRAFT
        assert cfg.significance_level == 0.05
        assert cfg.min_trades_per_variant == 30
        assert cfg.max_duration_days == 30
        assert cfg.success_metric == "pnl"
        assert cfg.created_at is not None


# ------------------------------------------------------------------
# 2. Fewer than 2 variants fails
# ------------------------------------------------------------------
class TestExperimentConfigTooFewVariants:
    def test_single_variant_rejected(self):
        with pytest.raises(ValidationError, match="At least 2 variants"):
            _make_config(
                variants=[
                    ExperimentVariant(
                        name="control", params={}, allocation_pct=100.0
                    )
                ]
            )


# ------------------------------------------------------------------
# 3. Allocation not summing to 100% fails
# ------------------------------------------------------------------
class TestExperimentConfigBadAllocation:
    def test_allocation_not_100(self):
        with pytest.raises(
            ValidationError, match="Variant allocation must sum to 100%"
        ):
            _make_config(variants=_make_variants(alloc=(60.0, 60.0)))


# ------------------------------------------------------------------
# 4. significance_level out of range fails
# ------------------------------------------------------------------
class TestExperimentConfigBadSignificance:
    def test_too_low(self):
        with pytest.raises(
            ValidationError, match="significance_level must be 0.01-0.10"
        ):
            _make_config(significance_level=0.001)

    def test_too_high(self):
        with pytest.raises(
            ValidationError, match="significance_level must be 0.01-0.10"
        ):
            _make_config(significance_level=0.5)


# ------------------------------------------------------------------
# 5. ExperimentResult serialization round-trip
# ------------------------------------------------------------------
class TestExperimentResultRoundTrip:
    def test_round_trip_all_fields(self):
        result = ExperimentResult(
            experiment_id="exp-001",
            variant_metrics=[
                VariantMetrics(
                    variant_name="control",
                    trade_count=50,
                    total_pnl=120.0,
                    avg_pnl=2.4,
                    win_rate=0.58,
                    sharpe=1.2,
                    profit_factor=1.8,
                    max_drawdown_pct=5.0,
                ),
                VariantMetrics(
                    variant_name="treatment",
                    trade_count=48,
                    total_pnl=180.0,
                    avg_pnl=3.75,
                    win_rate=0.62,
                    sharpe=1.5,
                    profit_factor=2.1,
                    max_drawdown_pct=4.2,
                ),
            ],
            p_value=0.03,
            effect_size=0.45,
            confidence_interval_95=(0.5, 2.1),
            winner="treatment",
            recommendation="adopt_treatment",
        )

        data = result.model_dump(mode="json")
        restored = ExperimentResult.model_validate(data)

        assert restored.experiment_id == "exp-001"
        assert restored.p_value == 0.03
        assert restored.effect_size == 0.45
        assert restored.confidence_interval_95 == (0.5, 2.1)
        assert restored.winner == "treatment"
        assert restored.recommendation == "adopt_treatment"
        assert len(restored.variant_metrics) == 2
        assert restored.variant_metrics[1].avg_pnl == 3.75


# ------------------------------------------------------------------
# 6. VariantMetrics defaults are sensible (zeros)
# ------------------------------------------------------------------
class TestVariantMetricsDefaults:
    def test_defaults_are_zero(self):
        vm = VariantMetrics(variant_name="control")
        assert vm.trade_count == 0
        assert vm.total_pnl == 0.0
        assert vm.avg_pnl == 0.0
        assert vm.win_rate == 0.0
        assert vm.sharpe == 0.0
        assert vm.profit_factor == 0.0
        assert vm.max_drawdown_pct == 0.0


# ------------------------------------------------------------------
# 7. ExperimentStatus enum values match expected strings
# ------------------------------------------------------------------
class TestExperimentStatusEnum:
    def test_enum_values(self):
        assert ExperimentStatus.DRAFT == "draft"
        assert ExperimentStatus.ACTIVE == "active"
        assert ExperimentStatus.CONCLUDED == "concluded"
        assert ExperimentStatus.CANCELLED == "cancelled"
        assert len(ExperimentStatus) == 4

    def test_experiment_type_values(self):
        assert ExperimentType.PARAMETER_AB == "parameter_ab"
        assert ExperimentType.FILTER_AB == "filter_ab"
        assert ExperimentType.ABLATION == "ablation"
        assert len(ExperimentType) == 3


# ------------------------------------------------------------------
# 8. source_suggestion_id links config to a suggestion
# ------------------------------------------------------------------
class TestExperimentConfigSuggestionLink:
    def test_source_suggestion_id(self):
        cfg = _make_config(source_suggestion_id="sug-abc123")
        assert cfg.source_suggestion_id == "sug-abc123"

    def test_source_suggestion_id_defaults_none(self):
        cfg = _make_config()
        assert cfg.source_suggestion_id is None
