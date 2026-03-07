# tests/test_experiment_manager.py
"""Tests for ExperimentManager — A/B testing lifecycle."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from schemas.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentVariant,
    VariantMetrics,
)
from skills.experiment_manager import ExperimentManager


def _make_config(experiment_id: str = "exp1", **kwargs) -> ExperimentConfig:
    """Helper to build a minimal ExperimentConfig."""
    defaults = dict(
        experiment_id=experiment_id,
        bot_id="bot1",
        title="Test experiment",
        variants=[
            ExperimentVariant(name="control", params={}, allocation_pct=50),
            ExperimentVariant(name="treatment", params={"fast_ma": 8}, allocation_pct=50),
        ],
        significance_level=0.05,
        min_trades_per_variant=30,
        max_duration_days=30,
    )
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


@pytest.fixture
def mgr(tmp_path: Path) -> ExperimentManager:
    return ExperimentManager(tmp_path, min_trades=30)


class TestExperimentManager:
    # ── creation & retrieval ─────────────────────────────────────

    def test_create_experiment_and_retrieve(self, mgr: ExperimentManager):
        config = _make_config("exp1")
        mgr.create_experiment(config)
        found = mgr.get_by_id("exp1")
        assert found is not None
        assert found.experiment_id == "exp1"
        assert found.status == ExperimentStatus.DRAFT
        assert found.title == "Test experiment"

    def test_create_experiment_deduplicates(self, mgr: ExperimentManager):
        config = _make_config("exp1")
        mgr.create_experiment(config)
        mgr.create_experiment(_make_config("exp1", title="Different title"))
        # Should still have only one, with original title
        all_exps = mgr._load_all()
        assert len(all_exps) == 1
        assert all_exps[0].title == "Test experiment"

    # ── status transitions ───────────────────────────────────────

    def test_activate_experiment_draft_to_active(self, mgr: ExperimentManager):
        mgr.create_experiment(_make_config("exp1"))
        mgr.activate_experiment("exp1")
        exp = mgr.get_by_id("exp1")
        assert exp.status == ExperimentStatus.ACTIVE
        assert exp.started_at is not None

    def test_activate_non_draft_raises(self, mgr: ExperimentManager):
        mgr.create_experiment(_make_config("exp1"))
        mgr.activate_experiment("exp1")
        with pytest.raises(ValueError, match="Cannot activate"):
            mgr.activate_experiment("exp1")

    def test_cancel_sets_status(self, mgr: ExperimentManager):
        mgr.create_experiment(_make_config("exp1"))
        mgr.cancel_experiment("exp1")
        exp = mgr.get_by_id("exp1")
        assert exp.status == ExperimentStatus.CANCELLED

    # ── variant data ingestion ───────────────────────────────────

    def test_ingest_variant_data(self, mgr: ExperimentManager):
        mgr.create_experiment(_make_config("exp1"))
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 3.0}]
        mgr.ingest_variant_data("exp1", "control", trades)
        pnls = mgr._load_variant_pnls("exp1", "control")
        assert pnls == [10.0, -5.0, 3.0]

    # ── analysis ─────────────────────────────────────────────────

    def test_analyze_with_sufficient_data(self, mgr: ExperimentManager):
        """40 trades per variant should produce a result with full metrics."""
        mgr.create_experiment(_make_config("exp1"))
        # Control: mean ~12
        control_trades = [{"pnl": v} for v in [10, 12, 14, 11, 13] * 8]
        # Treatment: mean ~17
        treatment_trades = [{"pnl": v} for v in [15, 17, 19, 16, 18] * 8]
        mgr.ingest_variant_data("exp1", "control", control_trades)
        mgr.ingest_variant_data("exp1", "treatment", treatment_trades)

        result = mgr.analyze_experiment("exp1")
        assert len(result.variant_metrics) == 2
        ctrl = result.variant_metrics[0]
        treat = result.variant_metrics[1]
        assert ctrl.trade_count == 40
        assert treat.trade_count == 40
        assert treat.avg_pnl > ctrl.avg_pnl
        # With clearly separated means, should be significant
        assert result.p_value is not None
        assert result.p_value < 0.05
        assert result.winner == "treatment"
        assert result.recommendation == "adopt_treatment"

    def test_analyze_with_insufficient_data(self, mgr: ExperimentManager):
        """5 trades per variant should give recommendation = 'extend'."""
        mgr.create_experiment(_make_config("exp1"))
        mgr.ingest_variant_data("exp1", "control", [{"pnl": 10}] * 5)
        mgr.ingest_variant_data("exp1", "treatment", [{"pnl": 15}] * 5)

        result = mgr.analyze_experiment("exp1")
        # Not enough trades for statistical test
        assert result.p_value is None
        assert result.recommendation == "extend"

    # ── statistical helpers ──────────────────────────────────────

    def test_welch_t_test_known_data(self, mgr: ExperimentManager):
        """Clearly different groups should yield a significant p-value."""
        control = [10, 12, 14, 11, 13] * 8  # mean 12, n=40
        treatment = [15, 17, 19, 16, 18] * 8  # mean 17, n=40
        p_value, ci = mgr._welch_t_test(control, treatment)
        assert p_value is not None
        assert p_value < 0.01  # very significant
        assert ci is not None
        # CI for difference should be positive (treatment > control)
        assert ci[0] > 0
        assert ci[1] > 0

    def test_cohens_d_known_data(self, mgr: ExperimentManager):
        """Groups with known separation should produce reasonable effect size."""
        group_a = [10.0, 12.0, 14.0, 11.0, 13.0] * 8
        group_b = [15.0, 17.0, 19.0, 16.0, 18.0] * 8
        d = mgr._cohens_d(group_a, group_b)
        # Mean diff = 5, stdev ~1.58 for each group -> d ~ 3.16
        assert d > 2.0  # large effect size

    # ── auto-conclusion ──────────────────────────────────────────

    def test_auto_conclude_significant_result(self, mgr: ExperimentManager):
        """Should auto-conclude when p < significance_level."""
        config = _make_config("exp1")
        mgr.create_experiment(config)
        mgr.activate_experiment("exp1")
        # Ingest clearly different data
        mgr.ingest_variant_data(
            "exp1", "control", [{"pnl": v} for v in [10, 12, 14, 11, 13] * 8]
        )
        mgr.ingest_variant_data(
            "exp1", "treatment", [{"pnl": v} for v in [15, 17, 19, 16, 18] * 8]
        )
        assert mgr.check_auto_conclusion("exp1") is True

    def test_auto_conclude_insufficient_trades(self, mgr: ExperimentManager):
        """Not enough trades should return False."""
        config = _make_config("exp1")
        mgr.create_experiment(config)
        mgr.activate_experiment("exp1")
        mgr.ingest_variant_data("exp1", "control", [{"pnl": 10}] * 5)
        mgr.ingest_variant_data("exp1", "treatment", [{"pnl": 15}] * 5)
        assert mgr.check_auto_conclusion("exp1") is False

    def test_auto_conclude_duration_exceeded(self, mgr: ExperimentManager):
        """Past max_duration_days should return True."""
        config = _make_config("exp1", max_duration_days=7)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp1")
        # Manually backdate started_at
        experiments = mgr._load_all()
        experiments[0].started_at = datetime.now(timezone.utc) - timedelta(days=10)
        mgr._save_all(experiments)
        assert mgr.check_auto_conclusion("exp1") is True

    # ── conclude & results ───────────────────────────────────────

    def test_conclude_sets_status_and_result(self, mgr: ExperimentManager):
        mgr.create_experiment(_make_config("exp1"))
        mgr.activate_experiment("exp1")

        result = ExperimentResult(
            experiment_id="exp1",
            variant_metrics=[
                VariantMetrics(variant_name="control", trade_count=40),
                VariantMetrics(variant_name="treatment", trade_count=40),
            ],
            winner="treatment",
            recommendation="adopt_treatment",
        )
        mgr.conclude_experiment("exp1", result)

        exp = mgr.get_by_id("exp1")
        assert exp.status == ExperimentStatus.CONCLUDED
        assert exp.concluded_at is not None
        # Result persisted
        results = mgr.get_results()
        assert len(results) == 1
        assert results[0].winner == "treatment"

    # ── filtering ────────────────────────────────────────────────

    def test_get_active_returns_only_active(self, mgr: ExperimentManager):
        mgr.create_experiment(_make_config("exp1"))  # DRAFT
        mgr.create_experiment(_make_config("exp2"))
        mgr.activate_experiment("exp2")  # ACTIVE
        mgr.create_experiment(_make_config("exp3"))
        mgr.cancel_experiment("exp3")  # CANCELLED

        active = mgr.get_active()
        assert len(active) == 1
        assert active[0].experiment_id == "exp2"
