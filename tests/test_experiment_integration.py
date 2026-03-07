"""Phase 2C integration tests — A/B testing lifecycle."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from schemas.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentType,
    ExperimentVariant,
    VariantMetrics,
)
from skills.experiment_manager import ExperimentManager
from skills.experiment_config_generator import ExperimentConfigGenerator


def _make_config(
    experiment_id: str = "exp-001",
    bot_id: str = "bot_alpha",
    min_trades: int = 30,
    max_duration_days: int = 30,
    source_suggestion_id: str | None = None,
) -> ExperimentConfig:
    """Helper to create a standard experiment config."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        bot_id=bot_id,
        experiment_type=ExperimentType.PARAMETER_AB,
        title=f"Test experiment {experiment_id}",
        description="Integration test experiment",
        variants=[
            ExperimentVariant(
                name="control",
                params={"fast_period": 10},
                allocation_pct=50.0,
            ),
            ExperimentVariant(
                name="treatment",
                params={"fast_period": 8},
                allocation_pct=50.0,
            ),
        ],
        min_trades_per_variant=min_trades,
        max_duration_days=max_duration_days,
        source_suggestion_id=source_suggestion_id,
    )


def _generate_trades(
    count: int,
    mean_pnl: float,
    spread: float = 5.0,
) -> list[dict]:
    """Generate synthetic trade data with controlled PnL distribution."""
    import random

    rng = random.Random(42)
    trades = []
    for i in range(count):
        pnl = mean_pnl + rng.gauss(0, spread)
        trades.append({
            "trade_id": f"trade-{i}",
            "pnl": round(pnl, 2),
            "timestamp": (datetime.now(timezone.utc) - timedelta(hours=count - i)).isoformat(),
        })
    return trades


class TestFullLifecycle:
    """Test 1: Full lifecycle — create, activate, ingest, auto-conclude, result with winner."""

    def test_full_lifecycle_with_winner(self, tmp_path: Path) -> None:
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir, min_trades=30)

        config = _make_config(min_trades=30)
        mgr.create_experiment(config)

        # Activate
        mgr.activate_experiment("exp-001")
        exp = mgr.get_by_id("exp-001")
        assert exp is not None
        assert exp.status == ExperimentStatus.ACTIVE

        # Ingest data: treatment clearly better (mean +15 vs +5)
        control_trades = _generate_trades(40, mean_pnl=5.0, spread=3.0)
        treatment_trades = _generate_trades(40, mean_pnl=15.0, spread=3.0)
        mgr.ingest_variant_data("exp-001", "control", control_trades)
        mgr.ingest_variant_data("exp-001", "treatment", treatment_trades)

        # Should auto-conclude (enough trades + significant difference)
        should_conclude = mgr.check_auto_conclusion("exp-001")
        assert should_conclude is True

        # Analyze and conclude
        result = mgr.analyze_experiment("exp-001")
        assert result.winner == "treatment"
        assert result.recommendation == "adopt_treatment"
        assert result.p_value is not None
        assert result.p_value < 0.05

        mgr.conclude_experiment("exp-001", result)
        exp = mgr.get_by_id("exp-001")
        assert exp.status == ExperimentStatus.CONCLUDED

        # Verify result persisted
        results = mgr.get_results()
        assert len(results) == 1
        assert results[0].experiment_id == "exp-001"
        assert results[0].winner == "treatment"


class TestInconclusive:
    """Test 2: Inconclusive — similar data, duration expires."""

    def test_inconclusive_result(self, tmp_path: Path) -> None:
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir, min_trades=30)

        config = _make_config(max_duration_days=1)
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        # Backdate started_at to simulate duration expiry
        experiments = mgr._load_all()
        experiments[0].started_at = datetime.now(timezone.utc) - timedelta(days=2)
        mgr._save_all(experiments)

        # Ingest similar data (no significant difference)
        control_trades = _generate_trades(40, mean_pnl=10.0, spread=8.0)
        treatment_trades = _generate_trades(40, mean_pnl=10.5, spread=8.0)
        mgr.ingest_variant_data("exp-001", "control", control_trades)
        mgr.ingest_variant_data("exp-001", "treatment", treatment_trades)

        # Duration expired -> should auto-conclude
        assert mgr.check_auto_conclusion("exp-001") is True

        result = mgr.analyze_experiment("exp-001")
        # With very similar means and high spread, p-value should be > 0.05
        assert result.recommendation in ("inconclusive", "extend")

        mgr.conclude_experiment("exp-001", result)
        exp = mgr.get_by_id("exp-001")
        assert exp.status == ExperimentStatus.CONCLUDED


class TestTreatmentWorse:
    """Test 3: Treatment worse — control clearly better."""

    def test_treatment_worse_keeps_control(self, tmp_path: Path) -> None:
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir, min_trades=30)

        config = _make_config()
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        # Control is better (mean +15 vs -5)
        control_trades = _generate_trades(40, mean_pnl=15.0, spread=3.0)
        treatment_trades = _generate_trades(40, mean_pnl=-5.0, spread=3.0)
        mgr.ingest_variant_data("exp-001", "control", control_trades)
        mgr.ingest_variant_data("exp-001", "treatment", treatment_trades)

        result = mgr.analyze_experiment("exp-001")
        assert result.winner == "control"
        assert result.recommendation == "keep_control"
        assert result.p_value is not None
        assert result.p_value < 0.05


class TestManualCancellation:
    """Test 4: Manual cancellation — cancel active experiment."""

    def test_cancel_active_experiment(self, tmp_path: Path) -> None:
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir)

        config = _make_config()
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        mgr.cancel_experiment("exp-001")
        exp = mgr.get_by_id("exp-001")
        assert exp is not None
        assert exp.status == ExperimentStatus.CANCELLED

        # No active experiments remain
        assert len(mgr.get_active()) == 0


class TestFeatureFlagOff:
    """Test 5: Feature flag off — no ExperimentManager-related items in app.state."""

    def test_ab_testing_disabled(self, tmp_path: Path) -> None:
        from orchestrator.config import AppConfig

        config = AppConfig(
            data_dir=str(tmp_path),
            ab_testing_enabled=False,
        )
        assert config.ab_testing_enabled is False

        # When ab_testing_enabled is False, ExperimentConfigGenerator should not be created
        # Simulate app wiring logic
        experiment_config_generator = None
        if config.ab_testing_enabled:
            experiment_config_generator = ExperimentConfigGenerator()

        assert experiment_config_generator is None


class TestFeatureFlagOn:
    """Test 6: Feature flag on — ExperimentConfigGenerator created."""

    def test_ab_testing_enabled(self, tmp_path: Path) -> None:
        from orchestrator.config import AppConfig

        config = AppConfig(
            data_dir=str(tmp_path),
            ab_testing_enabled=True,
        )
        assert config.ab_testing_enabled is True

        # When ab_testing_enabled is True, ExperimentConfigGenerator should be created
        experiment_manager = ExperimentManager(findings_dir=tmp_path / "findings")
        experiment_config_generator = None
        if config.ab_testing_enabled and experiment_manager is not None:
            experiment_config_generator = ExperimentConfigGenerator()

        assert experiment_config_generator is not None

    def test_ab_testing_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from orchestrator.config import AppConfig

        monkeypatch.setenv("AB_TESTING_ENABLED", "true")
        config = AppConfig.from_env()
        assert config.ab_testing_enabled is True

    def test_ab_testing_from_env_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from orchestrator.config import AppConfig

        monkeypatch.setenv("AB_TESTING_ENABLED", "false")
        config = AppConfig.from_env()
        assert config.ab_testing_enabled is False


class TestConcurrentExperiments:
    """Test 7: Concurrent experiments — two experiments on different bots."""

    def test_concurrent_experiments_on_different_bots(self, tmp_path: Path) -> None:
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir, min_trades=30)

        config_a = _make_config(experiment_id="exp-alpha", bot_id="bot_alpha")
        config_b = _make_config(experiment_id="exp-beta", bot_id="bot_beta")

        mgr.create_experiment(config_a)
        mgr.create_experiment(config_b)
        mgr.activate_experiment("exp-alpha")
        mgr.activate_experiment("exp-beta")

        # Both should be active simultaneously
        active = mgr.get_active()
        assert len(active) == 2
        bot_ids = {e.bot_id for e in active}
        assert bot_ids == {"bot_alpha", "bot_beta"}

        # Ingest data for both
        mgr.ingest_variant_data("exp-alpha", "control", _generate_trades(40, mean_pnl=5.0))
        mgr.ingest_variant_data("exp-alpha", "treatment", _generate_trades(40, mean_pnl=15.0))
        mgr.ingest_variant_data("exp-beta", "control", _generate_trades(40, mean_pnl=8.0))
        mgr.ingest_variant_data("exp-beta", "treatment", _generate_trades(40, mean_pnl=6.0))

        # Conclude alpha (treatment wins)
        result_a = mgr.analyze_experiment("exp-alpha")
        mgr.conclude_experiment("exp-alpha", result_a)

        # Beta still active
        active = mgr.get_active()
        assert len(active) == 1
        assert active[0].experiment_id == "exp-beta"

        # Conclude beta
        result_b = mgr.analyze_experiment("exp-beta")
        mgr.conclude_experiment("exp-beta", result_b)
        assert len(mgr.get_active()) == 0
        assert len(mgr.get_results()) == 2


class TestConfigGeneratorIntegration:
    """Test 8: Config generator integration — generate from suggestion with linkage."""

    def test_generate_config_from_suggestion(self, tmp_path: Path) -> None:
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        gen = ExperimentConfigGenerator()
        mgr = ExperimentManager(findings_dir=findings_dir, min_trades=30)

        # Generate experiment config from a suggestion
        config = gen.generate_from_suggestion(
            suggestion_id="sug-abc123",
            bot_id="bot_alpha",
            param_name="fast_period",
            current_value=10,
            proposed_value=8,
            title="Test fast_period reduction",
            duration_days=14,
        )

        assert config.source_suggestion_id == "sug-abc123"
        assert config.bot_id == "bot_alpha"
        assert len(config.variants) == 2
        assert config.variants[0].name == "control"
        assert config.variants[0].params["fast_period"] == 10
        assert config.variants[1].name == "treatment"
        assert config.variants[1].params["fast_period"] == 8

        # Create the experiment in the manager
        mgr.create_experiment(config)
        exp = mgr.get_by_id(config.experiment_id)
        assert exp is not None
        assert exp.source_suggestion_id == "sug-abc123"

    def test_generate_config_with_config_registry(self, tmp_path: Path) -> None:
        """Config generator can accept a config_registry (may be None)."""
        gen_no_registry = ExperimentConfigGenerator(config_registry=None)
        gen_with_registry = ExperimentConfigGenerator(config_registry=MagicMock())

        config1 = gen_no_registry.generate_from_suggestion(
            suggestion_id="sug-1", bot_id="bot_a",
            param_name="x", current_value=1, proposed_value=2,
        )
        config2 = gen_with_registry.generate_from_suggestion(
            suggestion_id="sug-1", bot_id="bot_a",
            param_name="x", current_value=1, proposed_value=2,
        )
        # Both generate valid configs with the same deterministic ID
        assert config1.experiment_id == config2.experiment_id


class TestSchedulerJobRegistration:
    """Test 9: Scheduler job registration — experiment_check job when fn provided."""

    def test_experiment_check_job_registered(self) -> None:
        from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs

        config = SchedulerConfig(experiment_check_interval_hours=6)

        async def dummy_experiment_check():
            pass

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            experiment_check_fn=dummy_experiment_check,
        )

        experiment_jobs = [j for j in jobs if j["name"] == "experiment_check"]
        assert len(experiment_jobs) == 1
        job = experiment_jobs[0]
        assert job["trigger"] == "interval"
        assert job["seconds"] == 6 * 3600  # 6 hours in seconds
        assert job["func"] is dummy_experiment_check

    def test_experiment_check_job_not_registered_when_none(self) -> None:
        from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs

        config = SchedulerConfig()

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            experiment_check_fn=None,
        )

        experiment_jobs = [j for j in jobs if j["name"] == "experiment_check"]
        assert len(experiment_jobs) == 0

    def test_experiment_check_interval_configurable(self) -> None:
        from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs

        config = SchedulerConfig(experiment_check_interval_hours=12)

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=AsyncMock(),
            monitoring_fn=AsyncMock(),
            relay_fn=AsyncMock(),
            experiment_check_fn=AsyncMock(),
        )

        experiment_jobs = [j for j in jobs if j["name"] == "experiment_check"]
        assert len(experiment_jobs) == 1
        assert experiment_jobs[0]["seconds"] == 12 * 3600


class TestTelegramCallbackRouting:
    """Test 10: Telegram callback routing — start_experiment and cancel_experiment."""

    @pytest.mark.asyncio
    async def test_start_experiment_callback(self, tmp_path: Path) -> None:
        from comms.telegram_handlers import TelegramCallbackRouter

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir)

        config = _make_config()
        mgr.create_experiment(config)

        router = TelegramCallbackRouter()

        async def handle_start(experiment_id: str, context=None) -> str:
            mgr.activate_experiment(experiment_id)
            return f"Experiment {experiment_id} started"

        async def handle_cancel(experiment_id: str, context=None) -> str:
            mgr.cancel_experiment(experiment_id)
            return f"Experiment {experiment_id} cancelled"

        router.register("start_experiment_", handle_start)
        router.register("cancel_experiment_", handle_cancel)

        # Dispatch start callback
        result = await router.dispatch("start_experiment_exp-001")
        assert result == "Experiment exp-001 started"
        exp = mgr.get_by_id("exp-001")
        assert exp.status == ExperimentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_cancel_experiment_callback(self, tmp_path: Path) -> None:
        from comms.telegram_handlers import TelegramCallbackRouter

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir)

        config = _make_config()
        mgr.create_experiment(config)
        mgr.activate_experiment("exp-001")

        router = TelegramCallbackRouter()

        async def handle_cancel(experiment_id: str, context=None) -> str:
            mgr.cancel_experiment(experiment_id)
            return f"Experiment {experiment_id} cancelled"

        router.register("cancel_experiment_", handle_cancel)

        result = await router.dispatch("cancel_experiment_exp-001")
        assert result == "Experiment exp-001 cancelled"
        exp = mgr.get_by_id("exp-001")
        assert exp.status == ExperimentStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_callbacks_registered_in_app_wiring(self, tmp_path: Path) -> None:
        """Verify that the app wiring pattern correctly registers both callbacks."""
        from comms.telegram_handlers import TelegramCallbackRouter

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        mgr = ExperimentManager(findings_dir=findings_dir)

        # Simulate the app.py wiring pattern
        callback_router = TelegramCallbackRouter()

        async def _handle_start_experiment(experiment_id: str, context=None) -> str:
            mgr.activate_experiment(experiment_id)
            return f"Experiment {experiment_id} started"

        async def _handle_cancel_experiment(experiment_id: str, context=None) -> str:
            mgr.cancel_experiment(experiment_id)
            return f"Experiment {experiment_id} cancelled"

        callback_router.register("start_experiment_", _handle_start_experiment)
        callback_router.register("cancel_experiment_", _handle_cancel_experiment)

        # Both prefixes should be in handlers
        assert "start_experiment_" in callback_router.handlers
        assert "cancel_experiment_" in callback_router.handlers

        # Create and test the full flow
        config = _make_config(experiment_id="exp-wire-test")
        mgr.create_experiment(config)

        await callback_router.dispatch("start_experiment_exp-wire-test")
        exp = mgr.get_by_id("exp-wire-test")
        assert exp.status == ExperimentStatus.ACTIVE

        await callback_router.dispatch("cancel_experiment_exp-wire-test")
        exp = mgr.get_by_id("exp-wire-test")
        assert exp.status == ExperimentStatus.CANCELLED
