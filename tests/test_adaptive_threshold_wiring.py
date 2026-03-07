"""Tests for adaptive threshold wiring — config, app, scheduler, context_builder, handlers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import AppConfig
from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs
from analysis.context_builder import ContextBuilder
from skills.threshold_learner import ThresholdLearner


class TestConfigWiring:
    """Test AppConfig reads ADAPTIVE_THRESHOLDS_ENABLED."""

    def test_default_disabled(self):
        """adaptive_thresholds_enabled defaults to False."""
        config = AppConfig()
        assert config.adaptive_thresholds_enabled is False

    def test_reads_from_env(self, monkeypatch):
        """Config reads ADAPTIVE_THRESHOLDS_ENABLED from env."""
        monkeypatch.setenv("ADAPTIVE_THRESHOLDS_ENABLED", "true")
        config = AppConfig.from_env()
        assert config.adaptive_thresholds_enabled is True

    def test_reads_false_from_env(self, monkeypatch):
        """Config reads ADAPTIVE_THRESHOLDS_ENABLED=false from env."""
        monkeypatch.setenv("ADAPTIVE_THRESHOLDS_ENABLED", "false")
        config = AppConfig.from_env()
        assert config.adaptive_thresholds_enabled is False


class TestAppWiring:
    """Test app.py creates ThresholdLearner based on feature flag."""

    def test_feature_flag_off_no_learner(self, tmp_path):
        """Feature flag off: no ThresholdLearner created."""
        config = AppConfig(data_dir=str(tmp_path), adaptive_thresholds_enabled=False)
        from orchestrator.app import create_app

        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.threshold_learner is None

    def test_feature_flag_on_creates_learner(self, tmp_path):
        """Feature flag on: ThresholdLearner created."""
        config = AppConfig(data_dir=str(tmp_path), adaptive_thresholds_enabled=True)
        from orchestrator.app import create_app

        app = create_app(db_dir=str(tmp_path), config=config)
        assert app.state.threshold_learner is not None
        assert isinstance(app.state.threshold_learner, ThresholdLearner)


class TestSchedulerWiring:
    """Test scheduler job registration for threshold learning."""

    def _noop(self):
        pass

    def test_job_registered_when_enabled(self):
        """Scheduler job registered when threshold_learning_fn provided."""
        config = SchedulerConfig()

        async def dummy():
            pass

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=dummy,
            monitoring_fn=dummy,
            relay_fn=dummy,
            threshold_learning_fn=dummy,
        )
        job_names = [j["name"] for j in jobs]
        assert "threshold_learning" in job_names

        tl_job = next(j for j in jobs if j["name"] == "threshold_learning")
        assert tl_job["trigger"] == "cron"
        assert tl_job["day_of_week"] == "sun"
        assert tl_job["hour"] == 9
        assert tl_job["minute"] == 30

    def test_job_not_registered_when_disabled(self):
        """Scheduler job NOT registered when threshold_learning_fn is None."""
        config = SchedulerConfig()

        async def dummy():
            pass

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=dummy,
            monitoring_fn=dummy,
            relay_fn=dummy,
            threshold_learning_fn=None,
        )
        job_names = [j["name"] for j in jobs]
        assert "threshold_learning" not in job_names


class TestContextBuilderWiring:
    """Test ContextBuilder loads threshold profile."""

    def test_load_threshold_profile_empty(self, tmp_path):
        """ContextBuilder returns empty dict when no learned_thresholds.jsonl."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        cb = ContextBuilder(memory_dir=memory_dir)
        result = cb.load_threshold_profile()
        assert result == {}

    def test_load_threshold_profile_with_data(self, tmp_path):
        """ContextBuilder loads threshold profiles from JSONL."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        profile_data = {
            "bot_id": "bot_a",
            "thresholds": {
                "alpha_decay:decay_threshold": {
                    "detector_name": "alpha_decay",
                    "bot_id": "bot_a",
                    "threshold_name": "decay_threshold",
                    "default_value": 0.3,
                    "learned_value": 0.15,
                    "sample_count": 20,
                    "confidence": 0.7,
                }
            },
            "total_outcomes_used": 20,
        }
        (findings_dir / "learned_thresholds.jsonl").write_text(
            json.dumps(profile_data) + "\n"
        )

        cb = ContextBuilder(memory_dir=memory_dir)
        result = cb.load_threshold_profile()
        assert result != {}
        assert "profiles" in result
        assert result["count"] == 1
        assert result["profiles"][0]["bot_id"] == "bot_a"

    def test_base_package_includes_threshold_profile(self, tmp_path):
        """base_package includes threshold_profile when available."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        profile_data = {
            "bot_id": "bot_a",
            "thresholds": {},
            "total_outcomes_used": 5,
        }
        (findings_dir / "learned_thresholds.jsonl").write_text(
            json.dumps(profile_data) + "\n"
        )

        cb = ContextBuilder(memory_dir=memory_dir)
        pkg = cb.base_package()
        assert "threshold_profile" in pkg.data
        assert pkg.data["threshold_profile"]["count"] == 1


class TestHandlersWiring:
    """Test Handlers receives threshold_learner parameter."""

    def test_handlers_accepts_threshold_learner(self, tmp_path):
        """Handlers constructor accepts threshold_learner parameter."""
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream
        from schemas.notifications import NotificationPreferences

        learner = MagicMock(spec=ThresholdLearner)

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot_a"],
            threshold_learner=learner,
        )
        assert handlers._threshold_learner is learner

    def test_handlers_without_threshold_learner(self, tmp_path):
        """Handlers constructor works without threshold_learner."""
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream
        from schemas.notifications import NotificationPreferences

        handlers = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=["bot_a"],
        )
        assert handlers._threshold_learner is None


class TestLearnThresholdsOnEmptyData:
    """Test that learn_thresholds runs without error on empty data."""

    def test_learn_thresholds_empty(self, tmp_path):
        """learn_thresholds runs without error on empty findings directory."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        learner = ThresholdLearner(findings_dir=findings_dir)
        result = learner.learn_thresholds()
        assert result == {}

    def test_get_threshold_no_profile(self, tmp_path):
        """get_threshold returns default when no profile exists."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        learner = ThresholdLearner(findings_dir=findings_dir)
        result = learner.get_threshold("alpha_decay", "decay_threshold", "bot_a", 0.3)
        assert result == 0.3
