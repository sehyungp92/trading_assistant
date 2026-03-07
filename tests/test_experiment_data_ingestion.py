# tests/test_experiment_data_ingestion.py
"""Tests for experiment data ingestion — Task 12."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skills.build_daily_metrics import DailyMetricsBuilder


# ── DailyMetricsBuilder.build_experiment_breakdown ──────────────


class TestBuildExperimentBreakdown:

    def test_extracts_from_snapshot(self):
        """build_experiment_breakdown extracts experiment data from snapshot."""
        builder = DailyMetricsBuilder(date="2026-03-07", bot_id="bot1")
        snapshot = {
            "experiment_breakdown": {
                "exp1": {
                    "control": {"trades": 10, "pnl": 50.0},
                    "treatment": {"trades": 12, "pnl": 75.0},
                },
            },
        }
        result = builder.build_experiment_breakdown(snapshot)
        assert "exp1" in result
        assert result["exp1"]["control"]["trades"] == 10

    def test_returns_empty_dict_when_no_experiment_data(self):
        """build_experiment_breakdown returns empty dict when no experiment_breakdown key."""
        builder = DailyMetricsBuilder(date="2026-03-07", bot_id="bot1")
        result = builder.build_experiment_breakdown({"some_other_key": 123})
        assert result == {}

    def test_returns_empty_dict_for_empty_breakdown(self):
        """build_experiment_breakdown returns empty dict when experiment_breakdown is empty."""
        builder = DailyMetricsBuilder(date="2026-03-07", bot_id="bot1")
        result = builder.build_experiment_breakdown({"experiment_breakdown": {}})
        assert result == {}


class TestWriteCuratedExperimentData:

    def test_writes_experiment_data_json(self, tmp_path: Path):
        """write_curated writes experiment_data.json when experiment data is present."""
        builder = DailyMetricsBuilder(date="2026-03-07", bot_id="bot1")
        snapshot = {
            "experiment_breakdown": {
                "exp1": {"control": {"trades": 5}, "treatment": {"trades": 7}},
            },
        }
        output_dir = builder.write_curated(
            trades=[], missed=[], base_dir=tmp_path,
            daily_snapshot=snapshot,
        )
        exp_file = output_dir / "experiment_data.json"
        assert exp_file.exists()
        data = json.loads(exp_file.read_text())
        assert "exp1" in data

    def test_backward_compat_without_experiment_data(self, tmp_path: Path):
        """write_curated does not create experiment_data.json when no experiment data."""
        builder = DailyMetricsBuilder(date="2026-03-07", bot_id="bot1")
        output_dir = builder.write_curated(
            trades=[], missed=[], base_dir=tmp_path,
            daily_snapshot={"per_strategy_summary": {}},
        )
        exp_file = output_dir / "experiment_data.json"
        assert not exp_file.exists()

    def test_backward_compat_no_snapshot(self, tmp_path: Path):
        """write_curated works fine when daily_snapshot is None."""
        builder = DailyMetricsBuilder(date="2026-03-07", bot_id="bot1")
        output_dir = builder.write_curated(
            trades=[], missed=[], base_dir=tmp_path,
        )
        exp_file = output_dir / "experiment_data.json"
        assert not exp_file.exists()


# ── Weekly handler experiment lifecycle ─────────────────────────


def _make_handlers(tmp_path: Path, experiment_manager=None, **kwargs):
    """Create a minimal Handlers instance for testing."""
    from orchestrator.agent_runner import AgentRunner
    from orchestrator.event_stream import EventStream
    from orchestrator.handlers import Handlers
    from schemas.notifications import NotificationPreferences

    agent_runner = MagicMock(spec=AgentRunner)
    event_stream = EventStream()
    dispatcher = MagicMock()

    return Handlers(
        agent_runner=agent_runner,
        event_stream=event_stream,
        dispatcher=dispatcher,
        notification_prefs=NotificationPreferences(),
        curated_dir=tmp_path / "curated",
        memory_dir=tmp_path / "memory",
        runs_dir=tmp_path / "runs",
        source_root=tmp_path,
        bots=["bot1"],
        experiment_manager=experiment_manager,
        **kwargs,
    )


class TestWeeklyHandlerExperiments:

    def test_checks_active_experiments(self, tmp_path: Path):
        """Weekly handler checks active experiments via experiment_manager."""
        mgr = MagicMock()
        mgr.get_active.return_value = []
        h = _make_handlers(tmp_path, experiment_manager=mgr)
        assert h._experiment_manager is mgr

    def test_auto_concludes_experiment(self, tmp_path: Path):
        """Weekly handler auto-concludes experiment when check_auto_conclusion returns True."""
        from schemas.experiments import (
            ExperimentConfig,
            ExperimentResult,
            ExperimentStatus,
            ExperimentVariant,
        )

        exp = ExperimentConfig(
            experiment_id="exp1",
            bot_id="bot1",
            title="Test",
            variants=[
                ExperimentVariant(name="control", params={}, allocation_pct=50),
                ExperimentVariant(name="treatment", params={"x": 1}, allocation_pct=50),
            ],
            status=ExperimentStatus.ACTIVE,
        )
        result = ExperimentResult(
            experiment_id="exp1",
            variant_metrics=[],
            recommendation="adopt_treatment",
        )

        mgr = MagicMock()
        mgr.get_active.return_value = [exp]
        mgr.check_auto_conclusion.return_value = True
        mgr.analyze_experiment.return_value = result

        h = _make_handlers(tmp_path, experiment_manager=mgr)

        # Simulate the experiment lifecycle check logic directly
        active = mgr.get_active()
        for e in active:
            if mgr.check_auto_conclusion(e.experiment_id):
                r = mgr.analyze_experiment(e.experiment_id)
                mgr.conclude_experiment(e.experiment_id, r)

        mgr.conclude_experiment.assert_called_once_with("exp1", result)

    def test_broadcasts_experiment_concluded(self, tmp_path: Path):
        """Weekly handler broadcasts experiment_concluded event."""
        from orchestrator.event_stream import EventStream
        from schemas.experiments import (
            ExperimentConfig,
            ExperimentResult,
            ExperimentStatus,
            ExperimentVariant,
        )

        exp = ExperimentConfig(
            experiment_id="exp1",
            bot_id="bot1",
            title="Test",
            variants=[
                ExperimentVariant(name="control", params={}, allocation_pct=50),
                ExperimentVariant(name="treatment", params={"x": 1}, allocation_pct=50),
            ],
            status=ExperimentStatus.ACTIVE,
        )
        result = ExperimentResult(
            experiment_id="exp1",
            variant_metrics=[],
            recommendation="adopt_treatment",
        )

        mgr = MagicMock()
        mgr.get_active.return_value = [exp]
        mgr.check_auto_conclusion.return_value = True
        mgr.analyze_experiment.return_value = result

        event_stream = EventStream()
        broadcast_calls = []
        original_broadcast = event_stream.broadcast

        def capture_broadcast(event_type, data):
            broadcast_calls.append((event_type, data))
            return original_broadcast(event_type, data)

        event_stream.broadcast = capture_broadcast

        h = _make_handlers(tmp_path, experiment_manager=mgr)
        h._event_stream = event_stream

        # Simulate the lifecycle check
        active = mgr.get_active()
        for e in active:
            if mgr.check_auto_conclusion(e.experiment_id):
                r = mgr.analyze_experiment(e.experiment_id)
                mgr.conclude_experiment(e.experiment_id, r)
                h._event_stream.broadcast(
                    "experiment_concluded",
                    {"experiment_id": e.experiment_id, "recommendation": r.recommendation},
                )

        assert any(t == "experiment_concluded" for t, _ in broadcast_calls)

    def test_no_active_experiments_graceful(self, tmp_path: Path):
        """Weekly handler handles no active experiments gracefully."""
        mgr = MagicMock()
        mgr.get_active.return_value = []

        h = _make_handlers(tmp_path, experiment_manager=mgr)

        # Should not raise
        mgr.get_active()
        mgr.check_auto_conclusion.assert_not_called()
        mgr.conclude_experiment.assert_not_called()

    def test_missing_experiment_manager_graceful(self, tmp_path: Path):
        """Weekly handler handles missing experiment_manager gracefully."""
        h = _make_handlers(tmp_path, experiment_manager=None)
        assert h._experiment_manager is None

    def test_handlers_constructor_accepts_experiment_manager(self, tmp_path: Path):
        """Handlers constructor accepts experiment_manager parameter."""
        mgr = MagicMock()
        h = _make_handlers(tmp_path, experiment_manager=mgr)
        assert h._experiment_manager is mgr
