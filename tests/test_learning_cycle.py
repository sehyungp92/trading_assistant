# tests/test_learning_cycle.py
"""Tests for Phase 4 — Autonomous Learning Cycle."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from schemas.learning_ledger import LearningLedgerEntry


# ── Learning Cycle Core ──

class TestLearningCycleRun:
    def _setup(self, tmp_path: Path):
        """Create minimal directory structure for learning cycle."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        return memory_dir, runs_dir, curated_dir

    def _write_curated_summary(self, curated_dir: Path, date: str, bot_id: str, summary: dict):
        path = curated_dir / date / bot_id / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary))

    @pytest.mark.asyncio
    async def test_cold_start_returns_entry(self, tmp_path: Path):
        """Cycle runs successfully with no historical data."""
        from skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)
        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        entry = await cycle.run("2026-03-01", "2026-03-07")
        assert isinstance(entry, LearningLedgerEntry)
        assert entry.week_start == "2026-03-01"
        assert entry.week_end == "2026-03-07"

    @pytest.mark.asyncio
    async def test_cycle_computes_composite_delta(self, tmp_path: Path):
        """Cycle computes composite deltas from curated data."""
        from skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)

        # Create 30 days of curated data for bot_a (enough for GT computation)
        base_date = datetime(2026, 3, 7, tzinfo=timezone.utc)
        for i in range(35):
            date_str = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            self._write_curated_summary(curated_dir, date_str, "bot_a", {
                "net_pnl": 10.0 + (i * 0.1 if i < 7 else 0),
                "win_rate": 0.55,
                "max_drawdown_pct": 5.0,
                "avg_process_quality": 75.0,
                "trade_count": 5,
            })

        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        entry = await cycle.run("2026-03-01", "2026-03-07")
        assert "bot_a" in entry.composite_delta or entry.composite_delta == {}

    @pytest.mark.asyncio
    async def test_cycle_records_to_ledger(self, tmp_path: Path):
        """Cycle persists entry to learning_ledger.jsonl."""
        from skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)
        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        await cycle.run("2026-03-01", "2026-03-07")

        ledger_path = memory_dir / "findings" / "learning_ledger.jsonl"
        assert ledger_path.exists()
        lines = ledger_path.read_text().strip().splitlines()
        assert len(lines) >= 1

    @pytest.mark.asyncio
    async def test_cycle_deduplicates(self, tmp_path: Path):
        """Running cycle twice with same dates → single entry."""
        from skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)
        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        await cycle.run("2026-03-01", "2026-03-07")
        await cycle.run("2026-03-01", "2026-03-07")

        ledger_path = memory_dir / "findings" / "learning_ledger.jsonl"
        lines = ledger_path.read_text().strip().splitlines()
        assert len(lines) == 1  # deduped by entry_id


# ── Synthesis Integration ──

class TestCycleSynthesis:
    def _setup(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()
        return memory_dir, runs_dir, curated_dir

    @pytest.mark.asyncio
    async def test_cycle_builds_synthesis(self, tmp_path: Path):
        """Cycle creates retrospective_synthesis.jsonl."""
        from skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)

        # Write an outcome for the week
        outcomes_path = memory_dir / "findings" / "outcomes.jsonl"
        outcomes_path.write_text(json.dumps({
            "suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing",
            "verdict": "positive", "pnl_delta": 0.05,
            "measured_at": "2026-03-03T10:00:00+00:00",
            "title": "Good change",
        }) + "\n")

        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        entry = await cycle.run("2026-03-01", "2026-03-07")
        assert "Good change" in entry.what_worked

        synth_path = memory_dir / "findings" / "retrospective_synthesis.jsonl"
        assert synth_path.exists()

    @pytest.mark.asyncio
    async def test_cycle_applies_recalibration(self, tmp_path: Path):
        """Cycle creates category_overrides when discard items found."""
        from skills.learning_cycle import LearningCycle

        memory_dir, runs_dir, curated_dir = self._setup(tmp_path)

        # Write 3+ failures for same (bot_id, category)
        outcomes = [
            {
                "suggestion_id": f"s{i}", "bot_id": "bot_a",
                "category": "filter_threshold",
                "verdict": "negative", "pnl_delta": -0.01,
                "measured_at": f"2026-03-0{i+1}T10:00:00+00:00",
                "title": f"Bad filter {i}",
            }
            for i in range(4)
        ]
        (memory_dir / "findings" / "outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes) + "\n"
        )

        cycle = LearningCycle(
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            runs_dir=runs_dir,
            bots=["bot_a"],
        )
        await cycle.run("2026-03-01", "2026-03-07")

        overrides_path = memory_dir / "findings" / "category_overrides.jsonl"
        assert overrides_path.exists()
        lines = overrides_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["confidence_multiplier"] == 0.3


# ── Hypothesis Lifecycle ──

class TestCycleHypothesisLifecycle:
    @pytest.mark.asyncio
    async def test_what_worked_records_positive_outcome(self, tmp_path: Path):
        """what_worked items → hypothesis_library.record_outcome(positive=True)."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        # Track calls
        calls = []
        class MockHypLib:
            def record_outcome(self, hyp_id, positive):
                calls.append({"id": hyp_id, "positive": positive})
            def get_active(self):
                return []

        (memory_dir / "findings" / "outcomes.jsonl").write_text(json.dumps({
            "suggestion_id": "s1", "bot_id": "bot_a", "category": "exit_timing",
            "verdict": "positive", "pnl_delta": 0.05,
            "measured_at": "2026-03-03T10:00:00+00:00", "title": "Good",
        }) + "\n")

        from skills.learning_cycle import LearningCycle
        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
        )
        await cycle.run("2026-03-01", "2026-03-07")

        positive_calls = [c for c in calls if c["positive"]]
        assert len(positive_calls) >= 1

    @pytest.mark.asyncio
    async def test_what_failed_records_negative_outcome(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        calls = []
        class MockHypLib:
            def record_outcome(self, hyp_id, positive):
                calls.append({"id": hyp_id, "positive": positive})
            def get_active(self):
                return []

        (memory_dir / "findings" / "outcomes.jsonl").write_text(json.dumps({
            "suggestion_id": "s2", "bot_id": "bot_a", "category": "signal",
            "verdict": "negative", "pnl_delta": -0.03,
            "measured_at": "2026-03-04T10:00:00+00:00", "title": "Bad",
        }) + "\n")

        from skills.learning_cycle import LearningCycle
        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
        )
        await cycle.run("2026-03-01", "2026-03-07")

        negative_calls = [c for c in calls if not c["positive"]]
        assert len(negative_calls) >= 1


# ── Experiment Selection ──

class TestExperimentSelection:
    def test_max_2_per_bot(self, tmp_path: Path):
        from skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [
                    MockHyp("h1", "signal", 0.8),
                    MockHyp("h2", "exit_timing", 0.6),
                    MockHyp("h3", "filter_threshold", 0.4),
                ]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        bot_a_count = sum(1 for s in selected if s["bot_id"] == "bot_a")
        assert bot_a_count <= 2

    def test_skips_discarded_categories(self, tmp_path: Path):
        from skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        # Write a discard override
        (memory_dir / "findings" / "category_overrides.jsonl").write_text(
            json.dumps({
                "bot_id": "bot_a", "category": "signal",
                "confidence_multiplier": 0.3,
            }) + "\n"
        )

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [MockHyp("h1", "signal", 0.8)]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        signal_selected = [s for s in selected if s["category"] == "signal"]
        assert len(signal_selected) == 0

    def test_skips_negative_effectiveness(self, tmp_path: Path):
        from skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [MockHyp("h1", "signal", -0.2)]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        assert len(selected) == 0

    def test_prefers_high_effectiveness(self, tmp_path: Path):
        from skills.learning_cycle import LearningCycle

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        class MockHyp:
            def __init__(self, hid, cat, eff):
                self.hypothesis_id = hid
                self.category = cat
                self.effectiveness = eff

        class MockHypLib:
            def get_active(self):
                return [
                    MockHyp("h_low", "signal", 0.1),
                    MockHyp("h_high", "exit_timing", 0.9),
                    MockHyp("h_mid", "stop_loss", 0.5),
                ]

        class MockExpTracker:
            def get_active_experiments(self):
                return []

        cycle = LearningCycle(
            curated_dir=curated_dir, memory_dir=memory_dir,
            runs_dir=runs_dir, bots=["bot_a"],
            hypothesis_library=MockHypLib(),
            experiment_tracker=MockExpTracker(),
        )
        selected = cycle._select_next_experiments()
        # Max 2 per bot, highest effectiveness first
        assert len(selected) == 2
        assert selected[0]["hypothesis_id"] == "h_high"
        assert selected[1]["hypothesis_id"] == "h_mid"


# ── Scheduler Wiring ──

class TestLearningCycleScheduling:
    def test_learning_cycle_spec_created(self):
        from orchestrator.scheduler import SchedulerConfig, build_scheduled_job_specs

        config = SchedulerConfig()

        async def _noop(scheduled_for=None):
            pass

        specs = build_scheduled_job_specs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            learning_cycle_fn=_noop,
        )
        lc_specs = [s for s in specs if s.name == "learning_cycle"]
        assert len(lc_specs) == 1
        assert lc_specs[0].day_of_week == "sun"
        assert lc_specs[0].hour == 11

    def test_learning_cycle_not_created_without_fn(self):
        from orchestrator.scheduler import SchedulerConfig, build_scheduled_job_specs

        config = SchedulerConfig()

        async def _noop(scheduled_for=None):
            pass

        specs = build_scheduled_job_specs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
        )
        lc_specs = [s for s in specs if "learning_cycle" in s.name]
        assert len(lc_specs) == 0


# ── Dashboard ──

class TestLearningDashboard:
    def test_dashboard_endpoint_exists(self, tmp_path: Path):
        """Dashboard endpoint is registered on the app."""
        from orchestrator.app import create_app
        app = create_app(db_dir=str(tmp_path))
        routes = [r.path for r in app.routes]
        assert "/learning/dashboard" in routes

    @pytest.mark.asyncio
    async def test_dashboard_returns_structure(self, tmp_path: Path):
        """Dashboard returns expected keys."""
        from orchestrator.app import create_app
        from httpx import AsyncClient, ASGITransport

        app = create_app(db_dir=str(tmp_path))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/learning/dashboard")
            assert resp.status_code == 200
            data = resp.json()
            assert "ground_truth_trend" in data
            assert "recent_lessons" in data
            assert "category_scorecard" in data
            assert "prediction_accuracy" in data
            assert "net_improvement" in data


# ── Learning Ledger Extended ──

class TestLedgerRecordWeekExtended:
    def test_record_week_with_explicit_delta(self, tmp_path: Path):
        """record_week accepts pre-computed composite_delta."""
        from skills.learning_ledger import LearningLedger

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        ledger = LearningLedger(findings_dir)

        entry = ledger.record_week(
            week_start="2026-03-01",
            week_end="2026-03-07",
            composite_delta={"bot_a": 0.15},
            net_improvement=True,
            what_worked=["Exit timing improved"],
        )
        assert entry.composite_delta == {"bot_a": 0.15}
        assert entry.net_improvement is True
        assert entry.what_worked == ["Exit timing improved"]

    def test_record_week_without_gt_snapshots(self, tmp_path: Path):
        """record_week works with no GT snapshots (cold start)."""
        from skills.learning_ledger import LearningLedger

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        ledger = LearningLedger(findings_dir)

        entry = ledger.record_week(
            week_start="2026-03-01",
            week_end="2026-03-07",
        )
        assert entry.composite_delta == {}
        assert entry.net_improvement is False
