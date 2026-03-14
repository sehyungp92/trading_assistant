# tests/test_action_wiring.py
"""Tests for Phase 2 — closing dead-end loops and wiring action on measurements."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.scheduler import (
    SchedulerConfig,
    build_scheduled_job_specs,
    create_scheduler_jobs,
)
from schemas.discovery import DiscoveryReport, StrategyIdea, TradeReference


# ── Discovery scheduling ──

class TestDiscoveryScheduling:
    def test_discovery_spec_created(self):
        config = SchedulerConfig()

        async def _noop(scheduled_for=None):
            pass

        specs = build_scheduled_job_specs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            discovery_fn=_noop,
        )
        discovery_specs = [s for s in specs if s.name == "discovery_analysis"]
        assert len(discovery_specs) == 1
        spec = discovery_specs[0]
        assert spec.day_of_week == "sat"
        assert spec.hour == 3
        assert spec.minute == 0
        assert spec.misfire_grace_time == 172800
        assert spec.catchup_limit == 1

    def test_discovery_not_created_without_fn(self):
        config = SchedulerConfig()

        async def _noop(scheduled_for=None):
            pass

        specs = build_scheduled_job_specs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
        )
        discovery_specs = [s for s in specs if "discovery" in s.name]
        assert len(discovery_specs) == 0

    def test_learning_cycle_spec_created(self):
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

    def test_legacy_discovery_job_created(self):
        config = SchedulerConfig()

        async def _noop():
            pass

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            discovery_fn=_noop,
        )
        discovery_jobs = [j for j in jobs if j["name"] == "discovery_analysis"]
        assert len(discovery_jobs) == 1
        assert discovery_jobs[0]["trigger"] == "cron"
        assert discovery_jobs[0]["day_of_week"] == "sat"

    def test_legacy_learning_cycle_job_created(self):
        config = SchedulerConfig()

        async def _noop():
            pass

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=_noop,
            monitoring_fn=_noop,
            relay_fn=_noop,
            learning_cycle_fn=_noop,
        )
        lc_jobs = [j for j in jobs if j["name"] == "learning_cycle"]
        assert len(lc_jobs) == 1
        assert lc_jobs[0]["day_of_week"] == "sun"
        assert lc_jobs[0]["hour"] == 11


# ── Strategy Ideation ──

class TestStrategyIdeation:
    def test_strategy_idea_schema(self):
        idea = StrategyIdea(
            idea_id="abc123",
            title="Regime-Filtered ORB Reversal",
            description="Enter counter-trend ORB trades only in mean-reverting regimes",
            edge_hypothesis="ORB failures in trending regimes are systematic",
            confidence=0.7,
        )
        assert idea.status == "proposed"
        assert idea.confidence == 0.7
        assert idea.proposed_at is not None

    def test_discovery_report_with_ideas(self):
        report = DiscoveryReport(
            run_id="disc-001",
            date="2026-03-01",
            strategy_ideas=[
                StrategyIdea(title="Test Idea", description="desc"),
            ],
        )
        assert len(report.strategy_ideas) == 1

    def test_strategy_ideas_persistence(self, tmp_path: Path):
        ideas_path = tmp_path / "findings" / "strategy_ideas.jsonl"
        ideas_path.parent.mkdir(parents=True)
        idea = {
            "idea_id": "test123",
            "title": "Test Strategy",
            "description": "A test strategy",
            "confidence": 0.8,
            "status": "proposed",
            "proposed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(ideas_path, "a") as f:
            f.write(json.dumps(idea) + "\n")

        lines = ideas_path.read_text().strip().splitlines()
        assert len(lines) == 1
        loaded = json.loads(lines[0])
        assert loaded["idea_id"] == "test123"


# ── Prediction recalibration ──

class TestPredictionRecalibration:
    def test_daily_prompt_contains_recalibration_instructions(self):
        from analysis.prompt_assembler import _FOCUSED_INSTRUCTIONS

        assert "PREDICTION TRACK RECORD" in _FOCUSED_INSTRUCTIONS
        assert "accuracy < 50%" in _FOCUSED_INSTRUCTIONS
        assert "accuracy > 70%" in _FOCUSED_INSTRUCTIONS

    def test_weekly_prompt_contains_recalibration_instructions(self):
        from analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "PREDICTION TRACK RECORD" in _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "systematic biases" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_daily_prompt_contains_ground_truth(self):
        from analysis.prompt_assembler import _FOCUSED_INSTRUCTIONS

        assert "GROUND TRUTH PERFORMANCE" in _FOCUSED_INSTRUCTIONS

    def test_weekly_prompt_contains_ground_truth(self):
        from analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "GROUND TRUTH PERFORMANCE" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_weekly_prompt_contains_synthesis(self):
        from analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "LEARNING SYNTHESIS" in _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "what_worked" in _FOCUSED_WEEKLY_INSTRUCTIONS
        assert "discard" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_weekly_prompt_contains_strategy_ideas(self):
        from analysis.weekly_prompt_assembler import _FOCUSED_WEEKLY_INSTRUCTIONS

        assert "STRATEGY IDEAS" in _FOCUSED_WEEKLY_INSTRUCTIONS

    def test_daily_prompt_contains_blocked_approaches(self):
        from analysis.prompt_assembler import _FOCUSED_INSTRUCTIONS

        assert "BLOCKED APPROACHES" in _FOCUSED_INSTRUCTIONS


# ── Discovery prompt ──

class TestDiscoveryPromptStrategyIdeation:
    def test_discovery_instructions_contain_strategy_ideation(self):
        from analysis.discovery_prompt_assembler import _DISCOVERY_INSTRUCTIONS

        assert "STRATEGY IDEATION" in _DISCOVERY_INSTRUCTIONS
        assert "edge hypothesis" in _DISCOVERY_INSTRUCTIONS
        assert "strategy_ideas" in _DISCOVERY_INSTRUCTIONS

    def test_discovery_structured_output_includes_strategy_ideas(self):
        from analysis.discovery_prompt_assembler import _DISCOVERY_INSTRUCTIONS

        assert '"strategy_ideas"' in _DISCOVERY_INSTRUCTIONS
        assert '"entry_logic"' in _DISCOVERY_INSTRUCTIONS
        assert '"exit_logic"' in _DISCOVERY_INSTRUCTIONS


# ── TransferProposalBuilder.create_from_reasoning ──

class TestTransferFromReasoning:
    def test_create_from_reasoning(self, tmp_path: Path):
        from skills.transfer_proposal_builder import TransferProposalBuilder

        # Need a mock pattern library
        class MockPatternLibrary:
            def load_active(self):
                return []

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        builder = TransferProposalBuilder(
            pattern_library=MockPatternLibrary(),
            curated_dir=curated_dir,
            bots=["bot_a", "bot_b", "bot_c"],
            findings_dir=findings_dir,
        )

        reasoning = {
            "suggestion_id": "s123",
            "transferable": True,
            "mechanism": "Exit timing improvement reduces tail losses",
            "category": "exit_timing",
            "bot_id": "bot_a",
        }

        targets = builder.create_from_reasoning(reasoning, source_bot="bot_a")
        assert "bot_b" in targets
        assert "bot_c" in targets
        assert "bot_a" not in targets

        # Check persistence
        proposals_path = findings_dir / "transfer_proposals.jsonl"
        assert proposals_path.exists()
        lines = proposals_path.read_text().strip().splitlines()
        assert len(lines) == 2  # bot_b and bot_c

    def test_create_from_reasoning_excludes_source_bot(self, tmp_path: Path):
        from skills.transfer_proposal_builder import TransferProposalBuilder

        class MockLib:
            def load_active(self):
                return []

        builder = TransferProposalBuilder(
            pattern_library=MockLib(),
            curated_dir=tmp_path,
            bots=["only_bot"],
            findings_dir=tmp_path / "findings",
        )

        targets = builder.create_from_reasoning(
            {"suggestion_id": "s1", "mechanism": "test"},
            source_bot="only_bot",
        )
        assert len(targets) == 0


# ── Context builder strategy ideas ──

class TestContextBuilderStrategyIdeas:
    def test_strategy_ideas_loaded_into_base_package(self, tmp_path: Path):
        from analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        ideas_path = memory_dir / "findings" / "strategy_ideas.jsonl"
        idea = {
            "idea_id": "i1", "title": "Test",
            "status": "proposed", "timestamp": "2026-02-01T00:00:00+00:00",
        }
        ideas_path.write_text(json.dumps(idea) + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "strategy_ideas" in pkg.data
        assert len(pkg.data["strategy_ideas"]) == 1

    def test_spurious_outcomes_loaded(self, tmp_path: Path):
        from analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        spurious_path = memory_dir / "findings" / "spurious_outcomes.jsonl"
        spurious_path.write_text(json.dumps({
            "suggestion_id": "s1",
            "mechanism": "Coincidental regime shift",
        }) + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "spurious_outcomes" in pkg.data
        assert len(pkg.data["spurious_outcomes"]) == 1
