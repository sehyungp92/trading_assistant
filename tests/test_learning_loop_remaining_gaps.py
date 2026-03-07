# tests/test_learning_loop_remaining_gaps.py
"""Tests for closing remaining learning loop gaps (C1-C5, M1-M3).

Covers:
- Brain feedback routing
- Worker feedback dispatch
- End-to-end feedback → tracker update
- Hypothesis lifecycle from feedback
- Pattern library ingestion
- Candidate hypothesis promotion
- Hypothesis ID instruction verification
- Daily instruction coverage
- Validation pattern aggregation
- Shared tier mapping
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from analysis.context_builder import ContextBuilder
from orchestrator.event_stream import EventStream
from orchestrator.handlers import Handlers
from orchestrator.orchestrator_brain import Action, ActionType, OrchestratorBrain
from orchestrator.worker import Worker
from schemas.agent_response import CATEGORY_TO_TIER, ParsedAnalysis, StructuralProposal
from schemas.corrections import CorrectionType
from schemas.suggestion_tracking import SuggestionRecord, SuggestionStatus
from skills.hypothesis_library import HypothesisLibrary
from skills.suggestion_tracker import SuggestionTracker


def _make_handlers(tmp_path, tracker=None, es=None):
    """Create a Handlers instance with defaults for testing."""
    memory_dir = tmp_path / "memory"
    findings_dir = memory_dir / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

    if tracker is None:
        tracker = SuggestionTracker(store_dir=findings_dir)
    if es is None:
        es = EventStream()

    return Handlers(
        agent_runner=MagicMock(),
        event_stream=es,
        dispatcher=AsyncMock(),
        notification_prefs=MagicMock(),
        curated_dir=tmp_path / "data" / "curated",
        memory_dir=memory_dir,
        runs_dir=tmp_path / "runs",
        source_root=tmp_path,
        bots=["bot1", "bot2"],
        suggestion_tracker=tracker,
    ), tracker, es


# ============================================================
# Phase 1: Brain feedback routing (C1)
# ============================================================


class TestBrainFeedbackRouting:
    def test_brain_routes_user_feedback_to_process_feedback(self):
        """Brain maps 'user_feedback' event_type → PROCESS_FEEDBACK action."""
        brain = OrchestratorBrain()
        event = {
            "event_type": "user_feedback",
            "event_id": "fb-001",
            "bot_id": "user",
            "payload": json.dumps({"text": "approve suggestion #abc123", "report_id": "r1"}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.PROCESS_FEEDBACK
        assert actions[0].details["text"] == "approve suggestion #abc123"

    def test_brain_feedback_passes_payload_details(self):
        """Payload is parsed and placed in action.details."""
        brain = OrchestratorBrain()
        event = {
            "event_type": "user_feedback",
            "event_id": "fb-002",
            "bot_id": "user",
            "payload": json.dumps({"text": "reject suggestion #xyz789", "report_id": "weekly-r2"}),
        }
        actions = brain.decide(event)
        assert actions[0].details["report_id"] == "weekly-r2"

    def test_brain_feedback_empty_payload(self):
        """Brain handles missing payload gracefully."""
        brain = OrchestratorBrain()
        event = {
            "event_type": "user_feedback",
            "event_id": "fb-003",
            "bot_id": "user",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.PROCESS_FEEDBACK


class TestWorkerFeedbackDispatch:
    @pytest.mark.asyncio
    async def test_worker_dispatches_to_on_feedback(self, tmp_path):
        """Worker dispatches PROCESS_FEEDBACK actions to on_feedback slot."""
        handler = AsyncMock()
        brain = OrchestratorBrain()

        from orchestrator.db.queue import EventQueue
        from orchestrator.task_registry import TaskRegistry

        q = EventQueue(db_path=str(tmp_path / "events.db"))
        reg = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
        await q.initialize()
        await reg.initialize()

        worker = Worker(queue=q, registry=reg, brain=brain)
        worker.on_feedback = handler

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-test",
            bot_id="user",
            details={"text": "approve suggestion #abc"},
        )
        await worker._dispatch(action)
        handler.assert_awaited_once_with(action)

        await q.close()
        await reg.close()

    @pytest.mark.asyncio
    async def test_worker_logs_when_no_feedback_handler(self, tmp_path, caplog):
        """Worker logs info when no on_feedback handler is set."""
        brain = OrchestratorBrain()

        from orchestrator.db.queue import EventQueue
        from orchestrator.task_registry import TaskRegistry

        q = EventQueue(db_path=str(tmp_path / "events.db"))
        reg = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
        await q.initialize()
        await reg.initialize()

        worker = Worker(queue=q, registry=reg, brain=brain)
        # on_feedback is None by default

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-noop",
            bot_id="user",
        )

        import logging
        with caplog.at_level(logging.INFO):
            await worker._dispatch(action)

        assert any("Feedback" in msg or "feedback" in msg for msg in caplog.messages)

        await q.close()
        await reg.close()


class TestEndToEndFeedbackRouting:
    @pytest.mark.asyncio
    async def test_feedback_event_to_tracker_update(self, tmp_path):
        """Full path: enqueue feedback event → brain routes → handle_feedback → tracker updated."""
        h, tracker, es = _make_handlers(tmp_path)

        # Pre-record a suggestion
        tracker.record(SuggestionRecord(
            suggestion_id="abc123",
            bot_id="bot1",
            title="Widen stop",
            tier="parameter",
            source_report_id="weekly-2026-03-01",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-e2e",
            bot_id="user",
            details={"text": "approve suggestion #abc123", "report_id": "weekly-2026-03-01"},
        )
        await h.handle_feedback(action)

        # Verify the suggestion was implemented
        all_recs = tracker.load_all()
        found = [r for r in all_recs if r["suggestion_id"] == "abc123"]
        assert len(found) == 1
        assert found[0]["status"] == SuggestionStatus.IMPLEMENTED.value


# ============================================================
# Phase 2: Hypothesis lifecycle from feedback (C2)
# ============================================================


class TestHypothesisLifecycleFromFeedback:
    @pytest.mark.asyncio
    async def test_accept_with_hypothesis_id_records_acceptance(self, tmp_path):
        """Accepting a suggestion linked to a hypothesis calls record_acceptance."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        # Seed hypothesis library
        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Record suggestion with hypothesis_id
        tracker.record(SuggestionRecord(
            suggestion_id="sugghyp1abc",
            bot_id="bot1",
            title="Switch to trailing stop",
            tier="hypothesis",
            source_report_id="weekly-test",
            hypothesis_id="h-exit-trailing",
        ))

        # Accept it
        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-hyp1",
            bot_id="user",
            details={"text": "approve suggestion #sugghyp1abc", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        # Verify hypothesis acceptance was recorded
        records = lib.get_all_records()
        trailing = [r for r in records if r.id == "h-exit-trailing"]
        assert len(trailing) == 1
        assert trailing[0].times_accepted >= 1

    @pytest.mark.asyncio
    async def test_reject_with_hypothesis_id_records_rejection(self, tmp_path):
        """Rejecting a suggestion linked to a hypothesis calls record_rejection."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        tracker.record(SuggestionRecord(
            suggestion_id="sugghyp2def",
            bot_id="bot1",
            title="Regime pause test",
            tier="hypothesis",
            source_report_id="weekly-test",
            hypothesis_id="h-regime-pause",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-hyp2",
            bot_id="user",
            details={"text": "reject suggestion #sugghyp2def", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        records = lib.get_all_records()
        pause = [r for r in records if r.id == "h-regime-pause"]
        assert len(pause) == 1
        assert pause[0].times_rejected >= 1

    @pytest.mark.asyncio
    async def test_accept_without_hypothesis_id_no_op(self, tmp_path):
        """Accepting a suggestion without hypothesis_id doesn't touch library."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()
        before = {r.id: r.times_accepted for r in lib.get_all_records()}

        tracker.record(SuggestionRecord(
            suggestion_id="suggnohyp01",
            bot_id="bot1",
            title="Simple tweak",
            tier="parameter",
            source_report_id="weekly-test",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-nohyp",
            bot_id="user",
            details={"text": "approve suggestion #suggnohyp01", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        after = {r.id: r.times_accepted for r in lib.get_all_records()}
        assert before == after

    @pytest.mark.asyncio
    async def test_auto_retirement_after_3_rejections(self, tmp_path):
        """Hypothesis auto-retires after 3 rejections with non-positive effectiveness."""
        h, tracker, es = _make_handlers(tmp_path)
        findings_dir = tmp_path / "memory" / "findings"

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Pre-set negative outcome + 2 rejections on hypothesis
        lib.record_outcome("h-fills-timing", positive=False)
        lib.record_rejection("h-fills-timing")
        lib.record_rejection("h-fills-timing")

        # Record and reject a 3rd time via feedback
        tracker.record(SuggestionRecord(
            suggestion_id="suggretire1",
            bot_id="bot1",
            title="Delay entry test",
            tier="hypothesis",
            source_report_id="weekly-test",
            hypothesis_id="h-fills-timing",
        ))

        action = Action(
            type=ActionType.PROCESS_FEEDBACK,
            event_id="fb-retire",
            bot_id="user",
            details={"text": "reject suggestion #suggretire1", "report_id": "weekly-test"},
        )
        await h.handle_feedback(action)

        records = lib.get_all_records()
        timing = [r for r in records if r.id == "h-fills-timing"]
        assert timing[0].status == "retired"


# ============================================================
# Phase 3: Pattern library ingestion (C3)
# ============================================================


class TestPatternLibraryIngestion:
    def test_structural_proposals_create_pattern_entries(self, tmp_path):
        """Structural proposals from parsed response create PatternEntry objects."""
        h, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(
                    bot_id="bot1",
                    title="Replace signal component X",
                    description="Signal X has decayed",
                    reversibility="moderate",
                    evidence="30-day correlation < 0.1",
                ),
                StructuralProposal(
                    bot_id="bot2",
                    title="Add regime gate for ranging",
                    description="Loses in ranging",
                    reversibility="easy",
                    evidence="20+ trades in ranging regime with negative PnL",
                ),
            ],
        )

        h._extract_and_record_patterns(parsed, ["bot1", "bot2"])

        from skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 2
        titles = {e.title for e in entries}
        assert "Replace signal component X" in titles
        assert "Add regime gate for ranging" in titles

    def test_pattern_dedup_by_title(self, tmp_path):
        """Same proposal title twice → only one pattern entry."""
        h, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(bot_id="bot1", title="Same proposal"),
            ],
        )

        h._extract_and_record_patterns(parsed, ["bot1", "bot2"])
        h._extract_and_record_patterns(parsed, ["bot1", "bot2"])

        from skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 1

    def test_pattern_entry_has_target_bots(self, tmp_path):
        """Pattern entry target_bots excludes the source bot."""
        h, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            structural_proposals=[
                StructuralProposal(bot_id="bot1", title="Filter restructure"),
            ],
        )

        h._extract_and_record_patterns(parsed, ["bot1", "bot2", "bot3"])

        from skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert entries[0].source_bot == "bot1"
        assert "bot1" not in entries[0].target_bots
        assert "bot2" in entries[0].target_bots
        assert "bot3" in entries[0].target_bots

    def test_empty_proposals_no_op(self, tmp_path):
        """No structural proposals → no pattern entries created."""
        h, tracker, es = _make_handlers(tmp_path)
        parsed = ParsedAnalysis(structural_proposals=[])
        h._extract_and_record_patterns(parsed, ["bot1"])

        from skills.pattern_library import PatternLibrary
        lib = PatternLibrary(tmp_path / "memory" / "findings")
        entries = lib.load_all()
        assert len(entries) == 0


# ============================================================
# Phase 4: Candidate hypothesis promotion (C4)
# ============================================================


class TestCandidateHypothesisPromotion:
    def test_candidate_promoted_after_2_proposals(self, tmp_path):
        """Candidate hypothesis with times_proposed >= 2 gets promoted to active."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(parents=True)

        lib = HypothesisLibrary(findings_dir)
        cid = lib.add_candidate(
            title="New pattern discovered",
            category="signal_decay",
            description="Root cause X seen 15 times",
        )

        # Simulate 2 proposals
        lib.record_proposal(cid)
        lib.record_proposal(cid)

        promoted = lib.promote_candidates()
        assert promoted == 1

        records = lib.get_all_records()
        candidate = [r for r in records if r.id == cid]
        assert candidate[0].status == "active"

    def test_candidate_not_promoted_with_1_proposal(self, tmp_path):
        """Candidate with only 1 proposal stays as candidate."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(parents=True)

        lib = HypothesisLibrary(findings_dir)
        cid = lib.add_candidate(
            title="Another pattern",
            category="filter_over_blocking",
            description="Some pattern",
        )

        lib.record_proposal(cid)

        promoted = lib.promote_candidates()
        assert promoted == 0

        records = lib.get_all_records()
        candidate = [r for r in records if r.id == cid]
        assert candidate[0].status == "candidate"

    def test_promoted_candidate_included_in_weekly_prompt(self, tmp_path):
        """Candidates (even with effectiveness 0) are included in weekly prompt merge."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(parents=True)

        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        cid = lib.add_candidate(
            title="Candidate for promotion",
            category="exit_timing",
            description="A candidate",
        )

        active = lib.get_active()
        candidate_records = [h for h in active if h.id == cid]
        assert len(candidate_records) == 1
        assert candidate_records[0].status == "candidate"
        # effectiveness == 0 since no outcomes
        assert candidate_records[0].effectiveness == 0


# ============================================================
# Phase 5: Hypothesis ID instruction verification (C5)
# ============================================================


class TestHypothesisIdInstructions:
    def test_weekly_instruction_contains_hypothesis_id_guidance(self):
        """Weekly instructions tell Claude to match hypothesis IDs."""
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "set hypothesis_id to that hypothesis" in _WEEKLY_INSTRUCTIONS
        assert "MUST check if any existing hypothesis matches" in _WEEKLY_INSTRUCTIONS

    def test_weekly_template_contains_required_note(self):
        """Structured output template says REQUIRED for hypothesis_id."""
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "REQUIRED: use id from structural_hypotheses" in _WEEKLY_INSTRUCTIONS

    def test_daily_template_contains_required_note(self):
        """Daily structured output template also has REQUIRED note."""
        from analysis.prompt_assembler import _INSTRUCTIONS

        assert "REQUIRED: use id from structural_hypotheses" in _INSTRUCTIONS


# ============================================================
# Phase 6: Moderate gaps (M1, M2, M3)
# ============================================================


class TestDailyInstructionCoverage:
    def test_daily_instructions_reference_failure_log(self):
        """Daily instructions now reference failure_log."""
        from analysis.prompt_assembler import _INSTRUCTIONS
        assert "failure_log" in _INSTRUCTIONS

    def test_daily_instructions_reference_consolidated_patterns(self):
        """Daily instructions now reference consolidated_patterns."""
        from analysis.prompt_assembler import _INSTRUCTIONS
        assert "consolidated_patterns" in _INSTRUCTIONS

    def test_daily_instructions_reference_hypothesis_track_record(self):
        """Daily instructions now reference hypothesis_track_record."""
        from analysis.prompt_assembler import _INSTRUCTIONS
        assert "hypothesis_track_record" in _INSTRUCTIONS

    def test_daily_instructions_reference_validation_patterns(self):
        """Daily instructions now reference validation_patterns."""
        from analysis.prompt_assembler import _INSTRUCTIONS
        assert "validation_patterns" in _INSTRUCTIONS


class TestValidationPatternAggregation:
    def test_load_validation_patterns_from_log(self, tmp_path):
        """ContextBuilder aggregates blocked suggestions from validation_log.jsonl."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        # Write sample validation log
        log_path = findings_dir / "validation_log.jsonl"
        entries = [
            {
                "date": "2026-03-01",
                "blocked_count": 2,
                "blocked_details": [
                    {"title": "Widen exit timing", "reason": "poor track record", "bot_id": "bot1"},
                    {"title": "Change filter threshold", "reason": "rejected before", "bot_id": "bot2"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "date": "2026-03-02",
                "blocked_count": 1,
                "blocked_details": [
                    {"title": "Adjust exit timing again", "reason": "low win rate", "bot_id": "bot1"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]
        with open(log_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        ctx = ContextBuilder(memory_dir)
        patterns = ctx.load_validation_patterns()

        assert "exit_timing" in patterns
        assert patterns["exit_timing"]["blocked_count"] == 2

    def test_empty_log_returns_empty_dict(self, tmp_path):
        """No validation log → empty dict."""
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        ctx = ContextBuilder(memory_dir)
        patterns = ctx.load_validation_patterns()
        assert patterns == {}

    def test_validation_patterns_in_base_package(self, tmp_path):
        """Validation patterns appear in base_package data when log exists."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)

        log_path = findings_dir / "validation_log.jsonl"
        entry = {
            "date": "2026-03-05",
            "blocked_count": 1,
            "blocked_details": [
                {"title": "Bad signal change", "reason": "blocked", "bot_id": "bot1"},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log_path.write_text(json.dumps(entry) + "\n")

        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "validation_patterns" in pkg.data


class TestSharedTierMapping:
    def test_category_to_tier_contains_all_categories(self):
        """CATEGORY_TO_TIER has entries for all known categories."""
        expected = {
            "exit_timing", "filter_threshold", "stop_loss",
            "signal", "structural", "position_sizing", "regime_gate",
        }
        assert set(CATEGORY_TO_TIER.keys()) == expected

    def test_scorer_tier_to_category_includes_legacy_mappings(self):
        """SuggestionScorer's _TIER_TO_CATEGORY includes strategy_variant and hypothesis."""
        from skills.suggestion_scorer import _TIER_TO_CATEGORY

        assert "strategy_variant" in _TIER_TO_CATEGORY
        assert "hypothesis" in _TIER_TO_CATEGORY
        assert _TIER_TO_CATEGORY["hypothesis"] == "structural"

    def test_handlers_uses_shared_mapping(self, tmp_path):
        """Handlers._record_agent_suggestions uses CATEGORY_TO_TIER from schemas."""
        h, tracker, es = _make_handlers(tmp_path)

        # Create a mock validation result with an approved structural suggestion
        mock_suggestion = MagicMock()
        mock_suggestion.title = "Restructure filter"
        mock_suggestion.bot_id = "bot1"
        mock_suggestion.category = "structural"
        mock_suggestion.evidence_summary = "Evidence"

        mock_validation = MagicMock()
        mock_validation.approved_suggestions = [mock_suggestion]

        id_map = h._record_agent_suggestions(mock_validation, "run-test")

        # Should be recorded with tier = "hypothesis" (from CATEGORY_TO_TIER)
        all_recs = tracker.load_all()
        assert len(all_recs) == 1
        assert all_recs[0]["tier"] == "hypothesis"


class TestWeeklyPromptValidationPatterns:
    def test_weekly_instructions_reference_validation_patterns(self):
        """Weekly instructions reference validation_patterns."""
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS
        assert "validation_patterns" in _WEEKLY_INSTRUCTIONS
