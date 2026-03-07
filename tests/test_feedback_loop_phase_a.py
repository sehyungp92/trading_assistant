# tests/test_feedback_loop_phase_a.py
"""Tests for Phase A: Close the Suggestion Lifecycle.

Covers: path fix, suggestion recording, dedup, accept/reject feedback, IDs in prompts.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from analysis.context_builder import ContextBuilder
from analysis.feedback_handler import FeedbackHandler
from schemas.corrections import CorrectionType
from schemas.suggestion_tracking import SuggestionRecord, SuggestionStatus
from skills.suggestion_tracker import SuggestionTracker


# --- A0: Path mismatch fix ---


class TestPathAlignment:
    """Verify tracker and context_builder share the same findings path."""

    def test_tracker_and_context_builder_share_path(self, tmp_path):
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        ctx = ContextBuilder(memory_dir)

        # Record a suggestion via tracker
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        # Context builder should see it via load_rejected_suggestions path
        # (same directory)
        assert tracker._suggestions_path.parent == (memory_dir / "findings")
        all_suggestions = tracker.load_all()
        assert len(all_suggestions) == 1

    def test_load_rejected_sees_tracker_data(self, tmp_path):
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))
        tracker.reject("s001", "Bad idea")

        ctx = ContextBuilder(memory_dir)
        rejected = ctx.load_rejected_suggestions()
        assert len(rejected) == 1
        assert rejected[0]["suggestion_id"] == "s001"

    def test_load_outcome_measurements_sees_tracker_data(self, tmp_path):
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        from schemas.suggestion_tracking import SuggestionOutcome

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record_outcome(SuggestionOutcome(
            suggestion_id="s001", implemented_date="2026-03-01",
            pnl_delta_7d=100.0,
        ))

        ctx = ContextBuilder(memory_dir)
        outcomes = ctx.load_outcome_measurements()
        assert len(outcomes) == 1
        assert outcomes[0]["suggestion_id"] == "s001"


# --- A1: Handlers accepts suggestion_tracker ---


class TestHandlersSuggestionTracker:
    def test_handlers_with_tracker(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )
        assert h._suggestion_tracker is tracker

    def test_handlers_without_tracker(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
        )
        assert h._suggestion_tracker is None


# --- A2: Suggestion recording in weekly handler ---


class TestRecordSuggestions:
    def test_record_suggestions_creates_records(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        # Create mock StrategySuggestion objects
        suggestions = [
            MagicMock(title="Widen stop", bot_id="bot1", tier=MagicMock(value="parameter"), description="Widen the stop loss"),
            MagicMock(title="Remove filter X", bot_id="bot2", tier=MagicMock(value="filter"), description="Filter too aggressive"),
        ]

        id_map = h._record_suggestions(suggestions, "weekly-2026-03-01")
        assert len(id_map) == 2
        # Verify persisted
        all_recs = tracker.load_all()
        assert len(all_recs) == 2

    def test_record_suggestions_deterministic_ids(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Test", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        ids1 = h._record_suggestions(suggestions, "run1")
        # Same run_id + suggestions produce same IDs
        tracker2 = SuggestionTracker(store_dir=tmp_path / "dir2")
        h2 = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker2,
        )
        ids2 = h2._record_suggestions(suggestions, "run1")
        assert list(ids1.keys()) == list(ids2.keys())

    def test_record_suggestions_no_duplicates_on_rerun(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Test", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        h._record_suggestions(suggestions, "run1")
        h._record_suggestions(suggestions, "run1")  # re-run

        all_recs = tracker.load_all()
        assert len(all_recs) == 1  # dedup prevents duplicate

    def test_record_suggestions_without_tracker(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
        )
        # Should return empty dict and not crash
        result = h._record_suggestions([MagicMock()], "run1")
        assert result == {}

    def test_record_suggestions_broadcasts_event(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        es = EventStream()
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=es,
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Test", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        h._record_suggestions(suggestions, "run1")

        recent = es.get_recent()
        event_types = [e.event_type for e in recent]
        assert "suggestions_recorded" in event_types


# --- A3: Deduplication ---


class TestSuggestionDedup:
    def test_same_id_twice_no_duplicate(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        rec = SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        )
        assert tracker.record(rec) is True
        assert tracker.record(rec) is False
        assert len(tracker.load_all()) == 1

    def test_different_ids_both_recorded(self, tmp_path):
        tracker = SuggestionTracker(store_dir=tmp_path)
        for sid in ["s001", "s002"]:
            result = tracker.record(SuggestionRecord(
                suggestion_id=sid, bot_id="bot1", title=f"Test {sid}",
                tier="parameter", source_report_id="r1",
            ))
            assert result is True
        assert len(tracker.load_all()) == 2


# --- A4: Accept/reject feedback patterns ---


class TestFeedbackPatterns:
    def test_parse_accept_suggestion(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("approve suggestion #abc123")
        assert correction.correction_type == CorrectionType.SUGGESTION_ACCEPT
        assert correction.target_id == "abc123"

    def test_parse_reject_suggestion(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("reject suggestion #def456")
        assert correction.correction_type == CorrectionType.SUGGESTION_REJECT
        assert correction.target_id == "def456"

    def test_parse_accept_recommendation(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("implement recommendation abc123")
        assert correction.correction_type == CorrectionType.SUGGESTION_ACCEPT
        assert correction.target_id == "abc123"

    def test_parse_decline_suggestion(self):
        handler = FeedbackHandler(report_id="r1")
        correction = handler.parse("skip suggestion xyz")
        assert correction.correction_type == CorrectionType.SUGGESTION_REJECT
        assert correction.target_id == "xyz"

    @pytest.mark.asyncio
    async def test_handle_feedback_accept_routes_to_tracker(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream
        from orchestrator.orchestrator_brain import Action, ActionType

        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="abc123", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test-ev",
            bot_id="system",
            details={"text": "approve suggestion #abc123", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        suggestions = tracker.load_all()
        match = [s for s in suggestions if s["suggestion_id"] == "abc123"]
        assert match[0]["status"] == "implemented"

    @pytest.mark.asyncio
    async def test_handle_feedback_reject_routes_to_tracker(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream
        from orchestrator.orchestrator_brain import Action, ActionType

        tracker = SuggestionTracker(store_dir=tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="def456", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)

        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=memory_dir,
            runs_dir=tmp_path / "runs",
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        action = Action(
            type=ActionType.SEND_NOTIFICATION,
            event_id="test-ev",
            bot_id="system",
            details={"text": "reject suggestion #def456", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        suggestions = tracker.load_all()
        match = [s for s in suggestions if s["suggestion_id"] == "def456"]
        assert match[0]["status"] == "rejected"


# --- A5: Suggestion IDs in weekly instructions ---


class TestWeeklyInstructions:
    def test_instructions_contain_suggestion_id(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "suggestion_id" in _WEEKLY_INSTRUCTIONS
        assert "approve suggestion" in _WEEKLY_INSTRUCTIONS

    def test_metadata_populated_after_recording(self, tmp_path):
        from orchestrator.handlers import Handlers
        from orchestrator.event_stream import EventStream

        tracker = SuggestionTracker(store_dir=tmp_path)
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=EventStream(),
            dispatcher=MagicMock(),
            notification_prefs=MagicMock(),
            curated_dir=tmp_path,
            memory_dir=tmp_path,
            runs_dir=tmp_path,
            source_root=tmp_path,
            bots=[],
            suggestion_tracker=tracker,
        )

        suggestions = [
            MagicMock(title="Widen stop", bot_id="b1", tier=MagicMock(value="parameter"), description=""),
        ]

        id_map = h._record_suggestions(suggestions, "run1")
        assert len(id_map) == 1
        # All IDs are 12 hex chars
        for sid in id_map:
            assert len(sid) == 12
