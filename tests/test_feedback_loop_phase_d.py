# tests/test_feedback_loop_phase_d.py
"""Tests for Phase D: Daily Context Enrichment.

Covers: active suggestions in daily prompt, lifecycle broadcasts.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from schemas.suggestion_tracking import SuggestionRecord, SuggestionStatus
from skills.suggestion_tracker import SuggestionTracker


# --- D0: Active suggestions in daily prompt ---


class TestActiveSuggestionsInDaily:
    def test_loads_only_non_rejected(self, tmp_path):
        from analysis.context_builder import ContextBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        tracker = SuggestionTracker(store_dir=findings_dir)
        # proposed
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Active one",
            tier="parameter", source_report_id="r1",
        ))
        # rejected
        tracker.record(SuggestionRecord(
            suggestion_id="s002", bot_id="bot1", title="Rejected one",
            tier="parameter", source_report_id="r1",
        ))
        tracker.reject("s002", "Bad idea")
        # deployed
        tracker.record(SuggestionRecord(
            suggestion_id="s003", bot_id="bot1", title="Deployed one",
            tier="parameter", source_report_id="r1",
        ))
        tracker.mark_deployed("s003")

        ctx = ContextBuilder(tmp_path)
        active = ctx.load_active_suggestions()
        ids = [s["suggestion_id"] for s in active]
        assert "s001" in ids  # proposed
        assert "s002" not in ids  # rejected
        assert "s003" in ids  # deployed

    def test_applies_temporal_window(self, tmp_path):
        from analysis.context_builder import ContextBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        tracker = SuggestionTracker(store_dir=findings_dir)
        # Create 40 suggestions (cap is 30)
        for i in range(40):
            tracker.record(SuggestionRecord(
                suggestion_id=f"s{i:03d}", bot_id="bot1", title=f"Test {i}",
                tier="parameter", source_report_id="r1",
            ))

        ctx = ContextBuilder(tmp_path)
        active = ctx.load_active_suggestions()
        assert len(active) <= 30

    def test_in_base_package(self, tmp_path):
        from analysis.context_builder import ContextBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (tmp_path / "policies" / "v1").mkdir(parents=True)

        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="s001", bot_id="bot1", title="Test",
            tier="parameter", source_report_id="r1",
        ))

        ctx = ContextBuilder(tmp_path)
        pkg = ctx.base_package()
        assert "active_suggestions" in pkg.data

    def test_daily_instructions_reference_active_suggestions(self):
        from analysis.prompt_assembler import _INSTRUCTIONS

        assert "active_suggestions" in _INSTRUCTIONS
        assert "DEPLOYED" in _INSTRUCTIONS


# --- D1: Lifecycle broadcasts ---


class TestLifecycleBroadcasts:
    @pytest.mark.asyncio
    async def test_accept_broadcasts_event(self, tmp_path):
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

        es = EventStream()
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=es,
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
            event_id="test",
            bot_id="system",
            details={"text": "approve suggestion #abc123", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        recent = es.get_recent()
        types = [e.event_type for e in recent]
        assert "suggestion_accepted" in types

    @pytest.mark.asyncio
    async def test_reject_broadcasts_event(self, tmp_path):
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

        es = EventStream()
        h = Handlers(
            agent_runner=MagicMock(),
            event_stream=es,
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
            event_id="test",
            bot_id="system",
            details={"text": "reject suggestion #def456", "report_id": "r1"},
        )
        await h.handle_feedback(action)

        recent = es.get_recent()
        types = [e.event_type for e in recent]
        assert "suggestion_rejected" in types

    def test_record_suggestions_broadcasts(self, tmp_path):
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
        types = [e.event_type for e in recent]
        assert "suggestions_recorded" in types
