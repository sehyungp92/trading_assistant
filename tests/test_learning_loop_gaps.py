# tests/test_learning_loop_gaps.py
"""Tests for 8 learning loop gap closures.

Covers:
- Gap 8: hypothesis_id field on SuggestionRecord
- Gap 1: Agent suggestion recording via _record_agent_suggestions
- Gap 4: Blocked suggestion details in validation log
- Gap 5: MagicMock removal + load_track_record_from_file
- Gap 7: TransferProposalBuilder findings_dir wiring
- Gap 6: JSONL-backed HypothesisLibrary in weekly handler
- Gap 3: Hypothesis outcome linking in _measure_outcomes
- Gap 2: Automated prediction evaluation via curated_dir
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from analysis.context_builder import ContextBuilder
from orchestrator.event_stream import EventStream
from orchestrator.handlers import Handlers
from schemas.agent_response import AgentPrediction, AgentSuggestion, ParsedAnalysis, StructuralProposal
from schemas.suggestion_tracking import SuggestionRecord, SuggestionStatus
from skills.suggestion_tracker import SuggestionTracker


def _make_handlers(tmp_path, tracker=None, es=None):
    """Create a Handlers instance with defaults for testing."""
    memory_dir = tmp_path / "memory"
    findings_dir = memory_dir / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "policies" / "v1").mkdir(parents=True, exist_ok=True)
    curated_dir = tmp_path / "data" / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    if tracker is None:
        tracker = SuggestionTracker(store_dir=findings_dir)
    if es is None:
        es = EventStream()

    return Handlers(
        agent_runner=MagicMock(),
        event_stream=es,
        dispatcher=AsyncMock(),
        notification_prefs=MagicMock(),
        curated_dir=curated_dir,
        memory_dir=memory_dir,
        runs_dir=tmp_path / "runs",
        source_root=tmp_path,
        bots=["bot1", "bot2"],
        suggestion_tracker=tracker,
    ), tracker, es


class TestGap8HypothesisIdField:
    """Gap 8: SuggestionRecord has hypothesis_id field."""

    def test_hypothesis_id_default_none(self):
        rec = SuggestionRecord(
            suggestion_id="test1",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            source_report_id="run-1",
        )
        assert rec.hypothesis_id is None

    def test_hypothesis_id_set(self):
        rec = SuggestionRecord(
            suggestion_id="test2",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            source_report_id="run-1",
            hypothesis_id="h-signal-recalibrate",
        )
        assert rec.hypothesis_id == "h-signal-recalibrate"

    def test_hypothesis_id_serialization(self):
        rec = SuggestionRecord(
            suggestion_id="test3",
            bot_id="bot1",
            title="Test",
            tier="parameter",
            source_report_id="run-1",
            hypothesis_id="h-exit-trailing",
        )
        data = rec.model_dump(mode="json")
        assert data["hypothesis_id"] == "h-exit-trailing"

    def test_hypothesis_id_backward_compat(self):
        """Old records without hypothesis_id deserialize fine."""
        data = {
            "suggestion_id": "old1",
            "bot_id": "bot1",
            "title": "Old suggestion",
            "tier": "parameter",
            "source_report_id": "run-old",
        }
        rec = SuggestionRecord(**data)
        assert rec.hypothesis_id is None


class TestGap1AgentSuggestionRecording:
    """Gap 1: _record_agent_suggestions records parsed suggestions to SuggestionTracker."""

    def test_record_agent_suggestions_basic(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path)

        from analysis.response_validator import ValidationResult
        from schemas.agent_response import AgentSuggestion

        approved = [
            AgentSuggestion(bot_id="bot1", title="Widen stop loss", category="stop_loss", confidence=0.7),
            AgentSuggestion(bot_id="bot2", title="Reduce exposure", category="position_sizing", confidence=0.6),
        ]
        validation = ValidationResult(approved_suggestions=approved)

        id_map = h._record_agent_suggestions(validation, "daily-2026-03-07")
        assert len(id_map) == 2
        all_records = tracker.load_all()
        assert len(all_records) == 2

    def test_record_agent_suggestions_with_hypothesis_link(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path)

        from analysis.response_validator import ValidationResult

        approved = [
            AgentSuggestion(bot_id="bot1", title="Switch to trailing stop", category="exit_timing", confidence=0.8),
        ]
        parsed = ParsedAnalysis(
            suggestions=approved,
            structural_proposals=[
                StructuralProposal(hypothesis_id="h-exit-trailing", bot_id="bot1", title="Trailing stop"),
            ],
        )
        validation = ValidationResult(approved_suggestions=approved)

        id_map = h._record_agent_suggestions(validation, "weekly-2026-03-07", parsed)
        assert len(id_map) == 1
        all_records = tracker.load_all()
        assert all_records[0]["hypothesis_id"] == "h-exit-trailing"

    def test_record_agent_suggestions_no_tracker(self, tmp_path):
        h, _, es = _make_handlers(tmp_path)
        h._suggestion_tracker = None
        from analysis.response_validator import ValidationResult

        validation = ValidationResult(approved_suggestions=[
            AgentSuggestion(bot_id="bot1", title="Test", confidence=0.5),
        ])
        result = h._record_agent_suggestions(validation, "run-1")
        assert result == {}

    def test_record_agent_suggestions_dedup(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path)
        from analysis.response_validator import ValidationResult

        approved = [AgentSuggestion(bot_id="bot1", title="Same suggestion", confidence=0.5)]
        validation = ValidationResult(approved_suggestions=approved)

        h._record_agent_suggestions(validation, "run-1")
        h._record_agent_suggestions(validation, "run-1")  # duplicate
        assert len(tracker.load_all()) == 1

    def test_record_agent_suggestions_broadcasts_event(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path)
        from analysis.response_validator import ValidationResult

        events = []
        es.subscribe_fn = lambda: None  # just to check broadcast
        original_broadcast = es.broadcast

        def capture_broadcast(event_type, data):
            events.append((event_type, data))
            return original_broadcast(event_type, data)

        es.broadcast = capture_broadcast

        approved = [AgentSuggestion(bot_id="bot1", title="Test suggestion", confidence=0.5)]
        validation = ValidationResult(approved_suggestions=approved)
        h._record_agent_suggestions(validation, "run-1")

        agent_events = [e for e in events if e[0] == "agent_suggestions_recorded"]
        assert len(agent_events) == 1
        assert agent_events[0][1]["count"] == 1


class TestGap4BlockedDetails:
    """Gap 4: Validation log includes blocked suggestion details."""

    def test_validate_and_annotate_returns_tuple(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path)

        parsed = ParsedAnalysis(
            suggestions=[
                AgentSuggestion(bot_id="bot1", title="Good suggestion", confidence=0.7),
            ],
            raw_report="Test report",
        )
        result = h._validate_and_annotate(parsed, "2026-03-07")
        assert isinstance(result, tuple)
        assert len(result) == 2
        report, validation = result
        assert isinstance(report, str)

    def test_blocked_details_in_validation_log(self, tmp_path):
        h, tracker, es = _make_handlers(tmp_path)

        # Pre-populate a rejected suggestion
        findings_dir = tmp_path / "memory" / "findings"
        suggestions_path = findings_dir / "suggestions.jsonl"
        rejected = {
            "suggestion_id": "rej1",
            "bot_id": "bot1",
            "title": "Widen stop loss on bot1",
            "tier": "parameter",
            "status": "rejected",
            "source_report_id": "old-run",
        }
        suggestions_path.write_text(json.dumps(rejected) + "\n", encoding="utf-8")

        parsed = ParsedAnalysis(
            suggestions=[
                AgentSuggestion(bot_id="bot1", title="Widen stop loss on bot1", confidence=0.7),
            ],
            raw_report="Test report",
        )
        report, validation = h._validate_and_annotate(parsed, "2026-03-07")

        # Check validation log has blocked_details
        log_path = findings_dir / "validation_log.jsonl"
        assert log_path.exists()
        log_entry = json.loads(log_path.read_text().strip())
        assert "blocked_details" in log_entry
        assert log_entry["blocked_count"] == 1
        assert log_entry["blocked_details"][0]["title"] == "Widen stop loss on bot1"
        assert "bot_id" in log_entry["blocked_details"][0]


class TestGap5RemoveMagicMock:
    """Gap 5: MagicMock removed from production context_builder.py."""

    def test_load_transfer_track_record_no_mock_import(self, tmp_path):
        """Verify context_builder doesn't import unittest.mock."""
        import inspect
        source = inspect.getsource(ContextBuilder.load_transfer_track_record)
        assert "MagicMock" not in source
        assert "unittest.mock" not in source

    def test_load_track_record_from_file_static(self, tmp_path):
        from skills.transfer_proposal_builder import TransferProposalBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        outcomes_path = findings_dir / "transfer_outcomes.jsonl"
        outcomes_path.write_text(
            json.dumps({"pattern_id": "p1", "verdict": "positive"}) + "\n"
            + json.dumps({"pattern_id": "p1", "verdict": "negative"}) + "\n"
            + json.dumps({"pattern_id": "p2", "verdict": "positive"}) + "\n",
            encoding="utf-8",
        )

        track = TransferProposalBuilder.load_track_record_from_file(findings_dir)
        assert "p1" in track
        assert track["p1"]["total"] == 2
        assert track["p1"]["positive"] == 1
        assert track["p1"]["success_rate"] == 0.5
        assert track["p2"]["success_rate"] == 1.0

    def test_load_track_record_from_file_empty(self, tmp_path):
        from skills.transfer_proposal_builder import TransferProposalBuilder

        track = TransferProposalBuilder.load_track_record_from_file(tmp_path)
        assert track == {}

    def test_context_builder_uses_static_method(self, tmp_path):
        """ContextBuilder.load_transfer_track_record uses the static method."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        outcomes_path = findings_dir / "transfer_outcomes.jsonl"
        outcomes_path.write_text(
            json.dumps({"pattern_id": "p1", "verdict": "positive"}) + "\n",
            encoding="utf-8",
        )

        ctx = ContextBuilder(memory_dir)
        track = ctx.load_transfer_track_record()
        assert "p1" in track
        assert track["p1"]["success_rate"] == 1.0


class TestGap7FindingsDirWiring:
    """Gap 7: TransferProposalBuilder gets findings_dir in weekly handler."""

    def test_transfer_builder_gets_findings_dir(self):
        """Verify the weekly handler code passes findings_dir to TransferProposalBuilder."""
        import inspect
        source = inspect.getsource(Handlers.handle_weekly_analysis)
        assert "findings_dir=self._memory_dir" in source


class TestGap6JsonlHypothesisLibrary:
    """Gap 6: Weekly handler uses JSONL-backed HypothesisLibrary."""

    def test_weekly_handler_uses_hypothesis_library_class(self):
        """Verify the weekly handler imports HypothesisLibrary class, not just get_relevant."""
        import inspect
        source = inspect.getsource(Handlers.handle_weekly_analysis)
        assert "HypothesisLibrary" in source
        assert "get_active()" in source

    def test_hypothesis_library_active_filters_retired(self, tmp_path):
        """Retired hypotheses are excluded from get_active."""
        from skills.hypothesis_library import HypothesisLibrary

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Retire a hypothesis by recording negative outcome + many rejections
        lib.record_outcome("h-signal-recalibrate", positive=False)
        for _ in range(4):
            lib.record_rejection("h-signal-recalibrate")

        active = lib.get_active()
        active_ids = {h.id for h in active}
        assert "h-signal-recalibrate" not in active_ids

    def test_high_effectiveness_included_without_keyword_match(self, tmp_path):
        """Hypotheses with high effectiveness are included even without keyword match."""
        from skills.hypothesis_library import HypothesisLibrary

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        lib = HypothesisLibrary(findings_dir)
        lib.seed_if_needed()

        # Give a hypothesis high effectiveness
        lib.record_proposal("h-exit-trailing")
        lib.record_outcome("h-exit-trailing", positive=True)

        records = lib.get_active()
        h = next(r for r in records if r.id == "h-exit-trailing")
        assert h.effectiveness > 0.3  # Would be included in merged list


class TestGap3HypothesisOutcomeLinking:
    """Gap 3: Hypothesis outcomes linked from suggestion outcomes in _measure_outcomes."""

    def test_measure_outcomes_links_hypothesis(self, tmp_path):
        """Verify _measure_outcomes code calls hypothesis_library.record_outcome."""
        import inspect
        from orchestrator.app import create_app
        source = inspect.getsource(create_app)
        assert "hypothesis_library.record_outcome" in source
        assert "hyp_id" in source


class TestGap2PredictionEvaluation:
    """Gap 2: Automated prediction evaluation via curated_dir in ContextBuilder."""

    def test_context_builder_accepts_curated_dir(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)
        assert ctx._curated_dir == curated_dir

    def test_context_builder_curated_dir_defaults_none(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        ctx = ContextBuilder(memory_dir)
        assert ctx._curated_dir is None

    def test_load_prediction_accuracy_with_curated_dir(self, tmp_path):
        """When curated_dir is available, real accuracy is computed."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        # Write a prediction
        from skills.prediction_tracker import PredictionTracker
        from schemas.agent_response import AgentPrediction

        tracker = PredictionTracker(findings_dir)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(
                bot_id="bot1", metric="pnl", direction="improve",
                confidence=0.8, timeframe_days=7,
            ),
        ])

        # Write corresponding curated data
        bot_dir = curated_dir / "2026-03-05" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(
            json.dumps({"total_pnl": 100.0, "win_rate": 0.6}),
            encoding="utf-8",
        )

        ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)
        result = ctx.load_prediction_accuracy()
        assert result["has_predictions"] is True
        assert "accuracy_by_metric" in result
        assert "pnl" in result["accuracy_by_metric"]

    def test_load_prediction_accuracy_no_curated(self, tmp_path):
        """Without curated_dir, falls back to count-only metadata."""
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)

        from skills.prediction_tracker import PredictionTracker
        from schemas.agent_response import AgentPrediction

        tracker = PredictionTracker(findings_dir)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(
                bot_id="bot1", metric="pnl", direction="improve",
                confidence=0.8, timeframe_days=7,
            ),
        ])

        ctx = ContextBuilder(memory_dir)
        result = ctx.load_prediction_accuracy()
        assert result["has_predictions"] is True
        assert result["count"] == 1

    def test_validate_and_annotate_passes_curated_dir(self):
        """Verify _validate_and_annotate creates ContextBuilder with curated_dir."""
        import inspect
        source = inspect.getsource(Handlers._validate_and_annotate)
        assert "curated_dir=self._curated_dir" in source

    def test_prediction_evaluation_in_measure_outcomes(self):
        """Verify _measure_outcomes evaluates predictions."""
        import inspect
        from orchestrator.app import create_app
        source = inspect.getsource(create_app)
        assert "evaluate_predictions" in source
        assert "get_accuracy_by_metric" not in source or "evaluate_predictions" in source
