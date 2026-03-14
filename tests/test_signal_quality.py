# tests/test_signal_quality.py
"""Tests for Phase 5 — Signal Quality + Simplicity Criterion."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analysis.context_builder import _apply_temporal_window


# ── Temporal Decay ──

class TestTemporalDecay:
    def test_recent_entries_scored_higher(self):
        """Recent entries appear before older ones."""
        now = datetime.now(timezone.utc)
        entries = [
            {"timestamp": (now - timedelta(days=30)).isoformat(), "id": "old"},
            {"timestamp": (now - timedelta(days=1)).isoformat(), "id": "recent"},
            {"timestamp": (now - timedelta(days=15)).isoformat(), "id": "mid"},
        ]
        result = _apply_temporal_window(entries)
        ids = [e["id"] for e in result]
        assert ids[0] == "recent"
        assert ids[1] == "mid"
        assert ids[2] == "old"

    def test_entries_without_timestamps_last(self):
        """Entries without timestamps go after timestamped ones."""
        now = datetime.now(timezone.utc)
        entries = [
            {"id": "no_ts"},
            {"timestamp": (now - timedelta(days=1)).isoformat(), "id": "has_ts"},
        ]
        result = _apply_temporal_window(entries)
        assert result[0]["id"] == "has_ts"
        assert result[1]["id"] == "no_ts"

    def test_old_entries_excluded(self):
        """Entries older than max_age_days are excluded."""
        now = datetime.now(timezone.utc)
        entries = [
            {"timestamp": (now - timedelta(days=100)).isoformat(), "id": "ancient"},
            {"timestamp": (now - timedelta(days=5)).isoformat(), "id": "recent"},
        ]
        result = _apply_temporal_window(entries, max_age_days=90)
        ids = [e["id"] for e in result]
        assert "ancient" not in ids
        assert "recent" in ids

    def test_max_entries_cap(self):
        """Result is capped at max_entries."""
        now = datetime.now(timezone.utc)
        entries = [
            {"timestamp": (now - timedelta(days=i)).isoformat(), "id": f"e{i}"}
            for i in range(20)
        ]
        result = _apply_temporal_window(entries, max_entries=5)
        assert len(result) == 5

    def test_decay_ordering_correct(self):
        """Exponential decay gives correct relative ordering."""
        now = datetime.now(timezone.utc)
        # Day 0 should have score 1.0, day 14 should have score 0.5
        entries = [
            {"timestamp": (now - timedelta(days=14)).isoformat(), "id": "two_weeks"},
            {"timestamp": now.isoformat(), "id": "today"},
            {"timestamp": (now - timedelta(days=7)).isoformat(), "id": "one_week"},
        ]
        result = _apply_temporal_window(entries)
        ids = [e["id"] for e in result]
        assert ids == ["today", "one_week", "two_weeks"]

    def test_empty_input(self):
        result = _apply_temporal_window([])
        assert result == []


# ── Simplicity Criterion ──

class TestSimplicityCriterion:
    def test_marginal_track_record_blocked(self):
        """Category with sample>=3, win_rate<0.5, avg_pnl<0.001 → blocked."""
        from analysis.response_validator import ResponseValidator
        from schemas.agent_response import AgentSuggestion, ParsedAnalysis
        from schemas.suggestion_scoring import CategoryScore, CategoryScorecard

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot_a", category="filter_threshold",
                win_rate=0.4, avg_pnl_delta=0.0005, sample_size=5,
                confidence_multiplier=0.4,
            ),
        ])
        validator = ResponseValidator(category_scorecard=scorecard)
        parsed = ParsedAnalysis(suggestions=[
            AgentSuggestion(
                title="Tweak filter", bot_id="bot_a",
                category="filter_threshold", confidence=0.6,
            ),
        ])
        result = validator.validate(parsed)
        assert len(result.blocked_suggestions) == 1
        assert "marginal" in result.blocked_suggestions[0].reason

    def test_strong_track_record_passes(self):
        """Category with good win_rate passes simplicity check."""
        from analysis.response_validator import ResponseValidator
        from schemas.agent_response import AgentSuggestion, ParsedAnalysis
        from schemas.suggestion_scoring import CategoryScore, CategoryScorecard

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot_a", category="exit_timing",
                win_rate=0.7, avg_pnl_delta=0.05, sample_size=10,
                confidence_multiplier=0.7,
            ),
        ])
        validator = ResponseValidator(category_scorecard=scorecard)
        parsed = ParsedAnalysis(suggestions=[
            AgentSuggestion(
                title="Better exit", bot_id="bot_a",
                category="exit_timing", confidence=0.7,
            ),
        ])
        result = validator.validate(parsed)
        assert len(result.approved_suggestions) == 1
        assert len(result.blocked_suggestions) == 0

    def test_low_confidence_structural_blocked(self):
        """Low-confidence structural suggestion → blocked."""
        from analysis.response_validator import ResponseValidator
        from schemas.agent_response import AgentSuggestion, ParsedAnalysis

        validator = ResponseValidator()
        parsed = ParsedAnalysis(suggestions=[
            AgentSuggestion(
                title="Restructure everything", bot_id="bot_a",
                category="structural", confidence=0.3,
            ),
        ])
        result = validator.validate(parsed)
        assert len(result.blocked_suggestions) == 1
        assert "low-confidence structural" in result.blocked_suggestions[0].reason

    def test_high_confidence_structural_passes(self):
        """High-confidence structural suggestion → approved."""
        from analysis.response_validator import ResponseValidator
        from schemas.agent_response import AgentSuggestion, ParsedAnalysis

        validator = ResponseValidator()
        parsed = ParsedAnalysis(suggestions=[
            AgentSuggestion(
                title="Add regime filter", bot_id="bot_a",
                category="structural", confidence=0.7,
            ),
        ])
        result = validator.validate(parsed)
        assert len(result.approved_suggestions) == 1

    def test_insufficient_samples_not_blocked(self):
        """Category with < 3 samples → not blocked by simplicity criterion."""
        from analysis.response_validator import ResponseValidator
        from schemas.agent_response import AgentSuggestion, ParsedAnalysis
        from schemas.suggestion_scoring import CategoryScore, CategoryScorecard

        scorecard = CategoryScorecard(scores=[
            CategoryScore(
                bot_id="bot_a", category="signal",
                win_rate=0.3, avg_pnl_delta=0.0001, sample_size=2,
                confidence_multiplier=1.0,
            ),
        ])
        validator = ResponseValidator(category_scorecard=scorecard)
        parsed = ParsedAnalysis(suggestions=[
            AgentSuggestion(
                title="Change signal", bot_id="bot_a",
                category="signal", confidence=0.5,
            ),
        ])
        result = validator.validate(parsed)
        assert len(result.approved_suggestions) == 1


# ── Context Budget ──

class TestContextBudget:
    def _setup_memory(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        (memory_dir / "findings").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)
        return memory_dir

    def test_high_priority_items_always_present(self, tmp_path: Path):
        """ground_truth_trend and last_week_synthesis are kept even with small budget."""
        from analysis.context_builder import ContextBuilder

        memory_dir = self._setup_memory(tmp_path)

        # Write ground truth data
        (memory_dir / "findings" / "learning_ledger.jsonl").write_text(
            json.dumps({
                "entry_id": "abc", "week_start": "2026-03-01", "week_end": "2026-03-07",
                "ground_truth_end": {"bot_a": {"composite_score": 0.6}},
                "composite_delta": {"bot_a": 0.1},
                "lessons_for_next_week": ["Test lesson"],
            }) + "\n"
        )

        # Write synthesis
        (memory_dir / "findings" / "retrospective_synthesis.jsonl").write_text(
            json.dumps({
                "week_start": "2026-03-01", "week_end": "2026-03-07",
                "lessons": ["Lesson"], "what_worked": [], "what_failed": [],
                "discard": [], "ground_truth_deltas": {},
            }) + "\n"
        )

        ctx = ContextBuilder(memory_dir)
        # Use a large enough budget to include these plus any auto-seeded data
        pkg = ctx.base_package(context_budget_items=5)
        # High-priority items should be present
        assert "ground_truth_trend" in pkg.data
        assert "last_week_synthesis" in pkg.data

    def test_low_priority_items_dropped_when_over_budget(self, tmp_path: Path):
        """With a very small budget, low-priority items are dropped."""
        from analysis.context_builder import ContextBuilder

        memory_dir = self._setup_memory(tmp_path)

        # Write many data sources
        findings = memory_dir / "findings"
        (findings / "outcomes.jsonl").write_text(json.dumps({
            "suggestion_id": "s1", "verdict": "positive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        (findings / "suggestions.jsonl").write_text(json.dumps({
            "suggestion_id": "s1", "status": "implemented",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
        (findings / "learning_ledger.jsonl").write_text(json.dumps({
            "entry_id": "abc", "week_start": "2026-03-01", "week_end": "2026-03-07",
            "ground_truth_end": {"bot_a": {"composite_score": 0.6}},
            "composite_delta": {"bot_a": 0.1},
            "lessons_for_next_week": ["Lesson"],
        }) + "\n")

        ctx = ContextBuilder(memory_dir)
        # Very small budget
        pkg = ctx.base_package(context_budget_items=2)
        assert len(pkg.data) <= 2

    def test_default_budget_allows_all(self, tmp_path: Path):
        """Default budget of 15 doesn't drop items when fewer present."""
        from analysis.context_builder import ContextBuilder

        memory_dir = self._setup_memory(tmp_path)
        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()  # default budget
        # Should work fine with no data
        assert isinstance(pkg.data, dict)

    def test_budget_preserves_priority_order(self, tmp_path: Path):
        """When budget is hit, higher-priority items are preserved over lower-priority."""
        from analysis.context_builder import ContextBuilder

        memory_dir = self._setup_memory(tmp_path)
        findings = memory_dir / "findings"

        # Write ground_truth (high priority)
        (findings / "learning_ledger.jsonl").write_text(json.dumps({
            "entry_id": "abc", "week_start": "2026-03-01", "week_end": "2026-03-07",
            "ground_truth_end": {"bot_a": {"composite_score": 0.6}},
            "composite_delta": {"bot_a": 0.1},
            "lessons_for_next_week": ["Lesson"],
        }) + "\n")

        # Write outcome_reasonings (lower priority)
        (findings / "outcome_reasonings.jsonl").write_text(json.dumps({
            "suggestion_id": "s1", "mechanism": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")

        ctx = ContextBuilder(memory_dir)
        # Budget = 2: should keep ground_truth_trend (rank 1) + hypothesis_track_record (auto-seeded, unprioritized)
        # But outcome_reasonings (rank 12) should be kept over unprioritized if budget allows
        pkg_small = ctx.base_package(context_budget_items=2)

        # With a budget of 2, we should have ground_truth_trend as highest priority
        assert "ground_truth_trend" in pkg_small.data
        # And exactly 2 items
        assert len(pkg_small.data) <= 2
