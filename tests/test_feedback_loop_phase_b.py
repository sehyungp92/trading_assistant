# tests/test_feedback_loop_phase_b.py
"""Tests for Phase B: Activate Outcome Feedback Loop.

Covers: prompt instructions, ForecastTracker, wiring into handlers/context_builder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from schemas.forecast_tracking import AccuracyTrend, ForecastMetaAnalysis, ForecastRecord
from skills.forecast_tracker import ForecastTracker


# --- B0 + B1: Instructions ---


class TestOutcomeInstructions:
    def test_weekly_instructions_reference_retrospective(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "RETROSPECTIVE QUESTIONS" in _WEEKLY_INSTRUCTIONS

    def test_weekly_instructions_reference_outcome_measurements(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "outcome_measurements" in _WEEKLY_INSTRUCTIONS

    def test_daily_instructions_reference_outcome_measurements(self):
        from analysis.prompt_assembler import _INSTRUCTIONS

        # Daily focused instructions reference quantified impact and evidence base
        assert "evidence base" in _INSTRUCTIONS


# --- B2: ForecastTracker ---


class TestForecastTracker:
    def test_record_and_load(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        rec = ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
            by_bot={"bot1": 0.8, "bot2": 0.6},
        )
        tracker.record_week(rec)
        loaded = tracker.load_all()
        assert len(loaded) == 1
        assert loaded[0].accuracy == 0.7

    def test_rolling_accuracy(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # 4 weeks of data
        for i in range(4):
            tracker.record_week(ForecastRecord(
                week_start=f"2026-02-{i * 7 + 3:02d}",
                week_end=f"2026-02-{i * 7 + 9:02d}",
                predictions_reviewed=10,
                correct_predictions=6 + i,  # 6, 7, 8, 9 correct
                accuracy=(6 + i) / 10,
            ))
        meta = tracker.compute_meta_analysis()
        assert meta.weeks_analyzed == 4
        # Total correct = 6+7+8+9 = 30 out of 40
        assert meta.rolling_accuracy_4w == 0.75
        assert meta.rolling_accuracy_12w == 0.75  # same since < 12 weeks

    def test_trend_detection_improving(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # Older weeks: low accuracy; recent: high
        for i in range(6):
            tracker.record_week(ForecastRecord(
                week_start=f"2026-01-{i * 7 + 6:02d}",
                week_end=f"2026-01-{i * 7 + 12:02d}",
                predictions_reviewed=10,
                correct_predictions=3 if i < 3 else 8,
                accuracy=0.3 if i < 3 else 0.8,
            ))
        meta = tracker.compute_meta_analysis()
        assert meta.trend == AccuracyTrend.IMPROVING

    def test_trend_detection_degrading(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # Older weeks: high; recent: low
        for i in range(6):
            tracker.record_week(ForecastRecord(
                week_start=f"2026-01-{i * 7 + 6:02d}",
                week_end=f"2026-01-{i * 7 + 12:02d}",
                predictions_reviewed=10,
                correct_predictions=8 if i < 3 else 3,
                accuracy=0.8 if i < 3 else 0.3,
            ))
        meta = tracker.compute_meta_analysis()
        assert meta.trend == AccuracyTrend.DEGRADING

    def test_per_bot_breakdown(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
            by_bot={"bot1": 0.9, "bot2": 0.5},
        ))
        meta = tracker.compute_meta_analysis()
        assert meta.accuracy_by_bot["bot1"] == 0.9
        assert meta.accuracy_by_bot["bot2"] == 0.5

    def test_calibration_computation(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        # Very high accuracy → positive calibration (under-confident)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=9,
            accuracy=0.9,
        ))
        meta = tracker.compute_meta_analysis()
        assert meta.calibration_adjustment > 0  # accuracy 0.9 > 0.5

    def test_empty_history_defaults(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        meta = tracker.compute_meta_analysis()
        assert meta.weeks_analyzed == 0
        assert meta.rolling_accuracy_4w == 0.0
        assert meta.trend == AccuracyTrend.STABLE

    def test_single_week_is_stable(self, tmp_path):
        tracker = ForecastTracker(tmp_path)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
        ))
        meta = tracker.compute_meta_analysis()
        assert meta.trend == AccuracyTrend.STABLE


# --- B3: Wiring ---


class TestForecastWiring:
    def test_context_builder_loads_forecast_meta(self, tmp_path):
        from analysis.context_builder import ContextBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        tracker = ForecastTracker(findings_dir)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
        ))

        ctx = ContextBuilder(tmp_path)
        meta = ctx.load_forecast_meta()
        assert meta["weeks_analyzed"] == 1
        assert meta["rolling_accuracy_4w"] == 0.7

    def test_base_package_includes_forecast_meta(self, tmp_path):
        from analysis.context_builder import ContextBuilder

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()
        (tmp_path / "policies" / "v1").mkdir(parents=True)

        tracker = ForecastTracker(findings_dir)
        tracker.record_week(ForecastRecord(
            week_start="2026-02-24",
            week_end="2026-03-02",
            predictions_reviewed=10,
            correct_predictions=7,
            accuracy=0.7,
        ))

        ctx = ContextBuilder(tmp_path)
        pkg = ctx.base_package()
        assert "forecast_meta_analysis" in pkg.data

    def test_weekly_instructions_reference_calibration(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "forecast_meta_analysis" in _WEEKLY_INSTRUCTIONS

    def test_base_package_includes_active_suggestions(self, tmp_path):
        from analysis.context_builder import ContextBuilder
        from schemas.suggestion_tracking import SuggestionRecord
        from skills.suggestion_tracker import SuggestionTracker

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
        assert len(pkg.data["active_suggestions"]) == 1

    def test_context_builder_no_forecast_on_empty(self, tmp_path):
        from analysis.context_builder import ContextBuilder

        (tmp_path / "findings").mkdir()
        (tmp_path / "policies" / "v1").mkdir(parents=True)

        ctx = ContextBuilder(tmp_path)
        pkg = ctx.base_package()
        assert "forecast_meta_analysis" not in pkg.data
