# tests/test_prediction_tracker.py
"""Tests for prediction tracking and evaluation."""
from __future__ import annotations

import json

import pytest

from schemas.agent_response import AgentPrediction
from skills.prediction_tracker import PredictionTracker


class TestPredictionTracker:
    def test_record_and_load(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        predictions = [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
            AgentPrediction(bot_id="bot2", metric="win_rate", direction="decline", confidence=0.6),
        ]
        tracker.record_predictions("2026-03-01", predictions)

        loaded = tracker.load_predictions("2026-03-01")
        assert len(loaded) == 2
        assert loaded[0].bot_id == "bot1"
        assert loaded[1].metric == "win_rate"

    def test_load_filtered_by_week(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
        ])
        tracker.record_predictions("2026-03-08", [
            AgentPrediction(bot_id="bot2", metric="pnl", direction="decline", confidence=0.7),
        ])

        week1 = tracker.load_predictions("2026-03-01")
        assert len(week1) == 1
        assert week1[0].bot_id == "bot1"

        all_predictions = tracker.load_predictions()
        assert len(all_predictions) == 2

    def test_evaluate_correct_prediction(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
        ])

        # Create curated data with positive PnL
        curated = tmp_path / "curated"
        bot_dir = curated / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({"total_pnl": 500.0}))

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 1
        assert evaluation.correct == 1
        assert evaluation.accuracy == 1.0
        assert evaluation.verdicts[0].status == "correct"

    def test_evaluate_incorrect_prediction(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
        ])

        curated = tmp_path / "curated"
        bot_dir = curated / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({"total_pnl": -200.0}))

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 1
        assert evaluation.correct == 0
        assert evaluation.accuracy == 0.0
        assert evaluation.verdicts[0].status == "incorrect"

    def test_evaluate_missing_curated_data(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
        ])

        curated = tmp_path / "curated"
        curated.mkdir()

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 0  # insufficient data excluded from total
        assert evaluation.verdicts[0].status == "insufficient_data"

    def test_confidence_weighted_accuracy(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
            AgentPrediction(bot_id="bot2", metric="pnl", direction="decline", confidence=0.3),
        ])

        curated = tmp_path / "curated"
        for bot_id, pnl in [("bot1", 100.0), ("bot2", 100.0)]:
            d = curated / "2026-03-01" / bot_id
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"total_pnl": pnl}))

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.total == 2
        assert evaluation.correct == 1  # bot1 correct, bot2 wrong
        # CW accuracy: (0.9 * 1 + 0.3 * 0) / (0.9 + 0.3) = 0.75
        assert evaluation.confidence_weighted_accuracy == pytest.approx(0.75, abs=0.01)

    def test_accuracy_by_metric(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.8),
            AgentPrediction(bot_id="bot1", metric="win_rate", direction="improve", confidence=0.7),
        ])

        curated = tmp_path / "curated"
        d = curated / "2026-03-01" / "bot1"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "total_pnl": 100.0,
            "win_rate": -0.05,  # decline, not improve
        }))

        evaluation = tracker.evaluate_predictions("2026-03-01", curated)
        assert evaluation.accuracy_by_metric.get("pnl") == 1.0
        assert evaluation.accuracy_by_metric.get("win_rate") == 0.0

    def test_empty_predictions(self, tmp_path):
        tracker = PredictionTracker(tmp_path)
        evaluation = tracker.evaluate_predictions("2026-03-01", tmp_path)
        assert evaluation.total == 0
        assert evaluation.accuracy == 0.0
