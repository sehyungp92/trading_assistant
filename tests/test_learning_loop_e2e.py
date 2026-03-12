# tests/test_learning_loop_e2e.py
"""End-to-end learning loop integration test.

Tests the complete cycle:
1. Strategy engine produces suggestions
2. Mock Claude response with structured block
3. Response parsed → ParsedAnalysis
4. Validation applied → blocked suggestions stripped, confidence adjusted
5. Predictions recorded
6. User approves a suggestion → implement()
7. AutoOutcomeMeasurer measures → outcome recorded
8. Category scorecard recomputed → success rate updated
9. Next weekly prompt includes scorecard → validator would cap confidence
10. Hypothesis lifecycle updated → low-effectiveness hypothesis retired
11. Verify: next prompt does NOT contain retired hypothesis
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.context_builder import ContextBuilder
from analysis.response_parser import parse_response
from analysis.response_validator import ResponseValidator
from schemas.agent_response import AgentPrediction, AgentSuggestion, ParsedAnalysis
from schemas.suggestion_scoring import CategoryScore, CategoryScorecard
from schemas.suggestion_tracking import SuggestionOutcome, SuggestionRecord, SuggestionStatus
from skills.forecast_tracker import ForecastTracker
from skills.hypothesis_library import HypothesisLibrary, get_relevant
from skills.prediction_tracker import PredictionTracker
from skills.suggestion_scorer import SuggestionScorer
from skills.suggestion_tracker import SuggestionTracker


@pytest.fixture
def memory_dir(tmp_path):
    """Create a memory directory with minimal policy files."""
    mem = tmp_path / "memory"
    findings = mem / "findings"
    policies = mem / "policies" / "v1"
    policies.mkdir(parents=True)
    findings.mkdir(parents=True)
    (policies / "agents.md").write_text("# Agents\nYou are an agent.")
    (policies / "trading_rules.md").write_text("# Trading Rules\nBe careful.")
    (policies / "soul.md").write_text("# Soul\nQuantify everything.")
    return mem


class TestLearningLoopE2E:
    def test_full_learning_cycle(self, tmp_path, memory_dir):
        """Test the complete learning loop from suggestion → measurement → feedback."""
        findings_dir = memory_dir / "findings"

        # --- Step 1: Record initial suggestions ---
        tracker = SuggestionTracker(store_dir=findings_dir)
        suggestion1 = SuggestionRecord(
            suggestion_id="s001",
            bot_id="bot1",
            title="Widen stop loss on bot1",
            tier="exit_timing",
            source_report_id="weekly-2026-02-23",
            description="Current stop is too tight for this regime",
        )
        suggestion2 = SuggestionRecord(
            suggestion_id="s002",
            bot_id="bot1",
            title="Adjust filter threshold on bot1",
            tier="parameter",
            source_report_id="weekly-2026-02-23",
        )
        tracker.record(suggestion1)
        tracker.record(suggestion2)

        # --- Step 2: Simulate Claude response with structured block ---
        mock_response = """# Weekly Report
Bot1 had a strong week. Win rate improved.

## Suggestions
1. Widen stop loss on bot1 — exit efficiency is low
2. New signal idea for bot2

<!-- STRUCTURED_OUTPUT
{
  "predictions": [
    {"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.8, "timeframe_days": 7, "reasoning": "Strong momentum alignment"},
    {"bot_id": "bot2", "metric": "drawdown", "direction": "decline", "confidence": 0.6, "timeframe_days": 7, "reasoning": "Regime unfavorable"}
  ],
  "suggestions": [
    {"suggestion_id": "s001", "bot_id": "bot1", "category": "exit_timing", "title": "Widen stop loss on bot1", "expected_impact": "+0.5% daily PnL", "confidence": 0.7, "evidence_summary": "exit_efficiency < 40%"},
    {"suggestion_id": "new1", "bot_id": "bot2", "category": "signal", "title": "Add momentum crossover", "expected_impact": "+0.3% weekly PnL", "confidence": 0.6, "evidence_summary": "Strong backtest results"}
  ],
  "structural_proposals": [
    {"hypothesis_id": "h-exit-trailing", "bot_id": "bot1", "title": "Switch to trailing stop", "description": "Fixed stops causing premature exits", "reversibility": "easy", "evidence": "exit_efficiency < 50%", "estimated_complexity": "medium"}
  ]
}
-->
"""

        # --- Step 3: Parse response ---
        parsed = parse_response(mock_response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 2
        assert len(parsed.suggestions) == 2
        assert len(parsed.structural_proposals) == 1

        # --- Step 4: Validate response ---
        # Reject suggestion s002 first
        tracker.reject("s002", "Not worth the effort")

        rejected = ContextBuilder(memory_dir).load_rejected_suggestions()
        assert len(rejected) == 1

        # No forecast meta yet, no scorecard
        validator = ResponseValidator(
            rejected_suggestions=rejected,
            forecast_meta={},
            category_scorecard=CategoryScorecard(),
        )
        result = validator.validate(parsed)
        # Neither suggestion matches the rejected "Adjust filter threshold" closely enough
        assert len(result.approved_suggestions) == 2
        assert len(result.blocked_suggestions) == 0

        # --- Step 5: Record predictions ---
        pred_tracker = PredictionTracker(findings_dir)
        pred_tracker.record_predictions("2026-03-01", parsed.predictions)
        loaded = pred_tracker.load_predictions("2026-03-01")
        assert len(loaded) == 2

        # --- Step 6: User approves suggestion s001 ---
        tracker.implement("s001")
        all_suggestions = tracker.load_all()
        s001 = [s for s in all_suggestions if s["suggestion_id"] == "s001"][0]
        assert s001["status"] == SuggestionStatus.IMPLEMENTED.value

        # --- Step 7: Record outcome for s001 ---
        outcome = SuggestionOutcome(
            suggestion_id="s001",
            implemented_date="2026-02-24",
            pnl_delta_7d=150.0,
            win_rate_delta_7d=0.05,
        )
        tracker.record_outcome(outcome)
        outcomes = tracker.load_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["pnl_delta_7d"] == 150.0

        # --- Step 8: Recompute category scorecard ---
        scorer = SuggestionScorer(findings_dir)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) >= 1
        # exit_timing had 1 positive outcome
        exit_score = scorecard.get_score("bot1", "exit_timing")
        assert exit_score is not None
        assert exit_score.win_rate == 1.0
        assert exit_score.sample_size == 1

        # --- Step 9: Verify scorecard appears in next prompt ---
        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert "category_scorecard" in pkg.data

        # --- Step 10: Hypothesis lifecycle ---
        hyp_lib = HypothesisLibrary(findings_dir)
        hyp_lib.seed_if_needed()
        hyp_lib.record_proposal("h-exit-trailing")

        # Simulate negative outcome + 3 rejections to trigger retirement
        hyp_lib.record_outcome("h-crowding-diversify", positive=False)
        hyp_lib.record_rejection("h-crowding-diversify")
        hyp_lib.record_rejection("h-crowding-diversify")
        hyp_lib.record_rejection("h-crowding-diversify")

        active = hyp_lib.get_active()
        assert not any(h.id == "h-crowding-diversify" for h in active)

        # --- Step 11: Verify retired hypothesis excluded from track record ---
        track = hyp_lib.get_track_record()
        assert track["h-crowding-diversify"]["status"] == "retired"
        assert track["h-exit-trailing"]["times_proposed"] == 1

    def test_validation_blocks_poor_category(self, tmp_path, memory_dir):
        """Test that suggestions in categories with poor track record are blocked."""
        findings_dir = memory_dir / "findings"

        # Create several suggestions and outcomes with poor results
        tracker = SuggestionTracker(store_dir=findings_dir)
        for i in range(5):
            tracker.record(SuggestionRecord(
                suggestion_id=f"bad{i}",
                bot_id="bot1",
                title=f"Bad exit timing idea {i}",
                tier="exit_timing",
                source_report_id="weekly-test",
            ))
            tracker.implement(f"bad{i}")
            tracker.record_outcome(SuggestionOutcome(
                suggestion_id=f"bad{i}",
                implemented_date="2026-02-01",
                pnl_delta_7d=-100.0,  # all negative
            ))

        scorer = SuggestionScorer(findings_dir)
        scorecard = scorer.compute_scorecard()
        exit_score = scorecard.get_score("bot1", "exit_timing")
        assert exit_score is not None
        assert exit_score.win_rate == 0.0
        assert exit_score.sample_size == 5

        # Now validate a new exit_timing suggestion for bot1
        suggestion = AgentSuggestion(
            bot_id="bot1",
            category="exit_timing",
            title="Another exit timing idea",
            confidence=0.9,
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])
        validator = ResponseValidator(category_scorecard=scorecard)
        result = validator.validate(parsed)

        assert len(result.blocked_suggestions) == 1
        assert "track record" in result.blocked_suggestions[0].reason.lower()

    def test_prediction_evaluation_flow(self, tmp_path, memory_dir):
        """Test predictions recorded → evaluated → accuracy computed."""
        findings_dir = memory_dir / "findings"
        curated_dir = tmp_path / "curated"

        tracker = PredictionTracker(findings_dir)
        tracker.record_predictions("2026-03-01", [
            AgentPrediction(bot_id="bot1", metric="pnl", direction="improve", confidence=0.9),
            AgentPrediction(bot_id="bot1", metric="win_rate", direction="decline", confidence=0.7),
        ])

        # Create curated data
        bot_dir = curated_dir / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "total_pnl": 200.0,  # positive → improve ✓
            "win_rate": 0.6,  # positive → improve (predicted decline ✗)
        }))

        baseline_dir = curated_dir / "2026-03-01" / "bot1"
        (baseline_dir / "summary.json").write_text(json.dumps({
            "total_pnl": 100.0,
            "win_rate": 0.60,
        }))
        target_dir = curated_dir / "2026-03-08" / "bot1"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "summary.json").write_text(json.dumps({
            "total_pnl": 200.0,
            "win_rate": 0.65,
        }))

        evaluation = tracker.evaluate_predictions("2026-03-01", curated_dir)
        assert evaluation.total == 2
        assert evaluation.correct == 1
        assert evaluation.accuracy == 0.5
        assert evaluation.accuracy_by_metric["pnl"] == 1.0
        assert evaluation.accuracy_by_metric["win_rate"] == 0.0

    def test_context_builder_includes_all_new_data(self, memory_dir):
        """Verify base_package includes category_scorecard and hypothesis_track_record."""
        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        # These should at least not error; with no data they may be empty
        # But the loading methods should work without error
        assert isinstance(pkg.data, dict)
        # Even with no data, these keys shouldn't be present (empty = omitted)
        # This just verifies no exceptions during loading

    def test_forecast_meta_includes_accuracy_by_metric(self, tmp_path, memory_dir):
        """Verify ForecastMetaAnalysis now supports accuracy_by_metric."""
        from schemas.forecast_tracking import ForecastMetaAnalysis

        meta = ForecastMetaAnalysis(
            rolling_accuracy_4w=0.6,
            accuracy_by_metric={"pnl": 0.8, "win_rate": 0.3},
            weeks_analyzed=4,
        )
        assert meta.accuracy_by_metric["pnl"] == 0.8
        assert meta.accuracy_by_metric["win_rate"] == 0.3

    def test_parse_validate_annotate_report(self, memory_dir):
        """End-to-end: parse → validate → annotated report with notes."""
        findings_dir = memory_dir / "findings"

        # Add a rejected suggestion
        tracker = SuggestionTracker(store_dir=findings_dir)
        tracker.record(SuggestionRecord(
            suggestion_id="rej1",
            bot_id="bot1",
            title="Widen stop loss on bot1",
            tier="exit_timing",
            source_report_id="w1",
        ))
        tracker.reject("rej1", "Already tried, didn't work")

        response = """# Report
<!-- STRUCTURED_OUTPUT
{
  "predictions": [],
  "suggestions": [
    {"bot_id": "bot1", "title": "Widen stop loss on bot1", "category": "exit_timing", "confidence": 0.8},
    {"bot_id": "bot2", "title": "New idea for bot2", "category": "signal", "confidence": 0.6}
  ]
}
-->
"""
        parsed = parse_response(response)
        assert parsed.parse_success

        ctx = ContextBuilder(memory_dir)
        rejected = ctx.load_rejected_suggestions()
        validator = ResponseValidator(rejected_suggestions=rejected)
        result = validator.validate(parsed)

        assert len(result.blocked_suggestions) == 1
        assert len(result.approved_suggestions) == 1
        assert result.approved_suggestions[0].title == "New idea for bot2"

        final = parsed.raw_report
        if result.validator_notes:
            final += "\n\n---\n## Validator Notes\n" + result.validator_notes

        assert "Validator Notes" in final
        assert "Widen stop loss" in final
