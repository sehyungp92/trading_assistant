# tests/test_response_parser.py
"""Tests for the structured response parser."""
from __future__ import annotations

import json

import pytest

from analysis.response_parser import parse_response
from schemas.agent_response import ParsedAnalysis


class TestParseResponse:
    """Tests for parse_response()."""

    def test_valid_structured_block(self):
        response = """# Daily Report
Some analysis text here.

<!-- STRUCTURED_OUTPUT
{
  "predictions": [
    {"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.8, "timeframe_days": 7, "reasoning": "Strong signal alignment"}
  ],
  "suggestions": [
    {"suggestion_id": "#abc123", "bot_id": "bot1", "category": "exit_timing", "title": "Widen stop loss", "expected_impact": "+0.3% daily PnL", "confidence": 0.7, "evidence_summary": "MFE > 2x realized PnL"}
  ],
  "structural_proposals": [
    {"hypothesis_id": "h-exit-trailing", "bot_id": "bot1", "title": "Switch to trailing stop", "description": "Fixed stops cause premature exits", "reversibility": "easy", "evidence": "exit_efficiency < 50%", "estimated_complexity": "medium"}
  ]
}
-->
"""
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 1
        assert parsed.predictions[0].bot_id == "bot1"
        assert parsed.predictions[0].metric == "pnl"
        assert parsed.predictions[0].direction == "improve"
        assert parsed.predictions[0].confidence == 0.8
        assert len(parsed.suggestions) == 1
        assert parsed.suggestions[0].title == "Widen stop loss"
        assert len(parsed.structural_proposals) == 1
        assert parsed.structural_proposals[0].hypothesis_id == "h-exit-trailing"
        assert parsed.raw_report == response

    def test_missing_block(self):
        response = "# Daily Report\nJust some text, no structured block."
        parsed = parse_response(response)
        assert parsed.parse_success is False
        assert parsed.predictions == []
        assert parsed.suggestions == []
        assert parsed.structural_proposals == []
        assert parsed.raw_report == response

    def test_malformed_json(self):
        response = """# Report
<!-- STRUCTURED_OUTPUT
{not valid json}
-->
"""
        parsed = parse_response(response)
        assert parsed.parse_success is False
        assert parsed.raw_report == response

    def test_extra_whitespace(self):
        response = """# Report

<!--   STRUCTURED_OUTPUT
  {
    "predictions": [],
    "suggestions": [],
    "structural_proposals": []
  }
  -->
"""
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert parsed.predictions == []
        assert parsed.suggestions == []

    def test_partial_fields(self):
        response = """# Report
<!-- STRUCTURED_OUTPUT
{
  "predictions": [
    {"bot_id": "bot1", "metric": "win_rate", "direction": "stable", "confidence": 0.5}
  ]
}
-->
"""
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 1
        assert parsed.predictions[0].timeframe_days == 7  # default
        assert parsed.predictions[0].reasoning == ""  # default
        assert parsed.suggestions == []  # missing = empty

    def test_empty_response(self):
        parsed = parse_response("")
        assert parsed.parse_success is False
        assert parsed.raw_report == ""

    def test_block_with_surrounding_markdown(self):
        response = """# Report Title
## Bot1 Analysis
Great performance today. Win rate improved.

## Suggestions
1. Widen stop loss on bot1

<!-- STRUCTURED_OUTPUT
{
  "predictions": [{"bot_id": "bot1", "metric": "sharpe", "direction": "improve", "confidence": 0.6}],
  "suggestions": [{"bot_id": "bot1", "title": "Widen stop", "category": "stop_loss", "confidence": 0.7}]
}
-->

## Footer
More text after the block.
"""
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 1
        assert len(parsed.suggestions) == 1
        assert "Footer" in parsed.raw_report

    def test_multiple_predictions_and_suggestions(self):
        data = {
            "predictions": [
                {"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.8},
                {"bot_id": "bot2", "metric": "drawdown", "direction": "decline", "confidence": 0.6},
            ],
            "suggestions": [
                {"bot_id": "bot1", "title": "Adjust filter", "category": "filter_threshold", "confidence": 0.5},
                {"bot_id": "bot2", "title": "Change exit", "category": "exit_timing", "confidence": 0.9},
            ],
        }
        response = f"# Report\n<!-- STRUCTURED_OUTPUT\n{json.dumps(data)}\n-->"
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 2
        assert len(parsed.suggestions) == 2


class TestJsonFenceFallback:
    """Multi-LLM parity: GPT/GLM models may emit ```json fences instead of
    HTML comment blocks.  The parser must accept both."""

    def test_json_fence_parsed(self):
        """A response with ```json fence (no comment block) should parse."""
        data = {
            "predictions": [
                {"bot_id": "bot1", "metric": "pnl", "direction": "improve", "confidence": 0.7}
            ],
            "suggestions": [
                {"bot_id": "bot1", "title": "Tighten filter", "category": "filter_threshold", "confidence": 0.6}
            ],
        }
        response = f"# Report\nGreat analysis.\n\n```json\n{json.dumps(data, indent=2)}\n```\n"
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.predictions) == 1
        assert parsed.predictions[0].bot_id == "bot1"
        assert len(parsed.suggestions) == 1
        assert parsed.suggestions[0].title == "Tighten filter"

    def test_comment_block_takes_priority_over_fence(self):
        """When both formats are present, comment block wins."""
        comment_data = {
            "predictions": [
                {"bot_id": "from_comment", "metric": "pnl", "direction": "improve", "confidence": 0.8}
            ],
        }
        fence_data = {
            "predictions": [
                {"bot_id": "from_fence", "metric": "pnl", "direction": "decline", "confidence": 0.3}
            ],
        }
        response = (
            f"# Report\n"
            f"```json\n{json.dumps(fence_data)}\n```\n\n"
            f"<!-- STRUCTURED_OUTPUT\n{json.dumps(comment_data)}\n-->\n"
        )
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert parsed.predictions[0].bot_id == "from_comment"

    def test_malformed_fence_falls_through(self):
        """Malformed ```json fence with no comment block → parse_success=False."""
        response = "# Report\n```json\n{not valid}\n```\n"
        parsed = parse_response(response)
        assert parsed.parse_success is False

    def test_fence_with_structural_proposals(self):
        """Verify full schema parsing works via fence fallback."""
        data = {
            "predictions": [],
            "suggestions": [],
            "structural_proposals": [
                {
                    "hypothesis_id": "h-trailing-stop",
                    "bot_id": "bot2",
                    "title": "Trailing stop",
                    "description": "Switch from fixed to trailing",
                    "reversibility": "easy",
                    "evidence": "exit_efficiency < 40%",
                }
            ],
        }
        response = f"# Weekly\n```json\n{json.dumps(data, indent=2)}\n```\n"
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert len(parsed.structural_proposals) == 1
        assert parsed.structural_proposals[0].hypothesis_id == "h-trailing-stop"

    def test_malformed_comment_falls_back_to_fence(self):
        """If comment block has bad JSON but fence is valid, use the fence."""
        fence_data = {
            "predictions": [
                {"bot_id": "bot1", "metric": "win_rate", "direction": "stable", "confidence": 0.5}
            ],
        }
        response = (
            f"# Report\n"
            f"<!-- STRUCTURED_OUTPUT\n{{broken json}}\n-->\n\n"
            f"```json\n{json.dumps(fence_data)}\n```\n"
        )
        parsed = parse_response(response)
        assert parsed.parse_success is True
        assert parsed.predictions[0].bot_id == "bot1"
