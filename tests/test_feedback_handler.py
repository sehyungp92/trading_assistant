# tests/test_feedback_handler.py
"""Tests for the human feedback handler."""
import json
from pathlib import Path

import pytest

from schemas.corrections import HumanCorrection, CorrectionType
from analysis.feedback_handler import FeedbackHandler


class TestFeedbackHandler:
    def test_parse_trade_reclassify(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse(
            "Trade #xyz wasn't bad — it was a planned hedge against spot"
        )
        assert correction.correction_type == CorrectionType.TRADE_RECLASSIFY
        assert correction.target_id == "xyz"
        assert correction.raw_text == "Trade #xyz wasn't bad — it was a planned hedge against spot"

    def test_parse_regime_override(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse(
            "Bot2's regime classification was wrong today, it was a slow trend not ranging"
        )
        assert correction.correction_type == CorrectionType.REGIME_OVERRIDE
        assert "bot2" in correction.target_id.lower()

    def test_parse_positive_reinforcement(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse(
            "Good catch on the volume filter, that's the third time this week"
        )
        assert correction.correction_type == CorrectionType.POSITIVE_REINFORCEMENT

    def test_parse_free_text(self):
        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse("Interesting day overall, keep monitoring ETH")
        assert correction.correction_type == CorrectionType.FREE_TEXT

    def test_write_to_jsonl(self, tmp_path: Path):
        corrections_path = tmp_path / "corrections.jsonl"
        handler = FeedbackHandler(report_id="daily-2026-03-01")

        correction = handler.parse("Trade #abc was actually fine")
        handler.write_correction(correction, corrections_path)

        lines = corrections_path.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["correction_type"] == "trade_reclassify"

    def test_appends_to_existing_file(self, tmp_path: Path):
        corrections_path = tmp_path / "corrections.jsonl"
        corrections_path.write_text('{"existing": true}\n')

        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse("Good catch on volume filter")
        handler.write_correction(correction, corrections_path)

        lines = corrections_path.read_text().strip().splitlines()
        assert len(lines) == 2
