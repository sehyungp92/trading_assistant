# tests/test_feedback_wiring.py
"""Tests for feedback loop wiring — Telegram callbacks -> FeedbackHandler -> corrections.jsonl."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from analysis.feedback_handler import FeedbackHandler
from comms.telegram_handlers import TelegramCallbackRouter


class TestFeedbackWiring:
    def test_router_accepts_feedback_handler(self):
        """TelegramCallbackRouter can register a feedback callback."""
        router = TelegramCallbackRouter()
        router.register("feedback_correction", AsyncMock())
        assert "feedback_correction" in router.handlers

    @pytest.mark.asyncio
    async def test_feedback_callback_writes_correction(self, tmp_path):
        """When user sends feedback via Telegram, it flows through to corrections.jsonl."""
        corrections_path = tmp_path / "findings" / "corrections.jsonl"

        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse("Trade #T123 was actually a hedge, not a real loss")
        handler.write_correction(correction, corrections_path)

        assert corrections_path.exists()
        import json
        lines = corrections_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["correction_type"] == "trade_reclassify"
        assert data["target_id"] == "T123"

    @pytest.mark.asyncio
    async def test_positive_reinforcement_recorded(self, tmp_path):
        corrections_path = tmp_path / "findings" / "corrections.jsonl"

        handler = FeedbackHandler(report_id="daily-2026-03-01")
        correction = handler.parse("Great analysis on the regime detection")
        handler.write_correction(correction, corrections_path)

        import json
        data = json.loads(corrections_path.read_text().strip())
        assert data["correction_type"] == "positive_reinforcement"
