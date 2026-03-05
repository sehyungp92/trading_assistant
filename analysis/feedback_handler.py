# analysis/feedback_handler.py
"""Feedback handler — parses human reply text into structured corrections.

Human replies on Telegram/Discord are parsed into HumanCorrection objects and
appended to memory/findings/corrections.jsonl. Future analysis prompts include
recent corrections as context (Ralph Loop V2).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from schemas.corrections import HumanCorrection, CorrectionType


class FeedbackHandler:
    """Parses human feedback text and writes structured corrections."""

    def __init__(self, report_id: str) -> None:
        self.report_id = report_id

    def parse(self, text: str) -> HumanCorrection:
        """Classify and parse a human feedback message."""
        # Pattern: Trade #<id> ...
        trade_match = re.search(r"[Tt]rade\s*#(\w+)", text)
        if trade_match:
            return HumanCorrection(
                correction_type=CorrectionType.TRADE_RECLASSIFY,
                original_report_id=self.report_id,
                target_id=trade_match.group(1),
                raw_text=text,
            )

        # Pattern: Bot<N>'s regime ... wrong
        regime_match = re.search(r"(bot\w+).*regime.*wrong", text, re.IGNORECASE)
        if regime_match:
            return HumanCorrection(
                correction_type=CorrectionType.REGIME_OVERRIDE,
                original_report_id=self.report_id,
                target_id=regime_match.group(1),
                raw_text=text,
            )

        # Pattern: Good catch, nice, well done, etc.
        positive_patterns = [
            r"good\s+catch",
            r"nice\s+catch",
            r"well\s+done",
            r"great\s+(analysis|catch|job)",
            r"spot\s+on",
        ]
        for pattern in positive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return HumanCorrection(
                    correction_type=CorrectionType.POSITIVE_REINFORCEMENT,
                    original_report_id=self.report_id,
                    raw_text=text,
                )

        # Default: free text
        return HumanCorrection(
            correction_type=CorrectionType.FREE_TEXT,
            original_report_id=self.report_id,
            raw_text=text,
        )

    def write_correction(self, correction: HumanCorrection, path: Path) -> None:
        """Append a correction to the JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(correction.model_dump(mode="json"), default=str) + "\n")
