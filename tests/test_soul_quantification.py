"""Tests for soul.md quantification reinforcement in prompts (C1)."""
from __future__ import annotations

from pathlib import Path

from analysis.prompt_assembler import DailyPromptAssembler, _INSTRUCTIONS
from analysis.weekly_prompt_assembler import WeeklyPromptAssembler, _WEEKLY_INSTRUCTIONS


def test_daily_prompt_includes_quantification_instruction():
    assert "QUANTIFICATION REQUIRED" in _INSTRUCTIONS
    assert "Expected return impact" in _INSTRUCTIONS
    assert "Drawdown impact" in _INSTRUCTIONS
    assert "Evidence base" in _INSTRUCTIONS


def test_weekly_prompt_includes_calmar_requirement():
    assert "QUANTIFICATION REQUIRED" in _WEEKLY_INSTRUCTIONS
    assert "Calmar ratio impact" in _WEEKLY_INSTRUCTIONS


def test_instruction_text_matches_soul_md_criteria():
    # soul.md says: max 3 actionable items, each specific/testable/quantified
    assert "max 3" in _INSTRUCTIONS
    assert "statistical significance" in _INSTRUCTIONS
    assert "trade count" in _INSTRUCTIONS
