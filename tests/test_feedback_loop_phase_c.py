# tests/test_feedback_loop_phase_c.py
"""Tests for Phase C: Structural Improvement Framework.

Covers: prescriptive consolidation, hypothesis library, cross-bot transfer proposals.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from schemas.memory import ConsolidationSummary, PatternCount


# --- C0: Prescriptive consolidation ---


class TestPrescriptiveConsolidation:
    def test_concentration_insight(self, tmp_path):
        from orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_bots=[PatternCount(category="bot", key="bot1", count=50)],
        )
        insights = consolidator._generate_insights(summary)
        assert any("Concentration alert" in i for i in insights)
        assert any("bot1" in i for i in insights)

    def test_recurring_issue_insight(self, tmp_path):
        from orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_root_causes=[PatternCount(category="root_cause", key="regime_mismatch", count=15)],
        )
        insights = consolidator._generate_insights(summary)
        assert any("Systemic pattern" in i for i in insights)
        assert any("regime_mismatch" in i for i in insights)

    def test_error_pattern_insight(self, tmp_path):
        from orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_error_types=[PatternCount(category="error_type", key="IndexError", count=8)],
        )
        insights = consolidator._generate_insights(summary)
        assert any("Blind spot" in i for i in insights)

    def test_no_insights_on_sparse_data(self, tmp_path):
        from orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path)
        summary = ConsolidationSummary(
            source_file="corrections.jsonl",
            total_entries=100,
            top_bots=[PatternCount(category="bot", key="bot1", count=10)],
            top_root_causes=[PatternCount(category="root_cause", key="normal_loss", count=3)],
            top_error_types=[PatternCount(category="error_type", key="ValueError", count=2)],
        )
        insights = consolidator._generate_insights(summary)
        assert len(insights) == 0

    def test_markdown_includes_insights_section(self, tmp_path):
        from orchestrator.memory_consolidator import MemoryConsolidator

        consolidator = MemoryConsolidator(findings_dir=tmp_path, threshold=1)

        # Write enough entries to trigger consolidation
        path = tmp_path / "corrections.jsonl"
        entries = []
        for i in range(5):
            entries.append(json.dumps({
                "bot_id": "bot1", "root_cause": "regime_mismatch",
                "correction_type": "trade_reclassify",
            }))
        # Add more for high root cause count
        for i in range(10):
            entries.append(json.dumps({
                "bot_id": "bot1", "root_cause": "regime_mismatch",
            }))
        path.write_text("\n".join(entries), encoding="utf-8")

        consolidator.consolidate("corrections.jsonl")

        md_path = tmp_path / "patterns_consolidated.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Actionable Insights" in content


# --- C1: Hypothesis library ---


class TestHypothesisLibrary:
    def test_get_all_returns_hypotheses(self):
        from skills.hypothesis_library import get_all

        all_h = get_all()
        assert len(all_h) >= 10

    def test_get_by_category_signal_decay(self):
        from skills.hypothesis_library import get_by_category

        results = get_by_category("signal_decay")
        assert len(results) >= 1
        assert all(h.category == "signal_decay" for h in results)

    def test_get_relevant_with_suggestions(self):
        from skills.hypothesis_library import get_relevant

        suggestions = [
            MagicMock(title="Signal decay on bot1", description="Win rate declining"),
            MagicMock(title="Filter over-blocking", description="Too many missed trades"),
        ]
        results = get_relevant(suggestions)
        categories = {h.category for h in results}
        assert "signal_decay" in categories
        assert "filter_over_blocking" in categories

    def test_get_relevant_empty_on_unknown(self):
        from skills.hypothesis_library import get_relevant

        suggestions = [MagicMock(title="Unknown thing", description="Something unrelated")]
        results = get_relevant(suggestions)
        assert len(results) == 0

    def test_no_duplicate_hypotheses(self):
        from skills.hypothesis_library import get_relevant

        # Multiple suggestions pointing to same category
        suggestions = [
            MagicMock(title="Signal decay A", description=""),
            MagicMock(title="Signal alpha B", description=""),
        ]
        results = get_relevant(suggestions)
        ids = [h.id for h in results]
        assert len(ids) == len(set(ids))

    def test_weekly_instructions_reference_hypotheses(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "STRUCTURAL HYPOTHESES" in _WEEKLY_INSTRUCTIONS
        assert "structural_hypotheses" in _WEEKLY_INSTRUCTIONS


# --- C2: Transfer proposals ---


class TestTransferProposals:
    def _make_pattern_library(self, tmp_path, patterns):
        from skills.pattern_library import PatternLibrary
        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        for p in patterns:
            lib.add(p)
        return lib

    def test_proposed_patterns_included_in_proposals(self, tmp_path):
        """PROPOSED patterns are now included in transfer proposals (not just VALIDATED)."""
        from skills.transfer_proposal_builder import TransferProposalBuilder
        from skills.pattern_library import PatternLibrary
        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.PROPOSED, source_bot="bot1",
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2"])
        proposals = builder.build_proposals()
        assert len(proposals) == 1
        assert proposals[0].target_bot == "bot2"

    def test_source_excluded_from_proposals(self, tmp_path):
        from skills.transfer_proposal_builder import TransferProposalBuilder
        from skills.pattern_library import PatternLibrary
        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED, source_bot="bot1",
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2"])
        proposals = builder.build_proposals()
        target_bots = [p.target_bot for p in proposals]
        assert "bot1" not in target_bots  # source bot excluded

    def test_already_targeted_excluded(self, tmp_path):
        from skills.transfer_proposal_builder import TransferProposalBuilder
        from skills.pattern_library import PatternLibrary
        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED, source_bot="bot1",
            target_bots=["bot2"],  # already targeted
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2", "bot3"])
        proposals = builder.build_proposals()
        target_bots = [p.target_bot for p in proposals]
        assert "bot2" not in target_bots
        assert "bot3" in target_bots

    def test_sorted_by_score(self, tmp_path):
        from skills.transfer_proposal_builder import TransferProposalBuilder
        from skills.pattern_library import PatternLibrary
        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test A", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED, source_bot="bot1",
        ))

        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2", "bot3"])
        proposals = builder.build_proposals()
        if len(proposals) >= 2:
            assert proposals[0].compatibility_score >= proposals[1].compatibility_score

    def test_weekly_instructions_reference_transfer(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "CROSS-BOT TRANSFER" in _WEEKLY_INSTRUCTIONS
        assert "transfer_proposals" in _WEEKLY_INSTRUCTIONS

    def test_eligible_bots_only(self, tmp_path):
        from skills.transfer_proposal_builder import TransferProposalBuilder
        from skills.pattern_library import PatternLibrary
        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        lib = PatternLibrary(tmp_path)
        lib.add(PatternEntry(
            pattern_id="p1", title="Test", category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.IMPLEMENTED, source_bot="bot1",
        ))

        # Only bot2 and bot3 are eligible
        builder = TransferProposalBuilder(lib, tmp_path / "curated", ["bot1", "bot2", "bot3"])
        proposals = builder.build_proposals()
        assert len(proposals) == 2
        assert {p.target_bot for p in proposals} == {"bot2", "bot3"}
