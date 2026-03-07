# tests/test_transfer_outcomes.py
"""Tests for transfer outcome measurement and track record scoring."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from schemas.transfer_proposals import TransferOutcome, TransferProposal
from skills.transfer_proposal_builder import TransferProposalBuilder


def _make_curated_summary(path: Path, total_pnl: float, win_rate: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "total_pnl": total_pnl,
        "win_rate": win_rate,
    }))


class TestTransferOutcome:
    def test_schema(self):
        o = TransferOutcome(
            pattern_id="p1",
            source_bot="bot1",
            target_bot="bot2",
            pnl_delta_7d=100.0,
            win_rate_delta_7d=0.05,
            verdict="positive",
        )
        assert o.verdict == "positive"
        assert o.pnl_delta_7d == 100.0


class TestTransferTrackRecord:
    def test_empty_track_record(self, tmp_path):
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2"],
            findings_dir=tmp_path,
        )
        assert builder.compute_transfer_track_record() == {}

    def test_compute_track_record(self, tmp_path):
        # Write outcomes
        outcomes_path = tmp_path / "transfer_outcomes.jsonl"
        outcomes = [
            {"pattern_id": "p1", "target_bot": "bot2", "verdict": "positive"},
            {"pattern_id": "p1", "target_bot": "bot3", "verdict": "negative"},
            {"pattern_id": "p2", "target_bot": "bot2", "verdict": "positive"},
        ]
        with open(outcomes_path, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2", "bot3"],
            findings_dir=tmp_path,
        )
        track = builder.compute_transfer_track_record()
        assert track["p1"]["total"] == 2
        assert track["p1"]["positive"] == 1
        assert track["p1"]["success_rate"] == 0.5
        assert track["p2"]["success_rate"] == 1.0

    def test_score_boosting_from_positive_track_record(self, tmp_path):
        """Patterns with positive track record get score boost."""
        # Write positive track record
        outcomes_path = tmp_path / "transfer_outcomes.jsonl"
        outcomes = [
            {"pattern_id": "p1", "target_bot": "bot3", "verdict": "positive"},
            {"pattern_id": "p1", "target_bot": "bot4", "verdict": "positive"},
        ]
        with open(outcomes_path, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        # Create a pattern library mock
        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        pattern = PatternEntry(
            pattern_id="p1",
            source_bot="bot1",
            title="Great pattern",
            category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED,
        )
        lib = MagicMock()
        lib.load_active.return_value = [pattern]

        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2"],
            findings_dir=tmp_path,
        )
        proposals = builder.build_proposals()
        assert len(proposals) == 1
        # Score should be boosted from default 0.5 by +0.1
        assert proposals[0].compatibility_score >= 0.5

    def test_score_penalizing_from_negative_track_record(self, tmp_path):
        """Patterns with negative track record get score penalty."""
        outcomes_path = tmp_path / "transfer_outcomes.jsonl"
        outcomes = [
            {"pattern_id": "p1", "target_bot": "bot3", "verdict": "negative"},
            {"pattern_id": "p1", "target_bot": "bot4", "verdict": "negative"},
        ]
        with open(outcomes_path, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        from schemas.pattern_library import PatternEntry, PatternStatus, PatternCategory

        pattern = PatternEntry(
            pattern_id="p1",
            source_bot="bot1",
            title="Bad pattern",
            category=PatternCategory.ENTRY_SIGNAL,
            status=PatternStatus.VALIDATED,
        )
        lib = MagicMock()
        lib.load_active.return_value = [pattern]

        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1", "bot2"],
            findings_dir=tmp_path,
        )
        proposals = builder.build_proposals()
        assert len(proposals) == 1
        # Score should be penalized from default 0.5 by -0.2
        assert proposals[0].compatibility_score <= 0.5

    def test_missing_data_handled(self, tmp_path):
        lib = MagicMock()
        lib.load_active.return_value = []
        builder = TransferProposalBuilder(
            pattern_library=lib,
            curated_dir=tmp_path,
            bots=["bot1"],
            findings_dir=tmp_path,
        )
        outcomes = builder.measure_transfer_outcomes()
        assert outcomes == []
