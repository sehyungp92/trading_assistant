"""Tests for enriched data in the analysis pipeline — prompt assembler + microstructure detector."""
import json
from pathlib import Path

import pytest

from analysis.prompt_assembler import DailyPromptAssembler, _CURATED_FILES
from analysis.strategy_engine import StrategyEngine
from schemas.strategy_suggestions import SuggestionTier


# ---------------------------------------------------------------------------
# 1. Daily assembler includes new enriched files
# ---------------------------------------------------------------------------

class TestDailyAssemblerEnrichedFiles:
    def test_curated_files_includes_filter_decisions(self):
        assert "filter_decisions.json" in _CURATED_FILES

    def test_curated_files_includes_indicator_snapshots(self):
        assert "indicator_snapshots.json" in _CURATED_FILES

    def test_curated_files_includes_orderbook_context(self):
        assert "orderbook_context.json" in _CURATED_FILES

    def test_assembler_handles_missing_enriched_files_gracefully(self, tmp_path):
        """Assembler should not error when enriched files are absent."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir()
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test")
        (memory_dir / "policies" / "v1" / "agents.md").write_text("test")
        (memory_dir / "findings").mkdir()

        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)

        # Write only the required summary.json — enriched files are absent
        (bot_dir / "summary.json").write_text(json.dumps({"total_pnl": 100}))

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )

        # Should not raise
        pkg = assembler.assemble()
        assert pkg is not None
        # Enriched keys should not be present in bot data
        bot_data = pkg.data.get("bot1", {})
        assert "filter_decisions" not in bot_data
        assert "indicator_snapshots" not in bot_data
        assert "orderbook_context" not in bot_data

    def test_assembler_loads_enriched_files_when_present(self, tmp_path):
        """When enriched files exist, they are loaded into the prompt package data."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "policies").mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir()
        (memory_dir / "policies" / "v1" / "soul.md").write_text("test")
        (memory_dir / "policies" / "v1" / "trading_rules.md").write_text("test")
        (memory_dir / "policies" / "v1" / "agents.md").write_text("test")
        (memory_dir / "findings").mkdir()

        curated_dir = tmp_path / "curated"
        bot_dir = curated_dir / "2026-03-01" / "bot1"
        bot_dir.mkdir(parents=True)

        (bot_dir / "summary.json").write_text(json.dumps({"total_pnl": 100}))
        (bot_dir / "filter_decisions.json").write_text(json.dumps({"vol_filter": {"pass_rate": 0.8}}))
        (bot_dir / "indicator_snapshots.json").write_text(json.dumps({"decisions": {"enter": 5}}))
        (bot_dir / "orderbook_context.json").write_text(json.dumps({"event_count": 20, "avg_spread_bps": 5.0}))

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot1"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )

        pkg = assembler.assemble()
        bot_data = pkg.data.get("bot1", {})
        assert "filter_decisions" in bot_data
        assert "indicator_snapshots" in bot_data
        assert "orderbook_context" in bot_data
        assert bot_data["orderbook_context"]["event_count"] == 20


# ---------------------------------------------------------------------------
# 2. Microstructure detector tests
# ---------------------------------------------------------------------------

class TestDetectMicrostructureIssues:
    def setup_method(self):
        self.engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")

    def test_high_spread_produces_suggestion(self):
        stats = {"event_count": 50, "avg_spread_bps": 25.0, "max_spread_bps": 40.0, "avg_imbalance_ratio": 1.0}
        suggestions = self.engine.detect_microstructure_issues("bot1", stats)
        assert len(suggestions) == 1
        assert "spread" in suggestions[0].title.lower()
        assert suggestions[0].tier == SuggestionTier.FILTER

    def test_low_imbalance_produces_suggestion(self):
        stats = {"event_count": 20, "avg_spread_bps": 5.0, "max_spread_bps": 10.0, "avg_imbalance_ratio": 0.3}
        suggestions = self.engine.detect_microstructure_issues("bot1", stats)
        assert len(suggestions) == 1
        assert "imbalance" in suggestions[0].title.lower()

    def test_normal_stats_no_suggestion(self):
        stats = {"event_count": 50, "avg_spread_bps": 10.0, "max_spread_bps": 15.0, "avg_imbalance_ratio": 1.2}
        suggestions = self.engine.detect_microstructure_issues("bot1", stats)
        assert len(suggestions) == 0

    def test_insufficient_events_no_suggestion(self):
        stats = {"event_count": 3, "avg_spread_bps": 50.0, "max_spread_bps": 80.0, "avg_imbalance_ratio": 0.1}
        suggestions = self.engine.detect_microstructure_issues("bot1", stats)
        assert len(suggestions) == 0

    def test_both_issues_produce_two_suggestions(self):
        stats = {"event_count": 30, "avg_spread_bps": 25.0, "max_spread_bps": 40.0, "avg_imbalance_ratio": 0.3}
        suggestions = self.engine.detect_microstructure_issues("bot1", stats)
        assert len(suggestions) == 2
        titles = {s.title for s in suggestions}
        assert any("spread" in t.lower() for t in titles)
        assert any("imbalance" in t.lower() for t in titles)

    def test_suggestions_have_detection_context(self):
        stats = {"event_count": 50, "avg_spread_bps": 25.0, "max_spread_bps": 40.0, "avg_imbalance_ratio": 0.3}
        suggestions = self.engine.detect_microstructure_issues("bot1", stats)
        for s in suggestions:
            assert s.detection_context is not None
            assert s.detection_context.detector_name == "microstructure"
            assert s.detection_context.bot_id == "bot1"

    def test_build_report_wires_orderbook_stats(self):
        """build_report should call detect_microstructure_issues when orderbook_stats provided."""
        from schemas.weekly_metrics import BotWeeklySummary

        summary = BotWeeklySummary(
            bot_id="bot1", total_pnl=100.0, trade_count=10,
            win_rate=0.6, avg_win=50.0, avg_loss=-30.0,
            max_drawdown_pct=5.0, sharpe_ratio=1.5,
            week_start="2026-02-23", week_end="2026-03-01",
        )
        ob_stats = {
            "bot1": {
                "event_count": 50,
                "avg_spread_bps": 30.0,
                "max_spread_bps": 50.0,
                "avg_imbalance_ratio": 1.0,
            }
        }
        report = self.engine.build_report(
            bot_summaries={"bot1": summary},
            orderbook_stats=ob_stats,
        )
        micro_suggestions = [s for s in report.suggestions if "spread" in s.title.lower()]
        assert len(micro_suggestions) >= 1
