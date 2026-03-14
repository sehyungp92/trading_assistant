"""Tests for generic context builder (H3)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.context_builder import ContextBuilder
from schemas.prompt_package import PromptPackage


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    """Create a minimal memory directory with policies and corrections."""
    policy_dir = tmp_path / "policies" / "v1"
    policy_dir.mkdir(parents=True)
    (policy_dir / "agents.md").write_text("You are a trading assistant.")
    (policy_dir / "trading_rules.md").write_text("Never risk more than 2%.")
    (policy_dir / "soul.md").write_text("Be cautious and precise.")

    findings_dir = tmp_path / "findings"
    findings_dir.mkdir()
    corrections = [
        {"date": "2026-02-28", "correction": "Bot-A filter was too aggressive"},
        {"date": "2026-03-01", "correction": "Risk limit should be 1.5% not 2%"},
    ]
    lines = [json.dumps(c) for c in corrections]
    (findings_dir / "corrections.jsonl").write_text("\n".join(lines))

    return tmp_path


class TestContextBuilder:
    def test_build_system_prompt_loads_all_policies(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        prompt = ctx.build_system_prompt()
        assert "--- agents.md ---" in prompt
        assert "trading assistant" in prompt
        assert "--- trading_rules.md ---" in prompt
        assert "--- soul.md ---" in prompt

    def test_build_system_prompt_skips_missing_files(self, tmp_path: Path):
        policy_dir = tmp_path / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agents.md").write_text("Agent only.")
        ctx = ContextBuilder(tmp_path)
        prompt = ctx.build_system_prompt()
        assert "--- agents.md ---" in prompt
        assert "trading_rules.md" not in prompt

    def test_load_corrections(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        corrections = ctx.load_corrections()
        assert len(corrections) == 2
        # Temporal decay sorts by recency (most recent first)
        assert corrections[0]["date"] == "2026-03-01"

    def test_load_corrections_returns_empty_when_missing(self, tmp_path: Path):
        ctx = ContextBuilder(tmp_path)
        corrections = ctx.load_corrections()
        assert corrections == []

    def test_list_policy_files(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        files = ctx.list_policy_files()
        assert len(files) == 3
        assert any("agents.md" in f for f in files)

    def test_runtime_metadata(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        meta = ctx.runtime_metadata()
        assert "assembled_at" in meta
        assert meta["timezone"] == "UTC"

    def test_base_package_returns_prompt_package(self, memory_dir: Path):
        ctx = ContextBuilder(memory_dir)
        pkg = ctx.base_package()
        assert isinstance(pkg, PromptPackage)
        assert "trading assistant" in pkg.system_prompt
        assert len(pkg.corrections) == 2
        assert len(pkg.context_files) == 3
        assert pkg.metadata["timezone"] == "UTC"

    def test_assemblers_use_context_builder(self, memory_dir: Path, tmp_path: Path):
        """Integration test: DailyPromptAssembler uses ContextBuilder."""
        from analysis.prompt_assembler import DailyPromptAssembler
        curated_dir = tmp_path / "curated"
        curated_dir.mkdir()

        assembler = DailyPromptAssembler(
            date="2026-03-01",
            bots=["bot-a"],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert isinstance(pkg, PromptPackage)
        assert "trading assistant" in pkg.system_prompt
        assert len(pkg.corrections) == 2
        assert "bot-a" in pkg.task_prompt


class TestLoadFailureLog:
    def test_loads_failure_log_entries(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        log_path = findings / "failure-log.jsonl"
        log_path.write_text(
            '{"error_type":"timeout","bot_id":"bot1","outcome":"known_fix"}\n'
            '{"error_type":"api_error","bot_id":"bot2","outcome":"needs_human"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        entries = ctx.load_failure_log()
        assert len(entries) == 2
        assert entries[0]["error_type"] == "timeout"

    def test_missing_failure_log_returns_empty(self, tmp_path):
        ctx = ContextBuilder(memory_dir=tmp_path)
        assert ctx.load_failure_log() == []


class TestLoadRejectedSuggestions:
    def test_loads_rejected_suggestions(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        suggestions_path = findings / "suggestions.jsonl"
        suggestions_path.write_text(
            '{"suggestion_id":"s001","bot_id":"bot1","title":"Widen stop","status":"rejected","rejection_reason":"No evidence"}\n'
            '{"suggestion_id":"s002","bot_id":"bot1","title":"Remove filter","status":"deployed"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        rejected = ctx.load_rejected_suggestions()
        assert len(rejected) == 1
        assert rejected[0]["title"] == "Widen stop"

    def test_missing_suggestions_file_returns_empty(self, tmp_path):
        ctx = ContextBuilder(memory_dir=tmp_path)
        assert ctx.load_rejected_suggestions() == []


class TestBasePackageWithFailureLog:
    def test_base_package_includes_failure_log(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "failure-log.jsonl").write_text(
            '{"error_type":"timeout","outcome":"known_fix"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "failure_log" in pkg.data
        assert len(pkg.data["failure_log"]) == 1

    def test_base_package_includes_rejected_suggestions(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "suggestions.jsonl").write_text(
            '{"suggestion_id":"s001","status":"rejected","rejection_reason":"x"}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "rejected_suggestions" in pkg.data
        assert len(pkg.data["rejected_suggestions"]) == 1


class TestLoadAllocationHistory:
    def test_loads_allocation_records(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "allocation_history.jsonl").write_text(
            '{"date":"2026-03-01","bot_id":"bot-a","allocation_pct":25.0}\n'
            '{"date":"2026-03-02","bot_id":"bot-a","allocation_pct":30.0}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        records = ctx.load_allocation_history()
        assert len(records) == 2
        # Temporal decay: most recent first
        assert records[0]["date"] == "2026-03-02"

    def test_missing_file_returns_empty(self, tmp_path):
        ctx = ContextBuilder(memory_dir=tmp_path)
        assert ctx.load_allocation_history() == []

    def test_temporal_decay_excludes_old_records(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "allocation_history.jsonl").write_text(
            '{"date":"2020-01-01","bot_id":"bot-a","allocation_pct":10.0}\n'
            '{"date":"2026-03-01","bot_id":"bot-a","allocation_pct":25.0}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        records = ctx.load_allocation_history()
        assert len(records) == 1
        assert records[0]["date"] == "2026-03-01"

    def test_base_package_includes_allocation_history(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "allocation_history.jsonl").write_text(
            '{"date":"2026-03-01","bot_id":"bot-a","allocation_pct":25.0}\n'
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "allocation_history" in pkg.data
        assert len(pkg.data["allocation_history"]) == 1

    def test_base_package_excludes_empty_allocation_history(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "allocation_history" not in pkg.data


class TestLoadSearchReports:
    def test_loads_recent_search_reports(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        reports = [
            {"suggestion_id": f"s{i}", "bot_id": "bot1", "param_name": "p1",
             "routing": "approve", "best_value": 0.7, "discard_reason": "",
             "exploration_summary": "ok", "searched_at": "2026-03-01T00:00:00Z"}
            for i in range(8)
        ]
        with (findings / "search_reports.jsonl").open("w") as f:
            for r in reports:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_search_reports(lookback_n=5)
        assert len(result) == 5  # Only last 5

    def test_filters_by_bot_id(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        reports = [
            {"bot_id": "bot1", "param_name": "p1", "routing": "approve"},
            {"bot_id": "bot2", "param_name": "p2", "routing": "discard"},
        ]
        with (findings / "search_reports.jsonl").open("w") as f:
            for r in reports:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_search_reports(bot_id="bot1")
        assert len(result) == 1
        assert result[0]["param_name"] == "p1"

    def test_base_package_includes_search_reports(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "search_reports.jsonl").write_text(
            json.dumps({"bot_id": "bot1", "param_name": "p1", "routing": "approve"}) + "\n"
        )
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "search_reports" in pkg.data
        assert len(pkg.data["search_reports"]) == 1


class TestLoadBacktestReliability:
    def test_loads_per_category_reliability(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        records = []
        for i in range(5):
            records.append({
                "suggestion_id": f"s{i}", "bot_id": "bot1",
                "param_category": "signal", "predicted_improvement": 1.1,
                "predicted_routing": "approve",
                "prediction_correct": i < 4,  # 4/5 = 0.80
            })
        with (findings / "backtest_calibration.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_backtest_reliability()
        assert "signal" in result
        assert result["signal"] == 0.8

    def test_excludes_categories_below_min_samples(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        records = [
            {"bot_id": "bot1", "param_category": "exit",
             "prediction_correct": True},
            {"bot_id": "bot1", "param_category": "exit",
             "prediction_correct": False},
        ]
        with (findings / "backtest_calibration.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        result = ctx.load_backtest_reliability()
        assert "exit" not in result  # Only 2 samples, need >= 3

    def test_base_package_includes_backtest_reliability(self, tmp_path):
        policies = tmp_path / "policies" / "v1"
        policies.mkdir(parents=True)
        findings = tmp_path / "findings"
        findings.mkdir()
        records = [
            {"bot_id": "bot1", "param_category": "signal",
             "prediction_correct": True}
            for _ in range(4)
        ]
        with (findings / "backtest_calibration.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        ctx = ContextBuilder(memory_dir=tmp_path)
        pkg = ctx.base_package()
        assert "backtest_reliability" in pkg.data
        assert pkg.data["backtest_reliability"]["signal"] == 1.0
