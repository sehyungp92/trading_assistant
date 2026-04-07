# tests/test_wfo_prompt_assembler.py
"""Tests for the WFO prompt assembler."""
import json
from pathlib import Path

from analysis.wfo_prompt_assembler import WFOPromptAssembler


class TestWFOPromptAssembler:
    def test_assembles_package(self, tmp_path: Path):
        policy_dir = tmp_path / "memory" / "policies" / "v1"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agent.md").write_text("You are the WFO analyst.")
        (policy_dir / "trading_rules.md").write_text("Max 15% drawdown.")
        (policy_dir / "soul.md").write_text("Conservative risk tolerance.")

        wfo_dir = tmp_path / "runs" / "wfo" / "bot2"
        wfo_dir.mkdir(parents=True)
        report = {"bot_id": "bot2", "recommendation": "adopt", "suggested_params": {"rsi": 35}}
        (wfo_dir / "wfo_report.json").write_text(json.dumps(report))

        assembler = WFOPromptAssembler(
            bot_id="bot2",
            memory_dir=tmp_path / "memory",
            wfo_output_dir=wfo_dir,
        )
        package = assembler.assemble()
        assert package.system_prompt
        assert package.task_prompt
        assert package.data
        assert package.instructions
        assert package.data["wfo_report"]["bot_id"] == "bot2"

    def test_handles_missing_files_gracefully(self, tmp_path: Path):
        assembler = WFOPromptAssembler(
            bot_id="bot1",
            memory_dir=tmp_path / "memory",
            wfo_output_dir=tmp_path / "nonexistent",
        )
        package = assembler.assemble()
        assert package.system_prompt is not None
        assert package.data == {}

    def test_includes_wfo_skill_context(self, tmp_path: Path):
        skill_dir = tmp_path / "memory" / "skills"
        skill_dir.mkdir(parents=True)
        (skill_dir / "wfo_pipeline.md").write_text("WFO pipeline instructions here.")

        assembler = WFOPromptAssembler(
            bot_id="bot1",
            memory_dir=tmp_path / "memory",
            wfo_output_dir=tmp_path / "runs",
        )
        package = assembler.assemble()
        assert "wfo_pipeline" in package.skill_context
