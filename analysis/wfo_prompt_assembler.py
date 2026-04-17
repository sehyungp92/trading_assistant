# analysis/wfo_prompt_assembler.py
"""WFO prompt assembler — builds context package for WFO analysis runtime.

Uses ContextBuilder for shared policy loading. Adds WFO-specific
data: optimization report, skill context, safety flags.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.context_builder import ContextBuilder
from schemas.prompt_package import PromptPackage

_WFO_INSTRUCTIONS = """\
1. Review the WFO report for bot {bot_id}
2. Verify the parameter changes make strategic sense (not just statistical)
3. Check cost sensitivity: are results fragile at higher costs?
4. Check robustness: stable across regimes and neighboring params?
5. Review safety flags and assess overall risk
6. Provide your recommendation: ADOPT / TEST_FURTHER / REJECT with reasoning
7. If ADOPT: summarize the param changes for a draft PR description
8. If TEST_FURTHER: specify what additional validation is needed
9. If REJECT: explain why and suggest alternative approaches
10. Output: wfo_analysis.md"""


class WFOPromptAssembler:
    """Assembles the full context package for a WFO analysis agent invocation."""

    def __init__(
        self,
        bot_id: str,
        memory_dir: Path,
        wfo_output_dir: Path,
    ) -> None:
        self.bot_id = bot_id
        self.memory_dir = memory_dir
        self.wfo_output_dir = wfo_output_dir
        self._ctx = ContextBuilder(memory_dir)

    def assemble(self, session_store=None) -> PromptPackage:
        """Build the complete WFO prompt package."""
        pkg = self._ctx.base_package(session_store=session_store, agent_type="wfo", bot_id=self.bot_id)
        pkg.task_prompt = self._build_task_prompt()
        pkg.data.update(self._load_data())
        pkg.instructions = _WFO_INSTRUCTIONS.format(bot_id=self.bot_id)
        pkg.skill_context = self._load_skill_context()
        pkg.context_files.extend(self._list_data_files())
        pkg.metadata["bot_ids"] = self.bot_id
        pkg.metadata["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return pkg

    def _build_task_prompt(self) -> str:
        return (
            f"Review the walk-forward optimization results for {self.bot_id}. "
            f"The WFO pipeline has run and produced a report with parameter recommendations, "
            f"cost sensitivity analysis, robustness scores, and safety flags. "
            f"Provide your assessment and recommendation."
        )

    def _load_data(self) -> dict:
        data: dict = {}
        report_path = self.wfo_output_dir / "wfo_report.json"
        if report_path.exists():
            try:
                data["wfo_report"] = json.loads(report_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return data

    def _load_skill_context(self) -> str:
        skill_path = self.memory_dir / "skills" / "wfo_pipeline.md"
        if skill_path.exists():
            return f"--- wfo_pipeline ---\n{skill_path.read_text()}"
        return ""

    def _list_data_files(self) -> list[str]:
        files: list[str] = []
        report_path = self.wfo_output_dir / "wfo_report.json"
        if report_path.exists():
            files.append(str(report_path))
        return files
