"""Prompt assembler — builds the context package for the daily analysis Claude invocation.

Follows the prompt structure from roadmap section 2.4:
  SYSTEM PROMPT: policies + corrections + patterns
  TASK PROMPT: "Analyze today's trading performance for all bots."
  DATA: curated summaries + risk card
  INSTRUCTIONS: structured analysis steps
"""
from __future__ import annotations

import json
from pathlib import Path

from analysis.context_builder import ContextBuilder
from schemas.prompt_package import PromptPackage

_CURATED_FILES = [
    "summary.json",
    "winners.json",
    "losers.json",
    "process_failures.json",
    "notable_missed.json",
    "regime_analysis.json",
    "filter_analysis.json",
    "root_cause_summary.json",
    "factor_attribution.json",
    "exit_efficiency.json",
    "hourly_performance.json",
    "slippage_stats.json",
]

_INSTRUCTIONS = """\
1. Start with portfolio-level picture (total PnL, drawdown, exposure, crowding alerts)
2. For each bot:
   a. What worked: winning pattern (regime + signal combo + process quality)
   b. What failed: distinguish PROCESS errors from NORMAL LOSSES using root cause tags
   c. Missed opportunities: quantify filter impact, note simulation assumptions
   d. Anomalies: anything statistically unusual
   e. Factor attribution: use signal factor data to identify which indicators drove PnL
   f. Exit efficiency: review exit timing scores and MAE/MFE ratios for optimization clues
3. Cross-bot patterns: correlation, regime alignment, crowding risk
4. Hourly performance: note time-of-day edges or dead zones from hourly_performance data
5. Slippage: review slippage_stats for regime-based cost patterns
6. Actionable items: max 3 specific, testable suggestions backed by factor attribution or exit efficiency data
   QUANTIFICATION REQUIRED: Every suggestion MUST include:
   (a) Expected return impact with a range (e.g., "+0.3% to +0.8% daily PnL")
   (b) Drawdown impact estimate (e.g., "max drawdown reduction of ~15%")
   (c) Evidence base: trade count, time period, and statistical significance
   Suggestions without quantification will be rejected.
7. Open risks: any CRITICAL/HIGH events that need human attention
8. Output: daily_report.md + report_checklist.json
9. Check the rejected_suggestions list (if present). Do NOT re-suggest anything that was previously rejected unless you have new evidence."""


class DailyPromptAssembler:
    """Assembles the full context package for a daily analysis agent invocation."""

    def __init__(
        self,
        date: str,
        bots: list[str],
        curated_dir: Path,
        memory_dir: Path,
        corrections_lookback_days: int = 30,
    ) -> None:
        self.date = date
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.corrections_lookback_days = corrections_lookback_days
        self._ctx = ContextBuilder(memory_dir)

    def assemble(self) -> PromptPackage:
        """Build the complete prompt package."""
        pkg = self._ctx.base_package()
        pkg.task_prompt = self._build_task_prompt()
        pkg.data.update(self._load_structured_data())
        pkg.instructions = _INSTRUCTIONS
        pkg.context_files.extend(self._list_data_files())
        return pkg

    def _build_task_prompt(self) -> str:
        bot_list = ", ".join(self.bots)
        return (
            f"Analyze today's ({self.date}) trading performance for all bots: {bot_list}.\n"
            f"Use the structured data provided. Follow the instructions exactly."
        )

    def _load_structured_data(self) -> dict:
        data: dict = {}
        date_dir = self.curated_dir / self.date

        for bot in self.bots:
            bot_dir = date_dir / bot
            bot_data: dict = {}
            for filename in _CURATED_FILES:
                path = bot_dir / filename
                if path.exists():
                    key = filename.replace(".json", "")
                    bot_data[key] = json.loads(path.read_text())
            data[bot] = bot_data

        risk_path = date_dir / "portfolio_risk_card.json"
        if risk_path.exists():
            data["portfolio_risk_card"] = json.loads(risk_path.read_text())

        return data

    def _list_data_files(self) -> list[str]:
        files: list[str] = []
        date_dir = self.curated_dir / self.date

        for bot in self.bots:
            bot_dir = date_dir / bot
            for filename in _CURATED_FILES:
                path = bot_dir / filename
                if path.exists():
                    files.append(str(path))

        risk_path = date_dir / "portfolio_risk_card.json"
        if risk_path.exists():
            files.append(str(risk_path))

        return files
