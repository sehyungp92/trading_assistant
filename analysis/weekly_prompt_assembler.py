# analysis/weekly_prompt_assembler.py
"""Weekly prompt assembler — builds context package for weekly analysis Claude invocation.

Similar to the daily prompt assembler (analysis/prompt_assembler.py) but with
weekly-specific data: 7 daily reports, weekly summary, week-over-week comparisons,
strategy refinement report, portfolio risk card series, and corrections.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

_WEEKLY_INSTRUCTIONS = """\
1. Trend analysis: Is performance improving or degrading? By bot? By regime?
2. Pattern recognition: Which signal+regime combos are consistently profitable?
3. Filter tuning signals: Across the week, which filters cost more than they saved?
4. Correlation analysis: How correlated are your bots? Is diversification working?
5. Process quality trends: Is average process quality improving? Which root causes most frequent?
6. Drawdown context: Was it a single event or systemic?
7. Week-over-week comparison: This week vs. last 4 weeks
8. Strategy refinement: Review the refinement report and highlight top suggestions
9. Actionable items: Max 5 specific, testable suggestions ranked by confidence
10. Output: weekly_report.md"""


class WeeklyPromptAssembler:
    """Assembles the full context package for a weekly analysis agent invocation."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        bots: list[str],
        curated_dir: Path,
        memory_dir: Path,
        runs_dir: Path,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.runs_dir = runs_dir

    def assemble(self) -> dict:
        """Build the complete weekly prompt package."""
        return {
            "system_prompt": self._build_system_prompt(),
            "task_prompt": self._build_task_prompt(),
            "data": self._load_data(),
            "instructions": _WEEKLY_INSTRUCTIONS,
            "corrections": self._load_corrections(),
            "context_files": self._list_context_files(),
        }

    def _build_system_prompt(self) -> str:
        parts: list[str] = []
        policy_dir = self.memory_dir / "policies" / "v1"
        for name in ["agents.md", "trading_rules.md", "soul.md"]:
            path = policy_dir / name
            if path.exists():
                parts.append(f"--- {name} ---\n{path.read_text()}")
        return "\n\n".join(parts)

    def _build_task_prompt(self) -> str:
        bot_list = ", ".join(self.bots)
        return (
            f"Produce the weekly summary for {self.week_start} to {self.week_end} "
            f"covering all bots: {bot_list}.\n"
            f"Use the weekly summary data, 7 daily reports, portfolio risk cards, "
            f"and strategy refinement report provided. Follow the instructions exactly."
        )

    def _load_data(self) -> dict:
        data: dict = {}

        # Weekly summary
        weekly_dir = self.curated_dir / "weekly" / self.week_start
        summary_path = weekly_dir / "weekly_summary.json"
        if summary_path.exists():
            data["weekly_summary"] = json.loads(summary_path.read_text())

        # Refinement report
        refinement_path = weekly_dir / "refinement_report.json"
        if refinement_path.exists():
            data["refinement_report"] = json.loads(refinement_path.read_text())

        # Week-over-week
        wow_path = weekly_dir / "week_over_week.json"
        if wow_path.exists():
            data["week_over_week"] = json.loads(wow_path.read_text())

        # 7 daily reports
        data["daily_reports"] = self._load_daily_reports()

        # 7 portfolio risk cards
        data["portfolio_risk_cards"] = self._load_risk_cards()

        return data

    def _load_daily_reports(self) -> list[dict]:
        reports: list[dict] = []
        for date_str in self._week_dates():
            run_dir = self.runs_dir / date_str / "daily-report"
            report_path = run_dir / "daily_report.md"
            if report_path.exists():
                reports.append({
                    "date": date_str,
                    "content": report_path.read_text(),
                })
        return reports

    def _load_risk_cards(self) -> list[dict]:
        cards: list[dict] = []
        for date_str in self._week_dates():
            card_path = self.curated_dir / date_str / "portfolio_risk_card.json"
            if card_path.exists():
                cards.append(json.loads(card_path.read_text()))
        return cards

    def _load_corrections(self) -> list[dict]:
        corrections_path = self.memory_dir / "findings" / "corrections.jsonl"
        if not corrections_path.exists():
            return []
        corrections: list[dict] = []
        for line in corrections_path.read_text().strip().splitlines():
            if line.strip():
                corrections.append(json.loads(line))
        return corrections

    def _list_context_files(self) -> list[str]:
        files: list[str] = []
        weekly_dir = self.curated_dir / "weekly" / self.week_start
        for name in ["weekly_summary.json", "refinement_report.json", "week_over_week.json"]:
            path = weekly_dir / name
            if path.exists():
                files.append(str(path))

        policy_dir = self.memory_dir / "policies" / "v1"
        for name in ["agents.md", "trading_rules.md", "soul.md"]:
            path = policy_dir / name
            if path.exists():
                files.append(str(path))

        return files

    def _week_dates(self) -> list[str]:
        """Generate the 7 date strings for this week."""
        start = datetime.strptime(self.week_start, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
