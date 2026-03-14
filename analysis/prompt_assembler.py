"""Prompt assembler — builds the context package for the daily analysis runtime invocation.

Uses deterministic triage to pre-process data and generate focused analytical
questions. Claude reasons about 3-5 significant events rather than mechanically
checking 29 items.
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
    "excursion_stats.json",
    "overlay_state_summary.json",
    "experiment_data.json",
    "signal_health.json",
    "fill_quality.json",
    "filter_decisions.json",
    "indicator_snapshots.json",
    "orderbook_stats.json",
]

# Focused instructions that require causal reasoning, not checklist narration.
_FOCUSED_INSTRUCTIONS = """\
You are analyzing today's trading data. A deterministic triage system has already
pre-processed the data and identified what deserves your attention.

## ROUTINE SUMMARY (pre-computed — do NOT regenerate this)
{routine_summary}

## SIGNIFICANT EVENTS REQUIRING YOUR ANALYSIS
{significant_events}

## YOUR ANALYTICAL TASKS

For each significant event above, you MUST:
1. State a hypothesis about WHY it happened
2. Identify confirming AND refuting evidence in the data
3. Check the evidence — does it support or undermine your hypothesis?
4. Rate your confidence (0.0–1.0) with explicit justification

{focus_questions}

## GROUND TRUTH PERFORMANCE (do not modify this evaluation)
If ground_truth_trend data is present, reference specific metric movements when
proposing changes. If a bot's composite is declining, prioritize diagnosis over
new proposals.

## YOUR PREDICTION TRACK RECORD
If prediction_accuracy_by_metric data is present, recalibrate accordingly:
- Metrics where your accuracy < 50%: reduce confidence to 0.2-0.4 or skip predictions
- Metrics where your accuracy > 70%: you may use confidence up to 0.8
- No data yet: use conservative confidence (0.3-0.5)
- Explain what you got wrong last time and why

## BLOCKED APPROACHES
If last_week_synthesis data contains a "discard" list, do NOT suggest those
approaches — they have failed repeatedly.

## PARAMETER SEARCH CONTEXT
If search_reports data is present, the autonomous inner loop has already tested
parameter neighborhoods. Do NOT propose parameter changes that overlap with what
the inner loop already explored — focus on structural changes or untested parameters.
If backtest_reliability data is present, categories marked as unreliable should be
addressed with structural changes rather than further parameter tuning.

## HYPOTHESIS TRACK RECORD
If hypothesis_track_record data is present, prioritize hypotheses with positive
effectiveness scores when making structural proposals. Do NOT propose changes
linked to hypotheses with effectiveness <= 0 or status="retired".

## ACTIVE EXPERIMENTS
If active_experiments data is present, do NOT propose changes that overlap with
experiments currently in progress — let them complete their observation window.
Reference experiment status when discussing related metrics.

## CONSTRAINTS
- Do NOT restate the routine summary — it's already computed above.
- Do NOT mechanically review every data file — focus only on what the triage flagged.
- Every suggestion MUST include quantified expected impact (PnL range, drawdown change)
  with evidence base (trade count, time period, statistical significance).
- Check rejected_suggestions: do NOT re-suggest previously rejected items without new evidence.
- Check active_suggestions: do NOT contradict DEPLOYED suggestions.
- Check category_scorecard: categories with win_rate < 30% (n≥5) need exceptional evidence.
- Suggestions without quantification will be rejected by the validator.

## STRUCTURED OUTPUT (REQUIRED)
At the END of your analysis, emit a structured data block.
This block is machine-parsed — do NOT omit it.
<!-- STRUCTURED_OUTPUT
{{
  "predictions": [
    {{"bot_id": "...", "metric": "pnl|win_rate|drawdown|sharpe", "direction": "improve|decline|stable", "confidence": 0.0-1.0, "timeframe_days": 7, "reasoning": "..."}}
  ],
  "suggestions": [
    {{"suggestion_id": "#abc123", "bot_id": "...", "category": "exit_timing|filter_threshold|stop_loss|signal|structural|position_sizing|regime_gate", "title": "...", "expected_impact": "...", "confidence": 0.0-1.0, "evidence_summary": "...", "proposed_value": 0.5, "target_param": "param_name"}}
  ],
  "structural_proposals": [
    {{"hypothesis_id": "REQUIRED: use id from structural_hypotheses if matching, else null", "bot_id": "...", "title": "...", "description": "...", "reversibility": "easy|moderate|hard", "evidence": "...", "estimated_complexity": "low|medium|high", "acceptance_criteria": [{{"metric": "...", "direction": "improve|not_degrade", "minimum_change": 0.0, "observation_window_days": 14, "minimum_trade_count": 20}}]}}
  ]
}}
-->"""

# Legacy instructions kept for backward compatibility (used when no triage is provided)
_INSTRUCTIONS = _FOCUSED_INSTRUCTIONS.format(
    routine_summary="(No triage data — review all bots manually)",
    significant_events="(No triage — review all curated data files for anomalies)",
    focus_questions="Analyze today's trading performance across all bots. Focus on anomalies, "
    "process failures, and actionable improvements.",
)


class DailyPromptAssembler:
    """Assembles the full context package for a daily analysis agent invocation."""

    def __init__(
        self,
        date: str,
        bots: list[str],
        curated_dir: Path,
        memory_dir: Path,
        corrections_lookback_days: int = 30,
        bot_configs: dict | None = None,
    ) -> None:
        self.date = date
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.corrections_lookback_days = corrections_lookback_days
        self.bot_configs = bot_configs
        self._ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)

    def assemble(self, triage_report=None) -> PromptPackage:
        """Build the complete prompt package.

        Args:
            triage_report: Optional TriageReport from DailyTriage. When provided,
                instructions are focused on the triage's significant events and
                questions. When None, uses fallback instructions.
        """
        pkg = self._ctx.base_package(bot_configs=self.bot_configs)
        pkg.task_prompt = self._build_task_prompt()
        pkg.data.update(self._load_structured_data(triage_report))
        pkg.instructions = self._build_instructions(triage_report)
        pkg.context_files.extend(self._list_data_files(triage_report))

        # Inject contradiction data if any
        contradictions = self._ctx.load_contradictions(self.date, self.bots, self.curated_dir)
        if contradictions:
            pkg.data["contradictions"] = contradictions

        return pkg

    def _build_instructions(self, triage_report=None) -> str:
        """Build instructions from triage report or use fallback."""
        if triage_report is None:
            return _INSTRUCTIONS

        # Format significant events
        event_lines = []
        for i, event in enumerate(triage_report.significant_events, 1):
            event_lines.append(
                f"{i}. **[{event.event_type.upper()}]** [{event.bot_id}] "
                f"(severity: {event.severity}) — {event.description}"
            )
        events_text = "\n".join(event_lines) if event_lines else "(No significant events detected — routine day)"

        # Format focus questions
        question_lines = []
        for i, q in enumerate(triage_report.focus_questions, 1):
            question_lines.append(f"{i}. {q}")
        questions_text = "\n".join(question_lines)

        return _FOCUSED_INSTRUCTIONS.format(
            routine_summary=triage_report.routine_summary,
            significant_events=events_text,
            focus_questions=questions_text,
        )

    def _build_task_prompt(self) -> str:
        bot_list = ", ".join(self.bots)
        return (
            f"Analyze today's ({self.date}) trading performance for all bots: {bot_list}.\n"
            f"Focus on the significant events identified by triage. Reason about causes, not just symptoms."
        )

    def _load_structured_data(self, triage_report=None) -> dict:
        data: dict = {}
        date_dir = self.curated_dir / self.date
        findings_dir = self.memory_dir / "findings"

        # Determine which files to load based on triage
        files_to_load = _CURATED_FILES
        if triage_report and triage_report.relevant_data_keys:
            # Always load summary + triage-relevant files
            relevant = set(triage_report.relevant_data_keys)
            relevant.add("summary.json")
            files_to_load = [f for f in _CURATED_FILES if f in relevant]

        for bot in self.bots:
            bot_dir = date_dir / bot
            bot_data: dict = {}
            for filename in files_to_load:
                path = bot_dir / filename
                if path.exists():
                    key = filename.replace(".json", "")
                    bot_data[key] = json.loads(path.read_text(encoding="utf-8"))

            # Inject signal factor rolling trends per bot
            factor_trends = self._ctx.load_signal_factor_history(bot, self.date, findings_dir)
            if factor_trends:
                bot_data["signal_factor_trends"] = factor_trends

            data[bot] = bot_data

        risk_path = date_dir / "portfolio_risk_card.json"
        if risk_path.exists():
            data["portfolio_risk_card"] = json.loads(risk_path.read_text(encoding="utf-8"))

        return data

    def _list_data_files(self, triage_report=None) -> list[str]:
        files: list[str] = []
        date_dir = self.curated_dir / self.date

        files_to_list = _CURATED_FILES
        if triage_report and triage_report.relevant_data_keys:
            relevant = set(triage_report.relevant_data_keys)
            relevant.add("summary.json")
            files_to_list = [f for f in _CURATED_FILES if f in relevant]

        for bot in self.bots:
            bot_dir = date_dir / bot
            for filename in files_to_list:
                path = bot_dir / filename
                if path.exists():
                    files.append(str(path))

        risk_path = date_dir / "portfolio_risk_card.json"
        if risk_path.exists():
            files.append(str(risk_path))

        return files
