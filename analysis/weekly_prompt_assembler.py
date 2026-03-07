# analysis/weekly_prompt_assembler.py
"""Weekly prompt assembler — builds context package for weekly analysis Claude invocation.

Uses ContextBuilder for shared policy/corrections loading. Adds weekly-specific
data: 7 daily reports, weekly summary, week-over-week comparisons,
strategy refinement report, portfolio risk card series.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from analysis.context_builder import ContextBuilder
from schemas.prompt_package import PromptPackage

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
   QUANTIFICATION REQUIRED: Each suggestion MUST quantify expected Calmar ratio impact.
   Include: (a) expected return change, (b) expected max drawdown change,
   (c) evidence base (trade count, time period, statistical confidence).
10. Output: weekly_report.md
11. Structural analysis: Based on 30-day root cause patterns, assess whether any bot's signal logic needs structural changes (not just parameter tuning). Specifically evaluate:
    - Has signal->outcome correlation declined? Does the signal need recalibration or replacement?
    - Are filters blocking high-quality signals? Should a filter be restructured rather than threshold-adjusted?
    - Are exits consistently premature? Should the exit strategy change (e.g., trailing stop)?
12. Parameter space proposals: If evidence suggests a new dimension should be optimized (e.g., adding a time-of-day gate, changing position sizing model), propose it with supporting evidence. These proposals require human approval.
13. Do NOT re-suggest anything in the rejected_suggestions list unless you present new quantitative evidence.
14. PORTFOLIO & STRATEGY ALLOCATION ASSESSMENT
    Review the allocation_analysis data and for each recommendation:
    a. Validate the quantitative rationale against your qualitative analysis
    b. Flag any recommendations you disagree with, explaining why
    c. For cross-bot allocation: assess whether rebalancing is warranted given market conditions
    d. For intra-bot proportions: consider strategy interactions not captured by correlation alone
    e. For same-instrument strategies (e.g., all NQ in momentum_trader): assess signal independence
    f. Highlight the top 3 allocation changes by expected impact
15. PORTFOLIO ALLOCATION ASSESSMENT FRAMEWORK
    For each bot and each strategy within each bot:
    a. Compute capital efficiency: PnL / (unit_risk × max_positions × days_active)
    b. Compute marginal Sharpe contribution: portfolio Sharpe with vs. without this strategy
    c. If capital efficiency differs >2x between strategies, propose reallocation with:
       - Current allocation and suggested allocation (as % of equity and R-units)
       - Expected Calmar ratio change
       - Minimum observation period before re-evaluating
    d. For same-instrument strategies (e.g., all NQ in momentum_trader), assess signal independence:
       - Entry overlap rate (% of entries within 2 bars of each other)
       - Return correlation at trade level (not just daily)
       - If correlation > 0.6, recommend consolidation or differentiation
16. REGIME-CONDITIONAL ANALYSIS
    Review the regime_conditional_analysis data and:
    a. Validate regime-specific allocation suggestions
    b. Identify regimes where specific strategies should be scaled down or paused
    c. Assess regime transition patterns — is the current regime distribution shifting?
    d. For each regime with >20 trades: confirm whether allocation adjustments are warranted
17. STRUCTURAL ASSESSMENT
    Review the structural_analysis data and:
    a. For decaying strategies: assess whether decay is structural or cyclical
    b. For architecture mismatches: evaluate whether the suggested change is appropriate
    c. For filter ROI: recommend filter changes only when net impact exceeds $500/week
    d. Prioritize structural proposals by reversibility — prefer low-effort, reversible changes
18. COORDINATOR INTERACTION EFFECTS (swing_trader only)
    Review the interaction_analysis data and:
    a. Assess net coordinator benefit — is the coordination system net positive?
    b. For each rule: evaluate whether the tightening/boosting parameters are optimal
    c. Assess overlay regime impact — should overlay signals gate more/fewer strategies?
19. ALLOCATION DRIFT ANALYSIS
    Review the allocation_drift data and:
    a. If total_drift_pct > 10%: flag as material drift requiring attention
    b. For persistent_drifters: assess whether drift is intentional or neglect
    c. If trend_direction is "increasing": warn that allocations are diverging
    d. Compare current recommended vs actual — highlight the largest gaps
    e. If drift has been stable and low (<5%), confirm allocations are well-tracked
20. CORRECTION PATTERNS
    Review correction_patterns data (if present): these are recurring human corrections
    clustered by type and target. Avoid repeating these mistakes in your analysis.
    For each pattern, acknowledge what was repeatedly corrected and adapt accordingly.
21. FILTER INTERACTION ANALYSIS
    Review filter_interaction data (if present): identify redundant filter pairs
    (high co-activation + similar blocking patterns) and recommend consolidation.
    Flag complementary pairs that together cover different failure modes.
22. EXIT STRATEGY SWEEP
    Review exit_sweep data (if present in simulation_results): compare the 12 exit
    configurations tested against baseline. Highlight the best-performing exit strategy
    per bot with expected improvement. Flag if the current exit mechanism is suboptimal
    (best alternative improvement > 10%). Note: these are proxy calculations using
    post-exit prices, not full backtests.
23. SUGGESTION IDS: Every suggestion in the weekly report MUST include its suggestion_id
    in brackets, e.g. "[#abc123] Widen stop loss on bot1". Users respond with
    "approve suggestion #abc123" to accept or "reject suggestion #abc123" to decline.
    Suggestion IDs are provided in metadata["suggestion_ids"].
24. RETROSPECTIVE REVIEW: Review weekly_retrospective data (if present) showing last
    week's prediction accuracy. Acknowledge accuracy rate. For incorrect predictions,
    explain what went wrong. If accuracy < 50%, flag this and lower confidence levels.
25. SUGGESTION OUTCOMES: Review outcome_measurements data (if present). For NEGATIVE
    outcomes, do NOT re-suggest similar approaches. For POSITIVE outcomes, reference
    as evidence for similar suggestions on other bots. Adjust confidence calibration
    based on track record.
26. FORECAST CALIBRATION: Review forecast_meta_analysis (if present). If rolling_accuracy_4w
    < 50%, apply a confidence haircut. For bots where accuracy < 40%, cap max confidence
    at 0.5. If calibration_adjustment is negative, you've been over-confident — lower scores.
27. STRUCTURAL HYPOTHESES: Review structural_hypotheses (if present). When proposing
    structural changes, you MUST check if any existing hypothesis matches your proposal.
    If a match exists, set hypothesis_id to that hypothesis's "id" field in your
    structured output. This is critical for tracking which hypotheses lead to improvements.
    The available hypothesis IDs and their effectiveness scores are listed in
    structural_hypotheses. Propose at most 2 structural changes per week. Each must
    include: expected impact, reversibility, minimum data needed.
    Check rejected_suggestions before proposing.
28. CROSS-BOT TRANSFER: Review transfer_proposals (if present). For proposals with
    compatibility > 0.7, recommend transfer with implementation notes. For 0.4-0.7,
    flag as "worth investigating". Propose new patterns for the library based on
    this week's findings.
29. CATEGORY SCORECARD: Review category_scorecard (if present). Categories with
    win_rate < 30% and sample_size >= 5 require exceptional new evidence to justify
    suggestions. Reference track record when proposing suggestions.
30. PREDICTION ACCURACY: Review prediction_accuracy_by_metric (if present). For metrics
    where accuracy < 40%, explicitly caveat predictions and lower confidence.
31. HYPOTHESIS TRACK RECORD: Review hypothesis_track_record (if present). Prioritize
    hypotheses with positive effectiveness scores. Do not propose retired hypotheses.
32. TRANSFER TRACK RECORD: Review transfer_track_record (if present). Favor patterns
    with proven cross-bot success. Avoid patterns with negative track record.
33. VALIDATION PATTERNS: Review validation_patterns (if present). These show which
    suggestion categories are consistently blocked by the validator. Avoid proposing
    suggestions in categories with high blocked_count unless you have strong new evidence.
34. STRUCTURED OUTPUT (REQUIRED): At the END of your analysis, emit a structured data block.
    This block is machine-parsed — do NOT omit it.
    <!-- STRUCTURED_OUTPUT
    {
      "predictions": [
        {"bot_id": "...", "metric": "pnl|win_rate|drawdown|sharpe", "direction": "improve|decline|stable", "confidence": 0.0-1.0, "timeframe_days": 7, "reasoning": "..."}
      ],
      "suggestions": [
        {"suggestion_id": "#abc123", "bot_id": "...", "category": "exit_timing|filter_threshold|stop_loss|signal|structural|position_sizing|regime_gate", "title": "...", "expected_impact": "...", "confidence": 0.0-1.0, "evidence_summary": "...", "proposed_value": 0.5, "target_param": "param_name"}
      ],
      "structural_proposals": [
        {"hypothesis_id": "REQUIRED: use id from structural_hypotheses if matching, else null", "bot_id": "...", "title": "...", "description": "...", "reversibility": "easy|moderate|hard", "evidence": "...", "estimated_complexity": "low|medium|high"}
      ]
    }
    -->"""


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
        self._ctx = ContextBuilder(memory_dir)

    def assemble(self) -> PromptPackage:
        """Build the complete weekly prompt package."""
        pkg = self._ctx.base_package()
        pkg.task_prompt = self._build_task_prompt()
        pkg.data.update(self._load_data())
        pkg.instructions = _WEEKLY_INSTRUCTIONS
        pkg.context_files.extend(self._list_data_files())
        return pkg

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

        weekly_dir = self.curated_dir / "weekly" / self.week_start
        summary_path = weekly_dir / "weekly_summary.json"
        if summary_path.exists():
            data["weekly_summary"] = json.loads(summary_path.read_text())

        refinement_path = weekly_dir / "refinement_report.json"
        if refinement_path.exists():
            data["refinement_report"] = json.loads(refinement_path.read_text())

        wow_path = weekly_dir / "week_over_week.json"
        if wow_path.exists():
            data["week_over_week"] = json.loads(wow_path.read_text())

        data["daily_reports"] = self._load_daily_reports()
        data["portfolio_risk_cards"] = self._load_risk_cards()

        # Load allocation analysis if present
        alloc_path = weekly_dir / "allocation_analysis.json"
        if alloc_path.exists():
            data["allocation_analysis"] = json.loads(alloc_path.read_text())

        # Load structural analysis if present
        structural_path = weekly_dir / "structural_analysis.json"
        if structural_path.exists():
            data["structural_analysis"] = json.loads(structural_path.read_text())

        # Load regime-conditional analysis if present
        regime_path = weekly_dir / "regime_conditional_analysis.json"
        if regime_path.exists():
            data["regime_conditional_analysis"] = json.loads(regime_path.read_text())

        # Load interaction analysis if present
        interaction_path = weekly_dir / "interaction_analysis.json"
        if interaction_path.exists():
            data["interaction_analysis"] = json.loads(interaction_path.read_text())

        # Load allocation drift analysis if present
        drift_path = weekly_dir / "allocation_drift.json"
        if drift_path.exists():
            data["allocation_drift"] = json.loads(drift_path.read_text())

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

    def _list_data_files(self) -> list[str]:
        files: list[str] = []
        weekly_dir = self.curated_dir / "weekly" / self.week_start
        for name in [
            "weekly_summary.json", "refinement_report.json", "week_over_week.json",
            "allocation_analysis.json", "structural_analysis.json",
            "regime_conditional_analysis.json", "interaction_analysis.json",
            "allocation_drift.json",
        ]:
            path = weekly_dir / name
            if path.exists():
                files.append(str(path))
        return files

    def _week_dates(self) -> list[str]:
        """Generate the 7 date strings for this week."""
        start = datetime.strptime(self.week_start, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
