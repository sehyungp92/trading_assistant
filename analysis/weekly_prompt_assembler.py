# analysis/weekly_prompt_assembler.py
"""Weekly prompt assembler — builds context package for weekly analysis runtime invocation.

Uses deterministic weekly triage for computed summaries and focused analytical
questions. Claude reasons about retrospective accuracy and discovers novel patterns
rather than mechanically reviewing 34 checklist items.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from analysis.context_builder import ContextBuilder
from schemas.prompt_package import PromptPackage

_FOCUSED_WEEKLY_INSTRUCTIONS = """\
You are analyzing a week of trading data. A deterministic triage system has
pre-computed summaries and identified what deserves your analytical attention.

## COMPUTED WEEK SUMMARY (pre-computed — do NOT regenerate)
{computed_summary}

## ANOMALIES DETECTED
{anomalies}

## RETROSPECTIVE QUESTIONS (about past decisions and predictions)
{retrospective_questions}

## DISCOVERY QUESTIONS (about novel patterns)
{discovery_questions}

## YOUR ANALYTICAL TASKS

For RETROSPECTIVE questions: review what was predicted/suggested, what actually
happened, and WHY the outcome differed from expectations. State lessons learned.

For DISCOVERY questions: look for patterns the automated detectors missed.
State a testable hypothesis, identify evidence for and against, and rate confidence.

## STRATEGY PROPOSALS
When proposing changes:
1. Each suggestion MUST quantify expected Calmar ratio impact (return change + drawdown change)
2. Evidence base required: trade count, time period, statistical significance
3. Check category_scorecard: categories with win_rate < 30% (n≥5) need exceptional evidence
4. Check rejected_suggestions: do NOT re-suggest without new evidence
5. Check hypothesis_track_record: prioritize hypotheses with positive effectiveness
6. Structural proposals MUST include acceptance_criteria with measurable metrics
7. Max 5 suggestions ranked by confidence. Each MUST have a suggestion_id.
8. Check validation_patterns: categories with 3+ blocks in 30 days need explicit differentiation

## PORTFOLIO IMPROVEMENT ASSESSMENT
Review portfolio-level data (if present) and propose at most 2 portfolio-level changes:
- **Family performance trajectory**: compare family_snapshots trends against allocation weights
- **Portfolio rolling metrics**: reference portfolio_rolling_metrics for Sharpe/Sortino/Calmar trends
- **Drawdown correlation risk**: check drawdown_correlation for systemic risk signals
- **Portfolio rule blocks**: review rule_blocks_summary for coordination system effectiveness
- **Strategy engine detector findings**: If refinement_report data is present, it contains
  pre-computed statistical findings from 16 automated detectors. For the top 5
  highest-confidence findings, you MUST state AGREE or DISAGREE with 1-sentence
  reasoning. Do NOT duplicate detector analysis — focus your effort on patterns
  the detectors cannot cover (structural issues, cross-bot interactions, novel market conditions)
- **Allocation analysis**: validate quantitative rationale against your qualitative analysis

Portfolio proposal requirements:
- All proposals must cite specific family/bot data and projected portfolio Calmar impact
- portfolio_allocation proposals require 60+ days of evidence
- portfolio_risk_cap and portfolio_drawdown_tier require 90+ days of evidence
- Never suggest removing drawdown tiers or loosening stop levels
- Maximum 15% allocation change per family per cycle, minimum 5% floor
- Check portfolio_outcomes for past portfolio change track record

## CROSS-BOT TRANSFER
Review transfer_proposals (if present):
- Compatibility > 0.7: recommend with implementation notes
- Compatibility 0.4–0.7: flag as "worth investigating"
- Check transfer_track_record to favor proven patterns

## GROUND TRUTH PERFORMANCE (do not modify this evaluation)
If ground_truth_trend data is present, your job is to improve these composite
scores. Reference specific metric movements when proposing changes. If a bot's
composite is declining, prioritize diagnosis over new proposals.

## YOUR PREDICTION TRACK RECORD
If prediction_accuracy_by_metric data is present, recalibrate accordingly:
- Metrics where your accuracy < 50%: reduce confidence to 0.2-0.4 or skip
- Metrics where your accuracy > 70%: you may use confidence up to 0.8
- Review your worst-performing prediction categories
- Identify systematic biases and state what you will do differently

## DIRECTIONAL BIAS AWARENESS
If forecast_meta_analysis contains directional_bias data:
- "optimistic" bias: you predict improvement more than reality — reduce improve predictions
- "pessimistic" bias: you predict decline more than reality — consider improve scenarios
- Acknowledge your bias before making predictions in affected metrics

## LAST WEEK'S LEARNING SYNTHESIS (what the data shows)
If last_week_synthesis data is present:
- what_worked: Double down on these approaches. Propose similar changes.
- what_failed: Do NOT retry unless conditions demonstrably changed.
- discard: These categories have failed repeatedly — do NOT suggest them.
- lessons: Incorporate these insights into your current analysis.
- ground_truth_deltas: Reference these for performance trajectory.

## STRATEGY IDEAS UNDER REVIEW
If strategy_ideas data is present, for each active idea:
- Assess whether this week's data strengthens or weakens the edge hypothesis
- If evidence is growing (15+ data points, confidence > 0.7), recommend a backtest
- If evidence is contradicted, recommend retiring the idea with an explanation

## ACTIVE EXPERIMENTS
If active_experiments data is present, do NOT propose changes that overlap with
experiments currently in progress — let them complete their observation window.
Reference experiment status when discussing related metrics.

## EXPERIMENT TRACK RECORD
If experiment_track_record data is present, use it to calibrate your confidence
in structural proposals. Categories with high pass rates deserve more aggressive
proposals. Categories with low pass rates need stronger evidence before proposing.

## BACKTEST RELIABILITY
If backtest_reliability data is present, categories with reliability < 0.50
should be addressed with structural changes rather than parameter tuning —
the inner loop's backtests are unreliable for those categories.

## OPTIMIZATION ALLOCATION
If optimization_allocation data is present, reference when proposing suggestions:
- Prefer categories with high value_per_suggestion
- Categories with negative value_per_suggestion require exceptional evidence
- Follow _recommendations for shifting effort between categories
- Categories with 0 positive outcomes in 3+ attempts should be avoided

## SEARCH SIGNAL QUALITY
If search_signal_summary data is present:
- approve_rate < 0.3: detector firing on noise — investigate threshold
- approve_rate > 0.7: search productive — more suggestions in this category worthwhile
- Reference specific bot:category approve_rates when proposing parameter changes

## CYCLE EFFECTIVENESS TREND
If `cycle_effectiveness_trend` data is present, it shows the normalized
effectiveness score (0.0-1.0) for recent cycles. Use this to calibrate ambition:
- Effectiveness trending up: current approach is working — propose incremental refinements
- Effectiveness trending down: something is off — diagnose before proposing more changes
- Effectiveness plateau: consider targeted experiments to break through

## SUGGESTION QUALITY TREND
If `suggestion_quality_trend` data is present, it shows whether suggestion
generation quality is improving over time (hit rate, high-value category ratio).
- Rising hit_rate: your suggestions are getting better — maintain approach
- Falling hit_rate: recalibrate — check which categories are dragging quality down
- Low high_value_ratio: too many suggestions in low-value categories — shift focus

## CONVERGENCE STATUS (learning loop health)
If `convergence_report` is present, it shows whether the learning system is
improving, degrading, oscillating, or stable across multiple dimensions.
- If OSCILLATING: avoid reversing last week's suggestions — let changes settle
- If DEGRADING: question current approach fundamentals before proposing more changes
- If IMPROVING: maintain current approach, propose incremental refinements only
- Reference specific dimension statuses when justifying confidence levels

## DISCOVERIES AND STRATEGY IDEAS
If `discoveries` and `strategy_ideas` are present:
- Reference discoveries that corroborate or contradict detector findings
- For strategy_ideas with status "under_review": assess edge strength and recommend
  either backtest validation or retirement
- Do not propose structural changes that overlap with active strategy ideas
- If a discovery has been corroborated by 2+ weeks of data, escalate to structural proposal

## OUTCOME REASONING
If outcome_reasonings data is present, reference the causal mechanisms that
drove past successes and failures. Propose changes that leverage proven
mechanisms and avoid mechanisms that have consistently failed.

## PARAMETER SEARCH RESULTS
If search_reports data is present, the autonomous inner loop has tested parameter
neighborhoods this week. Reference these results when evaluating parameter-level
suggestions — avoid re-proposing what the inner loop already explored.

## MACRO REGIME ANALYSIS
If macro_regime_context data is present in the base package:
- Report current macro regime (G=Recovery, R=Reflation, S=Infl Hedge, D=Defensive),
  confidence, and stress level
- Break down weekly performance by macro regime: P&L, win rate, expectancy per regime
- Cross-reference macro_regime_sensitivity from strategy profiles:
  e.g., DownturnDominator_v1 should be "disabled" in G/R but "full" in S/D
- Evaluate regime config effectiveness: is regime_unit_risk_mult appropriate?
  If losses persist despite reduced sizing → recommend more aggressive reduction
  If winning strongly with heavy reduction → may be too conservative
- If a regime transition occurred this week, measure transition cost (P&L in ±5d window)
- Note: stress_level is observational only (41% FPR). If reporting stress-stratified
  outcomes, caveat that the signal cannot reliably discriminate stress from normal volatility
- Reference regime_config_history for trend analysis: has config been stable or shifting?
- Evaluate whether current regime config values are appropriate and propose adjustments

## INTER-STRATEGY COORDINATION
If coordination_rules data is present:
- Evaluate whether coordination signals fired correctly (e.g., ATRSS entry → AKC_HELIX
  stop tightening). Did the coordination improve or hurt outcomes?
- Check cooldown pair behavior — did cooldowns prevent good setups or correctly block whipsaws?
- Assess direction filter agreement rates and quality of filtered trades
- Review stock_coordination for symbol collision events and sizing adjustments

## ARCHETYPE-RELATIVE EVALUATION
If strategy_profiles and archetype_expectations data are present:
- Evaluate each strategy against its archetype's expected ranges, not universal benchmarks
- Trend-followers with 40% win rate and 2.5R payoff are HEALTHY — do not suggest
  tightening stops to improve win rate
- Breakout strategies with 35% win rate are NORMAL — focus on cost-per-attempt
- Flag strategies performing below archetype floor in their PREFERRED regime as problematic
- Strategies underperforming in ADVERSE regimes is EXPECTED — do not propose changes
- Use portfolio_risk_config to validate that suggestions stay within risk bounds
- For strategies with `sub_engines` in their profile, compare performance across engines
  to identify which engines perform best in each regime/vol state combination.
- For strategies with `entry_types`, compare entry type win rates and payoff ratios.
- Reference the strategy's `analysis_focus` list for priority analytical dimensions.
- For mean_reversion_pullback archetype: high win rate + low payoff is expected —
  flag if win rate drops below archetype floor or if average loss exceeds 1.5x average win.

## SELF-ASSESSMENT
If self_assessment data is present, READ IT CAREFULLY. This summarizes your known
biases, weak categories, and recurring mistakes. You MUST:
- Acknowledge biases before making predictions in affected metrics
- Avoid or explicitly justify suggestions in weak categories
- Not repeat patterns listed in recurring corrections

## CONSTRAINTS (enforced by validator — violations are automatically stripped)
- Do NOT restate the computed summary — it's above.
- Focus analytical effort on the questions, not on re-summarizing data.
- BLOCKED: NEGATIVE outcome_measurements categories — do NOT re-suggest similar approaches.
- BLOCKED: structural proposals without acceptance_criteria with measurable metrics.
- BLOCKED: hypotheses with effectiveness <= 0 or status="retired" — do NOT re-propose.
- Overconfident predictions are capped by forecast_meta_analysis calibration data.
- outcome_measurements contains only HIGH/MEDIUM quality data. spurious_outcomes
  (if present) had confounding factors (concurrent changes, regime shifts,
  low/insufficient measurement quality) — treat as hypotheses, not evidence.

## STRUCTURED OUTPUT (REQUIRED)
At the END of your analysis, emit a structured data block.
CRITICAL: This block is machine-parsed by the learning system. If you omit it,
your suggestions and predictions are LOST and cannot improve future performance.
Always emit it, even if arrays are empty.
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
  ],
  "portfolio_proposals": [
    {{"proposal_type": "allocation_rebalance|risk_cap_change|coordination_change|drawdown_tier_change", "current_config": {{}}, "proposed_config": {{}}, "evidence_summary": "cite specific family metrics, correlation data, and time period", "expected_portfolio_calmar_delta": 0.0, "confidence": 0.0-1.0, "observation_window_days": 30}}
  ]
}}
-->"""

# Legacy instructions for when no triage is provided
_WEEKLY_INSTRUCTIONS = _FOCUSED_WEEKLY_INSTRUCTIONS.format(
    computed_summary="(No triage data — compute weekly summary manually from daily reports)",
    anomalies="(No triage — review all weekly data for structural patterns)",
    retrospective_questions="Review past predictions and suggestions. Which were accurate and which were not? Why?",
    discovery_questions="What patterns in this week's data might the automated detectors miss?",
)


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
        bot_configs: dict | None = None,
        strategy_registry=None,
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.runs_dir = runs_dir
        self.bot_configs = bot_configs
        self.strategy_registry = strategy_registry
        self._ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)

    def assemble(self, triage_report=None, session_store=None) -> PromptPackage:
        """Build the complete weekly prompt package.

        Args:
            triage_report: Optional WeeklyTriageReport from WeeklyTriage. When
                provided, instructions are focused on computed summaries and
                targeted questions. When None, uses fallback instructions.
            session_store: Optional SessionStore for loading session history.
        """
        pkg = self._ctx.base_package(
            session_store=session_store,
            agent_type="weekly_analysis",
            bot_configs=self.bot_configs,
            strategy_registry=self.strategy_registry,
        )
        pkg.task_prompt = self._build_task_prompt()
        pkg.data.update(self._load_data())
        pkg.instructions = self._build_instructions(triage_report)
        pkg.context_files.extend(self._list_data_files())
        return pkg

    def _build_instructions(self, triage_report=None) -> str:
        """Build instructions from triage report or use fallback."""
        if triage_report is None:
            return _WEEKLY_INSTRUCTIONS

        # Format anomalies
        anomaly_lines = []
        for i, a in enumerate(triage_report.anomalies, 1):
            bot_tag = f"[{a.bot_id}] " if a.bot_id else ""
            anomaly_lines.append(
                f"{i}. **[{a.anomaly_type.upper()}]** {bot_tag}"
                f"(severity: {a.severity}) — {a.description}"
            )
        anomalies_text = "\n".join(anomaly_lines) if anomaly_lines else "(No anomalies detected — stable week)"

        # Format retrospective questions
        retro_lines = []
        for i, q in enumerate(triage_report.retrospective_questions, 1):
            retro_lines.append(f"{i}. {q}")
        retro_text = "\n".join(retro_lines)

        # Format discovery questions
        disc_lines = []
        for i, q in enumerate(triage_report.discovery_questions, 1):
            disc_lines.append(f"{i}. {q}")
        disc_text = "\n".join(disc_lines)

        return _FOCUSED_WEEKLY_INSTRUCTIONS.format(
            computed_summary=triage_report.computed_summary,
            anomalies=anomalies_text,
            retrospective_questions=retro_text,
            discovery_questions=disc_text,
        )

    def _build_task_prompt(self) -> str:
        bot_list = ", ".join(self.bots)
        return (
            f"Produce the weekly summary for {self.week_start} to {self.week_end} "
            f"covering all bots: {bot_list}.\n"
            f"Focus on the retrospective and discovery questions. "
            f"Reason about WHY things happened, not just WHAT happened."
        )

    def _load_data(self) -> dict:
        data: dict = {}

        weekly_dir = self.curated_dir / "weekly" / self.week_start
        summary_path = weekly_dir / "weekly_summary.json"
        if summary_path.exists():
            data["weekly_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))

        refinement_path = weekly_dir / "refinement_report.json"
        if refinement_path.exists():
            data["refinement_report"] = json.loads(refinement_path.read_text(encoding="utf-8"))

        wow_path = weekly_dir / "week_over_week.json"
        if wow_path.exists():
            data["week_over_week"] = json.loads(wow_path.read_text(encoding="utf-8"))

        data["daily_reports"] = self._load_daily_reports()
        data["portfolio_risk_cards"] = self._load_risk_cards()

        # Load allocation analysis if present
        alloc_path = weekly_dir / "allocation_analysis.json"
        if alloc_path.exists():
            data["allocation_analysis"] = json.loads(alloc_path.read_text(encoding="utf-8"))

        # Load structural analysis if present
        structural_path = weekly_dir / "structural_analysis.json"
        if structural_path.exists():
            data["structural_analysis"] = json.loads(structural_path.read_text(encoding="utf-8"))

        # Load regime-conditional analysis if present
        regime_path = weekly_dir / "regime_conditional_analysis.json"
        if regime_path.exists():
            data["regime_conditional_analysis"] = json.loads(regime_path.read_text(encoding="utf-8"))

        # Load interaction analysis if present
        interaction_path = weekly_dir / "interaction_analysis.json"
        if interaction_path.exists():
            data["interaction_analysis"] = json.loads(interaction_path.read_text(encoding="utf-8"))

        # Load allocation drift analysis if present
        drift_path = weekly_dir / "allocation_drift.json"
        if drift_path.exists():
            data["allocation_drift"] = json.loads(drift_path.read_text(encoding="utf-8"))

        return data

    def _load_daily_reports(self) -> list[dict]:
        reports: list[dict] = []
        for date_str in self._week_dates():
            report_paths = self._find_daily_report_paths(date_str)
            if report_paths:
                reports.append({
                    "date": date_str,
                    "content": "\n\n".join(
                        path.read_text(encoding="utf-8") for path in report_paths
                    ),
                })
        return reports

    def _load_risk_cards(self) -> list[dict]:
        cards: list[dict] = []
        for date_str in self._week_dates():
            card_path = self.curated_dir / date_str / "portfolio_risk_card.json"
            if card_path.exists():
                cards.append(json.loads(card_path.read_text(encoding="utf-8")))
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
        for date_str in self._week_dates():
            for report_path in self._find_daily_report_paths(date_str):
                files.append(str(report_path))
        return files

    def _week_dates(self) -> list[str]:
        """Generate the 7 date strings for this week."""
        start = datetime.strptime(self.week_start, "%Y-%m-%d")
        return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

    def _find_daily_report_paths(self, date_str: str) -> list[Path]:
        paths: list[Path] = []
        for run_dir in sorted(self.runs_dir.glob(f"daily-{date_str}*")):
            if not run_dir.is_dir():
                continue
            report_path = run_dir / "daily_report.md"
            if report_path.exists():
                paths.append(report_path)
                continue
            fallback = run_dir / "response.md"
            if fallback.exists():
                paths.append(fallback)
        return paths

    def _build_computed_summary(self, triage_report) -> str:
        """Extract computed summary from triage report."""
        if not triage_report:
            return ""
        return triage_report.computed_summary
