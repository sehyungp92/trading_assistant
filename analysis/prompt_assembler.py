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
    "excursion_stats.json",
    "overlay_state_summary.json",
    "experiment_breakdown.json",
    "signal_health.json",
    "fill_quality.json",
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
6. Overlay analysis: when overlay_state_summary.json is present (swing_trader only),
   analyze the EMA crossover overlay's contribution.
   - Review overlay P&L relative to main strategy P&L
   - Correlate overlay state transitions with main strategy outcomes (informational only)
   - The overlay does NOT gate main strategy signals — it deploys idle capital independently
   - Flag if overlay positions are consistently losing during periods when main strategies are profitable
7. Excursion analysis: when excursion_stats.json is present, review MFE/MAE distributions.
   - Compare winner vs loser MFE/MAE profiles
   - High MAE on winners suggests stop placement is too tight
   - Low exit_efficiency (<50%) across winners suggests premature exits
   - This is bot-provided intra-trade data and supplements the post-exit proxy in exit_efficiency.json
8. Experiment breakdown (swing_trader): when experiment_breakdown.json is present,
   compare variant PnL, win_rate, and Sharpe across experiment groups.
   Flag underperforming variants. Note sample sizes — small samples reduce confidence.
9. Signal health (momentum_trader): when signal_health.json is present,
   review per-component metrics. Flag components with low stability (<0.5),
   low win_correlation (<0.1), or negative trend_during_trade on winning trades.
   These indicate signal components losing predictive power.
10. Fill quality (momentum_trader): when fill_quality.json is present,
   review entry vs exit slippage distributions. Flag adverse_selection_detected.
   Check per-symbol breakdown for outlier symbols. Quantify annualized slippage cost
   (avg_slippage_bps × estimated annual trade count × avg position size).
11. Actionable items: max 3 specific, testable suggestions backed by factor attribution or exit efficiency data
   QUANTIFICATION REQUIRED: Every suggestion MUST include:
   (a) Expected return impact with a range (e.g., "+0.3% to +0.8% daily PnL")
   (b) Drawdown impact estimate (e.g., "max drawdown reduction of ~15%")
   (c) Evidence base: trade count, time period, and statistical significance
   Suggestions without quantification will be rejected.
12. Open risks: any CRITICAL/HIGH events that need human attention
13. Output: daily_report.md + report_checklist.json
14. Check the rejected_suggestions list (if present). Do NOT re-suggest anything that was previously rejected unless you have new evidence.
15. Contradiction check: If 'contradictions' data is present, review each flagged item.
    Assess whether each contradiction indicates a genuine issue or an acceptable regime
    transition. Address each contradiction explicitly — do not ignore them silently.
16. Signal factor trends: If 'signal_factor_trends' data is present, review each factor's
    rolling 30d metrics. Factors with 'degrading' trend or 'below_threshold: true' should
    be explicitly called out as candidates for recalibration. Compare against today's
    factor_attribution data.
17. Correction patterns: If 'correction_patterns' data is present, review recurring human
    corrections. Avoid repeating these mistakes — adapt your analysis to address known blind spots.
18. Reference outcome_measurements (if present) when making suggestions. Only make
    high-confidence suggestions for approaches with proven POSITIVE track records.
19. Review active_suggestions (if present). Don't contradict IMPLEMENTED suggestions.
    For PROPOSED suggestions, note any supporting or contradicting evidence from today.
20. Review category_scorecard (if present). Categories with win_rate < 30% and sample_size >= 5
    require exceptional new evidence to justify suggestions in that category.
21. Review prediction_accuracy_by_metric (if present). For metrics where accuracy < 40%,
    explicitly caveat predictions and lower confidence.
22. Review failure_log (if present). Avoid analysis approaches that have previously failed.
23. Review consolidated_patterns (if present). These are systemic patterns discovered across
    findings — use them to ground your structural analysis.
24. Review hypothesis_track_record (if present). Prioritize hypotheses with positive
    effectiveness. Do not propose retired hypotheses.
25. Review validation_patterns (if present). These are recurring blocked suggestion categories —
    avoid proposing suggestions in categories that are consistently blocked.
26. STRUCTURED OUTPUT (REQUIRED): At the END of your analysis, emit a structured data block.
    This block is machine-parsed — do NOT omit it.
    <!-- STRUCTURED_OUTPUT
    {
      "predictions": [
        {"bot_id": "...", "metric": "pnl|win_rate|drawdown|sharpe", "direction": "improve|decline|stable", "confidence": 0.0-1.0, "timeframe_days": 7, "reasoning": "..."}
      ],
      "suggestions": [
        {"suggestion_id": "#abc123", "bot_id": "...", "category": "exit_timing|filter_threshold|stop_loss|signal|structural|position_sizing|regime_gate", "title": "...", "expected_impact": "...", "confidence": 0.0-1.0, "evidence_summary": "..."}
      ],
      "structural_proposals": [
        {"hypothesis_id": "REQUIRED: use id from structural_hypotheses if matching, else null", "bot_id": "...", "title": "...", "description": "...", "reversibility": "easy|moderate|hard", "evidence": "...", "estimated_complexity": "low|medium|high"}
      ]
    }
    -->"""


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

        # Inject contradiction data if any
        contradictions = self._ctx.load_contradictions(self.date, self.bots, self.curated_dir)
        if contradictions:
            pkg.data["contradictions"] = contradictions

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
        findings_dir = self.memory_dir / "findings"

        for bot in self.bots:
            bot_dir = date_dir / bot
            bot_data: dict = {}
            for filename in _CURATED_FILES:
                path = bot_dir / filename
                if path.exists():
                    key = filename.replace(".json", "")
                    bot_data[key] = json.loads(path.read_text())

            # Inject signal factor rolling trends per bot
            factor_trends = self._ctx.load_signal_factor_history(bot, self.date, findings_dir)
            if factor_trends:
                bot_data["signal_factor_trends"] = factor_trends

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
