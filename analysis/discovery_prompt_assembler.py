# analysis/discovery_prompt_assembler.py
"""Discovery prompt assembler — builds context for the discovery agent type.

Unlike daily/weekly analysis, the discovery agent gets access to raw trade
data (JSONL) and higher max_turns for iterative exploration. The goal is to
find patterns NOT covered by the 7 automated detectors.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from analysis.context_builder import ContextBuilder
from schemas.prompt_package import PromptPackage

_DISCOVERY_INSTRUCTIONS = """\
You are exploring raw trade data to discover patterns that the automated
detectors miss. You have access to raw JSONL trade files via Read/Grep/Glob.

## CONTEXT
The system has 7 automated detectors:
1. Alpha decay (signal correlation degradation)
2. Signal decay (win rate decline)
3. Exit timing issues (premature exits)
4. Correlation breakdown (cross-bot)
5. Time-of-day patterns
6. Drawdown patterns
7. Position sizing issues

Your job is to find patterns OUTSIDE this coverage.

## DATA ACCESS
Raw trade data is available in the curated directories listed in context_files.
Use Read/Grep/Glob to explore the JSONL files directly. Each line is a JSON
trade record with fields like: pnl, signal_strength, regime, entry_time,
exit_time, mae_pct, mfe_pct, position_size_pct, root_cause, etc.

## METHODOLOGY
1. Start with a broad scan: distribution shapes, outlier clusters, temporal patterns
2. Form a hypothesis about a pattern you observe
3. Test it: filter for confirming AND refuting data
4. If the pattern holds with >= 5 supporting data points, record it as a discovery

## ANTI-PATTERNS (do NOT do these)
- Do NOT restate curated summary statistics — those are already computed
- Do NOT report patterns with fewer than 5 supporting trades
- Do NOT report patterns that ARE covered by the 7 detectors above
- Do NOT report obvious observations (e.g., "winners have positive PnL")

## OUTPUT FORMAT
For each discovery, provide:
- Pattern description (1-2 sentences)
- Evidence: list of specific trade references (date, trade_id, PnL)
- Proposed root cause (from existing taxonomy or flag as "novel")
- Testable hypothesis that can be validated with future data
- Confidence rating (0.0-1.0) with justification
- Which automated detector relates (or "novel" if none)

## HYPOTHESIS CONTEXT
If existing_hypotheses data is present, check effectiveness scores before
proposing related structural changes. Prioritize hypotheses with positive
track records. Do NOT reinvestigate retired hypotheses (effectiveness <= 0).

## ACTIVE EXPERIMENTS
If active_experiments data is present, do NOT propose changes that conflict
with running experiments. Note any observations relevant to active experiments.

## STRATEGY IDEATION
Beyond pattern detection, consider whether the patterns you find suggest
an entirely new trading approach. A strategy idea should:
- Have a clear edge hypothesis (WHY would this work?)
- Be grounded in >= 10 supporting data points from the raw data
- Specify entry logic, exit logic, and applicable regimes
- NOT be a minor tweak to existing strategies — this is for genuinely new approaches

Only propose a strategy idea if you found something truly novel and well-supported.
Most discovery runs will NOT produce a strategy idea — that's expected.

## STRUCTURED OUTPUT (REQUIRED)
At the END, emit a structured data block:
<!-- STRUCTURED_OUTPUT
{
  "discoveries": [
    {
      "pattern_description": "...",
      "evidence": [{"date": "...", "bot_id": "...", "trade_id": "...", "pnl": 0.0, "signal_strength": 0.0, "regime": "...", "note": "..."}],
      "proposed_root_cause": "...",
      "testable_hypothesis": "...",
      "confidence": 0.0,
      "detector_coverage": "novel|alpha_decay|signal_decay|exit_timing|correlation|time_of_day|drawdown|position_sizing",
      "bot_id": "..."
    }
  ],
  "strategy_ideas": [
    {
      "title": "...",
      "description": "...",
      "edge_hypothesis": "...",
      "evidence": [{"date": "...", "bot_id": "...", "trade_id": "...", "pnl": 0.0, "regime": "...", "note": "..."}],
      "entry_logic": "...",
      "exit_logic": "...",
      "applicable_regimes": ["..."],
      "applicable_bots": ["..."],
      "confidence": 0.0
    }
  ],
  "structural_proposals": [
    {
      "hypothesis_id": "use id from existing_hypotheses if matching, else null",
      "bot_id": "...",
      "title": "...",
      "description": "...",
      "reversibility": "easy|moderate|hard",
      "evidence": "...",
      "estimated_complexity": "low|medium|high",
      "acceptance_criteria": [{"metric": "...", "direction": "improve|not_degrade", "minimum_change": 0.0, "observation_window_days": 14, "minimum_trade_count": 20}]
    }
  ]
}
-->"""


class DiscoveryPromptAssembler:
    """Assembles context for the discovery agent — raw data + exploration tools."""

    def __init__(
        self,
        date: str,
        bots: list[str],
        curated_dir: Path,
        memory_dir: Path,
        lookback_days: int = 30,
        bot_configs: dict | None = None,
    ) -> None:
        self.date = date
        self.bots = bots
        self.curated_dir = curated_dir
        self.memory_dir = memory_dir
        self.lookback_days = lookback_days
        self.bot_configs = bot_configs
        self._ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)

    def assemble(self) -> PromptPackage:
        """Build the complete prompt package for discovery analysis."""
        pkg = self._ctx.base_package(bot_configs=self.bot_configs)
        pkg.task_prompt = self._build_task_prompt()
        pkg.instructions = _DISCOVERY_INSTRUCTIONS
        pkg.data.update(self._load_discovery_context())
        pkg.context_files.extend(self._list_raw_data_files())
        return pkg

    def _build_task_prompt(self) -> str:
        bot_list = ", ".join(self.bots)
        return (
            f"Explore the last {self.lookback_days} days of raw trade data for bots: {bot_list}.\n"
            f"Find patterns not covered by the 7 automated detectors.\n"
            f"Use Read/Grep/Glob tools to examine the JSONL trade files listed in context_files.\n"
            f"Reference date: {self.date}."
        )

    def _load_discovery_context(self) -> dict:
        """Load lightweight context for discovery — existing detector findings."""
        data: dict = {}

        # Load latest regime analysis for context
        for bot in self.bots:
            bot_dir = self.curated_dir / self.date / bot
            regime_path = bot_dir / "regime_analysis.json"
            if regime_path.exists():
                try:
                    data.setdefault("regime_context", {})[bot] = json.loads(
                        regime_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError):
                    pass

        # Load hypothesis library for reference
        hypothesis_track = self._ctx.load_hypothesis_track_record()
        if hypothesis_track:
            data["existing_hypotheses"] = hypothesis_track

        # Load active experiments to avoid conflicts
        active_experiments = self._ctx.load_active_experiments()
        if active_experiments:
            data["active_experiments"] = active_experiments

        # Load experiment track record for calibration
        experiment_track_record = self._ctx.load_experiment_track_record()
        if experiment_track_record:
            data["experiment_track_record"] = experiment_track_record

        return data

    def _list_raw_data_files(self) -> list[str]:
        """List paths to raw JSONL trade files for the lookback window."""
        files: list[str] = []
        end = datetime.strptime(self.date, "%Y-%m-%d")
        for d in range(self.lookback_days):
            date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
            for bot in self.bots:
                trades_path = self.curated_dir / date_str / bot / "trades.jsonl"
                if trades_path.exists():
                    files.append(str(trades_path))
                missed_path = self.curated_dir / date_str / bot / "missed.jsonl"
                if missed_path.exists():
                    files.append(str(missed_path))
        return files
