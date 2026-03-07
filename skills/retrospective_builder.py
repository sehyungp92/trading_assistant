# skills/retrospective_builder.py
"""Weekly retrospective — compares last week's predictions/warnings to actual outcomes.

Deterministic pipeline. Produces a structured retrospective that Claude can use
to calibrate future analysis quality.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel


class PredictionOutcome(BaseModel):
    """A single prediction from a past report matched against actual outcome."""

    prediction_source: str  # "daily-2026-03-01" or "weekly-2026-02-24"
    prediction_text: str
    prediction_type: str  # "risk_warning", "suggestion", "regime_call", "alert"
    actual_outcome: str = ""  # What actually happened
    accuracy: str = ""  # "correct", "partially_correct", "incorrect", "unverifiable"


class WeeklyRetrospective(BaseModel):
    """Summary of prediction accuracy for the past week."""

    week_start: str
    week_end: str
    predictions_reviewed: int = 0
    correct: int = 0
    partially_correct: int = 0
    incorrect: int = 0
    unverifiable: int = 0
    accuracy_pct: float = 0.0
    predictions: list[PredictionOutcome] = []
    summary: str = ""


_SUGGESTION_MATCHERS: dict[str, dict] = {
    "stop": {
        "metrics": ["avg_mae_pct", "exit_efficiency"],
        "positive_direction": "decrease_mae_or_increase_efficiency",
    },
    "filter": {
        "metrics": ["missed_would_have_won"],
        "positive_direction": "decrease_missed_winners",
    },
    "regime": {
        "metrics": ["regime_pnl"],
        "positive_direction": "improvement_in_flagged_regime",
    },
    "hour": {
        "metrics": ["hourly_performance"],
        "positive_direction": "improvement_in_flagged_hours",
    },
    "position_siz": {
        "metrics": ["avg_win", "avg_loss"],
        "positive_direction": "loss_win_ratio_decrease",
    },
    "drawdown": {
        "metrics": ["max_drawdown_pct"],
        "positive_direction": "decrease",
    },
    "allocation": {
        "metrics": ["net_pnl"],
        "positive_direction": "increase",
    },
}


class RetrospectiveBuilder:
    """Builds a retrospective by comparing past reports to actual outcomes."""

    def __init__(
        self,
        runs_dir: Path,
        curated_dir: Path,
        memory_dir: Path,
    ) -> None:
        self._runs_dir = runs_dir
        self._curated_dir = curated_dir
        self._memory_dir = memory_dir

    def build(self, week_start: str, week_end: str) -> WeeklyRetrospective:
        """Build retrospective for the given week.

        Loads past daily/weekly reports from runs/ and compares predictions
        and warnings to actual outcomes from curated data.  Uses week-over-week
        deltas (prior week vs current week) rather than absolute thresholds
        for more precise accuracy assessment.
        """
        start = datetime.strptime(week_start, "%Y-%m-%d")
        end = datetime.strptime(week_end, "%Y-%m-%d")

        predictions = self._extract_predictions(start, end)
        outcomes = self._load_actual_outcomes(start, end)

        # Load prior-week outcomes for delta-based accuracy assessment
        prior_start = start - timedelta(days=7)
        prior_end = start - timedelta(days=1)
        prior_outcomes = self._load_actual_outcomes(prior_start, prior_end)

        # Match predictions to outcomes
        matched: list[PredictionOutcome] = []
        for pred in predictions:
            pred.actual_outcome = self._find_matching_outcome(pred, outcomes)
            pred.accuracy = self._assess_accuracy(pred, prior_outcomes, outcomes)
            matched.append(pred)

        correct = sum(1 for p in matched if p.accuracy == "correct")
        partial = sum(1 for p in matched if p.accuracy == "partially_correct")
        incorrect = sum(1 for p in matched if p.accuracy == "incorrect")
        unverifiable = sum(1 for p in matched if p.accuracy == "unverifiable")
        total = len(matched)

        accuracy_pct = (
            (correct + 0.5 * partial) / total * 100.0
            if total > 0
            else 0.0
        )

        return WeeklyRetrospective(
            week_start=week_start,
            week_end=week_end,
            predictions_reviewed=total,
            correct=correct,
            partially_correct=partial,
            incorrect=incorrect,
            unverifiable=unverifiable,
            accuracy_pct=round(accuracy_pct, 1),
            predictions=matched,
            summary=self._build_summary(correct, partial, incorrect, unverifiable, total),
        )

    def _extract_predictions(
        self, start: datetime, end: datetime,
    ) -> list[PredictionOutcome]:
        """Extract predictions from past daily report runs."""
        predictions: list[PredictionOutcome] = []

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            run_dir = self._runs_dir / f"daily-{date_str}"
            if run_dir.is_dir():
                predictions.extend(self._parse_run_predictions(run_dir, f"daily-{date_str}"))
            current += timedelta(days=1)

        # Also check previous weekly report
        prev_week = (start - timedelta(days=7)).strftime("%Y-%m-%d")
        weekly_run = self._runs_dir / f"weekly-{prev_week}"
        if weekly_run.is_dir():
            predictions.extend(self._parse_run_predictions(weekly_run, f"weekly-{prev_week}"))

        return predictions

    def _parse_run_predictions(
        self, run_dir: Path, source: str,
    ) -> list[PredictionOutcome]:
        """Parse predictions from a run directory's output files."""
        predictions: list[PredictionOutcome] = []

        # Look for structured output (suggestions, warnings)
        for output_file in run_dir.glob("*.json"):
            try:
                data = json.loads(output_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Extract suggestions
                    for suggestion in data.get("suggestions", []):
                        text = suggestion if isinstance(suggestion, str) else str(suggestion)
                        predictions.append(PredictionOutcome(
                            prediction_source=source,
                            prediction_text=text[:500],
                            prediction_type="suggestion",
                        ))
                    # Extract warnings/alerts
                    for warning in data.get("warnings", data.get("risk_warnings", [])):
                        text = warning if isinstance(warning, str) else str(warning)
                        predictions.append(PredictionOutcome(
                            prediction_source=source,
                            prediction_text=text[:500],
                            prediction_type="risk_warning",
                        ))
            except (json.JSONDecodeError, OSError):
                pass

        return predictions

    def _load_actual_outcomes(
        self, start: datetime, end: datetime,
    ) -> dict[str, list[dict]]:
        """Load actual outcomes from curated data for comparison.

        Loads summary.json plus additional metric files (exit_efficiency.json,
        filter_analysis.json, regime_analysis.json) for richer accuracy matching.
        """
        outcomes: dict[str, list[dict]] = {}
        _EXTRA_FILES = [
            "exit_efficiency.json",
            "filter_analysis.json",
            "regime_analysis.json",
            "hourly_performance.json",
        ]

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            date_dir = self._curated_dir / date_str
            if date_dir.is_dir():
                for bot_dir in date_dir.iterdir():
                    if bot_dir.is_dir():
                        summary_file = bot_dir / "summary.json"
                        if summary_file.exists():
                            try:
                                data = json.loads(summary_file.read_text(encoding="utf-8"))
                                # Merge additional metric files into the same dict
                                for extra in _EXTRA_FILES:
                                    extra_path = bot_dir / extra
                                    if extra_path.exists():
                                        try:
                                            key = extra.replace(".json", "")
                                            data[key] = json.loads(
                                                extra_path.read_text(encoding="utf-8")
                                            )
                                        except (json.JSONDecodeError, OSError):
                                            pass
                                outcomes.setdefault(date_str, []).append(data)
                            except (json.JSONDecodeError, OSError):
                                pass
            current += timedelta(days=1)

        return outcomes

    def _find_matching_outcome(
        self, prediction: PredictionOutcome, outcomes: dict,
    ) -> str:
        """Find the actual outcome matching a prediction using keyword→metric matching."""
        if not outcomes:
            return "No outcome data available for comparison"

        text_lower = prediction.prediction_text.lower()

        # Try keyword→metric matching first
        for keyword, matcher in _SUGGESTION_MATCHERS.items():
            if keyword not in text_lower:
                continue
            metric_values = self._extract_metric_values(outcomes, matcher["metrics"])
            if metric_values:
                parts = [f"Matched metrics for '{keyword}':"]
                for metric_name, value in metric_values.items():
                    parts.append(f"  {metric_name}={value:.4f}")
                return " ".join(parts)

        # Fallback: aggregate summary
        total_pnl = 0.0
        total_trades = 0
        for date_outcomes in outcomes.values():
            for outcome in date_outcomes:
                total_pnl += outcome.get("net_pnl", 0.0)
                total_trades += outcome.get("total_trades", 0)

        return f"Week total: {total_trades} trades, ${total_pnl:.2f} net PnL"

    @staticmethod
    def _extract_metric_values(
        outcomes: dict[str, list[dict]], metric_keys: list[str],
    ) -> dict[str, float]:
        """Extract average metric values across all outcome entries."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}

        for date_outcomes in outcomes.values():
            for outcome in date_outcomes:
                for key in metric_keys:
                    # Check top-level
                    val = outcome.get(key)
                    if isinstance(val, (int, float)):
                        totals[key] = totals.get(key, 0.0) + val
                        counts[key] = counts.get(key, 0) + 1
                        continue
                    # Check nested dicts (exit_efficiency, filter_analysis, etc.)
                    for sub_key, sub_val in outcome.items():
                        if isinstance(sub_val, dict) and key in sub_val:
                            nested = sub_val[key]
                            if isinstance(nested, (int, float)):
                                totals[key] = totals.get(key, 0.0) + nested
                                counts[key] = counts.get(key, 0) + 1

        return {k: totals[k] / counts[k] for k in totals if counts.get(k, 0) > 0}

    def _assess_accuracy(
        self,
        prediction: PredictionOutcome,
        prior_outcomes: dict[str, list[dict]] | None = None,
        current_outcomes: dict[str, list[dict]] | None = None,
    ) -> str:
        """Assess prediction accuracy using week-over-week metric deltas.

        Compares metric values from the week the prediction was made (prior)
        against the following week (current).  Falls back to absolute thresholds
        when prior-week data is unavailable.

        Classification logic (delta-based):
        - If metric improved >= 5% in the positive direction: "correct"
        - If metric improved > 0% but < 5%: "partially_correct"
        - If metric degraded > 0%: "incorrect"
        - If no data for comparison: "unverifiable"
        """
        if not prediction.actual_outcome or "No outcome data" in prediction.actual_outcome:
            return "unverifiable"

        # Check if the outcome was generated via keyword matching
        if not prediction.actual_outcome.startswith("Matched metrics"):
            return "unverifiable"

        text_lower = prediction.prediction_text.lower()

        # Find which keyword matched
        matched_keyword = None
        for keyword in _SUGGESTION_MATCHERS:
            if keyword in text_lower:
                matched_keyword = keyword
                break

        if not matched_keyword:
            return "unverifiable"

        matcher = _SUGGESTION_MATCHERS[matched_keyword]
        metric_keys = matcher["metrics"]
        direction = matcher["positive_direction"]

        # Try delta-based assessment first
        if prior_outcomes and current_outcomes:
            prior_vals = self._extract_metric_values(prior_outcomes, metric_keys)
            current_vals = self._extract_metric_values(current_outcomes, metric_keys)
            if prior_vals and current_vals:
                return self._classify_by_delta(direction, prior_vals, current_vals)

        # Fallback to absolute thresholds when prior data unavailable
        metric_values = self._parse_outcome_metrics(prediction.actual_outcome)
        if not metric_values:
            return "unverifiable"

        return self._classify_by_direction(direction, metric_values)

    @staticmethod
    def _parse_outcome_metrics(outcome_str: str) -> dict[str, float]:
        """Parse metric name=value pairs from an outcome string."""
        import re
        metrics: dict[str, float] = {}
        for match in re.finditer(r"(\w+)=([-+]?\d*\.?\d+)", outcome_str):
            metrics[match.group(1)] = float(match.group(2))
        return metrics

    @staticmethod
    def _classify_by_direction(direction: str, metrics: dict[str, float]) -> str:
        """Classify accuracy based on metric direction and 5% threshold."""
        # For each direction type, determine if metrics indicate improvement
        # We use the first available metric value as the signal

        values = list(metrics.values())
        if not values:
            return "unverifiable"

        primary = values[0]

        if direction == "decrease_mae_or_increase_efficiency":
            # For stop-related: lower MAE or higher exit efficiency is good
            # If exit_efficiency > 0.5 → likely improving (partially_correct)
            if primary > 0.55:
                return "partially_correct"
            elif primary < 0.3:
                return "incorrect"
            return "unverifiable"

        if direction == "decrease_missed_winners":
            # Lower missed_would_have_won count is good
            # We only have the absolute value, not delta — use threshold
            if primary <= 2:
                return "partially_correct"
            elif primary >= 10:
                return "incorrect"
            return "unverifiable"

        if direction in ("decrease", "loss_win_ratio_decrease"):
            # Lower is better — if metric is below threshold, improvement
            if primary < 0.95:
                return "partially_correct"
            elif primary > 1.05:
                return "incorrect"
            return "unverifiable"

        if direction == "increase":
            # Higher is better (net_pnl)
            if primary > 0:
                return "partially_correct"
            elif primary < 0:
                return "incorrect"
            return "unverifiable"

        if direction in ("improvement_in_flagged_regime", "improvement_in_flagged_hours"):
            # Positive PnL → improvement
            if primary > 0:
                return "partially_correct"
            elif primary < 0:
                return "incorrect"
            return "unverifiable"

        return "unverifiable"

    @staticmethod
    def _classify_by_delta(
        direction: str,
        prior: dict[str, float],
        current: dict[str, float],
    ) -> str:
        """Classify accuracy based on week-over-week metric delta.

        Uses the first metric available in both periods.  A >=5% improvement
        in the positive direction → "correct"; >0% → "partially_correct";
        degradation → "incorrect".
        """
        # Find a common metric between prior and current
        common_keys = set(prior) & set(current)
        if not common_keys:
            return "unverifiable"

        key = sorted(common_keys)[0]
        old_val = prior[key]
        new_val = current[key]

        # Compute relative change; avoid division by zero
        if old_val == 0.0:
            if new_val == 0.0:
                return "unverifiable"
            delta_pct = 100.0 if new_val > 0 else -100.0
        else:
            delta_pct = ((new_val - old_val) / abs(old_val)) * 100.0

        # Determine if the delta direction is "positive" per the matcher
        positive_means_increase = direction in (
            "increase",
            "improvement_in_flagged_regime",
            "improvement_in_flagged_hours",
            "decrease_mae_or_increase_efficiency",
        )
        positive_means_decrease = direction in (
            "decrease",
            "decrease_missed_winners",
            "loss_win_ratio_decrease",
        )

        if positive_means_decrease:
            delta_pct = -delta_pct  # flip: a decrease is positive

        if not positive_means_increase and not positive_means_decrease:
            # Unknown direction — fall back
            return "unverifiable"

        if delta_pct >= 5.0:
            return "correct"
        elif delta_pct > 0.0:
            return "partially_correct"
        elif delta_pct < -5.0:
            return "incorrect"
        else:
            return "unverifiable"

    def _build_summary(
        self,
        correct: int,
        partial: int,
        incorrect: int,
        unverifiable: int,
        total: int,
    ) -> str:
        if total == 0:
            return "No predictions to review this week."
        return (
            f"Reviewed {total} predictions: "
            f"{correct} correct, {partial} partially correct, "
            f"{incorrect} incorrect, {unverifiable} unverifiable. "
            f"Ask Claude to assess accuracy of unverifiable predictions "
            f"given the actual outcome data."
        )
