# skills/prediction_tracker.py
"""PredictionTracker — records and evaluates structured predictions.

Storage: memory/findings/predictions.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

from schemas.agent_response import AgentPrediction
from schemas.prediction_tracking import (
    PredictionEvaluation,
    PredictionRecord,
    PredictionVerdict,
)


class PredictionTracker:
    def __init__(self, findings_dir: Path) -> None:
        self._path = findings_dir / "predictions.jsonl"

    def record_predictions(
        self, week: str, predictions: list[AgentPrediction],
    ) -> None:
        """Append structured predictions for a given week/date."""
        if not predictions:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            for p in predictions:
                record = PredictionRecord(
                    bot_id=p.bot_id,
                    metric=p.metric,
                    direction=p.direction,
                    confidence=p.confidence,
                    timeframe_days=p.timeframe_days,
                    reasoning=p.reasoning,
                    week=week,
                )
                f.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")

    def load_predictions(self, week: str | None = None) -> list[PredictionRecord]:
        """Load predictions, optionally filtered by week."""
        if not self._path.exists():
            return []
        records: list[PredictionRecord] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if week and data.get("week") != week:
                continue
            records.append(PredictionRecord(**data))
        return records

    def evaluate_predictions(
        self, week: str, curated_dir: Path, baseline_date: str = "",
    ) -> PredictionEvaluation:
        """Evaluate predictions for a given week against actual curated data.

        Compares predicted direction with the realized metric change between the
        prediction's baseline date and its target horizon.
        """
        predictions = self.load_predictions(week)
        if not predictions:
            return PredictionEvaluation(week=week)

        verdicts: list[PredictionVerdict] = []
        for pred in predictions:
            anchor_date = self._parse_date(baseline_date or pred.week)
            actual = self._load_actual_metric_change(
                pred.bot_id,
                pred.metric,
                curated_dir,
                anchor_date,
                pred.timeframe_days,
            )
            if actual is None:
                verdicts.append(PredictionVerdict(
                    bot_id=pred.bot_id,
                    metric=pred.metric,
                    predicted_direction=pred.direction,
                    confidence=pred.confidence,
                    status="insufficient_data",
                ))
                continue

            actual_direction = (
                "improve" if actual > 0
                else "decline" if actual < 0
                else "stable"
            )
            # For drawdown, "decline" means drawdown increased, which is bad
            # but direction logic stays consistent with the metric direction
            correct = pred.direction == actual_direction
            verdicts.append(PredictionVerdict(
                bot_id=pred.bot_id,
                metric=pred.metric,
                predicted_direction=pred.direction,
                actual_direction=actual_direction,
                correct=correct,
                confidence=pred.confidence,
                status="correct" if correct else "incorrect",
            ))

        # Compute aggregates
        evaluated = [v for v in verdicts if v.status != "insufficient_data"]
        total = len(evaluated)
        correct_count = sum(1 for v in evaluated if v.correct)
        accuracy = correct_count / total if total > 0 else 0.0

        # Confidence-weighted accuracy
        conf_sum = sum(v.confidence for v in evaluated) if evaluated else 0.0
        cw_accuracy = (
            sum(v.confidence * (1 if v.correct else 0) for v in evaluated) / conf_sum
            if conf_sum > 0 else 0.0
        )

        # Per-metric accuracy
        by_metric: dict[str, list[bool]] = defaultdict(list)
        for v in evaluated:
            by_metric[v.metric].append(v.correct)
        accuracy_by_metric = {
            m: sum(vals) / len(vals) for m, vals in by_metric.items()
        }

        return PredictionEvaluation(
            week=week,
            verdicts=verdicts,
            total=total,
            correct=correct_count,
            accuracy=round(accuracy, 3),
            confidence_weighted_accuracy=round(cw_accuracy, 3),
            accuracy_by_metric={m: round(v, 3) for m, v in accuracy_by_metric.items()},
        )

    def get_accuracy_by_metric(self, curated_dir: Path) -> dict[str, float]:
        """Compute rolling accuracy per metric type across all evaluated weeks."""
        if not self._path.exists():
            return {}

        # Get all unique weeks
        weeks: set[str] = set()
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                data = json.loads(line)
                w = data.get("week", "")
                if w:
                    weeks.add(w)

        # Aggregate across weeks
        all_by_metric: dict[str, list[bool]] = defaultdict(list)
        for week in sorted(weeks):
            evaluation = self.evaluate_predictions(week, curated_dir)
            for v in evaluation.verdicts:
                if v.status != "insufficient_data":
                    all_by_metric[v.metric].append(v.correct)

        return {
            m: round(sum(vals) / len(vals), 3)
            for m, vals in all_by_metric.items()
            if vals
        }

    def _load_actual_metric_change(
        self,
        bot_id: str,
        metric: str,
        curated_dir: Path,
        baseline_date: date | None,
        timeframe_days: int,
    ) -> float | None:
        """Load realized change between the baseline and target dates."""
        if baseline_date is None:
            return None

        baseline_value = self._load_metric_for_date(
            bot_id, metric, curated_dir, baseline_date,
        )
        target_value = self._load_metric_for_date(
            bot_id, metric, curated_dir, baseline_date + timedelta(days=timeframe_days),
        )
        if baseline_value is None or target_value is None:
            return None
        return target_value - baseline_value

    def _load_metric_for_date(
        self,
        bot_id: str,
        metric: str,
        curated_dir: Path,
        target_date: date,
        max_offset_days: int = 3,
    ) -> float | None:
        """Load the closest available metric near a target date.

        We prefer exact matches, then later dates, then earlier dates. This
        tolerates weekends/market holidays without making evaluations depend on
        the current date.
        """
        for candidate in self._candidate_dates(target_date, max_offset_days):
            summary_path = curated_dir / candidate.strftime("%Y-%m-%d") / bot_id / "summary.json"
            if not summary_path.exists():
                continue
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            metric_value = self._extract_metric(data, metric)
            if metric_value is not None:
                return metric_value
        return None

    @staticmethod
    def _candidate_dates(target_date: date, max_offset_days: int) -> list[date]:
        dates = [target_date]
        for offset in range(1, max_offset_days + 1):
            dates.append(target_date + timedelta(days=offset))
            dates.append(target_date - timedelta(days=offset))
        return dates

    @staticmethod
    def _parse_date(raw: str) -> date | None:
        if not raw:
            return None
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _extract_metric(summary: dict, metric: str) -> float | None:
        """Extract a metric value from a daily summary."""
        metric_keys = {
            "pnl": ["total_pnl", "pnl", "net_pnl"],
            "win_rate": ["win_rate", "win_pct"],
            "drawdown": ["max_drawdown_pct", "max_drawdown", "drawdown"],
            "sharpe": ["sharpe_rolling_30d", "sharpe", "sharpe_ratio"],
        }
        for key in metric_keys.get(metric, [metric]):
            val = summary.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None
