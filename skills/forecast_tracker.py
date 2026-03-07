# skills/forecast_tracker.py
"""ForecastTracker — records weekly accuracy and computes rolling meta-analysis.

Storage: memory/findings/forecast_history.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from schemas.forecast_tracking import (
    AccuracyTrend,
    ForecastMetaAnalysis,
    ForecastRecord,
)


class ForecastTracker:
    def __init__(self, findings_dir: Path) -> None:
        self._path = findings_dir / "forecast_history.jsonl"

    def record_week(self, record: ForecastRecord) -> None:
        """Append a weekly forecast record."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")

    def load_all(self) -> list[ForecastRecord]:
        """Load all forecast records from disk."""
        if not self._path.exists():
            return []
        records: list[ForecastRecord] = []
        for line in self._path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                records.append(ForecastRecord(**json.loads(line)))
        return records

    def compute_meta_analysis(self) -> ForecastMetaAnalysis:
        """Compute rolling accuracy, trend, and calibration from all records."""
        records = self.load_all()
        if not records:
            return ForecastMetaAnalysis()

        # Sort by week_start descending
        records.sort(key=lambda r: r.week_start, reverse=True)

        weeks_analyzed = len(records)

        # Rolling accuracy: last 4 weeks
        last_4 = records[:4]
        acc_4w = self._weighted_accuracy(last_4)

        # Rolling accuracy: last 12 weeks
        last_12 = records[:12]
        acc_12w = self._weighted_accuracy(last_12)

        # Trend: compare first half vs second half of available data
        trend = self._compute_trend(records)

        # Per-bot accuracy: aggregate across all weeks
        bot_totals: dict[str, list[float]] = {}
        for rec in records:
            for bot_id, acc in rec.by_bot.items():
                bot_totals.setdefault(bot_id, []).append(acc)
        accuracy_by_bot = {
            bot: sum(accs) / len(accs) for bot, accs in bot_totals.items()
        }

        # Per-metric accuracy: aggregate by_type across all weeks
        metric_totals: dict[str, list[float]] = {}
        for rec in records:
            for metric, acc in rec.by_type.items():
                metric_totals.setdefault(metric, []).append(acc)
        accuracy_by_metric = {
            m: round(sum(accs) / len(accs), 3) for m, accs in metric_totals.items()
        }

        # Calibration adjustment: compare average confidence vs average accuracy
        # Positive = under-confident, Negative = over-confident
        avg_accuracy = acc_4w if weeks_analyzed >= 4 else acc_12w if weeks_analyzed >= 1 else 0.0
        # Assume 0.5 is the "neutral" confidence baseline
        calibration = avg_accuracy - 0.5  # clamped to [-1, 1]
        calibration = max(-1.0, min(1.0, calibration))

        return ForecastMetaAnalysis(
            rolling_accuracy_4w=round(acc_4w, 3),
            rolling_accuracy_12w=round(acc_12w, 3),
            trend=trend,
            accuracy_by_bot=accuracy_by_bot,
            accuracy_by_metric=accuracy_by_metric,
            calibration_adjustment=round(calibration, 3),
            weeks_analyzed=weeks_analyzed,
        )

    @staticmethod
    def _weighted_accuracy(records: list[ForecastRecord]) -> float:
        """Compute weighted accuracy from a list of records."""
        total_reviewed = sum(r.predictions_reviewed for r in records)
        if total_reviewed == 0:
            return 0.0
        total_correct = sum(r.correct_predictions for r in records)
        return total_correct / total_reviewed

    @staticmethod
    def _compute_trend(records: list[ForecastRecord]) -> AccuracyTrend:
        """Determine accuracy trend from record history."""
        if len(records) < 3:
            return AccuracyTrend.STABLE

        # Compare recent half vs older half
        mid = len(records) // 2
        recent = records[:mid]
        older = records[mid:]

        recent_acc = ForecastTracker._weighted_accuracy(recent)
        older_acc = ForecastTracker._weighted_accuracy(older)

        diff = recent_acc - older_acc
        if diff > 0.05:
            return AccuracyTrend.IMPROVING
        elif diff < -0.05:
            return AccuracyTrend.DEGRADING
        return AccuracyTrend.STABLE
