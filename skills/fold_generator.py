# skills/fold_generator.py
"""Fold generator — creates anchored or rolling temporal splits for WFO.

Anchored: IS always starts at data_start, IS end grows by step_days each fold.
Rolling: IS window slides forward by step_days, IS length stays constant.
Both: OOS immediately follows IS, OOS length is fixed.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from schemas.wfo_config import WFOConfig, WFOMethod
from schemas.wfo_results import FoldDefinition


class FoldGenerator:
    """Generates walk-forward optimization folds from a WFO config."""

    def __init__(self, config: WFOConfig) -> None:
        self._config = config

    def generate(self, data_start: str, data_end: str) -> list[FoldDefinition]:
        """Generate folds for the given data range. Returns empty list if insufficient data."""
        if self._config.method == WFOMethod.ANCHORED:
            folds = self._generate_anchored(data_start, data_end)
        else:
            folds = self._generate_rolling(data_start, data_end)

        if len(folds) < self._config.min_folds:
            return []
        return folds

    def _generate_anchored(self, data_start: str, data_end: str) -> list[FoldDefinition]:
        start = _parse(data_start)
        end = _parse(data_end)
        is_days = self._config.in_sample_days
        oos_days = self._config.out_of_sample_days
        step = self._config.step_days

        folds: list[FoldDefinition] = []
        is_end = start + timedelta(days=is_days)

        while is_end + timedelta(days=oos_days) <= end:
            oos_end = is_end + timedelta(days=oos_days)
            folds.append(FoldDefinition(
                fold_number=len(folds),
                is_start=_fmt(start),
                is_end=_fmt(is_end),
                oos_start=_fmt(is_end),
                oos_end=_fmt(oos_end),
            ))
            is_end += timedelta(days=step)

        return folds

    def _generate_rolling(self, data_start: str, data_end: str) -> list[FoldDefinition]:
        start = _parse(data_start)
        end = _parse(data_end)
        is_days = self._config.in_sample_days
        oos_days = self._config.out_of_sample_days
        step = self._config.step_days

        folds: list[FoldDefinition] = []
        is_start = start

        while True:
            is_end = is_start + timedelta(days=is_days)
            oos_end = is_end + timedelta(days=oos_days)
            if oos_end > end:
                break
            folds.append(FoldDefinition(
                fold_number=len(folds),
                is_start=_fmt(is_start),
                is_end=_fmt(is_end),
                oos_start=_fmt(is_end),
                oos_end=_fmt(oos_end),
            ))
            is_start += timedelta(days=step)

        return folds


def _parse(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")
