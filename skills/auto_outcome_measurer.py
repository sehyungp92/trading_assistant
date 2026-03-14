"""Automated outcome measurement for implemented suggestions.

Regime-controlled measurement: compares before/after performance with
regime matching, volatility controls, concurrent change detection,
and statistical significance estimation.
"""
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from schemas.outcome_measurement import (
    MeasurementQuality,
    OutcomeMeasurement,
    compute_measurement_quality,
    compute_significance,
)


class AutoOutcomeMeasurer:
    """Measures suggestion outcomes by comparing pre/post performance."""

    def __init__(
        self,
        curated_dir: Path,
        wfo_dir: Path | None = None,
        findings_dir: Path | None = None,
        calibration_tracker: "BacktestCalibrationTracker | None" = None,
    ) -> None:
        self._curated_dir = curated_dir
        self._wfo_dir = wfo_dir
        self._findings_dir = findings_dir
        self._calibration_tracker = calibration_tracker

    def measure(
        self,
        suggestion_id: str,
        bot_id: str,
        implemented_date: str,
        before_days: int = 7,
        after_days: int = 7,
    ) -> OutcomeMeasurement | None:
        """Compare performance before and after a suggestion was implemented.

        Now includes regime matching, volatility controls, concurrent change
        detection, and measurement quality assessment.
        """
        impl_date = datetime.strptime(implemented_date, "%Y-%m-%d")
        today = datetime.now(timezone.utc)

        before_start = impl_date - timedelta(days=before_days)
        after_end = impl_date + timedelta(days=after_days)

        before_summaries = self._load_summaries(bot_id, before_start, impl_date)
        after_summaries = self._load_summaries(bot_id, impl_date, after_end)

        if not before_summaries or not after_summaries:
            return None

        # Core metrics
        before_pnl = sum(self._summary_pnl(s) for s in before_summaries)
        after_pnl = sum(self._summary_pnl(s) for s in after_summaries)

        before_wins = sum(s.get("win_count", 0) for s in before_summaries)
        before_total = sum(s.get("total_trades", 0) for s in before_summaries)
        after_wins = sum(s.get("win_count", 0) for s in after_summaries)
        after_total = sum(s.get("total_trades", 0) for s in after_summaries)

        before_wr = before_wins / before_total if before_total > 0 else 0
        after_wr = after_wins / after_total if after_total > 0 else 0

        before_dd = max(
            (s.get("max_drawdown_pct", 0) for s in before_summaries), default=0
        )
        after_dd = max(
            (s.get("max_drawdown_pct", 0) for s in after_summaries), default=0
        )

        # Regime analysis
        before_regimes = self._load_regimes(bot_id, before_start, impl_date)
        after_regimes = self._load_regimes(bot_id, impl_date, after_end)
        before_regime = self._dominant_regime(before_regimes)
        after_regime = self._dominant_regime(after_regimes)
        regime_matched = before_regime == after_regime or not before_regime or not after_regime

        # Volatility
        before_vol = self._compute_volatility(before_summaries)
        after_vol = self._compute_volatility(after_summaries)
        vol_ratio = after_vol / before_vol if before_vol > 0 else 1.0

        # Concurrent changes
        concurrent = self._find_concurrent_changes(
            suggestion_id, bot_id, implemented_date, after_days
        )

        # Quality assessment
        quality = compute_measurement_quality(
            regime_matched=regime_matched,
            before_trade_count=before_total,
            after_trade_count=after_total,
            volatility_ratio=vol_ratio,
            concurrent_changes=concurrent,
        )

        # Effect significance
        daily_pnls_before = [self._summary_pnl(s) for s in before_summaries]
        daily_pnls_after = [self._summary_pnl(s) for s in after_summaries]
        effect = (after_pnl / max(len(after_summaries), 1)) - (
            before_pnl / max(len(before_summaries), 1)
        )
        noise = self._estimate_noise(daily_pnls_before + daily_pnls_after)
        sig = compute_significance(
            effect, noise, len(before_summaries), len(after_summaries),
        )

        measurement = OutcomeMeasurement(
            suggestion_id=suggestion_id,
            implemented_date=implemented_date,
            measurement_date=today.strftime("%Y-%m-%d"),
            window_days=after_days,
            pnl_before=before_pnl,
            pnl_after=after_pnl,
            win_rate_before=before_wr,
            win_rate_after=after_wr,
            drawdown_before=before_dd,
            drawdown_after=after_dd,
            before_regime=before_regime,
            after_regime=after_regime,
            regime_matched=regime_matched,
            before_volatility=round(before_vol, 6),
            after_volatility=round(after_vol, 6),
            volatility_ratio=round(vol_ratio, 4),
            before_trade_count=before_total,
            after_trade_count=after_total,
            concurrent_changes=concurrent,
            measurement_quality=quality,
            effect_size=round(effect, 4),
            noise_estimate=round(noise, 4),
            significance_score=round(sig, 4),
        )

        # Feed back to calibration tracker (approximate composite delta)
        if self._calibration_tracker:
            delta = self._estimate_composite_delta(measurement)
            self._calibration_tracker.record_outcome(suggestion_id, delta)

        return measurement

    @staticmethod
    def _estimate_composite_delta(m: OutcomeMeasurement) -> float:
        """Approximate composite delta aligned with inner loop formula.

        Mirrors: 0.4*calmar + 0.3*profit_factor - 0.3*drawdown.
        Uses ratio-based comparison (after/before) to avoid scale explosion.
        NOT GroundTruthComputer — clearly an approximate calibration signal.
        """
        window = max(m.window_days, 1)

        # Calmar proxy: annualized daily PnL / drawdown (ratio-based)
        before_daily = m.pnl_before / window
        after_daily = m.pnl_after / window
        _DD_FLOOR = 0.001  # floor to avoid division by near-zero
        before_dd = max(m.drawdown_before, _DD_FLOOR)
        after_dd = max(m.drawdown_after, _DD_FLOOR)
        calmar_before = (before_daily * 365) / before_dd
        calmar_after = (after_daily * 365) / after_dd
        # Ratio-based improvement (matches inner loop style), clamped
        calmar_ratio = calmar_after / calmar_before if calmar_before != 0 else 1.0
        calmar_imp = max(-3.0, min(3.0, calmar_ratio - 1.0))

        # Profit factor proxy: PnL efficiency relative to drawdown (ratio-based)
        pnl_eff_before = m.pnl_before / (before_dd * window)
        pnl_eff_after = m.pnl_after / (after_dd * window)
        pf_ratio = pnl_eff_after / pnl_eff_before if pnl_eff_before != 0 else 1.0
        pf_imp = max(-3.0, min(3.0, pf_ratio - 1.0))

        # Drawdown increase (higher drawdown is worse)
        dd_inc = m.drawdown_after - m.drawdown_before

        return 0.4 * calmar_imp + 0.3 * pf_imp - 0.3 * max(0.0, dd_inc)

    def detect_parameter_changes(self, bot_id: str) -> list[dict]:
        """Detect parameter changes from WFO reports over time."""
        if not self._wfo_dir or not self._wfo_dir.exists():
            return []

        reports = []
        for f in sorted(self._wfo_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("bot_id") == bot_id:
                    reports.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        changes = []
        for i in range(1, len(reports)):
            prev = reports[i - 1].get("suggested_params", {})
            curr = reports[i].get("suggested_params", {})
            if prev != curr:
                changes.append({
                    "date": reports[i].get("date", "unknown"),
                    "previous_params": prev,
                    "new_params": curr,
                })
        return changes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_pnl(summary: dict) -> float:
        """Prefer net PnL because it reflects fees/slippage in live trading."""
        value = summary.get("net_pnl")
        if value is None:
            value = summary.get("gross_pnl", 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _load_summaries(
        self, bot_id: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Load daily summaries for a date range."""
        summaries = []
        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            summary_path = self._curated_dir / date_str / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass
            current += timedelta(days=1)
        return summaries

    def _load_regimes(
        self, bot_id: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Load regime_analysis.json for each day in the window."""
        regimes = []
        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            path = self._curated_dir / date_str / bot_id / "regime_analysis.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    regimes.append(data)
                except (json.JSONDecodeError, OSError):
                    pass
            current += timedelta(days=1)
        return regimes

    @staticmethod
    def _dominant_regime(regime_data: list[dict]) -> str:
        """Determine the dominant regime from a list of daily regime analyses."""
        if not regime_data:
            return ""
        counts: dict[str, int] = {}
        for day in regime_data:
            regime = day.get("dominant_regime", "") or day.get("regime", "")
            if regime:
                counts[regime] = counts.get(regime, 0) + 1
        if not counts:
            return ""
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    @classmethod
    def _compute_volatility(cls, summaries: list[dict]) -> float:
        """Compute daily PnL standard deviation as a volatility proxy."""
        pnls = [cls._summary_pnl(s) for s in summaries]
        if len(pnls) < 2:
            return 0.0
        return statistics.stdev(pnls)

    def _find_concurrent_changes(
        self,
        suggestion_id: str,
        bot_id: str,
        implemented_date: str,
        window_days: int,
    ) -> list[str]:
        """Find other suggestions deployed within the measurement window."""
        if not self._findings_dir:
            return []
        suggestions_path = self._findings_dir / "suggestions.jsonl"
        if not suggestions_path.exists():
            return []

        impl_date = datetime.strptime(implemented_date, "%Y-%m-%d")
        window_start = impl_date - timedelta(days=window_days)
        window_end = impl_date + timedelta(days=window_days)

        concurrent: list[str] = []
        try:
            for line in suggestions_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("suggestion_id") == suggestion_id:
                    continue
                if rec.get("bot_id") != bot_id:
                    continue
                status = rec.get("status", "")
                if status not in ("deployed", "implemented", "measured"):
                    continue
                deployed_str = rec.get("deployed_at") or rec.get("resolved_at", "")
                if not deployed_str:
                    continue
                try:
                    deployed_dt = datetime.fromisoformat(
                        deployed_str.replace("Z", "+00:00")
                    )
                    if deployed_dt.tzinfo:
                        deployed_dt = deployed_dt.replace(tzinfo=None)
                    if window_start <= deployed_dt <= window_end:
                        concurrent.append(rec.get("suggestion_id", "unknown"))
                except (ValueError, TypeError):
                    pass
        except (OSError, json.JSONDecodeError):
            pass
        return concurrent

    @staticmethod
    def _estimate_noise(daily_pnls: list[float]) -> float:
        """Estimate noise as the standard deviation of daily PnLs."""
        if len(daily_pnls) < 2:
            return 0.0
        return statistics.stdev(daily_pnls)
