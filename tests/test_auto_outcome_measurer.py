# tests/test_auto_outcome_measurer.py
"""Tests for automated suggestion outcome measurement."""
import json
from datetime import datetime, timezone
from pathlib import Path

from schemas.outcome_measurement import OutcomeMeasurement, Verdict


class TestOutcomeMeasurementSchema:
    def test_verdict_positive(self):
        m = OutcomeMeasurement(
            suggestion_id="s1",
            implemented_date="2026-02-20",
            measurement_date="2026-03-01",
            window_days=7,
            pnl_before=100.0, pnl_after=200.0,
            win_rate_before=0.5, win_rate_after=0.6,
            drawdown_before=5.0, drawdown_after=4.0,
            before_trade_count=10, after_trade_count=10,
        )
        assert m.verdict == Verdict.POSITIVE
        assert m.pnl_delta == 100.0

    def test_verdict_negative(self):
        m = OutcomeMeasurement(
            suggestion_id="s2",
            implemented_date="2026-02-20",
            measurement_date="2026-03-01",
            window_days=7,
            pnl_before=200.0, pnl_after=50.0,
            win_rate_before=0.6, win_rate_after=0.4,
            before_trade_count=10, after_trade_count=10,
        )
        assert m.verdict == Verdict.NEGATIVE

    def test_verdict_neutral(self):
        m = OutcomeMeasurement(
            suggestion_id="s3",
            implemented_date="2026-02-20",
            measurement_date="2026-03-01",
            window_days=7,
            pnl_before=100.0, pnl_after=102.0,
            win_rate_before=0.5, win_rate_after=0.51,
            before_trade_count=10, after_trade_count=10,
        )
        assert m.verdict == Verdict.NEUTRAL


class TestAutoOutcomeMeasurer:
    def test_measure_suggestion_outcome(self, tmp_path):
        from skills.auto_outcome_measurer import AutoOutcomeMeasurer

        # Create curated summaries for before/after periods
        self._write_summaries(tmp_path, "2026-02-15", "bot1", pnl=50, wins=5, total=10)
        self._write_summaries(tmp_path, "2026-02-16", "bot1", pnl=60, wins=6, total=10)
        self._write_summaries(tmp_path, "2026-02-25", "bot1", pnl=100, wins=8, total=10)
        self._write_summaries(tmp_path, "2026-02-26", "bot1", pnl=90, wins=7, total=10)

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure(
            suggestion_id="s1",
            bot_id="bot1",
            implemented_date="2026-02-20",
            before_days=7,
            after_days=7,
        )
        assert isinstance(result, OutcomeMeasurement)
        assert result.pnl_after > result.pnl_before

    def test_insufficient_data_returns_none(self, tmp_path):
        from skills.auto_outcome_measurer import AutoOutcomeMeasurer

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure(
            suggestion_id="s1",
            bot_id="bot1",
            implemented_date="2026-02-20",
            before_days=7,
            after_days=7,
        )
        assert result is None

    def test_detect_parameter_change(self, tmp_path):
        from skills.auto_outcome_measurer import AutoOutcomeMeasurer

        # Write WFO reports with different params
        wfo_dir = tmp_path / "wfo"
        wfo_dir.mkdir()
        (wfo_dir / "2026-02-15.json").write_text(json.dumps({
            "bot_id": "bot1", "suggested_params": {"stop_pct": 2.0},
            "recommendation": "ADOPT",
        }))
        (wfo_dir / "2026-03-01.json").write_text(json.dumps({
            "bot_id": "bot1", "suggested_params": {"stop_pct": 3.0},
            "recommendation": "ADOPT",
        }))

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path, wfo_dir=wfo_dir)
        changes = measurer.detect_parameter_changes("bot1")
        assert len(changes) >= 1

    def test_measure_prefers_net_pnl(self, tmp_path):
        from skills.auto_outcome_measurer import AutoOutcomeMeasurer

        self._write_summaries(tmp_path, "2026-02-15", "bot1", pnl=100, wins=5, total=10, net_pnl=80)
        self._write_summaries(tmp_path, "2026-02-25", "bot1", pnl=150, wins=8, total=10, net_pnl=140)

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure(
            suggestion_id="s1",
            bot_id="bot1",
            implemented_date="2026-02-20",
            before_days=7,
            after_days=7,
        )

        assert result is not None
        assert result.pnl_before == 80
        assert result.pnl_after == 140

    def _write_summaries(
        self,
        base: Path,
        date: str,
        bot_id: str,
        pnl: float,
        wins: int,
        total: int,
        net_pnl: float | None = None,
    ):
        bot_dir = base / date / bot_id
        bot_dir.mkdir(parents=True, exist_ok=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "date": date, "bot_id": bot_id,
            "gross_pnl": pnl,
            "net_pnl": pnl if net_pnl is None else net_pnl,
            "win_count": wins,
            "total_trades": total,
            "max_drawdown_pct": 3.0,
        }))
