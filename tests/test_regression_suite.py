# tests/test_regression_suite.py
"""Tests for the regression suite — golden day loader + regression checks."""
import json
from pathlib import Path

import pytest

from tests.golden_days.loader import GoldenDay, load_golden_days


@pytest.fixture
def golden_day_dir(tmp_path: Path) -> Path:
    """Create a single golden day with known data."""
    day_dir = tmp_path / "2026-02-15"
    day_dir.mkdir()

    # Raw events
    raw_dir = day_dir / "raw_events"
    raw_dir.mkdir()
    (raw_dir / "trades.json").write_text(json.dumps([
        {
            "trade_id": "t1", "bot_id": "bot1", "pair": "BTCUSDT",
            "side": "LONG", "pnl": 200.0, "pnl_pct": 2.0,
            "entry_price": 50000.0, "exit_price": 51000.0,
            "position_size": 0.1, "entry_time": "2026-02-15T10:00:00Z",
            "exit_time": "2026-02-15T14:00:00Z",
            "entry_signal": "EMA cross", "exit_reason": "TAKE_PROFIT",
            "market_regime": "trending_up",
            "process_quality_score": 90, "root_causes": ["normal_win"],
        },
        {
            "trade_id": "t2", "bot_id": "bot1", "pair": "ETHUSDT",
            "side": "SHORT", "pnl": -80.0, "pnl_pct": -1.5,
            "entry_price": 3000.0, "exit_price": 3045.0,
            "position_size": 0.5, "entry_time": "2026-02-15T11:00:00Z",
            "exit_time": "2026-02-15T13:00:00Z",
            "entry_signal": "RSI divergence", "exit_reason": "STOP_LOSS",
            "market_regime": "ranging",
            "process_quality_score": 45, "root_causes": ["regime_mismatch"],
        },
    ]))
    (raw_dir / "missed.json").write_text(json.dumps([
        {
            "bot_id": "bot1", "pair": "BTCUSDT", "signal": "Volume breakout",
            "blocked_by": "volatility_filter", "hypothetical_entry": 50500.0,
            "outcome_24h": 400.0, "confidence": 0.7,
            "assumption_tags": ["mid_fill", "5bps_slippage"],
        },
    ]))

    # Expected classifications
    (day_dir / "expected_classifications.json").write_text(json.dumps({
        "t1": {"root_causes": ["normal_win"], "process_quality_score": 90},
        "t2": {"root_causes": ["regime_mismatch"], "process_quality_score": 45},
    }))

    # Expected curated metrics
    expected_dir = day_dir / "expected_curated"
    expected_dir.mkdir()
    (expected_dir / "summary.json").write_text(json.dumps({
        "bot_id": "bot1", "total_trades": 2, "win_count": 1,
        "loss_count": 1, "net_pnl": 120.0,
    }))

    # Human feedback
    (day_dir / "human_feedback.json").write_text(json.dumps([
        {"correction_type": "positive_reinforcement", "raw_text": "Good analysis"},
    ]))

    # Reference report
    (day_dir / "reference_report.md").write_text(
        "# Daily Report 2026-02-15\nBot1 had a good day. Process failure on t2."
    )

    # Metadata
    (day_dir / "metadata.json").write_text(json.dumps({
        "date": "2026-02-15",
        "bots": ["bot1"],
        "top_anomaly": "volume_spike",
        "biggest_loss_driver": "regime_mismatch",
        "crowding_alerts": [],
    }))

    return tmp_path


class TestGoldenDayLoader:
    def test_loads_golden_day(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        assert len(days) == 1

    def test_golden_day_has_trades(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert len(day.trades) == 2
        assert day.trades[0]["trade_id"] == "t1"

    def test_golden_day_has_missed(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert len(day.missed) == 1

    def test_golden_day_has_expected_classifications(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert "t1" in day.expected_classifications
        assert day.expected_classifications["t2"]["root_causes"] == ["regime_mismatch"]

    def test_golden_day_has_metadata(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert day.date == "2026-02-15"
        assert day.top_anomaly == "volume_spike"
        assert day.biggest_loss_driver == "regime_mismatch"

    def test_golden_day_has_reference_report(self, golden_day_dir: Path):
        days = load_golden_days(golden_day_dir)
        day = days[0]
        assert "Process failure on t2" in day.reference_report
