# tests/test_wfo_trade_loading.py
"""Tests for _load_trades_for_wfo in Handlers."""
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.handlers import Handlers


def _make_handlers(curated_dir: Path) -> Handlers:
    """Build a Handlers instance with minimal mocks for trade loading tests."""
    return Handlers(
        agent_runner=MagicMock(),
        event_stream=MagicMock(),
        dispatcher=AsyncMock(),
        notification_prefs=MagicMock(),
        curated_dir=curated_dir,
        memory_dir=curated_dir.parent / "memory",
        runs_dir=curated_dir.parent / "runs",
        source_root=curated_dir.parent,
        bots=["bot_alpha"],
    )


def _sample_trade(bot_id: str = "bot_alpha", trade_id: str = "t1") -> dict:
    return {
        "trade_id": trade_id,
        "bot_id": bot_id,
        "pair": "BTCUSDT",
        "side": "LONG",
        "entry_time": "2026-03-01T10:00:00",
        "exit_time": "2026-03-01T11:00:00",
        "entry_price": 50000.0,
        "exit_price": 50500.0,
        "position_size": 0.1,
        "pnl": 50.0,
        "pnl_pct": 1.0,
    }


def _sample_missed(bot_id: str = "bot_alpha") -> dict:
    return {
        "bot_id": bot_id,
        "pair": "ETHUSDT",
        "signal": "momentum_cross",
        "signal_strength": 0.85,
        "blocked_by": "spread_filter",
    }


class TestLoadTradesForWFO:
    def test_returns_empty_when_curated_dir_missing(self, tmp_path):
        h = _make_handlers(tmp_path / "nonexistent")
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert trades == []
        assert missed == []

    def test_returns_empty_when_no_bot_data(self, tmp_path):
        curated = tmp_path / "curated"
        (curated / "2026-03-01" / "other_bot").mkdir(parents=True)
        h = _make_handlers(curated)
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert trades == []
        assert missed == []

    def test_loads_trades_from_jsonl(self, tmp_path):
        curated = tmp_path / "curated"
        bot_dir = curated / "2026-03-01" / "bot_alpha"
        bot_dir.mkdir(parents=True)
        (bot_dir / "trades.jsonl").write_text(
            json.dumps(_sample_trade()) + "\n" + json.dumps(_sample_trade(trade_id="t2")) + "\n",
            encoding="utf-8",
        )
        h = _make_handlers(curated)
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert len(trades) == 2
        assert trades[0].trade_id == "t1"
        assert trades[1].trade_id == "t2"
        assert missed == []

    def test_loads_missed_from_jsonl(self, tmp_path):
        curated = tmp_path / "curated"
        bot_dir = curated / "2026-03-01" / "bot_alpha"
        bot_dir.mkdir(parents=True)
        (bot_dir / "missed.jsonl").write_text(
            json.dumps(_sample_missed()) + "\n",
            encoding="utf-8",
        )
        h = _make_handlers(curated)
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert trades == []
        assert len(missed) == 1
        assert missed[0].pair == "ETHUSDT"

    def test_loads_across_multiple_dates(self, tmp_path):
        curated = tmp_path / "curated"
        for date in ["2026-03-01", "2026-03-02", "2026-03-03"]:
            bot_dir = curated / date / "bot_alpha"
            bot_dir.mkdir(parents=True)
            (bot_dir / "trades.jsonl").write_text(
                json.dumps(_sample_trade(trade_id=f"t-{date}")) + "\n",
                encoding="utf-8",
            )
        h = _make_handlers(curated)
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert len(trades) == 3

    def test_skips_bad_json_lines(self, tmp_path):
        curated = tmp_path / "curated"
        bot_dir = curated / "2026-03-01" / "bot_alpha"
        bot_dir.mkdir(parents=True)
        (bot_dir / "trades.jsonl").write_text(
            json.dumps(_sample_trade()) + "\n"
            "not valid json\n"
            + json.dumps(_sample_trade(trade_id="t2")) + "\n",
            encoding="utf-8",
        )
        h = _make_handlers(curated)
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert len(trades) == 2  # bad line skipped

    def test_skips_empty_lines(self, tmp_path):
        curated = tmp_path / "curated"
        bot_dir = curated / "2026-03-01" / "bot_alpha"
        bot_dir.mkdir(parents=True)
        (bot_dir / "trades.jsonl").write_text(
            json.dumps(_sample_trade()) + "\n\n\n",
            encoding="utf-8",
        )
        h = _make_handlers(curated)
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert len(trades) == 1

    def test_ignores_non_directory_files_in_curated(self, tmp_path):
        curated = tmp_path / "curated"
        curated.mkdir(parents=True)
        (curated / "README.md").write_text("ignore me", encoding="utf-8")
        bot_dir = curated / "2026-03-01" / "bot_alpha"
        bot_dir.mkdir(parents=True)
        (bot_dir / "trades.jsonl").write_text(
            json.dumps(_sample_trade()) + "\n",
            encoding="utf-8",
        )
        h = _make_handlers(curated)
        trades, missed = h._load_trades_for_wfo("bot_alpha")
        assert len(trades) == 1
