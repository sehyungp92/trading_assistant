"""Tests for event-schema lineage fields — backward-compat round-trip."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from schemas.events import EventMetadata, MissedOpportunityEvent, TradeEvent


def _trade_kwargs() -> dict:
    return {
        "trade_id": "t1",
        "bot_id": "bot_a",
        "pair": "BTC/USDT",
        "side": "LONG",
        "entry_time": datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        "exit_time": datetime(2026, 5, 1, 11, 0, tzinfo=timezone.utc),
        "entry_price": 50000.0,
        "exit_price": 50500.0,
        "position_size": 0.1,
        "pnl": 50.0,
        "pnl_pct": 1.0,
    }


def _missed_kwargs() -> dict:
    return {
        "bot_id": "bot_a",
        "pair": "BTC/USDT",
        "signal": "RSI_OVERSOLD",
    }


def test_trade_event_round_trips_without_lineage() -> None:
    trade = TradeEvent(**_trade_kwargs())
    payload = trade.model_dump(mode="json")
    redo = TradeEvent(**payload)
    assert redo.trade_id == "t1"
    assert redo.deployment_id is None
    assert redo.experiment_id is None


def test_trade_event_round_trips_with_lineage() -> None:
    kwargs = _trade_kwargs()
    kwargs.update(
        deployment_id="dep_001",
        experiment_id="exp_001",
        variant_id="A",
        parameter_set_id="ps_42",
        strategy_version="v3",
        config_version="2026-05",
        signal_generation_version="sg_7",
        code_sha="abc1234",
    )
    trade = TradeEvent(**kwargs)
    payload = json.loads(json.dumps(trade.model_dump(mode="json"), default=str))
    redo = TradeEvent(**payload)
    assert redo.deployment_id == "dep_001"
    assert redo.experiment_id == "exp_001"
    assert redo.variant_id == "A"
    assert redo.parameter_set_id == "ps_42"
    assert redo.strategy_version == "v3"
    assert redo.config_version == "2026-05"
    assert redo.signal_generation_version == "sg_7"
    assert redo.code_sha == "abc1234"


def test_trade_event_ignores_unknown_fields() -> None:
    """Forward-compat: an old reader should accept newer payloads with
    extra fields — model_config="ignore" was added for migration windows."""
    kwargs = _trade_kwargs()
    kwargs["future_field_we_dont_know"] = {"any": "shape"}
    trade = TradeEvent(**kwargs)
    assert not hasattr(trade, "future_field_we_dont_know")


def test_missed_event_round_trips_with_lineage() -> None:
    kwargs = _missed_kwargs()
    kwargs.update(
        deployment_id="dep_002",
        experiment_id="exp_002",
        variant_id="B",
    )
    evt = MissedOpportunityEvent(**kwargs)
    redo = MissedOpportunityEvent(**evt.model_dump(mode="json"))
    assert redo.deployment_id == "dep_002"
    assert redo.experiment_id == "exp_002"
    assert redo.variant_id == "B"


def test_missed_event_round_trips_without_lineage() -> None:
    evt = MissedOpportunityEvent(**_missed_kwargs())
    redo = MissedOpportunityEvent(**evt.model_dump(mode="json"))
    assert redo.deployment_id is None
    assert redo.experiment_id is None
