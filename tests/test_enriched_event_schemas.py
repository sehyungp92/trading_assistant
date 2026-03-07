"""Tests for enriched event schemas (indicator snapshots, order book, filters, param changes)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from schemas.enriched_events import (
    ChangeSource,
    FilterDecisionEvent,
    IndicatorSnapshot,
    OrderBookContext,
    ParameterChangeEvent,
)


TS = datetime(2026, 3, 7, 14, 0, tzinfo=timezone.utc)


class TestIndicatorSnapshot:
    def test_serialization_roundtrip(self):
        """IndicatorSnapshot serializes to JSON and back with all fields intact."""
        snap = IndicatorSnapshot(
            bot_id="bot1",
            pair="BTCUSDT",
            timestamp=TS,
            indicators={"rsi_14": 45.2, "ema_20": 50100.0, "atr_14": 320.5},
            signal_name="ema_cross",
            signal_strength=0.78,
            decision="enter",
            context={"regime": "trending_up"},
        )
        data = json.loads(snap.model_dump_json())
        restored = IndicatorSnapshot.model_validate(data)
        assert restored.bot_id == "bot1"
        assert restored.pair == "BTCUSDT"
        assert restored.indicators["rsi_14"] == pytest.approx(45.2)
        assert restored.signal_name == "ema_cross"
        assert restored.signal_strength == pytest.approx(0.78)
        assert restored.decision == "enter"
        assert restored.context == {"regime": "trending_up"}

    def test_empty_indicators(self):
        """IndicatorSnapshot defaults to empty indicators dict and skip decision."""
        snap = IndicatorSnapshot(
            bot_id="bot2",
            pair="ETHUSDT",
            timestamp=TS,
        )
        assert snap.indicators == {}
        assert snap.decision == "skip"
        assert snap.signal_name == ""
        assert snap.signal_strength == 0.0
        assert snap.context == {}


class TestOrderBookContext:
    def test_imbalance_ratio_bid_heavy(self):
        """imbalance_ratio > 1 when bid depth exceeds ask depth."""
        ob = OrderBookContext(
            bot_id="bot1",
            pair="BTCUSDT",
            timestamp=TS,
            best_bid=50000.0,
            best_ask=50010.0,
            spread_bps=2.0,
            bid_depth_10bps=150.0,
            ask_depth_10bps=100.0,
        )
        assert ob.imbalance_ratio == pytest.approx(1.5)

    def test_imbalance_ratio_zero_ask_depth(self):
        """imbalance_ratio returns 0.0 when ask_depth_10bps is 0."""
        ob = OrderBookContext(
            bot_id="bot1",
            pair="BTCUSDT",
            timestamp=TS,
            best_bid=50000.0,
            best_ask=50010.0,
            bid_depth_10bps=150.0,
            ask_depth_10bps=0.0,
        )
        assert ob.imbalance_ratio == 0.0

    def test_with_trade_context_and_related_trade_id(self):
        """OrderBookContext can store trade context and related trade ID."""
        ob = OrderBookContext(
            bot_id="bot1",
            pair="ETHUSDT",
            timestamp=TS,
            best_bid=3000.0,
            best_ask=3001.0,
            spread_bps=3.3,
            bid_depth_10bps=200.0,
            ask_depth_10bps=180.0,
            trade_context="entry",
            related_trade_id="t_abc123",
        )
        assert ob.trade_context == "entry"
        assert ob.related_trade_id == "t_abc123"
        # Verify serialization includes computed field
        data = json.loads(ob.model_dump_json())
        assert "imbalance_ratio" in data
        restored = OrderBookContext.model_validate(data)
        assert restored.trade_context == "entry"
        assert restored.related_trade_id == "t_abc123"


class TestFilterDecisionEvent:
    def test_margin_pct_passing_filter(self):
        """margin_pct is positive when actual_value exceeds threshold (passed)."""
        fde = FilterDecisionEvent(
            bot_id="bot1",
            pair="BTCUSDT",
            timestamp=TS,
            filter_name="min_volume",
            passed=True,
            threshold=1000.0,
            actual_value=1200.0,
        )
        assert fde.passed is True
        # (1200 - 1000) / 1000 * 100 = 20%
        assert fde.margin_pct == pytest.approx(20.0)

    def test_margin_pct_failing_filter(self):
        """margin_pct is negative when actual_value is below threshold (blocked)."""
        fde = FilterDecisionEvent(
            bot_id="bot1",
            pair="BTCUSDT",
            timestamp=TS,
            filter_name="min_volume",
            passed=False,
            threshold=1000.0,
            actual_value=800.0,
        )
        assert fde.passed is False
        # (800 - 1000) / 1000 * 100 = -20%
        assert fde.margin_pct == pytest.approx(-20.0)

    def test_margin_pct_zero_threshold(self):
        """margin_pct returns 0.0 when threshold is zero to avoid division by zero."""
        fde = FilterDecisionEvent(
            bot_id="bot1",
            pair="BTCUSDT",
            timestamp=TS,
            filter_name="spread_filter",
            passed=True,
            threshold=0.0,
            actual_value=5.0,
        )
        assert fde.margin_pct == 0.0


class TestParameterChangeEvent:
    def test_roundtrip_all_change_sources(self):
        """ParameterChangeEvent serializes and deserializes with every ChangeSource."""
        for source in ChangeSource:
            pce = ParameterChangeEvent(
                bot_id="bot1",
                param_name="rsi_period",
                old_value=14,
                new_value=21,
                change_source=source,
                timestamp=TS,
            )
            data = json.loads(pce.model_dump_json())
            restored = ParameterChangeEvent.model_validate(data)
            assert restored.change_source == source
            assert restored.old_value == 14
            assert restored.new_value == 21

    def test_pr_merge_with_commit_sha_and_pr_url(self):
        """ParameterChangeEvent with pr_merge source includes commit and PR metadata."""
        pce = ParameterChangeEvent(
            bot_id="bot1",
            param_name="take_profit_pct",
            old_value=2.0,
            new_value=2.5,
            change_source=ChangeSource.PR_MERGE,
            timestamp=TS,
            commit_sha="abc123def456",
            pr_url="https://github.com/user/repo/pull/42",
        )
        assert pce.change_source == ChangeSource.PR_MERGE
        assert pce.commit_sha == "abc123def456"
        assert pce.pr_url == "https://github.com/user/repo/pull/42"
        # Round-trip
        data = json.loads(pce.model_dump_json())
        restored = ParameterChangeEvent.model_validate(data)
        assert restored.commit_sha == "abc123def456"
        assert restored.pr_url == "https://github.com/user/repo/pull/42"
