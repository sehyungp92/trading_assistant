# tests/test_counterfactual_simulator.py
"""Tests for counterfactual trade replay simulator."""
from datetime import datetime, timezone

import pytest

from schemas.events import TradeEvent, MissedOpportunityEvent
from schemas.counterfactual import ScenarioType, CounterfactualScenario, CounterfactualResult


def _make_trade(trade_id, pnl, regime="trending", pair="BTC/USDT"):
    return TradeEvent(
        trade_id=trade_id, bot_id="bot1", pair=pair,
        side="LONG",
        entry_time=datetime(2026, 3, 1, tzinfo=timezone.utc),
        exit_time=datetime(2026, 3, 1, 1, tzinfo=timezone.utc),
        entry_price=100, exit_price=100 + pnl, position_size=1,
        pnl=pnl, pnl_pct=pnl, market_regime=regime,
        spread_at_entry=2.0,
    )


def _make_missed(blocked_by, outcome_24h):
    return MissedOpportunityEvent(
        bot_id="bot1", pair="BTC/USDT", signal="momentum",
        blocked_by=blocked_by, outcome_24h=outcome_24h,
        hypothetical_entry=100.0, confidence=0.8, assumption_tags=[],
    )


class TestCounterfactualSchema:
    def test_scenario_creation(self):
        s = CounterfactualScenario(
            scenario_type=ScenarioType.REMOVE_FILTER,
            description="Remove volume filter",
            parameters={"filter_name": "volume_filter"},
        )
        assert s.scenario_type == ScenarioType.REMOVE_FILTER

    def test_result_model(self):
        r = CounterfactualResult(
            scenario=CounterfactualScenario(
                scenario_type=ScenarioType.ADD_REGIME_GATE,
                description="Add ranging gate",
                parameters={"regime": "ranging"},
            ),
            baseline_pnl=100.0, modified_pnl=150.0,
            baseline_trade_count=20, modified_trade_count=15,
        )
        assert r.delta_pnl == 50.0


class TestCounterfactualSimulator:
    def test_remove_filter_includes_missed_ops(self):
        from skills.counterfactual_simulator import CounterfactualSimulator

        trades = [_make_trade("t1", 10.0), _make_trade("t2", -5.0)]
        missed = [
            _make_missed("volume_filter", 20.0),
            _make_missed("volume_filter", -8.0),
            _make_missed("rsi_filter", 15.0),  # different filter, not included
        ]
        sim = CounterfactualSimulator()
        result = sim.simulate_remove_filter(trades, missed, "volume_filter")
        assert isinstance(result, CounterfactualResult)
        # Baseline: 10 - 5 = 5
        assert result.baseline_pnl == 5.0
        # Modified: 10 - 5 + 20 - 8 = 17
        assert result.modified_pnl == 17.0
        assert result.modified_trade_count == 4  # 2 original + 2 unfiltered

    def test_add_regime_gate_excludes_trades(self):
        from skills.counterfactual_simulator import CounterfactualSimulator

        trades = [
            _make_trade("t1", 10.0, "trending"),
            _make_trade("t2", 8.0, "trending"),
            _make_trade("t3", -15.0, "ranging"),
            _make_trade("t4", -12.0, "ranging"),
        ]
        sim = CounterfactualSimulator()
        result = sim.simulate_regime_gate(trades, [], "ranging")
        assert result.baseline_pnl == -9.0  # 10+8-15-12
        assert result.modified_pnl == 18.0  # 10+8
        assert result.modified_trade_count == 2

    def test_exclude_trades_by_criteria(self):
        from skills.counterfactual_simulator import CounterfactualSimulator

        trades = [
            _make_trade("t1", 10.0, pair="BTC/USDT"),
            _make_trade("t2", -20.0, pair="DOGE/USDT"),
            _make_trade("t3", 5.0, pair="ETH/USDT"),
        ]
        sim = CounterfactualSimulator()
        result = sim.simulate_exclude(trades, [], lambda t: t.pair == "DOGE/USDT")
        assert result.modified_pnl == 15.0
        assert result.modified_trade_count == 2

    def test_computes_win_rate_delta(self):
        from skills.counterfactual_simulator import CounterfactualSimulator

        trades = [
            _make_trade("t1", 10.0, "trending"),
            _make_trade("t2", -5.0, "ranging"),
            _make_trade("t3", -3.0, "ranging"),
        ]
        sim = CounterfactualSimulator()
        result = sim.simulate_regime_gate(trades, [], "ranging")
        # Baseline win rate: 1/3 = 0.333
        assert result.baseline_win_rate == pytest.approx(1 / 3, abs=0.01)
        # Modified: 1/1 = 1.0
        assert result.modified_win_rate == 1.0

    def test_empty_trades_returns_zero_baseline(self):
        from skills.counterfactual_simulator import CounterfactualSimulator

        sim = CounterfactualSimulator()
        result = sim.simulate_remove_filter([], [], "volume_filter")
        assert result.baseline_pnl == 0.0
        assert result.modified_pnl == 0.0
