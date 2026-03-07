"""Tests for enriched event ingestion — brain routing + daily metrics builder summaries."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orchestrator.orchestrator_brain import ActionType, OrchestratorBrain
from schemas.events import TradeEvent, MissedOpportunityEvent
from skills.build_daily_metrics import DailyMetricsBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: str, event_id: str = "e1", bot_id: str = "bot1",
                payload: dict | None = None) -> dict:
    """Build a minimal event dict for the brain."""
    evt = {"event_type": event_type, "event_id": event_id, "bot_id": bot_id}
    if payload is not None:
        evt["payload"] = json.dumps(payload)
    return evt


def _make_trade(trade_id: str, bot_id: str, pnl: float, **kwargs) -> TradeEvent:
    """Helper to create a TradeEvent with sensible defaults."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        trade_id=trade_id,
        bot_id=bot_id,
        pair="BTCUSDT",
        side="LONG",
        entry_time=now,
        exit_time=now,
        entry_price=50000.0,
        exit_price=50000.0 + pnl,
        position_size=1.0,
        pnl=pnl,
        pnl_pct=pnl / 50000.0 * 100,
        entry_signal="EMA cross",
        exit_reason="TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
        market_regime="trending_up",
        process_quality_score=85,
        root_causes=["normal_win"] if pnl > 0 else ["normal_loss"],
    )
    defaults.update(kwargs)
    return TradeEvent(**defaults)


# ===========================================================================
# 1. Brain routing tests
# ===========================================================================

class TestBrainEnrichedEventRouting:
    """Verify that the brain correctly routes the 4 new enriched event types."""

    def test_indicator_snapshot_routes_to_queue_for_daily(self):
        brain = OrchestratorBrain()
        actions = brain.decide(_make_event("indicator_snapshot"))
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_orderbook_context_routes_to_queue_for_daily(self):
        brain = OrchestratorBrain()
        actions = brain.decide(_make_event("orderbook_context"))
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_filter_decision_routes_to_queue_for_daily(self):
        brain = OrchestratorBrain()
        actions = brain.decide(_make_event("filter_decision"))
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_parameter_change_non_critical_routes_to_queue_for_daily(self):
        brain = OrchestratorBrain()
        actions = brain.decide(_make_event(
            "parameter_change",
            payload={"param_name": "lookback_window", "old_value": 20, "new_value": 30},
        ))
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_parameter_change_risk_param_routes_to_alert_immediate(self):
        brain = OrchestratorBrain()
        for param_name in ["risk_per_trade", "max_position_size", "kill_switch_enabled",
                           "trailing_stop_pct"]:
            actions = brain.decide(_make_event(
                "parameter_change",
                event_id=f"e_{param_name}",
                payload={"param_name": param_name, "old_value": 1, "new_value": 2},
            ))
            assert actions[0].type == ActionType.ALERT_IMMEDIATE, (
                f"Expected ALERT_IMMEDIATE for safety-critical param '{param_name}'"
            )
            assert actions[0].details is not None

    def test_parameter_change_is_safety_critical_flag(self):
        """Explicit is_safety_critical flag in payload triggers ALERT_IMMEDIATE."""
        brain = OrchestratorBrain()
        actions = brain.decide(_make_event(
            "parameter_change",
            payload={"param_name": "some_obscure_param", "is_safety_critical": True},
        ))
        assert actions[0].type == ActionType.ALERT_IMMEDIATE


# ===========================================================================
# 2. DailyMetricsBuilder summary tests
# ===========================================================================

class TestFilterDecisionSummary:
    def test_computes_per_filter_stats(self):
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        events = [
            {"filter_name": "vol_filter", "passed": True, "margin_pct": 10.0},
            {"filter_name": "vol_filter", "passed": False, "margin_pct": -3.0},
            {"filter_name": "vol_filter", "passed": True, "margin_pct": 2.0},
            {"filter_name": "regime_filter", "passed": False, "margin_pct": -8.0},
        ]
        result = builder.build_filter_decision_summary(events)

        assert "vol_filter" in result
        assert "regime_filter" in result

        vol = result["vol_filter"]
        assert vol["total_decisions"] == 3
        assert vol["pass_count"] == 2
        assert vol["block_count"] == 1
        assert vol["pass_rate"] == pytest.approx(2 / 3)
        assert vol["avg_margin_pct"] == pytest.approx((10.0 - 3.0 + 2.0) / 3)
        assert vol["min_margin_pct"] == -3.0
        # margins with abs < 5: -3.0 and 2.0
        assert vol["near_threshold_count"] == 2

        regime = result["regime_filter"]
        assert regime["total_decisions"] == 1
        assert regime["block_count"] == 1
        assert regime["pass_rate"] == 0.0

    def test_empty_events_returns_empty_dict(self):
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        assert builder.build_filter_decision_summary([]) == {}


class TestIndicatorSnapshotSummary:
    def test_aggregates_indicator_distributions(self):
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        events = [
            {"indicators": {"rsi_14": 30.0, "atr_20": 1.5}, "decision": "enter"},
            {"indicators": {"rsi_14": 70.0, "atr_20": 2.0}, "decision": "skip"},
            {"indicators": {"rsi_14": 50.0}, "decision": "exit"},
        ]
        result = builder.build_indicator_snapshot_summary(events)

        assert result["decisions"]["enter"] == 1
        assert result["decisions"]["skip"] == 1
        assert result["decisions"]["exit"] == 1

        rsi = result["indicators"]["rsi_14"]
        assert rsi["count"] == 3
        assert rsi["mean"] == pytest.approx(50.0)
        assert rsi["min"] == 30.0
        assert rsi["max"] == 70.0

        atr = result["indicators"]["atr_20"]
        assert atr["count"] == 2
        assert atr["mean"] == pytest.approx(1.75)


class TestOrderbookSummary:
    def test_computes_spread_and_imbalance_averages(self):
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        events = [
            {"spread_bps": 2.0, "imbalance_ratio": 1.2, "trade_context": "entry"},
            {"spread_bps": 4.0, "imbalance_ratio": 0.8, "trade_context": "exit"},
            {"spread_bps": 3.0, "imbalance_ratio": 1.0, "trade_context": None},
        ]
        result = builder.build_orderbook_summary(events)

        assert result["event_count"] == 3
        assert result["avg_spread_bps"] == pytest.approx(3.0)
        assert result["max_spread_bps"] == 4.0
        assert result["avg_imbalance_ratio"] == pytest.approx(1.0)
        assert result["entry_count"] == 1
        assert result["exit_count"] == 1

    def test_empty_events_returns_empty_dict(self):
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        assert builder.build_orderbook_summary([]) == {}


# ===========================================================================
# 3. Backward compatibility
# ===========================================================================

class TestBuilderBackwardCompatibility:
    def test_write_curated_works_without_enriched_events(self, tmp_path):
        """Existing callers that pass no enriched events should still work."""
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list[MissedOpportunityEvent] = []

        output_dir = builder.write_curated(trades, missed, tmp_path)

        # Core files must still be written
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "regime_analysis.json").exists()

        # Enriched files should NOT be written when no enriched events provided
        assert not (output_dir / "filter_decisions.json").exists()
        assert not (output_dir / "indicator_snapshots.json").exists()
        assert not (output_dir / "orderbook_context.json").exists()

    def test_write_curated_writes_enriched_files_when_provided(self, tmp_path):
        """When enriched events are provided, their summary files are written."""
        builder = DailyMetricsBuilder(date="2026-03-01", bot_id="bot1")
        trades = [_make_trade("t1", "bot1", 100.0)]
        missed: list[MissedOpportunityEvent] = []

        filter_events = [{"filter_name": "vol", "passed": True, "margin_pct": 5.0}]
        indicator_events = [{"indicators": {"rsi": 50.0}, "decision": "skip"}]
        orderbook_events = [{"spread_bps": 2.0, "imbalance_ratio": 1.0, "trade_context": "entry"}]

        output_dir = builder.write_curated(
            trades, missed, tmp_path,
            filter_decision_events=filter_events,
            indicator_snapshot_events=indicator_events,
            orderbook_context_events=orderbook_events,
        )

        assert (output_dir / "filter_decisions.json").exists()
        assert (output_dir / "indicator_snapshots.json").exists()
        assert (output_dir / "orderbook_context.json").exists()

        # Verify content is valid JSON
        filter_data = json.loads((output_dir / "filter_decisions.json").read_text())
        assert "vol" in filter_data
        assert filter_data["vol"]["total_decisions"] == 1
