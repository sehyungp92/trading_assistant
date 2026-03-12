import json

import pytest

from orchestrator.orchestrator_brain import OrchestratorBrain, ActionType


@pytest.fixture
def brain() -> OrchestratorBrain:
    return OrchestratorBrain()


class TestOrchestratorBrain:
    def test_trade_event_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "t001",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": json.dumps({"trade_id": "t001", "pnl": 50.0}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_error_critical_triggers_immediate_alert(self, brain: OrchestratorBrain):
        event = {
            "event_id": "err001",
            "bot_id": "bot3",
            "event_type": "error",
            "payload": json.dumps({"severity": "CRITICAL", "message": "crash"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.ALERT_IMMEDIATE for a in actions)

    def test_error_high_triggers_triage(self, brain: OrchestratorBrain):
        event = {
            "event_id": "err002",
            "bot_id": "bot2",
            "event_type": "error",
            "payload": json.dumps({"severity": "HIGH", "message": "repeated timeout"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.SPAWN_TRIAGE for a in actions)

    def test_error_medium_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "err003",
            "bot_id": "bot1",
            "event_type": "error",
            "payload": json.dumps({"severity": "MEDIUM", "message": "timeout"}),
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_missed_opportunity_queued(self, brain: OrchestratorBrain):
        event = {
            "event_id": "m001",
            "bot_id": "bot1",
            "event_type": "missed_opportunity",
            "payload": json.dumps({"signal": "EMA cross"}),
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_heartbeat_updates_tracking(self, brain: OrchestratorBrain):
        event = {
            "event_id": "hb001",
            "bot_id": "bot1",
            "event_type": "heartbeat",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.UPDATE_HEARTBEAT

    def test_coordinator_action_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ca001",
            "bot_id": "swing_trader",
            "event_type": "coordinator_action",
            "source_strategy": "ATRSS",
            "target_strategy": "Helix",
            "action": "tighten_stops",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].bot_id == "swing_trader"
        assert actions[0].details["event_type"] == "coordinator_action"
        assert actions[0].details["payload"]["action"] == "tighten_stops"

    def test_daily_snapshot_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ds001",
            "bot_id": "bot1",
            "event_type": "daily_snapshot",
            "payload": json.dumps({"total_pnl": 120.0}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY
        assert actions[0].details["event_type"] == "daily_snapshot"

    def test_order_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ord001",
            "bot_id": "bot1",
            "event_type": "order",
            "payload": json.dumps({"order_id": "o1", "side": "BUY"}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_process_quality_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pq001",
            "bot_id": "bot2",
            "event_type": "process_quality",
            "payload": json.dumps({"score": 85}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_bot_error_critical_alerts_immediately(self, brain: OrchestratorBrain):
        event = {
            "event_id": "be001",
            "bot_id": "bot3",
            "event_type": "bot_error",
            "payload": json.dumps({"severity": "CRITICAL", "message": "OOM"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.ALERT_IMMEDIATE for a in actions)

    def test_bot_error_high_triggers_triage(self, brain: OrchestratorBrain):
        event = {
            "event_id": "be002",
            "bot_id": "bot2",
            "event_type": "bot_error",
            "payload": json.dumps({"severity": "HIGH", "error_type": "api_timeout"}),
        }
        actions = brain.decide(event)
        assert any(a.type == ActionType.SPAWN_TRIAGE for a in actions)

    def test_post_exit_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pe001",
            "bot_id": "bot1",
            "event_type": "post_exit",
            "payload": json.dumps({"trade_id": "t1", "post_exit_pnl": 15.0}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_portfolio_rule_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "pr001",
            "bot_id": "bot1",
            "event_type": "portfolio_rule",
            "payload": json.dumps({"rule": "max_exposure", "triggered": True}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_market_snapshot_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "ms001",
            "bot_id": "bot1",
            "event_type": "market_snapshot",
            "payload": json.dumps({"btc_price": 65000}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_exit_movement_queued_for_daily(self, brain: OrchestratorBrain):
        event = {
            "event_id": "em001",
            "bot_id": "bot1",
            "event_type": "exit_movement",
            "payload": json.dumps({"trade_id": "t1", "movement_bps": 25}),
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.QUEUE_FOR_DAILY

    def test_truly_unknown_type_still_logged(self, brain: OrchestratorBrain):
        """Unrecognized event types still get LOG_UNKNOWN."""
        event = {
            "event_id": "u002",
            "bot_id": "bot1",
            "event_type": "totally_new_type",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.LOG_UNKNOWN

    def test_unknown_event_type_logged(self, brain: OrchestratorBrain):
        event = {
            "event_id": "u001",
            "bot_id": "bot1",
            "event_type": "something_new",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.LOG_UNKNOWN
