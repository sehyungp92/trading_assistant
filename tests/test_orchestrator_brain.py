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

    def test_unknown_event_type_logged(self, brain: OrchestratorBrain):
        event = {
            "event_id": "u001",
            "bot_id": "bot1",
            "event_type": "something_new",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert actions[0].type == ActionType.LOG_UNKNOWN
