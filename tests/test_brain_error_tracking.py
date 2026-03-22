"""Tests for brain error frequency tracking — prevents error storm resource waste."""
import json as jsonmod
from orchestrator.orchestrator_brain import OrchestratorBrain, ActionType
from tests.factories import make_error_event


def _make_error_event(event_id, bot_id="bot1", severity="HIGH", error_type="ConnectionError"):
    base = make_error_event(event_id=event_id, bot_id=bot_id, severity=severity, error_type=error_type)
    return {
        "event_type": "error",
        "event_id": base["event_id"],
        "bot_id": base["bot_id"],
        "payload": jsonmod.dumps({"severity": base["severity"], "error_type": base["error_type"]}),
    }


class TestErrorFrequencyTracking:
    def test_first_high_error_spawns_triage(self):
        brain = OrchestratorBrain()
        actions = brain.decide(_make_error_event("e1"))
        assert len(actions) == 1
        assert actions[0].type == ActionType.SPAWN_TRIAGE

    def test_duplicate_errors_consolidated_into_one_triage(self):
        brain = OrchestratorBrain()
        a1 = brain.decide(_make_error_event("e1", error_type="ConnectionError"))
        assert a1[0].type == ActionType.SPAWN_TRIAGE
        a2 = brain.decide(_make_error_event("e2", error_type="ConnectionError"))
        assert len(a2) == 1
        assert a2[0].type == ActionType.QUEUE_FOR_DAILY

    def test_different_error_types_not_suppressed(self):
        brain = OrchestratorBrain()
        brain.decide(_make_error_event("e1", error_type="ConnectionError"))
        a2 = brain.decide(_make_error_event("e2", error_type="TimeoutError"))
        assert a2[0].type == ActionType.SPAWN_TRIAGE

    def test_three_same_errors_creates_urgency_flag(self):
        brain = OrchestratorBrain()
        brain.decide(_make_error_event("e1", error_type="ConnectionError"))
        brain.decide(_make_error_event("e2", error_type="ConnectionError"))
        a3 = brain.decide(_make_error_event("e3", error_type="ConnectionError"))
        assert a3[0].type == ActionType.SPAWN_TRIAGE
        assert a3[0].details.get("urgency") == "error_storm"
        assert a3[0].details.get("error_count") >= 3

    def test_different_bots_tracked_separately(self):
        brain = OrchestratorBrain()
        brain.decide(_make_error_event("e1", bot_id="bot1", error_type="X"))
        a2 = brain.decide(_make_error_event("e2", bot_id="bot2", error_type="X"))
        assert a2[0].type == ActionType.SPAWN_TRIAGE

    def test_critical_errors_never_suppressed(self):
        brain = OrchestratorBrain()
        brain.decide(_make_error_event("e1", severity="CRITICAL", error_type="X"))
        a2 = brain.decide(_make_error_event("e2", severity="CRITICAL", error_type="X"))
        assert a2[0].type == ActionType.ALERT_IMMEDIATE
