# tests/test_wfo_orchestrator_wiring.py
"""Tests for WFO orchestrator wiring — brain, worker, scheduler."""
import asyncio

from orchestrator.orchestrator_brain import OrchestratorBrain, ActionType
from orchestrator.worker import Worker
from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


class TestBrainWFORouting:
    def test_wfo_trigger_spawns_wfo(self):
        brain = OrchestratorBrain()
        event = {
            "event_id": "wfo-bot2-2026-03-01",
            "event_type": "wfo_trigger",
            "bot_id": "bot2",
            "payload": "{}",
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.SPAWN_WFO
        assert actions[0].bot_id == "bot2"


class TestWorkerWFODispatch:
    def test_dispatches_to_wfo_handler(self):
        from unittest.mock import AsyncMock, MagicMock

        from orchestrator.db.queue import EventQueue
        from orchestrator.task_registry import TaskRegistry

        queue = AsyncMock(spec=EventQueue)
        queue.peek.return_value = [
            {
                "event_id": "wfo-bot2-2026-03-01",
                "event_type": "wfo_trigger",
                "bot_id": "bot2",
                "payload": "{}",
            },
        ]

        registry = MagicMock(spec=TaskRegistry)
        brain = OrchestratorBrain()
        worker = Worker(queue, registry, brain)
        handler = AsyncMock()
        worker.on_wfo = handler

        asyncio.get_event_loop().run_until_complete(worker.process_batch(limit=1))
        handler.assert_called_once()
        action = handler.call_args[0][0]
        assert action.type == ActionType.SPAWN_WFO


class TestSchedulerWFOConfig:
    def test_scheduler_config_has_wfo_fields(self):
        cfg = SchedulerConfig()
        assert hasattr(cfg, "wfo_day_of_week")
        assert hasattr(cfg, "wfo_hour")

    def test_scheduler_creates_wfo_job(self):
        async def noop():
            pass

        cfg = SchedulerConfig()
        jobs = create_scheduler_jobs(
            cfg,
            worker_fn=noop,
            monitoring_fn=noop,
            relay_fn=noop,
            wfo_fn=noop,
        )
        wfo_jobs = [j for j in jobs if j["name"] == "wfo"]
        assert len(wfo_jobs) == 1
        assert wfo_jobs[0]["trigger"] == "cron"

    def test_scheduler_omits_wfo_without_fn(self):
        async def noop():
            pass

        cfg = SchedulerConfig()
        jobs = create_scheduler_jobs(
            cfg,
            worker_fn=noop,
            monitoring_fn=noop,
            relay_fn=noop,
        )
        wfo_jobs = [j for j in jobs if j["name"] == "wfo"]
        assert len(wfo_jobs) == 0
