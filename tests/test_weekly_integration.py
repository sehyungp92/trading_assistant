# tests/test_weekly_integration.py
"""Integration test: weekly analysis trigger flows through brain → worker → handler."""
import pytest

from orchestrator.orchestrator_brain import OrchestratorBrain, ActionType


class TestWeeklyBrainRouting:
    def test_weekly_trigger_routes_to_spawn_weekly(self):
        brain = OrchestratorBrain()
        event = {
            "event_type": "weekly_summary_trigger",
            "event_id": "weekly-2026-03-01",
            "bot_id": "",
            "payload": '{"week_start":"2026-02-23","week_end":"2026-03-01"}',
        }
        actions = brain.decide(event)
        assert len(actions) == 1
        assert actions[0].type == ActionType.SPAWN_WEEKLY_SUMMARY
        assert actions[0].details["week_start"] == "2026-02-23"


class TestWeeklyWorkerDispatch:
    @pytest.mark.asyncio
    async def test_weekly_handler_called(self, tmp_path):
        from orchestrator.db.queue import EventQueue
        from orchestrator.db.connection import create_connection
        from orchestrator.task_registry import TaskRegistry
        from orchestrator.worker import Worker

        db_path = str(tmp_path / "test.db")
        queue = EventQueue(db_path)
        await queue.initialize()
        registry = TaskRegistry(db_path)
        await registry.initialize()
        brain = OrchestratorBrain()

        worker = Worker(queue=queue, registry=registry, brain=brain)

        called = {"value": False}

        async def mock_weekly_handler(action):
            called["value"] = True

        worker.on_weekly_analysis = mock_weekly_handler

        # Enqueue a weekly trigger event
        await queue.enqueue({
            "event_id": "weekly-2026-03-01",
            "event_type": "weekly_summary_trigger",
            "bot_id": "",
            "payload": "{}",
            "exchange_timestamp": "2026-03-01T08:00:00+00:00",
            "received_at": "2026-03-01T08:00:01+00:00",
        })

        processed = await worker.process_batch(limit=1)
        assert processed == 1
        assert called["value"] is True

        await queue.close()
        await registry.close()


class TestWeeklySchedulerConfig:
    def test_weekly_analysis_job_created(self):
        from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs

        async def noop():
            pass

        config = SchedulerConfig(
            weekly_analysis_day_of_week="sun",
            weekly_analysis_hour=8,
            weekly_analysis_minute=0,
        )
        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=noop,
            monitoring_fn=noop,
            relay_fn=noop,
            weekly_analysis_fn=noop,
        )
        weekly_jobs = [j for j in jobs if j["name"] == "weekly_analysis"]
        assert len(weekly_jobs) == 1
        assert weekly_jobs[0]["trigger"] == "cron"
        assert weekly_jobs[0]["day_of_week"] == "sun"
        assert weekly_jobs[0]["hour"] == 8
