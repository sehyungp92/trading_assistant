import json
from unittest.mock import AsyncMock

import pytest

from orchestrator.worker import Worker
from orchestrator.orchestrator_brain import OrchestratorBrain
from orchestrator.db.queue import EventQueue
from orchestrator.task_registry import TaskRegistry


@pytest.fixture
async def queue(tmp_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_path / "queue.db"))
    await q.initialize()
    return q


@pytest.fixture
async def registry(tmp_path) -> TaskRegistry:
    r = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
    await r.initialize()
    return r


@pytest.fixture
def brain() -> OrchestratorBrain:
    return OrchestratorBrain()


@pytest.fixture
def worker(queue, registry, brain) -> Worker:
    return Worker(queue=queue, registry=registry, brain=brain)


class TestWorker:
    async def test_process_pending_events(self, worker: Worker, queue: EventQueue):
        await queue.enqueue({
            "event_id": "e001",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": json.dumps({"trade_id": "t001"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 1

        # Event should be acked
        pending = await queue.peek(limit=10)
        assert len(pending) == 0

    async def test_critical_error_calls_alert_handler(self, worker: Worker, queue: EventQueue):
        worker.on_alert = AsyncMock()

        await queue.enqueue({
            "event_id": "err001",
            "bot_id": "bot3",
            "event_type": "error",
            "payload": json.dumps({"severity": "CRITICAL", "message": "crash"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        await worker.process_batch(limit=10)
        worker.on_alert.assert_called_once()

    async def test_heartbeat_calls_heartbeat_handler(self, worker: Worker, queue: EventQueue):
        worker.on_heartbeat = AsyncMock()

        await queue.enqueue({
            "event_id": "hb001",
            "bot_id": "bot1",
            "event_type": "heartbeat",
            "payload": "{}",
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        await worker.process_batch(limit=10)
        worker.on_heartbeat.assert_called_once()

    async def test_empty_queue_returns_zero(self, worker: Worker):
        processed = await worker.process_batch(limit=10)
        assert processed == 0
