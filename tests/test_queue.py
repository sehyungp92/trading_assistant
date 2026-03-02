import json
from datetime import datetime, timezone

import pytest

from orchestrator.db.queue import EventQueue


@pytest.fixture
async def queue(tmp_db_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_db_path))
    await q.initialize()
    return q


def _make_event(event_id: str = "abc123", bot_id: str = "bot1", event_type: str = "trade") -> dict:
    return {
        "event_id": event_id,
        "bot_id": bot_id,
        "event_type": event_type,
        "payload": json.dumps({"trade_id": "t001", "pnl": 50.0}),
        "exchange_timestamp": datetime(2026, 3, 1, 14, 0, tzinfo=timezone.utc).isoformat(),
        "received_at": datetime.now(timezone.utc).isoformat(),
    }


class TestEventQueue:
    async def test_enqueue_and_dequeue(self, queue: EventQueue):
        event = _make_event()
        inserted = await queue.enqueue(event)
        assert inserted is True

        pending = await queue.peek(limit=10)
        assert len(pending) == 1
        assert pending[0]["event_id"] == "abc123"

    async def test_idempotent_insert(self, queue: EventQueue):
        """Inserting the same event_id twice does not create duplicates."""
        event = _make_event(event_id="dup001")
        first = await queue.enqueue(event)
        second = await queue.enqueue(event)
        assert first is True
        assert second is False  # duplicate silently ignored

        pending = await queue.peek(limit=10)
        assert len(pending) == 1

    async def test_ack_removes_from_pending(self, queue: EventQueue):
        event = _make_event(event_id="ack001")
        await queue.enqueue(event)
        await queue.ack("ack001")

        pending = await queue.peek(limit=10)
        assert len(pending) == 0

    async def test_batch_enqueue(self, queue: EventQueue):
        events = [_make_event(event_id=f"batch{i}") for i in range(5)]
        result = await queue.enqueue_batch(events)
        assert result.inserted == 5
        assert result.duplicates == 0

        pending = await queue.peek(limit=10)
        assert len(pending) == 5

    async def test_batch_enqueue_with_duplicates(self, queue: EventQueue):
        events = [_make_event(event_id="same")] * 3
        result = await queue.enqueue_batch(events)
        assert result.inserted == 1
        assert result.duplicates == 2

    async def test_peek_respects_limit(self, queue: EventQueue):
        for i in range(10):
            await queue.enqueue(_make_event(event_id=f"limit{i}"))

        pending = await queue.peek(limit=3)
        assert len(pending) == 3

    async def test_watermark_tracking(self, queue: EventQueue):
        """Watermark tracks the latest acked event for a bot."""
        await queue.enqueue(_make_event(event_id="w1", bot_id="bot1"))
        await queue.enqueue(_make_event(event_id="w2", bot_id="bot1"))
        await queue.ack("w1")
        await queue.update_watermark("bot1", "w1")

        wm = await queue.get_watermark("bot1")
        assert wm == "w1"
