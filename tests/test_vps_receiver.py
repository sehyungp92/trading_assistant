import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.adapters.vps_receiver import VPSReceiver
from orchestrator.db.queue import EventQueue
from relay.app import create_relay_app
from relay.auth import compute_hmac


SHARED_SECRET = "test-secret"


@pytest.fixture
async def relay_app(tmp_path):
    app = create_relay_app(
        db_path=str(tmp_path / "relay.db"),
        shared_secrets={"bot1": SHARED_SECRET},
    )
    await app.state.store.initialize()
    yield app
    await app.state.store.close()


@pytest.fixture
async def local_queue(tmp_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_path / "local.db"))
    await q.initialize()
    return q


def _make_receiver(relay_app, local_queue: EventQueue, **kwargs) -> VPSReceiver:
    """Create a VPSReceiver that routes HTTP through the in-process relay app."""
    transport = ASGITransport(app=relay_app)

    def client_factory():
        return AsyncClient(transport=transport, base_url="http://relay")

    return VPSReceiver(
        relay_url="http://relay",
        local_queue=local_queue,
        _client_factory=client_factory,
        **kwargs,
    )


async def _seed_relay(relay_app, count: int = 3):
    """Push events to the relay for the receiver to pull."""
    transport = ASGITransport(app=relay_app)
    async with AsyncClient(transport=transport, base_url="http://relay") as client:
        for i in range(count):
            payload = {
                "bot_id": "bot1",
                "events": [
                    {
                        "event_id": f"pull{i}",
                        "bot_id": "bot1",
                        "event_type": "trade",
                        "payload": json.dumps({"trade_id": f"t{i}"}),
                        "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                    },
                ],
            }
            sig = compute_hmac(json.dumps(payload, sort_keys=True), SHARED_SECRET)
            await client.post("/events", json=payload, headers={"X-Signature": sig})


class TestVPSReceiver:
    async def test_pull_events_into_local_queue(self, relay_app, local_queue):
        await _seed_relay(relay_app, count=3)

        receiver = _make_receiver(relay_app, local_queue)
        pulled = await receiver.pull_and_store()
        assert pulled == 3

        pending = await local_queue.peek(limit=10)
        assert len(pending) == 3

    async def test_pull_empty_relay(self, relay_app, local_queue):
        receiver = _make_receiver(relay_app, local_queue)
        pulled = await receiver.pull_and_store()
        assert pulled == 0

    async def test_ack_after_pull(self, relay_app, local_queue):
        await _seed_relay(relay_app, count=2)

        receiver = _make_receiver(relay_app, local_queue)
        await receiver.pull_and_store()

        # Second pull should find nothing (acked)
        pulled = await receiver.pull_and_store()
        assert pulled == 0


class TestWatermarkPersistence:
    async def test_watermark_persisted_to_db(self, relay_app, local_queue):
        await _seed_relay(relay_app, count=3)

        receiver = _make_receiver(relay_app, local_queue)
        await receiver.pull_and_store()

        watermark = await local_queue.get_watermark("relay")
        assert watermark == "pull2"

    async def test_watermark_loaded_on_next_pull(self, relay_app, local_queue):
        """A second VPSReceiver instance resumes from persisted watermark."""
        await _seed_relay(relay_app, count=2)

        receiver1 = _make_receiver(relay_app, local_queue)
        await receiver1.pull_and_store()

        # Seed more events after the first batch
        transport = ASGITransport(app=relay_app)
        async with AsyncClient(transport=transport, base_url="http://relay") as client:
            payload = {
                "bot_id": "bot1",
                "events": [
                    {
                        "event_id": "pull_new",
                        "bot_id": "bot1",
                        "event_type": "trade",
                        "payload": json.dumps({"trade_id": "new"}),
                        "exchange_timestamp": "2026-03-01T15:00:00+00:00",
                    },
                ],
            }
            sig = compute_hmac(json.dumps(payload, sort_keys=True), SHARED_SECRET)
            await client.post("/events", json=payload, headers={"X-Signature": sig})

        # New receiver loads watermark from DB — only gets the new event
        receiver2 = _make_receiver(relay_app, local_queue)
        pulled = await receiver2.pull_and_store()
        assert pulled == 1

        pending = await local_queue.peek(limit=10)
        event_ids = [e["event_id"] for e in pending]
        assert "pull_new" in event_ids


class TestPollRetry:
    async def test_poll_handles_connection_error(self, relay_app, local_queue):
        """Relay unreachable returns 0, no exception propagated."""

        def bad_factory():
            return AsyncClient(base_url="http://localhost:1")

        receiver = VPSReceiver(
            relay_url="http://localhost:1",
            local_queue=local_queue,
            _client_factory=bad_factory,
            timeout=0.5,
        )
        result = await receiver.poll()
        assert result == 0
        assert receiver._consecutive_failures == 1

    async def test_poll_handles_http_error(self, relay_app, local_queue):
        """Relay returns 500 → returns 0, no exception."""
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def error_handler(request):
            return PlainTextResponse("Internal Server Error", status_code=500)

        error_app = Starlette(routes=[Route("/events", error_handler)])
        transport = ASGITransport(app=error_app)

        def client_factory():
            return AsyncClient(transport=transport, base_url="http://relay")

        receiver = VPSReceiver(
            relay_url="http://relay",
            local_queue=local_queue,
            _client_factory=client_factory,
        )
        result = await receiver.poll()
        assert result == 0
        assert receiver._consecutive_failures == 1

    async def test_poll_resets_failure_count_on_success(self, relay_app, local_queue):
        receiver = _make_receiver(relay_app, local_queue)
        receiver._consecutive_failures = 5

        result = await receiver.poll()
        assert result == 0  # empty relay, but no error
        assert receiver._consecutive_failures == 0


class TestDrain:
    async def test_drain_pulls_multiple_batches(self, relay_app, local_queue):
        """Seeds 5 events, drains with batch_size=2 — should pull all in 3 batches."""
        await _seed_relay(relay_app, count=5)

        receiver = _make_receiver(relay_app, local_queue)
        total = await receiver.drain(batch_size=2, max_batches=10)
        assert total == 5

        pending = await local_queue.peek(limit=10)
        assert len(pending) == 5

    async def test_drain_stops_when_empty(self, relay_app, local_queue):
        """Drain with no events returns 0 immediately."""
        receiver = _make_receiver(relay_app, local_queue)
        total = await receiver.drain()
        assert total == 0

    async def test_drain_caps_at_max_batches(self, relay_app, local_queue):
        """Drain respects max_batches limit."""
        await _seed_relay(relay_app, count=5)

        receiver = _make_receiver(relay_app, local_queue)
        # batch_size=1, max_batches=3 → should only get 3 events
        total = await receiver.drain(batch_size=1, max_batches=3)
        assert total == 3
