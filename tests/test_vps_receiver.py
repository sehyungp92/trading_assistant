import json

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.adapters.vps_receiver import VPSReceiver
from orchestrator.db.queue import EventQueue
from relay.app import create_relay_app
from relay.auth import compute_hmac


SHARED_SECRET = "test-secret"


@pytest.fixture
async def relay_client(tmp_path):
    app = create_relay_app(
        db_path=str(tmp_path / "relay.db"),
        shared_secrets={"bot1": SHARED_SECRET},
    )
    # Manually initialize/close store since httpx ASGITransport does not trigger lifespan
    await app.state.store.initialize()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://relay") as client:
        yield client
    await app.state.store.close()


@pytest.fixture
async def local_queue(tmp_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_path / "local.db"))
    await q.initialize()
    return q


async def _seed_relay(relay_client: AsyncClient, count: int = 3):
    """Push events to the relay for the receiver to pull."""
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
        await relay_client.post("/events", json=payload, headers={"X-Signature": sig})


class TestVPSReceiver:
    async def test_pull_events_into_local_queue(self, relay_client, local_queue):
        await _seed_relay(relay_client, count=3)

        receiver = VPSReceiver(relay_client=relay_client, local_queue=local_queue)
        pulled = await receiver.pull_and_store()
        assert pulled == 3

        pending = await local_queue.peek(limit=10)
        assert len(pending) == 3

    async def test_pull_empty_relay(self, relay_client, local_queue):
        receiver = VPSReceiver(relay_client=relay_client, local_queue=local_queue)
        pulled = await receiver.pull_and_store()
        assert pulled == 0

    async def test_ack_after_pull(self, relay_client, local_queue):
        await _seed_relay(relay_client, count=2)

        receiver = VPSReceiver(relay_client=relay_client, local_queue=local_queue)
        await receiver.pull_and_store()

        # Second pull should find nothing (acked)
        pulled = await receiver.pull_and_store()
        assert pulled == 0
