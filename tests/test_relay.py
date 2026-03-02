import json

import pytest
from httpx import ASGITransport, AsyncClient

from relay.app import create_relay_app
from relay.auth import compute_hmac, verify_hmac


SHARED_SECRET = "test-secret-key-12345"


@pytest.fixture
async def relay_client(tmp_db_path):
    app = create_relay_app(db_path=str(tmp_db_path), shared_secrets={"bot1": SHARED_SECRET, "bot2": SHARED_SECRET})
    # Manually initialize/close store since httpx ASGITransport does not trigger lifespan
    await app.state.store.initialize()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await app.state.store.close()


def _sign_payload(payload: dict, secret: str) -> str:
    body = json.dumps(payload, sort_keys=True)
    return compute_hmac(body, secret)


class TestRelayAuth:
    def test_compute_hmac(self):
        sig = compute_hmac("test body", "secret")
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex digest

    def test_verify_hmac(self):
        body = "test body"
        secret = "secret"
        sig = compute_hmac(body, secret)
        assert verify_hmac(body, sig, secret) is True

    def test_verify_hmac_rejects_bad_sig(self):
        assert verify_hmac("body", "bad_signature", "secret") is False


class TestRelayEndpoints:
    async def test_post_events(self, relay_client: AsyncClient):
        payload = {
            "bot_id": "bot1",
            "events": [
                {
                    "event_id": "e001",
                    "bot_id": "bot1",
                    "event_type": "trade",
                    "payload": json.dumps({"trade_id": "t001"}),
                    "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                },
            ],
        }
        sig = _sign_payload(payload, SHARED_SECRET)
        resp = await relay_client.post(
            "/events",
            json=payload,
            headers={"X-Signature": sig},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 1
        assert data["duplicates"] == 0

    async def test_post_rejects_bad_signature(self, relay_client: AsyncClient):
        payload = {"bot_id": "bot1", "events": []}
        resp = await relay_client.post(
            "/events",
            json=payload,
            headers={"X-Signature": "invalid"},
        )
        assert resp.status_code == 401

    async def test_post_rejects_unknown_bot(self, relay_client: AsyncClient):
        payload = {"bot_id": "unknown_bot", "events": []}
        resp = await relay_client.post(
            "/events",
            json=payload,
            headers={"X-Signature": "anything"},
        )
        assert resp.status_code == 401

    async def test_get_events_with_watermark(self, relay_client: AsyncClient):
        # Post two events
        for eid in ["w1", "w2"]:
            payload = {
                "bot_id": "bot1",
                "events": [
                    {
                        "event_id": eid,
                        "bot_id": "bot1",
                        "event_type": "trade",
                        "payload": "{}",
                        "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                    },
                ],
            }
            sig = _sign_payload(payload, SHARED_SECRET)
            await relay_client.post("/events", json=payload, headers={"X-Signature": sig})

        # Pull all
        resp = await relay_client.get("/events")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) == 2

    async def test_ack_watermark(self, relay_client: AsyncClient):
        # Post an event
        payload = {
            "bot_id": "bot1",
            "events": [
                {
                    "event_id": "ack1",
                    "bot_id": "bot1",
                    "event_type": "trade",
                    "payload": "{}",
                    "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                },
            ],
        }
        sig = _sign_payload(payload, SHARED_SECRET)
        await relay_client.post("/events", json=payload, headers={"X-Signature": sig})

        # Ack it
        resp = await relay_client.post("/ack", json={"watermark": "ack1"})
        assert resp.status_code == 200

        # Pull again — should get nothing after watermark
        resp = await relay_client.get("/events", params={"since": "ack1"})
        data = resp.json()
        assert len(data["events"]) == 0

    async def test_dedup_on_relay(self, relay_client: AsyncClient):
        """Posting the same event_id twice only stores one."""
        for _ in range(2):
            payload = {
                "bot_id": "bot1",
                "events": [
                    {
                        "event_id": "dup1",
                        "bot_id": "bot1",
                        "event_type": "trade",
                        "payload": "{}",
                        "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                    },
                ],
            }
            sig = _sign_payload(payload, SHARED_SECRET)
            await relay_client.post("/events", json=payload, headers={"X-Signature": sig})

        resp = await relay_client.get("/events")
        assert len(resp.json()["events"]) == 1
