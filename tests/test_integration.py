import json

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.app import create_app


@pytest.fixture
async def client(tmp_path):
    app = create_app(db_dir=str(tmp_path))
    # Manually initialize queue and registry since httpx ASGITransport
    # does not trigger FastAPI lifespan events
    await app.state.queue.initialize()
    await app.state.registry.initialize()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await app.state.queue.close()
    await app.state.registry.close()


class TestOrchestratorApp:
    async def test_health_endpoint(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "scheduler" in data
        assert "timestamp" in data

    async def test_ingest_event(self, client: AsyncClient):
        """Direct event ingest for testing without relay."""
        event = {
            "event_id": "direct001",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": json.dumps({"trade_id": "t001"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
        }
        resp = await client.post("/ingest", json=event)
        assert resp.status_code == 200
        data = resp.json()
        assert data["inserted"] is True

    async def test_ingest_duplicate_event(self, client: AsyncClient):
        event = {
            "event_id": "dup001",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": json.dumps({"trade_id": "t001"}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
        }
        await client.post("/ingest", json=event)
        resp = await client.post("/ingest", json=event)
        data = resp.json()
        assert data["inserted"] is False

    async def test_ingest_backfills_required_queue_timestamps(self, client: AsyncClient):
        event = {
            "event_id": "direct002",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": {"trade_id": "t002"},
        }

        resp = await client.post("/ingest", json=event)

        assert resp.status_code == 200
        pending = (await client.get("/events/pending")).json()
        stored = next(item for item in pending if item["event_id"] == "direct002")
        assert stored["exchange_timestamp"] == stored["received_at"]
        assert json.loads(stored["payload"]) == {"trade_id": "t002"}

    async def test_feedback_enqueues_queue_timestamps(self, client: AsyncClient):
        resp = await client.post("/feedback", json={
            "text": "please tighten stops",
            "bot_id": "bot1",
            "report_id": "report-123",
        })

        assert resp.status_code == 200
        event_id = resp.json()["event_id"]
        pending = (await client.get("/events/pending")).json()
        stored = next(item for item in pending if item["event_id"] == event_id)
        assert stored["event_type"] == "user_feedback"
        assert stored["exchange_timestamp"]
        assert stored["received_at"]
        assert json.loads(stored["payload"]) == {
            "text": "please tighten stops",
            "report_id": "report-123",
        }

    async def test_list_tasks(self, client: AsyncClient):
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_pending_events(self, client: AsyncClient):
        resp = await client.get("/events/pending")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
