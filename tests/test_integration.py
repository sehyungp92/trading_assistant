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

    async def test_list_tasks(self, client: AsyncClient):
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    async def test_pending_events(self, client: AsyncClient):
        resp = await client.get("/events/pending")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
