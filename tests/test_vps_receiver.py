"""Tests for VPSReceiver — uses a fake in-memory relay (no relay/ import)."""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from orchestrator.adapters.vps_receiver import VPSReceiver
from orchestrator.db.queue import EventQueue


# ---------------------------------------------------------------------------
# Fake relay — implements the HTTP contract without importing relay/
# ---------------------------------------------------------------------------

class FakeRelay:
    """Minimal in-memory relay implementing GET /events + POST /ack."""

    def __init__(self):
        self.events: list[dict] = []
        self._acked_through: int = -1  # index into self.events

    def seed(self, count: int = 3, prefix: str = "pull"):
        for i in range(count):
            self.events.append({
                "event_id": f"{prefix}{i}",
                "bot_id": "bot1",
                "event_type": "trade",
                "payload": json.dumps({"trade_id": f"t{i}"}),
                "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            })

    async def handle_get_events(self, request: Request) -> JSONResponse:
        since = request.query_params.get("since")
        limit = int(request.query_params.get("limit", "100"))

        start = 0
        if since:
            for idx, e in enumerate(self.events):
                if e["event_id"] == since:
                    start = idx + 1
                    break

        results = []
        for e in self.events[start:]:
            if results and len(results) >= limit:
                break
            results.append(e)
        return JSONResponse({"events": results})

    async def handle_ack(self, request: Request) -> JSONResponse:
        body = await request.json()
        watermark = body["watermark"]
        for idx, e in enumerate(self.events):
            if e["event_id"] == watermark:
                self._acked_through = idx
                break
        return JSONResponse({"status": "ok"})

    def as_app(self) -> Starlette:
        return Starlette(routes=[
            Route("/events", self.handle_get_events),
            Route("/ack", self.handle_ack, methods=["POST"]),
        ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_relay():
    return FakeRelay()


@pytest.fixture
def relay_app(fake_relay):
    return fake_relay.as_app()


@pytest.fixture
async def local_queue(tmp_path):
    q = EventQueue(db_path=str(tmp_path / "local.db"))
    await q.initialize()
    yield q
    await q.close()


def _make_receiver(relay_app, local_queue: EventQueue, **kwargs) -> VPSReceiver:
    transport = ASGITransport(app=relay_app)

    def client_factory():
        return AsyncClient(transport=transport, base_url="http://relay")

    return VPSReceiver(
        relay_url="http://relay",
        local_queue=local_queue,
        _client_factory=client_factory,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVPSReceiver:
    async def test_pull_events_into_local_queue(self, fake_relay, relay_app, local_queue):
        fake_relay.seed(count=3)

        receiver = _make_receiver(relay_app, local_queue)
        pulled = await receiver.pull_and_store()
        assert pulled == 3

        pending = await local_queue.peek(limit=10)
        assert len(pending) == 3

    async def test_pull_empty_relay(self, relay_app, local_queue):
        receiver = _make_receiver(relay_app, local_queue)
        pulled = await receiver.pull_and_store()
        assert pulled == 0

    async def test_ack_after_pull(self, fake_relay, relay_app, local_queue):
        fake_relay.seed(count=2)

        receiver = _make_receiver(relay_app, local_queue)
        await receiver.pull_and_store()

        # Second pull should find nothing (acked)
        pulled = await receiver.pull_and_store()
        assert pulled == 0


class TestWatermarkPersistence:
    async def test_watermark_persisted_to_db(self, fake_relay, relay_app, local_queue):
        fake_relay.seed(count=3)

        receiver = _make_receiver(relay_app, local_queue)
        await receiver.pull_and_store()

        watermark = await local_queue.get_watermark("relay")
        assert watermark == "pull2"

    async def test_watermark_loaded_on_next_pull(self, fake_relay, relay_app, local_queue):
        """A second VPSReceiver instance resumes from persisted watermark."""
        fake_relay.seed(count=2)

        receiver1 = _make_receiver(relay_app, local_queue)
        await receiver1.pull_and_store()

        # Seed more events after the first batch
        fake_relay.events.append({
            "event_id": "pull_new",
            "bot_id": "bot1",
            "event_type": "trade",
            "payload": json.dumps({"trade_id": "new"}),
            "exchange_timestamp": "2026-03-01T15:00:00+00:00",
        })

        # New receiver loads watermark from DB — only gets the new event
        receiver2 = _make_receiver(relay_app, local_queue)
        pulled = await receiver2.pull_and_store()
        assert pulled == 1

        pending = await local_queue.peek(limit=10)
        event_ids = [e["event_id"] for e in pending]
        assert "pull_new" in event_ids


class TestPollRetry:
    async def test_poll_handles_connection_error(self, local_queue):
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

    async def test_poll_handles_http_error(self, local_queue):
        """Relay returns 500 → returns 0, no exception."""

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


class TestLatencyRecording:
    async def test_pull_records_latency_when_tracker_provided(
        self, fake_relay, relay_app, local_queue,
    ):
        """VPSReceiver records latency for each event when tracker is present."""
        from orchestrator.latency_tracker import LatencyTracker

        fake_relay.seed(count=2)
        tracker = LatencyTracker()

        transport = ASGITransport(app=relay_app)
        receiver = VPSReceiver(
            relay_url="http://relay",
            local_queue=local_queue,
            latency_tracker=tracker,
            _client_factory=lambda: AsyncClient(transport=transport, base_url="http://relay"),
        )
        pulled = await receiver.pull_and_store()
        assert pulled == 2

        stats = tracker.get_stats("bot1")
        assert stats.sample_count == 2


class TestApiKeyAuth:
    async def test_api_key_sent_in_headers(self, fake_relay, local_queue):
        """When api_key is set, X-Api-Key header is included in requests."""
        fake_relay.seed(count=1)
        captured_headers: dict = {}

        async def capture_handler(request: Request) -> JSONResponse:
            captured_headers.update(dict(request.headers))
            return await fake_relay.handle_get_events(request)

        capture_app = Starlette(routes=[
            Route("/events", capture_handler),
            Route("/ack", fake_relay.handle_ack, methods=["POST"]),
        ])
        transport = ASGITransport(app=capture_app)

        receiver = VPSReceiver(
            relay_url="http://relay",
            local_queue=local_queue,
            api_key="test-key-123",
            _client_factory=lambda: AsyncClient(transport=transport, base_url="http://relay",
                                                headers={"X-Api-Key": "test-key-123"}),
        )
        await receiver.pull_and_store()
        assert captured_headers.get("x-api-key") == "test-key-123"

    async def test_no_api_key_when_empty(self, fake_relay, local_queue):
        """When api_key is empty, no X-Api-Key header is sent."""
        fake_relay.seed(count=1)
        captured_headers: dict = {}

        async def capture_handler(request: Request) -> JSONResponse:
            captured_headers.update(dict(request.headers))
            return await fake_relay.handle_get_events(request)

        capture_app = Starlette(routes=[
            Route("/events", capture_handler),
            Route("/ack", fake_relay.handle_ack, methods=["POST"]),
        ])
        transport = ASGITransport(app=capture_app)

        receiver = VPSReceiver(
            relay_url="http://relay",
            local_queue=local_queue,
            api_key="",
            _client_factory=lambda: AsyncClient(transport=transport, base_url="http://relay"),
        )
        await receiver.pull_and_store()
        assert "x-api-key" not in captured_headers

    async def test_401_triggers_poll_failure(self, local_queue):
        """Relay returning 401 is handled gracefully by poll()."""

        async def unauthorized_handler(request):
            return PlainTextResponse("Unauthorized", status_code=401)

        error_app = Starlette(routes=[Route("/events", unauthorized_handler)])
        transport = ASGITransport(app=error_app)

        receiver = VPSReceiver(
            relay_url="http://relay",
            local_queue=local_queue,
            api_key="wrong-key",
            _client_factory=lambda: AsyncClient(transport=transport, base_url="http://relay"),
        )
        result = await receiver.poll()
        assert result == 0
        assert receiver._consecutive_failures == 1

    async def test_make_client_includes_api_key(self):
        """_make_client() sets X-Api-Key header when api_key is provided."""
        from unittest.mock import MagicMock

        queue_mock = MagicMock()
        receiver = VPSReceiver(
            relay_url="http://relay.example.com",
            local_queue=queue_mock,
            api_key="my-secret-key",
        )
        client = receiver._make_client()
        assert client.headers.get("x-api-key") == "my-secret-key"
        await client.aclose()

    async def test_make_client_no_api_key_header_when_empty(self):
        """_make_client() does not set X-Api-Key when api_key is empty."""
        from unittest.mock import MagicMock

        queue_mock = MagicMock()
        receiver = VPSReceiver(
            relay_url="http://relay.example.com",
            local_queue=queue_mock,
            api_key="",
        )
        client = receiver._make_client()
        assert "x-api-key" not in client.headers
        await client.aclose()


class TestDrain:
    async def test_drain_pulls_multiple_batches(self, fake_relay, relay_app, local_queue):
        """Seeds 5 events, drains with batch_size=2 — should pull all in 3 batches."""
        fake_relay.seed(count=5)

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

    async def test_drain_caps_at_max_batches(self, fake_relay, relay_app, local_queue):
        """Drain respects max_batches limit."""
        fake_relay.seed(count=5)

        receiver = _make_receiver(relay_app, local_queue)
        # batch_size=1, max_batches=3 → should only get 3 events
        total = await receiver.drain(batch_size=1, max_batches=3)
        assert total == 3
