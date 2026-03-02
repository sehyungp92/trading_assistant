"""VPS Receiver — pulls events from relay VPS into local event queue.

Protocol:
  1. GET /events?since=<watermark> from relay
  2. Store into local EventQueue (dedup handled by queue)
  3. POST /ack with new watermark to relay
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from orchestrator.db.queue import EventQueue

logger = logging.getLogger(__name__)


class VPSReceiver:
    def __init__(
        self,
        relay_client: httpx.AsyncClient,
        local_queue: EventQueue,
    ) -> None:
        self._client = relay_client
        self._queue = local_queue
        self._last_watermark: str | None = None

    async def pull_and_store(self, limit: int = 100) -> int:
        """Pull new events from relay, store locally, ack on relay. Returns count pulled."""
        params: dict = {"limit": limit}
        if self._last_watermark:
            params["since"] = self._last_watermark

        resp = await self._client.get("/events", params=params)
        resp.raise_for_status()

        events = resp.json().get("events", [])
        if not events:
            return 0

        # Add received_at timestamp for local tracking
        now = datetime.now(timezone.utc).isoformat()
        for event in events:
            event.setdefault("received_at", now)

        result = await self._queue.enqueue_batch(events)
        logger.info("Pulled %d events (%d new, %d dup)", len(events), result.inserted, result.duplicates)

        # Ack the last event on relay
        last_event_id = events[-1]["event_id"]
        await self._client.post("/ack", json={"watermark": last_event_id})
        self._last_watermark = last_event_id

        return result.inserted
