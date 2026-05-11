"""VPS Receiver — pulls events from relay VPS into local event queue.

Protocol:
  1. GET /events?since=<watermark> from relay
  2. Store into local EventQueue (dedup handled by queue)
  3. POST /ack with new watermark to relay
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from orchestrator.db.queue import EventQueue

logger = logging.getLogger(__name__)


class VPSReceiver:
    def __init__(
        self,
        relay_url: str,
        local_queue: EventQueue,
        watermark_key: str = "relay",
        timeout: float = 30.0,
        *,
        api_key: str = "",
        latency_tracker=None,
        _client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._relay_url = relay_url
        self._queue = local_queue
        self._watermark_key = watermark_key
        self._timeout = timeout
        self._api_key = api_key
        self._client_factory = _client_factory
        self._latency_tracker = latency_tracker
        self._consecutive_failures: int = 0

    def _make_client(self) -> httpx.AsyncClient:
        if self._client_factory:
            return self._client_factory()
        headers = {}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        return httpx.AsyncClient(base_url=self._relay_url, timeout=self._timeout, headers=headers)

    async def pull_and_store(self, limit: int = 100) -> int:
        """Pull new events from relay, store locally, ack on relay. Returns count pulled."""
        watermark = await self._queue.get_watermark(self._watermark_key)

        params: dict = {"limit": limit}
        if watermark:
            params["since"] = watermark

        async with self._make_client() as client:
            resp = await client.get("/events", params=params)
            resp.raise_for_status()

            events = resp.json().get("events", [])
            if not events:
                return 0

            # Add received_at timestamp and normalize relay payloads for the
            # local SQLite queue. Some relays return decoded JSON objects while
            # EventQueue stores payload as text.
            now = datetime.now(timezone.utc).isoformat()
            normalized_events = [
                self._normalize_relay_event(event, received_at=now)
                for event in events
            ]
            for event in normalized_events:
                # Record latency if tracker available
                if self._latency_tracker:
                    ex_ts = event.get("exchange_timestamp", "")
                    rx_ts = event.get("received_at", "")
                    if ex_ts and rx_ts:
                        self._latency_tracker.record(
                            event.get("bot_id", "unknown"), ex_ts, rx_ts,
                        )

            result = await self._queue.enqueue_batch(normalized_events)
            logger.info(
                "Pulled %d events (%d new, %d dup)",
                len(events), result.inserted, result.duplicates,
            )

            # Ack the last event on relay
            last_event_id = normalized_events[-1]["event_id"]
            await client.post("/ack", json={"watermark": last_event_id})
            await self._queue.update_watermark(self._watermark_key, last_event_id)

            return result.inserted

    def _normalize_relay_event(self, event: dict, *, received_at: str) -> dict:
        normalized = dict(event)
        normalized.setdefault("received_at", received_at)

        payload = normalized.get("payload", {})
        payload_obj = payload
        if isinstance(payload, str):
            try:
                payload_obj = json.loads(payload)
            except json.JSONDecodeError:
                payload_obj = payload
        elif isinstance(payload, (dict, list)):
            normalized["payload"] = json.dumps(payload, default=str)
        else:
            normalized["payload"] = json.dumps(payload, default=str)

        normalized.setdefault(
            "exchange_timestamp",
            self._extract_exchange_timestamp(payload_obj) or received_at,
        )
        return normalized

    @staticmethod
    def _extract_exchange_timestamp(payload: object) -> str:
        if not isinstance(payload, dict):
            return ""
        for key in ("exchange_timestamp", "timestamp", "period_end"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for meta_key in ("metadata", "event_metadata"):
            meta = payload.get(meta_key)
            if isinstance(meta, dict):
                value = meta.get("exchange_timestamp")
                if isinstance(value, str) and value:
                    return value
        return ""

    async def poll(self) -> int:
        """Pull with retry-safe error handling. Returns events pulled, 0 on failure."""
        try:
            pulled = await self.pull_and_store()
            self._consecutive_failures = 0
            return pulled
        except Exception as exc:
            self._consecutive_failures += 1
            delay = min(2 ** self._consecutive_failures, 300)
            logger.warning(
                "Relay poll failed (attempt %d, next backoff %ds): %s",
                self._consecutive_failures, delay, exc,
            )
            return 0

    async def drain(self, batch_size: int = 100, max_batches: int = 100) -> int:
        """Pull all pending events from relay. For startup catch-up."""
        total = 0
        for _ in range(max_batches):
            pulled = await self.pull_and_store(limit=batch_size)
            total += pulled
            if pulled < batch_size:
                break
        if total > 0:
            logger.info("Startup drain: pulled %d events from relay", total)
        return total
