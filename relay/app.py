"""Relay VPS service — minimal FastAPI app for event buffering.

Receives HMAC-signed event batches from bot sidecars.
Exposes pull endpoint with watermark-based ack for home gateway.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from relay.auth import verify_hmac
from relay.db.store import RelayStore


class EventBatch(BaseModel):
    bot_id: str
    events: list[dict]


class AckRequest(BaseModel):
    watermark: str


def create_relay_app(db_path: str = "relay.db", shared_secrets: dict[str, str] | None = None) -> FastAPI:
    """Factory function so tests can inject a temp DB path and secrets."""
    secrets = shared_secrets or {}
    store = RelayStore(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.initialize()
        yield
        await store.close()

    app = FastAPI(title="Trading Assistant Relay", lifespan=lifespan)
    app.state.store = store  # expose for test access

    @app.post("/events")
    async def receive_events(request: Request, x_signature: str = Header(...)):
        body_bytes = await request.body()
        body_str = body_bytes.decode()
        data = json.loads(body_str)
        bot_id = data.get("bot_id", "")

        # Verify bot is known and signature is valid
        secret = secrets.get(bot_id)
        if not secret:
            raise HTTPException(status_code=401, detail=f"Unknown bot: {bot_id}")

        # Verify against canonicalized JSON (sorted keys)
        canonical = json.dumps(data, sort_keys=True)
        if not verify_hmac(canonical, x_signature, secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

        events = data.get("events", [])
        accepted, duplicates = await store.store_events(events)
        return {"accepted": accepted, "duplicates": duplicates}

    @app.get("/events")
    async def pull_events(since: Optional[str] = None, limit: int = 100):
        events = await store.get_events(since=since, limit=limit)
        return {"events": events}

    @app.post("/ack")
    async def ack_events(req: AckRequest):
        await store.ack_up_to(req.watermark)
        return {"status": "ok", "watermark": req.watermark}

    return app
