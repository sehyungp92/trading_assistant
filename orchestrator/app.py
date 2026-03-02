"""Orchestrator FastAPI entry point — wires all Phase 1 components together.

Run with: uvicorn orchestrator.app:app --reload
For production, use create_app() factory to configure paths.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI

from orchestrator.db.queue import EventQueue
from orchestrator.orchestrator_brain import OrchestratorBrain
from orchestrator.task_registry import TaskRegistry
from orchestrator.worker import Worker


def create_app(db_dir: str = ".") -> FastAPI:
    """Factory function. Tests inject a temp directory for DB files."""
    db_path = Path(db_dir)
    queue = EventQueue(db_path=str(db_path / "events.db"))
    registry = TaskRegistry(db_path=str(db_path / "tasks.db"))
    brain = OrchestratorBrain()
    worker = Worker(queue=queue, registry=registry, brain=brain)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await queue.initialize()
        await registry.initialize()
        yield
        await queue.close()
        await registry.close()

    app = FastAPI(title="Trading Assistant Orchestrator", lifespan=lifespan)

    # Expose on app.state so test fixtures can manually initialize/close
    # (httpx ASGITransport does not trigger lifespan events)
    app.state.queue = queue
    app.state.registry = registry
    app.state.worker = worker

    @app.get("/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.post("/ingest")
    async def ingest_event(event: dict):
        """Direct event ingest — bypasses relay, useful for testing."""
        event.setdefault("received_at", datetime.now(timezone.utc).isoformat())
        inserted = await queue.enqueue(event)
        return {"inserted": inserted, "event_id": event.get("event_id")}

    @app.get("/events/pending")
    async def pending_events(limit: int = 20):
        return await queue.peek(limit=limit)

    @app.get("/tasks")
    async def list_tasks(status: str | None = None):
        if status:
            from schemas.tasks import TaskStatus
            return [t.model_dump(mode="json") for t in await registry.list_by_status(TaskStatus(status))]
        return []

    @app.post("/process")
    async def trigger_processing(limit: int = 10):
        """Manually trigger event processing (for testing, normally done by scheduler)."""
        processed = await worker.process_batch(limit=limit)
        return {"processed": processed}

    return app


# Default app instance for `uvicorn orchestrator.app:app`
app = create_app()
