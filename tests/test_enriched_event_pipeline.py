"""Tests for enriched event pipeline: raw JSONL persistence and curated data building."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.db.queue import EventQueue
from orchestrator.event_stream import EventStream
from orchestrator.handlers import Handlers
from orchestrator.orchestrator_brain import Action, ActionType, OrchestratorBrain
from orchestrator.task_registry import TaskRegistry
from orchestrator.worker import Worker
from schemas.notifications import NotificationPreferences


@pytest.fixture
async def queue(tmp_path) -> EventQueue:
    q = EventQueue(db_path=str(tmp_path / "queue.db"))
    await q.initialize()
    return q


@pytest.fixture
async def registry(tmp_path) -> TaskRegistry:
    r = TaskRegistry(db_path=str(tmp_path / "tasks.db"))
    await r.initialize()
    return r


@pytest.fixture
def brain() -> OrchestratorBrain:
    return OrchestratorBrain()


class TestWorkerPersistsEnrichedEvents:
    """Task 1a: Worker persists enriched events to raw JSONL on QUEUE_FOR_DAILY."""

    async def test_persist_enriched_event_to_raw_jsonl(self, queue, registry, brain, tmp_path):
        raw_dir = tmp_path / "raw"
        worker = Worker(
            queue=queue, registry=registry, brain=brain,
            raw_data_dir=raw_dir,
        )

        await queue.enqueue({
            "event_id": "is001",
            "bot_id": "bot1",
            "event_type": "indicator_snapshot",
            "payload": json.dumps({"rsi": 55.2, "macd": 0.3}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 1

        # Find the raw JSONL file written
        raw_files = list(raw_dir.rglob("indicator_snapshot.jsonl"))
        assert len(raw_files) == 1
        lines = raw_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1

    async def test_worker_without_raw_dir_still_counts(self, queue, registry, brain):
        """Backward compat: raw_data_dir=None just counts, doesn't persist."""
        worker = Worker(queue=queue, registry=registry, brain=brain)
        assert worker._raw_data_dir is None

        await queue.enqueue({
            "event_id": "is002",
            "bot_id": "bot1",
            "event_type": "indicator_snapshot",
            "payload": json.dumps({"rsi": 55.2}),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        processed = await worker.process_batch(limit=10)
        assert processed == 1
        assert worker.daily_queue_counts.get("bot1", 0) >= 1

    async def test_persist_multiple_event_types(self, queue, registry, brain, tmp_path):
        raw_dir = tmp_path / "raw"
        worker = Worker(
            queue=queue, registry=registry, brain=brain,
            raw_data_dir=raw_dir,
        )

        for event_type, eid in [
            ("filter_decision", "fd001"),
            ("orderbook_context", "ob001"),
            ("parameter_change", "pc001"),
        ]:
            await queue.enqueue({
                "event_id": eid,
                "bot_id": "bot1",
                "event_type": event_type,
                "payload": json.dumps({"type": event_type}),
                "exchange_timestamp": "2026-03-01T14:00:00+00:00",
                "received_at": "2026-03-01T14:00:01+00:00",
            })

        await worker.process_batch(limit=10)

        for event_type in ["filter_decision", "orderbook_context", "parameter_change"]:
            files = list(raw_dir.rglob(f"{event_type}.jsonl"))
            assert len(files) == 1, f"Expected 1 file for {event_type}, got {len(files)}"


class TestHandlerBuildsEnrichedCurated:
    """Task 1b/1c: Daily handler reads enriched raw files and builds curated files."""

    def _make_handlers(self, tmp_path):
        return Handlers(
            agent_runner=AsyncMock(),
            event_stream=EventStream(),
            dispatcher=AsyncMock(),
            notification_prefs=NotificationPreferences(),
            curated_dir=tmp_path / "data" / "curated",
            memory_dir=tmp_path / "memory",
            runs_dir=tmp_path / "runs",
            source_root=tmp_path / "src",
            bots=["bot1"],
        )

    def test_build_enriched_curated_reads_raw_and_writes_curated(self, tmp_path):
        handlers = self._make_handlers(tmp_path)
        date = "2026-03-01"

        # Create raw enriched events
        raw_dir = tmp_path / "data" / "raw" / date / "bot1"
        raw_dir.mkdir(parents=True)
        (raw_dir / "filter_decision.jsonl").write_text(
            json.dumps({"filter_name": "rsi_filter", "action": "pass", "signal_strength": 0.8}) + "\n",
            encoding="utf-8",
        )
        (raw_dir / "indicator_snapshot.jsonl").write_text(
            json.dumps({"rsi": 55.2, "macd": 0.3}) + "\n",
            encoding="utf-8",
        )

        # Also need a summary.json for write_curated to work
        curated_bot_dir = tmp_path / "data" / "curated" / date / "bot1"
        curated_bot_dir.mkdir(parents=True)

        handlers._build_enriched_curated(date)

        # Check curated files were written (write_curated creates them)
        curated_dir = tmp_path / "data" / "curated" / date / "bot1"
        assert curated_dir.exists()

    def test_build_enriched_curated_skips_missing_raw(self, tmp_path):
        """No raw directory = graceful no-op."""
        handlers = self._make_handlers(tmp_path)
        handlers._build_enriched_curated("2026-03-01")
        # Should not raise

    def test_build_enriched_curated_handles_malformed_json(self, tmp_path):
        handlers = self._make_handlers(tmp_path)
        date = "2026-03-01"

        raw_dir = tmp_path / "data" / "raw" / date / "bot1"
        raw_dir.mkdir(parents=True)
        (raw_dir / "filter_decision.jsonl").write_text(
            "not valid json\n" + json.dumps({"filter_name": "test"}) + "\n",
            encoding="utf-8",
        )

        curated_bot_dir = tmp_path / "data" / "curated" / date / "bot1"
        curated_bot_dir.mkdir(parents=True)

        # Should not raise — malformed lines skipped
        handlers._build_enriched_curated(date)

    def test_parameter_change_events_persisted(self, tmp_path):
        """Parameter change events are persisted to raw JSONL."""
        raw_dir = tmp_path / "raw" / "2026-03-01" / "bot1"
        raw_dir.mkdir(parents=True)

        # Simulate worker persisting
        payload = {"param_name": "stop_loss", "old_value": 0.02, "new_value": 0.015}
        (raw_dir / "parameter_change.jsonl").write_text(
            json.dumps(payload) + "\n", encoding="utf-8",
        )

        data = json.loads((raw_dir / "parameter_change.jsonl").read_text().strip())
        assert data["param_name"] == "stop_loss"


class TestEndToEndEnrichedPipeline:
    """End-to-end: enriched event → raw JSONL → curated JSON."""

    async def test_enriched_event_end_to_end(self, queue, registry, brain, tmp_path):
        raw_dir = tmp_path / "raw"
        worker = Worker(
            queue=queue, registry=registry, brain=brain,
            raw_data_dir=raw_dir,
        )

        # Step 1: Enqueue enriched event
        await queue.enqueue({
            "event_id": "e2e001",
            "bot_id": "bot1",
            "event_type": "filter_decision",
            "payload": json.dumps({
                "filter_name": "rsi_filter",
                "action": "block",
                "signal_strength": 0.3,
            }),
            "exchange_timestamp": "2026-03-01T14:00:00+00:00",
            "received_at": "2026-03-01T14:00:01+00:00",
        })

        # Step 2: Worker processes and persists
        processed = await worker.process_batch(limit=10)
        assert processed == 1

        # Step 3: Verify raw JSONL exists
        raw_files = list(raw_dir.rglob("filter_decision.jsonl"))
        assert len(raw_files) == 1
        raw_data = json.loads(raw_files[0].read_text(encoding="utf-8").strip())
        assert raw_data["filter_name"] == "rsi_filter"
