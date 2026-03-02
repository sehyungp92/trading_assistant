"""SQLite-backed event queue with idempotent deduplication.

Every event has a deterministic event_id (hash of bot_id + timestamp + type + payload_key).
Duplicate inserts are silently ignored via INSERT OR IGNORE.
"""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from orchestrator.db.connection import create_connection, initialize_schema


@dataclass
class BatchResult:
    inserted: int
    duplicates: int


class EventQueue:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await create_connection(self._db_path)
        await initialize_schema(self._db)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Call initialize() first"
        return self._db

    async def enqueue(self, event: dict) -> bool:
        """Insert a single event. Returns True if inserted, False if duplicate."""
        cursor = await self.db.execute(
            """INSERT OR IGNORE INTO events
               (event_id, bot_id, event_type, payload, exchange_timestamp, received_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event["event_id"],
                event["bot_id"],
                event["event_type"],
                event["payload"],
                event["exchange_timestamp"],
                event["received_at"],
            ),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def enqueue_batch(self, events: list[dict]) -> BatchResult:
        """Insert a batch of events with idempotent dedup. Returns counts."""
        inserted = 0
        for event in events:
            cursor = await self.db.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, bot_id, event_type, payload, exchange_timestamp, received_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event["event_id"],
                    event["bot_id"],
                    event["event_type"],
                    event["payload"],
                    event["exchange_timestamp"],
                    event["received_at"],
                ),
            )
            inserted += cursor.rowcount
        await self.db.commit()
        return BatchResult(inserted=inserted, duplicates=len(events) - inserted)

    async def peek(self, limit: int = 10) -> list[dict]:
        """Get pending events without changing their status."""
        cursor = await self.db.execute(
            "SELECT * FROM events WHERE status = 'pending' ORDER BY created_at ASC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def ack(self, event_id: str) -> None:
        """Mark an event as acknowledged/processed."""
        await self.db.execute(
            "UPDATE events SET status = 'acked', processed_at = datetime('now') WHERE event_id = ?",
            (event_id,),
        )
        await self.db.commit()

    async def update_watermark(self, bot_id: str, last_event_id: str) -> None:
        """Update the watermark for relay pull protocol."""
        await self.db.execute(
            """INSERT INTO watermarks (bot_id, last_event_id, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(bot_id) DO UPDATE SET
                 last_event_id = excluded.last_event_id,
                 updated_at = excluded.updated_at""",
            (bot_id, last_event_id),
        )
        await self.db.commit()

    async def get_watermark(self, bot_id: str) -> str | None:
        """Get the last acked event_id for a bot."""
        cursor = await self.db.execute(
            "SELECT last_event_id FROM watermarks WHERE bot_id = ?",
            (bot_id,),
        )
        row = await cursor.fetchone()
        return row["last_event_id"] if row else None
