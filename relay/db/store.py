"""Relay-side event storage with idempotent deduplication."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class RelayStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        self._db.row_factory = aiosqlite.Row
        schema = _SCHEMA_PATH.read_text()
        await self._db.executescript(schema)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None
        return self._db

    async def store_events(self, events: list[dict]) -> tuple[int, int]:
        """Store events with dedup. Returns (accepted, duplicates)."""
        accepted = 0
        for event in events:
            cursor = await self.db.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, bot_id, event_type, payload, exchange_timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    event["event_id"],
                    event["bot_id"],
                    event["event_type"],
                    event["payload"],
                    event["exchange_timestamp"],
                ),
            )
            accepted += cursor.rowcount
        await self.db.commit()
        return accepted, len(events) - accepted

    async def get_events(self, since: str | None = None, limit: int = 100) -> list[dict]:
        """Pull unacked events, optionally after a watermark."""
        if since:
            cursor = await self.db.execute(
                """SELECT * FROM events
                   WHERE acked = 0 AND rowid > (SELECT rowid FROM events WHERE event_id = ?)
                   ORDER BY rowid ASC LIMIT ?""",
                (since, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM events WHERE acked = 0 ORDER BY rowid ASC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def ack_up_to(self, watermark: str) -> None:
        """Mark all events up to and including watermark as acked."""
        await self.db.execute(
            """UPDATE events SET acked = 1
               WHERE rowid <= (SELECT rowid FROM events WHERE event_id = ?)""",
            (watermark,),
        )
        await self.db.commit()
