"""SQLite connection factory with WAL mode for concurrent reads."""

from pathlib import Path

import aiosqlite

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def create_connection(db_path: str) -> aiosqlite.Connection:
    """Create and initialize a SQLite connection with WAL mode."""
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    db.row_factory = aiosqlite.Row
    return db


async def initialize_schema(db: aiosqlite.Connection) -> None:
    """Run the schema.sql file to create tables."""
    schema = _SCHEMA_PATH.read_text()
    await db.executescript(schema)
    await db.commit()
