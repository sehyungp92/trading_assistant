import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import aiosqlite


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary SQLite database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def tmp_db(tmp_db_path: Path) -> aiosqlite.Connection:
    """Provide an initialized temporary SQLite connection."""
    async with aiosqlite.connect(tmp_db_path) as db:
        yield db
