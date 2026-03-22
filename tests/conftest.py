from pathlib import Path

import pytest
import aiosqlite

# Import shared fixtures so they are available to all test files.
# pytest auto-discovers fixtures defined in conftest.py or imported here.
from tests.fixtures import (  # noqa: F401
    data_dir,
    event_stream,
    memory_dir,
    memory_dir_with_policies,
    mock_event_stream,
    sample_package,
    session_store,
)


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary SQLite database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def tmp_db(tmp_db_path: Path) -> aiosqlite.Connection:
    """Provide an initialized temporary SQLite connection."""
    async with aiosqlite.connect(tmp_db_path) as db:
        yield db
