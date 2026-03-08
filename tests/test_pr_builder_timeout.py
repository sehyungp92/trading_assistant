"""Tests for PRBuilder subprocess timeout (Task 6)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from skills.github_pr import PRBuilder


@pytest.fixture
def builder() -> PRBuilder:
    return PRBuilder(dry_run=True)


class TestSubprocessTimeout:
    """6a: Timeout on _run_cmd."""

    async def test_timeout_kills_process(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        # Run a command that takes long and should timeout
        code, stdout, stderr = await builder._run_cmd(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            timeout_seconds=1,
        )
        assert code == 1
        assert "timed out" in stderr

    async def test_normal_command_succeeds(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        code, stdout, stderr = await builder._run_cmd(
            [sys.executable, "-c", "print('hello')"],
            cwd=tmp_path,
            timeout_seconds=10,
        )
        assert code == 0
        assert "hello" in stdout


class TestPublicAPI:
    """6b: Public run_gh_command API."""

    async def test_run_gh_command_delegates_to_run_cmd(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        code, stdout, stderr = await builder.run_gh_command(
            [sys.executable, "-c", "print('via_public')"],
            cwd=tmp_path,
            timeout_seconds=10,
        )
        assert code == 0
        assert "via_public" in stdout

    async def test_run_gh_command_timeout(self, tmp_path):
        builder = PRBuilder(dry_run=False)
        code, _, stderr = await builder.run_gh_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            timeout_seconds=1,
        )
        assert code == 1
        assert "timed out" in stderr
