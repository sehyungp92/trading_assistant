"""Tests for subagent manager (M3)."""
from __future__ import annotations
import asyncio
import pytest
from orchestrator.subagent import SubagentManager

class TestSubagentManager:
    async def test_spawn_returns_agent_id(self):
        mgr = SubagentManager(max_concurrent=3)
        async def work():
            await asyncio.sleep(0.1)
        agent_id = await mgr.spawn("daily_analysis", work)
        assert agent_id is not None
        assert "daily_analysis" in agent_id

    async def test_spawn_respects_max_concurrent(self):
        mgr = SubagentManager(max_concurrent=2)
        async def long_work():
            await asyncio.sleep(10)

        id1 = await mgr.spawn("task1", long_work)
        id2 = await mgr.spawn("task2", long_work)
        id3 = await mgr.spawn("task3", long_work)  # Should be rejected

        assert id1 is not None
        assert id2 is not None
        assert id3 is None

        await mgr.cancel_all()

    async def test_get_running(self):
        mgr = SubagentManager()
        async def work():
            await asyncio.sleep(10)

        await mgr.spawn("wfo", work)
        running = mgr.get_running()
        assert len(running) == 1
        assert running[0].agent_type == "wfo"
        assert running[0].is_running

        await mgr.cancel_all()

    async def test_cancel_stops_agent(self):
        mgr = SubagentManager()
        async def long_work():
            await asyncio.sleep(10)

        agent_id = await mgr.spawn("wfo", long_work)
        assert len(mgr.get_running()) == 1

        cancelled = await mgr.cancel(agent_id)
        assert cancelled is True
        assert len(mgr.get_running()) == 0

    async def test_cancel_nonexistent_returns_false(self):
        mgr = SubagentManager()
        result = await mgr.cancel("nonexistent")
        assert result is False

    async def test_completed_agent_not_in_running(self):
        mgr = SubagentManager()
        async def quick_work():
            return "done"

        agent_id = await mgr.spawn("fast", quick_work)
        await asyncio.sleep(0.1)  # Let it complete
        assert len(mgr.get_running()) == 0
        assert len(mgr.get_all()) == 1

    async def test_cancel_all(self):
        mgr = SubagentManager(max_concurrent=5)
        async def work():
            await asyncio.sleep(10)

        for i in range(3):
            await mgr.spawn(f"task-{i}", work)

        assert len(mgr.get_running()) == 3
        cancelled = await mgr.cancel_all()
        assert cancelled == 3
        assert len(mgr.get_running()) == 0

    async def test_agent_failure_tracked(self):
        mgr = SubagentManager()
        async def failing_work():
            raise ValueError("boom")

        agent_id = await mgr.spawn("failing", failing_work)
        await asyncio.sleep(0.1)

        # Should be in all but not in running
        assert len(mgr.get_running()) == 0
        assert len(mgr.get_all()) == 1
