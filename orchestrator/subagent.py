from __future__ import annotations
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

@dataclass
class SubagentInfo:
    """Metadata about a running subagent."""
    id: str
    agent_type: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self.task is not None and not self.task.done()

class SubagentManager:
    """Manages background async tasks with concurrency limits."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._max_concurrent = max_concurrent
        self._agents: dict[str, SubagentInfo] = {}

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def get_running(self) -> list[SubagentInfo]:
        """Get all currently running subagents."""
        return [a for a in self._agents.values() if a.is_running]

    def get_all(self) -> list[SubagentInfo]:
        """Get all subagents (running and completed)."""
        return list(self._agents.values())

    async def spawn(
        self,
        agent_type: str,
        coro: Callable[[], Awaitable[object]],
    ) -> str | None:
        """Spawn a new subagent. Returns agent_id or None if at capacity."""
        running = self.get_running()
        if len(running) >= self._max_concurrent:
            logger.warning(
                "Cannot spawn %s: %d/%d slots used",
                agent_type, len(running), self._max_concurrent,
            )
            return None

        agent_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        task = asyncio.create_task(self._run_agent(agent_id, coro))
        info = SubagentInfo(id=agent_id, agent_type=agent_type, task=task)
        self._agents[agent_id] = info
        logger.info("Spawned subagent %s (type=%s)", agent_id, agent_type)
        return agent_id

    async def _run_agent(self, agent_id: str, coro: Callable[[], Awaitable[object]]) -> object:
        """Wrapper that logs completion/failure."""
        try:
            result = await coro()
            logger.info("Subagent %s completed", agent_id)
            return result
        except asyncio.CancelledError:
            logger.info("Subagent %s cancelled", agent_id)
            raise
        except Exception:
            logger.exception("Subagent %s failed", agent_id)
            raise

    async def cancel(self, agent_id: str) -> bool:
        """Cancel a running subagent. Returns True if cancelled."""
        info = self._agents.get(agent_id)
        if info is None or not info.is_running:
            return False
        info.task.cancel()
        try:
            await info.task
        except (asyncio.CancelledError, Exception):
            pass
        return True

    async def cancel_all(self) -> int:
        """Cancel all running subagents. Returns count cancelled."""
        cancelled = 0
        for info in self.get_running():
            if await self.cancel(info.id):
                cancelled += 1
        return cancelled
