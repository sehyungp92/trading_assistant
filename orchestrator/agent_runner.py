"""Agent runner — invokes Claude CLI with PromptPackage, records sessions.

Bridge between the prompt assemblers (which produce PromptPackage) and the
Claude CLI (`claude -p`). Creates a run directory with data files, invokes
Claude with the assembled prompt, captures output, and records the session.

Uses the Claude Code Max subscription via the `claude` CLI, NOT the
Anthropic Python API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.session_store import SessionStore
from orchestrator.skills_registry import SkillsRegistry
from schemas.prompt_package import PromptPackage

logger = logging.getLogger(__name__)

# Default timeout for Claude CLI invocations (10 minutes)
_DEFAULT_TIMEOUT_SECONDS = 600


@dataclass
class AgentResult:
    """Result of a Claude CLI invocation."""

    response: str
    run_dir: Path
    cost_usd: float = 0.0
    duration_ms: int = 0
    session_id: str = ""
    success: bool = True
    error: str = ""


class AgentRunner:
    """Invokes Claude CLI with a PromptPackage and records the session."""

    def __init__(
        self,
        runs_dir: Path,
        session_store: SessionStore,
        claude_path: str = "claude",
        default_model: str = "sonnet",
        default_max_turns: int = 5,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        skills_registry: SkillsRegistry | None = None,
        enforce_skills: bool = False,
    ) -> None:
        self._runs_dir = Path(runs_dir)
        self._session_store = session_store
        self._claude_path = claude_path
        self._default_model = default_model
        self._default_max_turns = default_max_turns
        self._timeout_seconds = timeout_seconds
        self._skills_registry = skills_registry
        self._enforce_skills = enforce_skills

    async def invoke(
        self,
        agent_type: str,
        prompt_package: PromptPackage,
        run_id: str,
        model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        """Invoke Claude CLI with the given prompt package.

        Args:
            agent_type: Type of agent (daily_analysis, weekly_analysis, wfo, triage).
            prompt_package: The assembled prompt package.
            run_id: Unique run identifier (e.g., "daily-2026-03-02").
            model: Override default model (sonnet/opus).
            max_turns: Override default max turns.
            allowed_tools: Restrict Claude's tool access.

        Returns:
            AgentResult with the response and metadata.
        """
        # Skills registry enforcement
        if self._skills_registry:
            check = self._skills_registry.check_action(agent_type, "generate_report")
            if not check.allowed:
                msg = f"SkillsRegistry denied action for {agent_type}: {check.reason}"
                if self._enforce_skills:
                    raise PermissionError(msg)
                logger.warning(msg)

        session_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write data files
        self._write_run_files(run_dir, prompt_package)

        # Build CLI command
        cmd = self._build_command(
            prompt_package=prompt_package,
            model=model or self._default_model,
            max_turns=max_turns or self._default_max_turns,
            allowed_tools=allowed_tools,
        )

        logger.info("Invoking Claude CLI for %s (run_id=%s)", agent_type, run_id)
        start = datetime.now(timezone.utc)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(run_dir),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            elapsed = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            logger.error("Claude CLI timed out after %ds for %s", self._timeout_seconds, run_id)
            return AgentResult(
                response="",
                run_dir=run_dir,
                duration_ms=elapsed,
                session_id=session_id,
                success=False,
                error=f"Timeout after {self._timeout_seconds}s",
            )

        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            logger.error("Claude CLI exited with code %d for %s: %s", process.returncode, run_id, stderr)
            return AgentResult(
                response=stdout,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=session_id,
                success=False,
                error=f"Exit code {process.returncode}: {stderr[:500]}",
            )

        # Parse output
        result = self._parse_output(stdout, run_dir, elapsed_ms, session_id)

        # Write response file
        response_path = run_dir / "response.md"
        response_path.write_text(result.response, encoding="utf-8")

        # Record session
        self._session_store.record_invocation(
            session_id=session_id,
            agent_type=agent_type,
            prompt_package=prompt_package.model_dump(mode="json"),
            response=result.response,
            token_usage={},
            duration_ms=result.duration_ms,
            metadata={
                "run_id": run_id,
                "model": model or self._default_model,
                "cost_usd": result.cost_usd,
            },
        )

        logger.info(
            "Claude CLI completed for %s (run_id=%s, duration=%dms, cost=$%.4f)",
            agent_type, run_id, result.duration_ms, result.cost_usd,
        )
        return result

    def _write_run_files(self, run_dir: Path, package: PromptPackage) -> None:
        """Write prompt package data files into the run directory."""
        # Write each top-level data key as a separate JSON file
        for key, value in package.data.items():
            file_path = run_dir / f"{key}.json"
            file_path.write_text(
                json.dumps(value, indent=2, default=str), encoding="utf-8"
            )

        # Write instructions
        if package.instructions:
            (run_dir / "instructions.md").write_text(package.instructions, encoding="utf-8")

        # Write system prompt for reference
        if package.system_prompt:
            (run_dir / "system_prompt.md").write_text(package.system_prompt, encoding="utf-8")

    def _build_command(
        self,
        prompt_package: PromptPackage,
        model: str,
        max_turns: int,
        allowed_tools: list[str] | None,
    ) -> list[str]:
        """Build the Claude CLI command arguments."""
        cmd = [
            self._claude_path,
            "-p", prompt_package.task_prompt,
            "--output-format", "json",
            "--no-session-persistence",
            "--model", model,
            "--max-turns", str(max_turns),
        ]

        if prompt_package.system_prompt:
            cmd.extend(["--system-prompt", prompt_package.system_prompt])

        if allowed_tools:
            cmd.extend(["--allowed-tools", ",".join(allowed_tools)])

        return cmd

    def _parse_output(
        self, stdout: str, run_dir: Path, elapsed_ms: int, session_id: str,
    ) -> AgentResult:
        """Parse Claude CLI JSON output."""
        try:
            data = json.loads(stdout)
            response = data.get("result", stdout)
            cost_usd = data.get("cost_usd", 0.0)
            duration_ms = data.get("duration_ms", elapsed_ms)
            return AgentResult(
                response=response,
                run_dir=run_dir,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                session_id=session_id,
                success=True,
            )
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat stdout as plain text response
            return AgentResult(
                response=stdout.strip(),
                run_dir=run_dir,
                cost_usd=0.0,
                duration_ms=elapsed_ms,
                session_id=session_id,
                success=True,
            )
