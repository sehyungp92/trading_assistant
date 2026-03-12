"""Agent runner for configured CLI-backed analysis providers."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tomllib
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.agent_preferences import AgentPreferencesManager
from orchestrator.event_stream import EventStream
from orchestrator.skills_registry import SkillsRegistry
from orchestrator.session_store import SessionStore
from schemas.agent_preferences import (
    AgentPreferences,
    AgentPreferencesView,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
)
from schemas.prompt_package import PromptPackage

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 600
_CLAUDE_RUNTIME = "claude_cli"
_CODEX_RUNTIME = "codex_cli"
_CODEX_SANDBOX = "read-only"
_CLI_PREVIEW_TIMEOUT_SECONDS = 5
_PROVIDER_STATUS_CACHE_TTL_SECONDS = 30
_PROGRESS_PREVIEW_LIMIT = 200
_CODEX_AUTH_MODE = "chatgpt"
_ANTHROPIC_ENV_KEYS_TO_CLEAR = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "API_TIMEOUT_MS",
)
_OPENAI_ENV_KEYS_TO_CLEAR = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
    "OPENAI_MODEL",
    "OPENAI_DEFAULT_MODEL",
)
_ALL_ENV_KEYS_TO_CLEAR = _ANTHROPIC_ENV_KEYS_TO_CLEAR + _OPENAI_ENV_KEYS_TO_CLEAR


@dataclass
class AgentResult:
    """Result of a provider invocation."""

    response: str
    run_dir: Path
    cost_usd: float = 0.0
    duration_ms: int = 0
    session_id: str = ""
    success: bool = True
    error: str = ""
    provider: str = ""
    runtime: str = ""
    requested_model: str | None = None
    effective_model: str = ""
    first_output_ms: int = 0
    stream_event_count: int = 0
    tool_call_count: int = 0
    auth_mode: str = ""


@dataclass
class _InvocationSpec:
    command: str
    args: list[str]
    env: dict[str, str]
    runtime: str
    provider: AgentProvider
    requested_model: str | None
    effective_model: str
    parse_mode: str = "json"


@dataclass
class _ClaudeAuthStatus:
    available: bool
    reason: str = ""
    auth_method: str = ""
    subscription_type: str = ""


@dataclass
class _ClaudeStreamState:
    raw_lines: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    final_result: str = ""
    resolved_session_id: str = ""
    cost_usd: float = 0.0
    first_output_ms: int = 0
    stream_event_count: int = 0
    tool_call_count: int = 0


@dataclass
class _CodexAuthStatus:
    available: bool
    reason: str = ""
    auth_mode: str = ""
    diagnostics: list[str] = field(default_factory=list)


class AgentRunner:
    """Invokes the configured provider runtime with a PromptPackage."""

    def __init__(
        self,
        runs_dir: Path,
        session_store: SessionStore,
        claude_command: str = "claude",
        claude_command_args: list[str] | None = None,
        codex_command: str = "codex",
        codex_command_args: list[str] | None = None,
        default_model: str = "sonnet",
        default_max_turns: int = 5,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        skills_registry: SkillsRegistry | None = None,
        enforce_skills: bool = False,
        preferences: AgentPreferences | None = None,
        zai_api_key: str = "",
        openrouter_api_key: str = "",
        event_stream: EventStream | None = None,
    ) -> None:
        self._runs_dir = Path(runs_dir)
        self._session_store = session_store
        self._claude_command = claude_command
        self._claude_command_args = list(claude_command_args or [])
        self._codex_command = codex_command
        self._codex_command_args = list(codex_command_args or [])
        self._default_model = default_model
        self._default_max_turns = default_max_turns
        self._timeout_seconds = timeout_seconds
        self._skills_registry = skills_registry
        self._enforce_skills = enforce_skills
        self._zai_api_key = zai_api_key
        self._openrouter_api_key = openrouter_api_key
        self._event_stream = event_stream
        self._preferences = AgentPreferencesManager(
            preferences
            or AgentPreferences(
                default=AgentSelection(
                    provider=AgentProvider.CLAUDE_MAX,
                    model=default_model,
                )
            )
        )
        self._preferences.set_provider_status_resolver(self.get_provider_statuses)
        self._provider_status_cache: dict[AgentProvider, ProviderReadiness] = {}
        self._provider_status_checked_at: dict[AgentProvider, datetime] = {}

    def get_preferences(self) -> AgentPreferences:
        return self._preferences.get_preferences()

    def get_preferences_view(self) -> AgentPreferencesView:
        return self._preferences.build_view()

    def update_preferences(self, preferences: AgentPreferences) -> None:
        errors = self._preferences.unavailable_reasons(preferences)
        if errors:
            raise ValueError("; ".join(errors))
        self._preferences.set_preferences(preferences)

    def get_provider_statuses(self) -> list[ProviderReadiness]:
        return [self._get_provider_status(provider) for provider in AgentProvider]

    def invalidate_provider_status_cache(
        self,
        provider: AgentProvider | None = None,
    ) -> None:
        """Clear cached provider readiness results."""
        if provider is None:
            self._provider_status_cache.clear()
            self._provider_status_checked_at.clear()
            return
        self._provider_status_cache.pop(provider, None)
        self._provider_status_checked_at.pop(provider, None)

    async def invoke(
        self,
        agent_type: str,
        prompt_package: PromptPackage,
        run_id: str,
        model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        """Invoke the configured runtime with the given prompt package."""
        workflow = self._resolve_workflow(agent_type)
        selection, requested_model = self._preferences.resolve_selection(workflow, model)
        return await self.invoke_with_selection(
            agent_type=agent_type,
            prompt_package=prompt_package,
            run_id=run_id,
            selection=selection,
            requested_model=requested_model,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
        )

    async def invoke_with_selection(
        self,
        agent_type: str,
        prompt_package: PromptPackage,
        run_id: str,
        selection: AgentSelection,
        requested_model: str | None = None,
        max_turns: int | None = None,
        allowed_tools: list[str] | None = None,
    ) -> AgentResult:
        """Invoke the runtime with an explicit provider selection."""
        self._check_skill_access(agent_type)

        session_id = f"{run_id}-{uuid.uuid4().hex[:8]}"
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_files(run_dir, prompt_package)

        status = self._get_provider_status(selection.provider)
        auth_mode = self._auth_mode_for_provider(selection.provider)
        if status.available and status.reason:
            logger.warning(
                "Provider %s diagnostics for %s: %s",
                selection.provider.value,
                agent_type,
                status.reason,
            )
        if not status.available:
            logger.error(
                "Provider %s unavailable for %s (run_id=%s): %s",
                selection.provider.value,
                agent_type,
                run_id,
                status.reason,
            )
            result = AgentResult(
                response="",
                run_dir=run_dir,
                duration_ms=0,
                session_id=session_id,
                success=False,
                error=status.reason or f"{selection.provider.value} is unavailable",
                provider=selection.provider.value,
                runtime=status.runtime,
                requested_model=requested_model,
                effective_model=selection.model or "",
                auth_mode=auth_mode,
            )
            self._broadcast_runtime_event(
                "agent_invocation_failed",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": result.provider,
                    "runtime": result.runtime,
                    "model": result.effective_model,
                    "error": result.error[:_PROGRESS_PREVIEW_LIMIT],
                },
            )
            return result

        invocation = self._build_invocation(
            prompt_package=prompt_package,
            selection=selection,
            requested_model=requested_model,
            max_turns=max_turns or self._default_max_turns,
            allowed_tools=allowed_tools,
        )

        logger.info(
            "Invoking agent runtime for %s (run_id=%s, provider=%s, runtime=%s, model=%s)",
            agent_type,
            run_id,
            invocation.provider.value,
            invocation.runtime,
            invocation.effective_model,
        )
        self._broadcast_runtime_event(
            "agent_invocation_started",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": invocation.provider.value,
                "runtime": invocation.runtime,
                "model": invocation.effective_model,
                **({"diagnostics": status.reason} if status.reason else {}),
            },
        )

        try:
            process = await asyncio.create_subprocess_exec(
                invocation.command,
                *invocation.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(run_dir),
                env=invocation.env,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            logger.error("Could not start agent runtime for %s: %s", run_id, exc)
            result = AgentResult(
                response="",
                run_dir=run_dir,
                duration_ms=0,
                session_id=session_id,
                success=False,
                error=str(exc),
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                auth_mode=auth_mode,
            )
            self._broadcast_runtime_event(
                "agent_invocation_failed",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": result.provider,
                    "runtime": result.runtime,
                    "model": result.effective_model,
                    "error": result.error[:_PROGRESS_PREVIEW_LIMIT],
                },
            )
            return result

        if invocation.parse_mode == "stream-json":
            result = await self._invoke_claude_stream(
                process=process,
                run_id=run_id,
                agent_type=agent_type,
                run_dir=run_dir,
                session_id=session_id,
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                auth_mode=auth_mode,
            )
        elif invocation.parse_mode == "jsonl-stream":
            result = await self._invoke_codex_stream(
                process=process,
                run_id=run_id,
                agent_type=agent_type,
                run_dir=run_dir,
                session_id=session_id,
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                auth_mode=auth_mode,
            )
        else:
            result = await self._invoke_buffered_process(
                process=process,
                run_dir=run_dir,
                session_id=session_id,
                provider=invocation.provider.value,
                runtime=invocation.runtime,
                requested_model=invocation.requested_model,
                effective_model=invocation.effective_model,
                parse_mode=invocation.parse_mode,
                auth_mode=auth_mode,
            )

        response_path = run_dir / "response.md"
        if result.response:
            response_path.write_text(result.response, encoding="utf-8")

        if not result.success:
            self._broadcast_runtime_event(
                "agent_invocation_failed",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": result.provider,
                    "runtime": result.runtime,
                    "model": result.effective_model,
                    "duration_ms": result.duration_ms,
                    "error": result.error[:_PROGRESS_PREVIEW_LIMIT],
                },
            )
            return result

        self._session_store.record_invocation(
            session_id=result.session_id or session_id,
            agent_type=agent_type,
            prompt_package=prompt_package.model_dump(mode="json"),
            response=result.response,
            token_usage={},
            duration_ms=result.duration_ms,
            metadata={
                "run_id": run_id,
                "provider": invocation.provider.value,
                "runtime": invocation.runtime,
                "requested_model": invocation.requested_model,
                "effective_model": invocation.effective_model,
                "cost_usd": result.cost_usd,
                "first_output_ms": result.first_output_ms,
                "stream_event_count": result.stream_event_count,
                "tool_call_count": result.tool_call_count,
                "auth_mode": result.auth_mode,
            },
        )

        self._broadcast_runtime_event(
            "agent_invocation_completed",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": result.provider,
                "runtime": result.runtime,
                "model": result.effective_model,
                "duration_ms": result.duration_ms,
                "cost_usd": result.cost_usd,
                "first_output_ms": result.first_output_ms,
                "stream_event_count": result.stream_event_count,
                "tool_call_count": result.tool_call_count,
            },
        )
        logger.info(
            "Agent runtime completed for %s (run_id=%s, provider=%s, duration=%dms, cost=$%.4f)",
            agent_type,
            run_id,
            invocation.provider.value,
            result.duration_ms,
            result.cost_usd,
        )
        return result

    def _check_skill_access(self, agent_type: str) -> None:
        if self._skills_registry:
            check = self._skills_registry.check_action(agent_type, "generate_report")
            if not check.allowed:
                msg = f"SkillsRegistry denied action for {agent_type}: {check.reason}"
                if self._enforce_skills:
                    raise PermissionError(msg)
                logger.warning(msg)

    def _write_run_files(self, run_dir: Path, package: PromptPackage) -> None:
        for key, value in package.data.items():
            file_path = run_dir / f"{key}.json"
            file_path.write_text(
                json.dumps(value, indent=2, default=str), encoding="utf-8"
            )

        if package.instructions:
            (run_dir / "instructions.md").write_text(package.instructions, encoding="utf-8")

        if package.system_prompt:
            (run_dir / "system_prompt.md").write_text(package.system_prompt, encoding="utf-8")

    def _build_invocation(
        self,
        prompt_package: PromptPackage,
        selection: AgentSelection,
        requested_model: str | None,
        max_turns: int,
        allowed_tools: list[str] | None,
    ) -> _InvocationSpec:
        if selection.provider == AgentProvider.CODEX_PRO:
            return self._build_codex_invocation(prompt_package, selection, requested_model)
        return self._build_claude_invocation(
            prompt_package=prompt_package,
            selection=selection,
            requested_model=requested_model,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
        )

    def _build_claude_invocation(
        self,
        prompt_package: PromptPackage,
        selection: AgentSelection,
        requested_model: str | None,
        max_turns: int,
        allowed_tools: list[str] | None,
    ) -> _InvocationSpec:
        resolved_command = self._require_resolved_command(self._claude_command)
        cli_model = self._resolve_claude_cli_model(selection)
        args = [
            *self._claude_command_args,
            "-p",
            prompt_package.task_prompt,
            "--output-format",
            "stream-json",
            "--no-session-persistence",
            "--model",
            cli_model,
            "--max-turns",
            str(max_turns),
        ]

        if prompt_package.system_prompt:
            args.extend(["--append-system-prompt", prompt_package.system_prompt])

        if allowed_tools:
            args.extend(["--allowed-tools", ",".join(allowed_tools)])

        return _InvocationSpec(
            command=resolved_command,
            args=args,
            env=self._build_env(
                self._claude_env_overrides(selection),
                clear_keys=_ANTHROPIC_ENV_KEYS_TO_CLEAR,
            ),
            runtime=_CLAUDE_RUNTIME,
            provider=selection.provider,
            requested_model=requested_model,
            effective_model=selection.model or "",
            parse_mode="stream-json",
        )

    def _build_codex_invocation(
        self,
        prompt_package: PromptPackage,
        selection: AgentSelection,
        requested_model: str | None,
    ) -> _InvocationSpec:
        resolved_command = self._require_resolved_command(self._codex_command)
        prompt = self._merge_codex_prompt(prompt_package)
        args = [
            *self._codex_command_args,
            "exec",
            "--json",
            "--color",
            "never",
            "--sandbox",
            _CODEX_SANDBOX,
            "--skip-git-repo-check",
            "--model",
            selection.model or "",
            prompt,
        ]
        return _InvocationSpec(
            command=resolved_command,
            args=args,
            env=self._build_env(clear_keys=_OPENAI_ENV_KEYS_TO_CLEAR),
            runtime=_CODEX_RUNTIME,
            provider=selection.provider,
            requested_model=requested_model,
            effective_model=selection.model or "",
            parse_mode="jsonl-stream",
        )

    def _build_env(
        self,
        overrides: dict[str, str] | None = None,
        *,
        clear_keys: tuple[str, ...] | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        for key in clear_keys or _ALL_ENV_KEYS_TO_CLEAR:
            env.pop(key, None)
        if overrides:
            env.update(overrides)
        return env

    def _claude_env_overrides(self, selection: AgentSelection) -> dict[str, str]:
        model = selection.model or ""
        if selection.provider == AgentProvider.ZAI_CODING_PLAN:
            return {
                "ANTHROPIC_AUTH_TOKEN": self._zai_api_key,
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model or "glm-4.7",
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model or "glm-4.7",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air",
                "API_TIMEOUT_MS": "3000000",
            }
        if selection.provider == AgentProvider.OPENROUTER:
            effective = model or "anthropic/claude-sonnet-4.5"
            return {
                "ANTHROPIC_AUTH_TOKEN": self._openrouter_api_key,
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": effective,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": effective,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": effective,
            }
        return {}

    def _resolve_claude_cli_model(self, selection: AgentSelection) -> str:
        model = (selection.model or self._default_model).strip()
        if selection.provider == AgentProvider.CLAUDE_MAX:
            return model
        lowered = model.lower()
        if "opus" in lowered:
            return "opus"
        if "haiku" in lowered or "air" in lowered:
            return "haiku"
        return "sonnet"

    def _merge_codex_prompt(self, prompt_package: PromptPackage) -> str:
        prompt_parts: list[str] = []
        system_prompt = prompt_package.system_prompt.strip()
        if system_prompt:
            prompt_parts.append(f"System instructions:\n{system_prompt}")
        prompt_parts.append(prompt_package.task_prompt)
        return "\n\n".join(part for part in prompt_parts if part).strip()

    async def _invoke_buffered_process(
        self,
        process: asyncio.subprocess.Process,
        run_dir: Path,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        parse_mode: str,
        auth_mode: str,
    ) -> AgentResult:
        start = datetime.now(timezone.utc)
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()
            elapsed = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            logger.error(
                "Agent runtime timed out after %ds for %s",
                self._timeout_seconds,
                run_dir.name,
            )
            return AgentResult(
                response="",
                run_dir=run_dir,
                duration_ms=elapsed,
                session_id=session_id,
                success=False,
                error=f"Timeout after {self._timeout_seconds}s",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                auth_mode=auth_mode,
            )

        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            logger.error(
                "Agent runtime exited with code %d for %s: %s",
                process.returncode,
                run_dir.name,
                stderr,
            )
            return AgentResult(
                response=stdout,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=session_id,
                success=False,
                error=f"Exit code {process.returncode}: {(stderr or stdout)[:500]}",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                auth_mode=auth_mode,
            )

        return self._parse_output(
            stdout=stdout,
            run_dir=run_dir,
            elapsed_ms=elapsed_ms,
            session_id=session_id,
            parse_mode=parse_mode,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            auth_mode=auth_mode,
        )

    async def _invoke_claude_stream(
        self,
        process: asyncio.subprocess.Process,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> AgentResult:
        start = datetime.now(timezone.utc)
        stdout_path = run_dir / "claude-session.jsonl"
        stderr_path = run_dir / "claude-stderr.log"

        stdout_task = asyncio.create_task(
            self._consume_claude_stdout(
                stdout=process.stdout,
                output_path=stdout_path,
                started_at=start,
                run_id=run_id,
                agent_type=agent_type,
                provider=provider,
                runtime=runtime,
            )
        )
        stderr_task = asyncio.create_task(self._consume_stderr(process.stderr, stderr_path))

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=self._timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            logger.error(
                "Agent runtime timed out after %ds for %s",
                self._timeout_seconds,
                run_id,
            )
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()

        stream_state = await stdout_task
        stderr_text = await stderr_task
        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        response = (
            stream_state.final_result.strip()
            or "\n".join(text for text in stream_state.texts if text).strip()
            or "\n".join(stream_state.raw_lines).strip()
        )
        resolved_session_id = stream_state.resolved_session_id or session_id

        if timed_out:
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Timeout after {self._timeout_seconds}s",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        if process.returncode != 0:
            logger.error(
                "Agent runtime exited with code %d for %s: %s",
                process.returncode,
                run_id,
                stderr_text,
            )
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Exit code {process.returncode}: {(stderr_text or response)[:500]}",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        return AgentResult(
            response=response,
            run_dir=run_dir,
            cost_usd=stream_state.cost_usd,
            duration_ms=elapsed_ms,
            session_id=resolved_session_id,
            success=True,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            first_output_ms=stream_state.first_output_ms,
            stream_event_count=stream_state.stream_event_count,
            tool_call_count=stream_state.tool_call_count,
            auth_mode=auth_mode,
        )

    async def _invoke_codex_stream(
        self,
        process: asyncio.subprocess.Process,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> AgentResult:
        start = datetime.now(timezone.utc)
        stdout_path = run_dir / "codex-session.jsonl"
        stderr_path = run_dir / "codex-stderr.log"

        stdout_task = asyncio.create_task(
            self._consume_codex_stdout(
                stdout=process.stdout,
                output_path=stdout_path,
                started_at=start,
                run_id=run_id,
                agent_type=agent_type,
                provider=provider,
                runtime=runtime,
            )
        )
        stderr_task = asyncio.create_task(self._consume_stderr(process.stderr, stderr_path))

        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=self._timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            logger.error(
                "Agent runtime timed out after %ds for %s",
                self._timeout_seconds,
                run_id,
            )
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()

        stream_state = await stdout_task
        stderr_text = await stderr_task
        elapsed_ms = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
        response = (
            stream_state.final_result.strip()
            or "\n".join(text for text in stream_state.texts if text).strip()
            or "\n".join(stream_state.raw_lines).strip()
        )
        resolved_session_id = stream_state.resolved_session_id or session_id

        if timed_out:
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Timeout after {self._timeout_seconds}s",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        if process.returncode != 0:
            logger.error(
                "Agent runtime exited with code %d for %s: %s",
                process.returncode,
                run_id,
                stderr_text,
            )
            return AgentResult(
                response=response,
                run_dir=run_dir,
                duration_ms=elapsed_ms,
                session_id=resolved_session_id,
                success=False,
                error=f"Exit code {process.returncode}: {(stderr_text or response)[:500]}",
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                first_output_ms=stream_state.first_output_ms,
                stream_event_count=stream_state.stream_event_count,
                tool_call_count=stream_state.tool_call_count,
                auth_mode=auth_mode,
            )

        return AgentResult(
            response=response,
            run_dir=run_dir,
            cost_usd=stream_state.cost_usd,
            duration_ms=elapsed_ms,
            session_id=resolved_session_id,
            success=True,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            first_output_ms=stream_state.first_output_ms,
            stream_event_count=stream_state.stream_event_count,
            tool_call_count=stream_state.tool_call_count,
            auth_mode=auth_mode,
        )

    async def _consume_claude_stdout(
        self,
        stdout: asyncio.StreamReader | None,
        output_path: Path,
        started_at: datetime,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> _ClaudeStreamState:
        state = _ClaudeStreamState()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as raw_file:
            if stdout is None:
                return state

            while True:
                line_bytes = await stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                if state.first_output_ms == 0:
                    state.first_output_ms = int(
                        (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
                    )
                state.raw_lines.append(line)
                state.stream_event_count += 1
                raw_file.write(line + "\n")
                self._parse_claude_stream_line(
                    line=line,
                    state=state,
                    run_id=run_id,
                    agent_type=agent_type,
                    provider=provider,
                    runtime=runtime,
                )
        return state

    async def _consume_codex_stdout(
        self,
        stdout: asyncio.StreamReader | None,
        output_path: Path,
        started_at: datetime,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> _ClaudeStreamState:
        state = _ClaudeStreamState()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as raw_file:
            if stdout is None:
                return state

            while True:
                line_bytes = await stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                if state.first_output_ms == 0:
                    state.first_output_ms = int(
                        (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
                    )
                state.raw_lines.append(line)
                state.stream_event_count += 1
                raw_file.write(line + "\n")
                self._parse_codex_stream_line(
                    line=line,
                    state=state,
                    run_id=run_id,
                    agent_type=agent_type,
                    provider=provider,
                    runtime=runtime,
                )
        return state

    async def _consume_stderr(
        self,
        stderr: asyncio.StreamReader | None,
        output_path: Path,
    ) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if stderr is None:
            output_path.write_text("", encoding="utf-8")
            return ""

        stderr_bytes = await stderr.read()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        output_path.write_text(stderr_text, encoding="utf-8")
        return stderr_text

    def _parse_claude_stream_line(
        self,
        line: str,
        state: _ClaudeStreamState,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> None:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            self._broadcast_runtime_event(
                "agent_invocation_progress",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": provider,
                    "runtime": runtime,
                    "kind": "raw",
                    "preview": self._truncate_preview(line),
                },
            )
            return

        if not isinstance(parsed, dict):
            return

        self._capture_session_identifier(parsed, state)
        self._capture_cost(parsed, state)

        event_type = str(parsed.get("type", "")).lower()
        if event_type == "assistant":
            message = parsed.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = str(block.get("type", "")).lower()
                        if block_type == "text":
                            self._record_assistant_text(
                                state=state,
                                text=block.get("text"),
                                run_id=run_id,
                                agent_type=agent_type,
                                provider=provider,
                                runtime=runtime,
                            )
                        elif block_type == "tool_use":
                            state.tool_call_count += 1
                            tool_name = str(block.get("name", "")).strip()
                            tool_input = block.get("input")
                            preview = (
                                self._truncate_preview(json.dumps(tool_input, default=str))
                                if tool_input is not None
                                else ""
                            )
                            self._broadcast_runtime_event(
                                "agent_invocation_progress",
                                {
                                    "run_id": run_id,
                                    "agent_type": agent_type,
                                    "provider": provider,
                                    "runtime": runtime,
                                    "kind": "tool_call",
                                    "tool_name": tool_name,
                                    "preview": preview,
                                },
                            )
        elif event_type == "user":
            message = parsed.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if str(block.get("type", "")).lower() != "tool_result":
                            continue
                        self._broadcast_runtime_event(
                            "agent_invocation_progress",
                            {
                                "run_id": run_id,
                                "agent_type": agent_type,
                                "provider": provider,
                                "runtime": runtime,
                                "kind": "tool_result",
                                "preview": self._extract_tool_result_preview(block),
                            },
                        )
        elif event_type == "result":
            result_text = parsed.get("result")
            if isinstance(result_text, str) and result_text.strip():
                state.final_result = result_text.strip()
                self._broadcast_runtime_event(
                    "agent_invocation_progress",
                    {
                        "run_id": run_id,
                        "agent_type": agent_type,
                        "provider": provider,
                        "runtime": runtime,
                        "kind": "result",
                        "preview": self._truncate_preview(result_text),
                    },
                )

        item = parsed.get("item")
        if isinstance(item, dict):
            item_text = item.get("text")
            item_type = str(item.get("type", "")).lower()
            if isinstance(item_text, str) and item_text.strip() and (
                not item_type or "message" in item_type
            ):
                self._record_assistant_text(
                    state=state,
                    text=item_text,
                    run_id=run_id,
                    agent_type=agent_type,
                    provider=provider,
                    runtime=runtime,
                )

    def _parse_codex_stream_line(
        self,
        line: str,
        state: _ClaudeStreamState,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> None:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            self._broadcast_runtime_event(
                "agent_invocation_progress",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": provider,
                    "runtime": runtime,
                    "kind": "raw",
                    "preview": self._truncate_preview(line),
                },
            )
            return

        if not isinstance(parsed, dict):
            return

        self._capture_session_identifier(parsed, state)
        self._capture_cost(parsed, state)

        result_text = parsed.get("result")
        if isinstance(result_text, str) and result_text.strip():
            state.final_result = result_text.strip()
            self._broadcast_runtime_event(
                "agent_invocation_progress",
                {
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "provider": provider,
                    "runtime": runtime,
                    "kind": "result",
                    "preview": self._truncate_preview(result_text),
                },
            )

        item = parsed.get("item")
        if not isinstance(item, dict):
            return

        item_type = str(item.get("type", "")).lower()
        item_text = item.get("text")
        if isinstance(item_text, str) and item_text.strip() and (
            not item_type or "message" in item_type
        ):
            self._record_assistant_text(
                state=state,
                text=item_text,
                run_id=run_id,
                agent_type=agent_type,
                provider=provider,
                runtime=runtime,
            )

        if "tool" not in item_type:
            return

        kind = "tool_result" if "result" in item_type or "output" in item_type else "tool_call"
        if kind == "tool_call":
            state.tool_call_count += 1
        self._broadcast_runtime_event(
            "agent_invocation_progress",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": provider,
                "runtime": runtime,
                "kind": kind,
                "tool_name": str(item.get("name") or item.get("tool_name") or "").strip(),
                "preview": self._extract_codex_item_preview(item),
            },
        )

    def _record_assistant_text(
        self,
        state: _ClaudeStreamState,
        text: object,
        run_id: str,
        agent_type: str,
        provider: str,
        runtime: str,
    ) -> None:
        if not isinstance(text, str):
            return
        cleaned = text.strip()
        if not cleaned:
            return
        state.texts.append(cleaned)
        self._broadcast_runtime_event(
            "agent_invocation_progress",
            {
                "run_id": run_id,
                "agent_type": agent_type,
                "provider": provider,
                "runtime": runtime,
                "kind": "assistant_text",
                "preview": self._truncate_preview(cleaned),
            },
        )

    def _capture_session_identifier(self, parsed: dict, state: _ClaudeStreamState) -> None:
        for key in ("session_id", "thread_id"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                state.resolved_session_id = value.strip()
                break

    def _capture_cost(self, parsed: dict, state: _ClaudeStreamState) -> None:
        direct_cost = parsed.get("cost_usd") or parsed.get("total_cost_usd")
        if isinstance(direct_cost, (int, float)):
            state.cost_usd = float(direct_cost)
            return
        usage = parsed.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_cost_usd") or usage.get("cost_usd")
            if isinstance(total, (int, float)):
                state.cost_usd = float(total)

    def _extract_tool_result_preview(self, block: dict) -> str:
        content = block.get("content")
        if isinstance(content, str):
            return self._truncate_preview(content)
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            return self._truncate_preview(" ".join(chunks))
        return self._truncate_preview(json.dumps(content, default=str))

    def _extract_codex_item_preview(self, item: dict) -> str:
        for key in ("arguments", "args", "input", "output", "content", "text"):
            value = item.get(key)
            if value in (None, ""):
                continue
            if isinstance(value, str):
                return self._truncate_preview(value)
            return self._truncate_preview(json.dumps(value, default=str))
        return self._truncate_preview(json.dumps(item, default=str))

    def _parse_output(
        self,
        stdout: str,
        run_dir: Path,
        elapsed_ms: int,
        session_id: str,
        parse_mode: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> AgentResult:
        if parse_mode == "jsonl":
            return self._parse_jsonl_output(
                stdout=stdout,
                run_dir=run_dir,
                elapsed_ms=elapsed_ms,
                session_id=session_id,
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                auth_mode=auth_mode,
            )
        return self._parse_json_output(
            stdout=stdout,
            run_dir=run_dir,
            elapsed_ms=elapsed_ms,
            session_id=session_id,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            auth_mode=auth_mode,
        )

    def _parse_json_output(
        self,
        stdout: str,
        run_dir: Path,
        elapsed_ms: int,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> AgentResult:
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
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                auth_mode=auth_mode,
            )
        except (json.JSONDecodeError, TypeError):
            return AgentResult(
                response=stdout.strip(),
                run_dir=run_dir,
                cost_usd=0.0,
                duration_ms=elapsed_ms,
                session_id=session_id,
                success=True,
                provider=provider,
                runtime=runtime,
                requested_model=requested_model,
                effective_model=effective_model,
                auth_mode=auth_mode,
            )

    def _parse_jsonl_output(
        self,
        stdout: str,
        run_dir: Path,
        elapsed_ms: int,
        session_id: str,
        provider: str,
        runtime: str,
        requested_model: str | None,
        effective_model: str,
        auth_mode: str,
    ) -> AgentResult:
        texts: list[str] = []
        resolved_session_id = session_id
        usage_total = 0.0
        stream_event_count = 0
        tool_call_count = 0
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            stream_event_count += 1
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                thread_id = parsed.get("thread_id")
                if isinstance(thread_id, str) and thread_id.strip():
                    resolved_session_id = thread_id.strip()
                usage = parsed.get("usage")
                if isinstance(usage, dict):
                    total = usage.get("total_cost_usd") or usage.get("cost_usd")
                    if isinstance(total, (int, float)):
                        usage_total = float(total)
                item = parsed.get("item")
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).lower()
                    if "tool" in item_type:
                        tool_call_count += 1
                    text = item.get("text")
                    if isinstance(text, str) and (
                        not item_type or "message" in item_type
                    ):
                        texts.append(text)

        response = "\n".join(texts).strip() or stdout.strip()
        return AgentResult(
            response=response,
            run_dir=run_dir,
            cost_usd=usage_total,
            duration_ms=elapsed_ms,
            session_id=resolved_session_id,
            success=True,
            provider=provider,
            runtime=runtime,
            requested_model=requested_model,
            effective_model=effective_model,
            stream_event_count=stream_event_count,
            tool_call_count=tool_call_count,
            auth_mode=auth_mode,
        )

    def _resolve_workflow(self, agent_type: str) -> AgentWorkflow | None:
        try:
            return AgentWorkflow(agent_type)
        except ValueError:
            return None

    def _get_provider_status(self, provider: AgentProvider) -> ProviderReadiness:
        cached = self._provider_status_cache.get(provider)
        checked_at = self._provider_status_checked_at.get(provider)
        if cached is not None and (
            checked_at is None
            or (datetime.now(timezone.utc) - checked_at).total_seconds()
            < _PROVIDER_STATUS_CACHE_TTL_SECONDS
        ):
            return cached

        if provider == AgentProvider.CLAUDE_MAX:
            auth_status = self._claude_max_auth_status()
            status = ProviderReadiness(
                provider=provider,
                available=auth_status.available,
                runtime=_CLAUDE_RUNTIME,
                reason=auth_status.reason,
            )
        elif provider == AgentProvider.CODEX_PRO:
            available, reason = self._command_ready(
                self._codex_command,
                self._codex_command_args,
                strict_execution=True,
            )
            if not available:
                status = ProviderReadiness(
                    provider=provider,
                    available=False,
                    runtime=_CODEX_RUNTIME,
                    reason=reason,
                )
            else:
                auth_status = self._codex_auth_status()
                status = ProviderReadiness(
                    provider=provider,
                    available=auth_status.available,
                    runtime=_CODEX_RUNTIME,
                    reason=auth_status.reason or "; ".join(auth_status.diagnostics),
                )
        elif provider == AgentProvider.ZAI_CODING_PLAN:
            cli_available, cli_reason = self._command_ready(
                self._claude_command,
                self._claude_command_args,
                strict_execution=True,
            )
            status = ProviderReadiness(
                provider=provider,
                available=bool(self._zai_api_key.strip()) and cli_available,
                runtime=_CLAUDE_RUNTIME,
                reason=(
                    ""
                    if self._zai_api_key.strip() and cli_available
                    else (
                        "ZAI_API_KEY is not configured"
                        if not self._zai_api_key.strip()
                        else cli_reason
                    )
                ),
            )
        else:
            cli_available, cli_reason = self._command_ready(
                self._claude_command,
                self._claude_command_args,
                strict_execution=True,
            )
            status = ProviderReadiness(
                provider=provider,
                available=bool(self._openrouter_api_key.strip()) and cli_available,
                runtime=_CLAUDE_RUNTIME,
                reason=(
                    ""
                    if self._openrouter_api_key.strip() and cli_available
                    else (
                        "OPENROUTER_API_KEY is not configured"
                        if not self._openrouter_api_key.strip()
                        else cli_reason
                    )
                ),
            )

        self._provider_status_cache[provider] = status
        self._provider_status_checked_at[provider] = datetime.now(timezone.utc)
        return status

    def _claude_max_auth_status(self) -> _ClaudeAuthStatus:
        resolved = self._resolve_command(self._claude_command)
        if not resolved:
            return _ClaudeAuthStatus(
                available=False,
                reason=f"Command not found: {self._claude_command}",
            )

        try:
            result = subprocess.run(
                [resolved, *self._claude_command_args, "auth", "status"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_CLI_PREVIEW_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return _ClaudeAuthStatus(
                available=False,
                reason=f"Claude auth check failed for {self._claude_command}: {exc}",
            )
        except subprocess.TimeoutExpired:
            return _ClaudeAuthStatus(
                available=False,
                reason=f"Claude auth check timed out for {self._claude_command}",
            )

        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            return _ClaudeAuthStatus(
                available=False,
                reason=(
                    "Claude Max login check failed: "
                    f"{details or f'exit code {result.returncode}'}"
                ),
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return _ClaudeAuthStatus(
                available=False,
                reason="Claude auth status returned invalid JSON",
            )

        logged_in = bool(payload.get("loggedIn"))
        auth_method = str(payload.get("authMethod", "")).strip()
        subscription_type = str(payload.get("subscriptionType", "")).strip().lower()

        if not logged_in:
            return _ClaudeAuthStatus(
                available=False,
                reason="Claude Max login required: run 'claude auth login'",
                auth_method=auth_method,
                subscription_type=subscription_type,
            )
        if auth_method != "claude.ai":
            return _ClaudeAuthStatus(
                available=False,
                reason=f"Claude Max requires claude.ai auth (found {auth_method or 'unknown'})",
                auth_method=auth_method,
                subscription_type=subscription_type,
            )
        if subscription_type != "max":
            found = subscription_type or "unknown"
            return _ClaudeAuthStatus(
                available=False,
                reason=f"Claude Max subscription required (found {found})",
                auth_method=auth_method,
                subscription_type=subscription_type,
            )
        return _ClaudeAuthStatus(
            available=True,
            auth_method=auth_method,
            subscription_type=subscription_type,
        )

    def _codex_auth_status(self) -> _CodexAuthStatus:
        codex_home = self._codex_home()
        diagnostics = self._codex_runtime_diagnostics(codex_home)
        auth_path = codex_home / "auth.json"
        if not auth_path.exists():
            return _CodexAuthStatus(
                available=False,
                reason=f"Codex ChatGPT auth file not found: {auth_path}",
                diagnostics=diagnostics,
            )

        try:
            payload = json.loads(auth_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return _CodexAuthStatus(
                available=False,
                reason=f"Codex auth check failed for {auth_path}: {exc}",
                diagnostics=diagnostics,
            )

        auth_mode = str(payload.get("auth_mode", "")).strip().lower()
        if auth_mode != _CODEX_AUTH_MODE:
            return _CodexAuthStatus(
                available=False,
                reason=(
                    "Codex Pro requires chatgpt auth mode "
                    f"(found {auth_mode or 'unknown'})"
                ),
                auth_mode=auth_mode,
                diagnostics=diagnostics,
            )
        if payload.get("OPENAI_API_KEY"):
            return _CodexAuthStatus(
                available=False,
                reason="Codex Pro requires ChatGPT auth without OPENAI_API_KEY fallback",
                auth_mode=auth_mode,
                diagnostics=diagnostics,
            )
        if not self._codex_has_tokens(payload):
            return _CodexAuthStatus(
                available=False,
                reason=(
                    "Codex Pro requires local ChatGPT tokens in auth.json; "
                    "re-authenticate before using the subscription-backed profile"
                ),
                auth_mode=auth_mode,
                diagnostics=diagnostics,
            )
        return _CodexAuthStatus(
            available=True,
            auth_mode=auth_mode,
            diagnostics=diagnostics,
        )

    def _codex_has_tokens(self, payload: dict) -> bool:
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return False
        for key in ("access_token", "id_token", "refresh_token"):
            value = tokens.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return False

    def _codex_runtime_diagnostics(self, codex_home: Path) -> list[str]:
        diagnostics: list[str] = []
        config_path = codex_home / "config.toml"
        if config_path.exists():
            try:
                config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                config = {}
            effort = str(config.get("model_reasoning_effort", "")).strip().lower()
            if effort == "xhigh":
                diagnostics.append(
                    "Codex config uses model_reasoning_effort=xhigh, which can increase "
                    "orchestrator latency"
                )

        codex_log = self._find_latest_codex_log()
        if codex_log is not None:
            try:
                tail = self._read_text_tail(codex_log).lower()
            except OSError:
                tail = ""
            if "database is locked" in tail or "slow statement" in tail:
                diagnostics.append(
                    "Recent Codex logs show shared state lock/slow-write symptoms under "
                    f"{codex_home}"
                )
        return diagnostics

    def _find_latest_codex_log(self) -> Path | None:
        if os.name != "nt":
            return None
        logs_root = Path.home() / "AppData" / "Roaming" / "Code" / "logs"
        if not logs_root.exists():
            return None
        candidates = sorted(
            logs_root.glob("**/openai.chatgpt/Codex.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _read_text_tail(self, path: Path, max_bytes: int = 65536) -> str:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")

    def _codex_home(self) -> Path:
        raw = os.environ.get("CODEX_HOME", "").strip()
        return Path(raw).expanduser() if raw else (Path.home() / ".codex")

    def _auth_mode_for_provider(self, provider: AgentProvider) -> str:
        if provider == AgentProvider.CLAUDE_MAX:
            return "claude.ai:max"
        if provider == AgentProvider.CODEX_PRO:
            return self._codex_auth_status().auth_mode or "unknown"
        if provider == AgentProvider.ZAI_CODING_PLAN:
            return "anthropic-compat:zai"
        if provider == AgentProvider.OPENROUTER:
            return "anthropic-compat:openrouter"
        return "local-cli"

    def _command_ready(
        self,
        command: str,
        command_args: list[str] | None = None,
        strict_execution: bool = False,
    ) -> tuple[bool, str]:
        resolved = self._resolve_command(command)
        if not resolved:
            return False, f"Command not found: {command}"
        if not strict_execution:
            return True, ""

        try:
            result = subprocess.run(
                [resolved, *(command_args or []), "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=_CLI_PREVIEW_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return False, f"Command preflight failed for {command}: {exc}"
        except subprocess.TimeoutExpired:
            return False, f"Command preflight timed out for {command}"

        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            return False, (
                f"Command preflight failed for {command}: "
                f"{details or f'exit code {result.returncode}'}"
            )
        return True, ""

    def _resolve_command(self, command: str) -> str | None:
        candidate = command.strip()
        if not candidate:
            return None

        resolved = shutil.which(candidate)
        if resolved:
            resolved_path = Path(resolved)
            if os.name == "nt" and not resolved_path.suffix:
                for suffix in (".cmd", ".exe", ".bat"):
                    suffixed = Path(f"{resolved}{suffix}")
                    if suffixed.exists():
                        return str(suffixed)
            return str(resolved_path)

        direct = Path(candidate).expanduser()
        if direct.exists():
            return str(direct)

        if os.name == "nt" and not direct.suffix:
            for suffix in (".cmd", ".exe", ".bat"):
                suffixed = Path(f"{candidate}{suffix}")
                if suffixed.exists():
                    return str(suffixed)
        return None

    def _require_resolved_command(self, command: str) -> str:
        resolved = self._resolve_command(command)
        if resolved is None:
            raise FileNotFoundError(f"Command not found: {command}")
        return resolved

    def _broadcast_runtime_event(self, event_type: str, data: dict) -> None:
        if self._event_stream is None:
            return
        self._event_stream.broadcast(event_type, data)

    def _truncate_preview(self, text: str) -> str:
        preview = text.strip().replace("\r", " ").replace("\n", " ")
        if len(preview) <= _PROGRESS_PREVIEW_LIMIT:
            return preview
        return preview[: _PROGRESS_PREVIEW_LIMIT - 3] + "..."
