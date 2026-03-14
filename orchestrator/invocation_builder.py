"""CLI invocation construction and environment isolation for agent runtimes."""
from __future__ import annotations

import os
from dataclasses import dataclass

from orchestrator.agent_preferences import DEFAULT_PROVIDER_MODELS
from orchestrator.provider_auth import ProviderAuthChecker
from schemas.agent_preferences import AgentProvider, AgentSelection
from schemas.prompt_package import PromptPackage

_CLAUDE_RUNTIME = "claude_cli"
_CODEX_RUNTIME = "codex_cli"
_CODEX_SANDBOX = "read-only"

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
class InvocationSpec:
    command: str
    args: list[str]
    env: dict[str, str]
    runtime: str
    provider: AgentProvider
    requested_model: str | None
    effective_model: str
    parse_mode: str = "json"


class InvocationBuilder:
    """Constructs CLI invocations for each provider runtime."""

    def __init__(
        self,
        auth_checker: ProviderAuthChecker,
        claude_command: str = "claude",
        claude_command_args: list[str] | None = None,
        codex_command: str = "codex",
        codex_command_args: list[str] | None = None,
        default_model: str = "sonnet",
    ) -> None:
        self._auth_checker = auth_checker
        self._claude_command = claude_command
        self._claude_command_args = claude_command_args or []
        self._codex_command = codex_command
        self._codex_command_args = codex_command_args or []
        self._default_model = default_model

    def build(
        self,
        prompt_package: PromptPackage,
        selection: AgentSelection,
        requested_model: str | None,
        max_turns: int,
        allowed_tools: list[str] | None,
    ) -> InvocationSpec:
        if selection.provider == AgentProvider.CODEX_PRO:
            return self.build_codex(prompt_package, selection, requested_model)
        return self.build_claude(
            prompt_package=prompt_package,
            selection=selection,
            requested_model=requested_model,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
        )

    def build_claude(
        self,
        prompt_package: PromptPackage,
        selection: AgentSelection,
        requested_model: str | None,
        max_turns: int,
        allowed_tools: list[str] | None,
    ) -> InvocationSpec:
        resolved_command = self._auth_checker.require_resolved_command(self._claude_command)
        cli_model = self.resolve_claude_cli_model(selection)
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

        return InvocationSpec(
            command=resolved_command,
            args=args,
            env=self.build_env(
                self.claude_env_overrides(selection),
                clear_keys=_ANTHROPIC_ENV_KEYS_TO_CLEAR,
            ),
            runtime=_CLAUDE_RUNTIME,
            provider=selection.provider,
            requested_model=requested_model,
            effective_model=selection.model or "",
            parse_mode="stream-json",
        )

    def build_codex(
        self,
        prompt_package: PromptPackage,
        selection: AgentSelection,
        requested_model: str | None,
    ) -> InvocationSpec:
        resolved_command = self._auth_checker.require_resolved_command(self._codex_command)
        system_prompt = prompt_package.system_prompt.strip()
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
        ]
        if system_prompt:
            args.extend(["--instructions", system_prompt])
        args.append(prompt_package.task_prompt)
        return InvocationSpec(
            command=resolved_command,
            args=args,
            env=self.build_env(clear_keys=_OPENAI_ENV_KEYS_TO_CLEAR),
            runtime=_CODEX_RUNTIME,
            provider=selection.provider,
            requested_model=requested_model,
            effective_model=selection.model or "",
            parse_mode="jsonl-stream",
        )

    def build_env(
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

    def claude_env_overrides(self, selection: AgentSelection) -> dict[str, str]:
        model = selection.model or ""
        if selection.provider == AgentProvider.ZAI_CODING_PLAN:
            return {
                "ANTHROPIC_AUTH_TOKEN": self._auth_checker.zai_api_key,
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": model or DEFAULT_PROVIDER_MODELS[AgentProvider.ZAI_CODING_PLAN],
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model or DEFAULT_PROVIDER_MODELS[AgentProvider.ZAI_CODING_PLAN],
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air",
                "API_TIMEOUT_MS": "3000000",
            }
        if selection.provider == AgentProvider.OPENROUTER:
            effective = model or DEFAULT_PROVIDER_MODELS[AgentProvider.OPENROUTER]
            return {
                "ANTHROPIC_AUTH_TOKEN": self._auth_checker.openrouter_api_key,
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
                "ANTHROPIC_DEFAULT_SONNET_MODEL": effective,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": effective,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": effective,
            }
        return {}

    def resolve_claude_cli_model(self, selection: AgentSelection) -> str:
        model = (selection.model or self._default_model).strip()
        if selection.provider == AgentProvider.CLAUDE_MAX:
            return model
        lowered = model.lower()
        if "opus" in lowered:
            return "opus"
        if "haiku" in lowered or "air" in lowered:
            return "haiku"
        return "sonnet"

    def merge_codex_prompt(self, prompt_package: PromptPackage) -> str:
        prompt_parts: list[str] = []
        system_prompt = prompt_package.system_prompt.strip()
        if system_prompt:
            prompt_parts.append(f"System instructions:\n{system_prompt}")
        prompt_parts.append(prompt_package.task_prompt)
        return "\n\n".join(part for part in prompt_parts if part).strip()
