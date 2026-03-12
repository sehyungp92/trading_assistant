"""Preference manager for agent runtime provider selection."""
from __future__ import annotations

from collections.abc import Callable

from schemas.agent_preferences import (
    AgentPreferences,
    AgentPreferencesView,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
)

DEFAULT_PROVIDER_MODELS: dict[AgentProvider, str] = {
    AgentProvider.CLAUDE_MAX: "sonnet",
    AgentProvider.CODEX_PRO: "gpt-5.4",
    AgentProvider.ZAI_CODING_PLAN: "glm-4.7",
    AgentProvider.OPENROUTER: "anthropic/claude-sonnet-4.5",
}

WORKFLOW_ORDER: tuple[AgentWorkflow, ...] = (
    AgentWorkflow.DAILY_ANALYSIS,
    AgentWorkflow.WEEKLY_ANALYSIS,
    AgentWorkflow.WFO,
    AgentWorkflow.TRIAGE,
)


class AgentPreferencesManager:
    """Manages persisted preferences and resolves effective selections."""

    def __init__(
        self,
        preferences: AgentPreferences | None = None,
        provider_status_resolver: Callable[[], list[ProviderReadiness]] | None = None,
    ) -> None:
        self._preferences = preferences or AgentPreferences()
        self._provider_status_resolver = provider_status_resolver

    def set_provider_status_resolver(
        self, resolver: Callable[[], list[ProviderReadiness]] | None
    ) -> None:
        self._provider_status_resolver = resolver

    def get_preferences(self) -> AgentPreferences:
        return self._preferences.model_copy(deep=True)

    def set_preferences(self, preferences: AgentPreferences) -> None:
        self._preferences = preferences.model_copy(deep=True)

    def resolve_selection(
        self,
        workflow: AgentWorkflow | None,
        model_override: str | None = None,
    ) -> tuple[AgentSelection, str | None]:
        requested = self._requested_selection(workflow)
        requested_model = model_override.strip() if model_override and model_override.strip() else requested.model
        return (
            AgentSelection(
                provider=requested.provider,
                model=requested_model or DEFAULT_PROVIDER_MODELS[requested.provider],
            ),
            requested_model,
        )

    def build_view(self) -> AgentPreferencesView:
        effective = {
            workflow: self.resolve_selection(workflow)[0]
            for workflow in WORKFLOW_ORDER
        }
        providers = self._provider_status_resolver() if self._provider_status_resolver else []
        return AgentPreferencesView(
            default=self._preferences.default.model_copy(deep=True),
            overrides={
                workflow: selection.model_copy(deep=True) if selection is not None else None
                for workflow, selection in self._preferences.overrides.items()
            },
            effective=effective,
            providers=providers,
        )

    def unavailable_reasons(self, preferences: AgentPreferences) -> list[str]:
        if self._provider_status_resolver is None:
            return []
        provider_status = {
            status.provider: status
            for status in self._provider_status_resolver()
        }
        errors: list[str] = []

        default_status = provider_status.get(preferences.default.provider)
        if default_status and not default_status.available:
            errors.append(
                f"default provider '{preferences.default.provider.value}' is unavailable: "
                f"{default_status.reason or 'unknown reason'}"
            )

        for workflow, selection in preferences.overrides.items():
            if selection is None:
                continue
            status = provider_status.get(selection.provider)
            if status and not status.available:
                errors.append(
                    f"override for '{workflow.value}' uses unavailable provider "
                    f"'{selection.provider.value}': {status.reason or 'unknown reason'}"
                )

        return errors

    def _requested_selection(self, workflow: AgentWorkflow | None) -> AgentSelection:
        if workflow is not None:
            override = self._preferences.overrides.get(workflow)
            if override is not None:
                return override
        return self._preferences.default
