"""Preference manager for agent runtime provider selection."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from orchestrator.provider_cooldown import ProviderCooldownTracker
from schemas.agent_preferences import (
    AgentPreferences,
    AgentPreferencesView,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
    WorkflowTuning,
)

DEFAULT_PROVIDER_MODELS: dict[AgentProvider, str] = {
    AgentProvider.CLAUDE_MAX: "sonnet",
    AgentProvider.CODEX_PRO: "gpt-5.4",
    AgentProvider.ZAI_CODING_PLAN: "glm-5",
    AgentProvider.OPENROUTER: "minimax/minimax-m2.5",
}

WORKFLOW_ORDER: tuple[AgentWorkflow, ...] = (
    AgentWorkflow.DAILY_ANALYSIS,
    AgentWorkflow.WEEKLY_ANALYSIS,
    AgentWorkflow.WFO,
    AgentWorkflow.TRIAGE,
)

DEFAULT_WORKFLOW_TUNING: dict[AgentWorkflow, WorkflowTuning] = {
    AgentWorkflow.DAILY_ANALYSIS: WorkflowTuning(timeout_seconds=600, max_turns=5),
    AgentWorkflow.WEEKLY_ANALYSIS: WorkflowTuning(timeout_seconds=900, max_turns=8),
    AgentWorkflow.WFO: WorkflowTuning(timeout_seconds=1200, max_turns=15),
    AgentWorkflow.TRIAGE: WorkflowTuning(timeout_seconds=300, max_turns=4),
}


class AgentPreferencesManager:
    """Manages persisted preferences and resolves effective selections."""

    def __init__(
        self,
        preferences: AgentPreferences | None = None,
        provider_status_resolver: Callable[[], list[ProviderReadiness]] | None = None,
        findings_dir: Path | None = None,
    ) -> None:
        self._preferences = preferences or AgentPreferences()
        self._provider_status_resolver = provider_status_resolver
        self._findings_dir = Path(findings_dir) if findings_dir is not None else None

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
        explicit_override = model_override.strip() if model_override and model_override.strip() else None
        requested_model = explicit_override or requested.model
        selection = AgentSelection(
            provider=requested.provider,
            model=requested_model or DEFAULT_PROVIDER_MODELS[requested.provider],
        )
        if workflow is not None and explicit_override is None:
            learned = self._resolve_learned_selection(workflow, selection)
            if learned is not None:
                selection = learned
                requested_model = selection.model
        return selection, requested_model

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

    def resolve_tuning(
        self,
        workflow: AgentWorkflow | None,
        *,
        max_turns_override: int | None = None,
        allowed_tools_override: list[str] | None = None,
    ) -> WorkflowTuning:
        """Resolve effective tuning for a workflow.

        Merges: caller overrides > user preferences > built-in defaults.
        """
        default = DEFAULT_WORKFLOW_TUNING.get(workflow) if workflow else None
        user = self._preferences.workflow_tuning.get(workflow) if workflow else None

        def _attr(obj: WorkflowTuning | None, name: str):
            return getattr(obj, name, None) if obj is not None else None

        def _first_set(*values):
            for v in values:
                if v is not None:
                    return v
            return None

        timeout = _first_set(_attr(user, "timeout_seconds"), _attr(default, "timeout_seconds"))
        max_turns = _first_set(max_turns_override, _attr(user, "max_turns"), _attr(default, "max_turns"))
        allowed_tools = _first_set(allowed_tools_override, _attr(user, "allowed_tools"), _attr(default, "allowed_tools"))

        return WorkflowTuning(
            timeout_seconds=timeout,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
        )

    def resolve_with_fallbacks(
        self,
        workflow: AgentWorkflow | None,
        model_override: str | None = None,
        cooldown_tracker: ProviderCooldownTracker | None = None,
    ) -> list[tuple[AgentSelection, str | None]]:
        """Return ordered list of (selection, requested_model) candidates.

        The primary selection is first, followed by fallback_chain entries,
        skipping providers that are in cooldown or unavailable.
        """
        primary_selection, requested_model = self.resolve_selection(workflow, model_override)

        # Gather provider availability
        provider_status: dict[AgentProvider, bool] = {}
        if self._provider_status_resolver:
            for status in self._provider_status_resolver():
                provider_status[status.provider] = status.available

        candidates: list[tuple[AgentSelection, str | None]] = []
        seen_providers: set[AgentProvider] = set()

        def _add_candidate(provider: AgentProvider, model: str | None, req_model: str | None) -> None:
            if provider in seen_providers:
                return
            seen_providers.add(provider)
            if cooldown_tracker and cooldown_tracker.is_cooled_down(provider):
                return
            if provider_status.get(provider) is False:
                return
            effective_model = model or DEFAULT_PROVIDER_MODELS.get(provider, "")
            candidates.append((
                AgentSelection(provider=provider, model=effective_model),
                req_model,
            ))

        # Primary first
        _add_candidate(primary_selection.provider, primary_selection.model, requested_model)

        # Fallback chain
        for entry in self._preferences.fallback_chain:
            _add_candidate(entry.provider, entry.model, None)

        return candidates

    def _requested_selection(self, workflow: AgentWorkflow | None) -> AgentSelection:
        if workflow is not None:
            override = self._preferences.overrides.get(workflow)
            if override is not None:
                return override
        return self._preferences.default

    def _resolve_learned_selection(
        self,
        workflow: AgentWorkflow,
        requested: AgentSelection,
    ) -> AgentSelection | None:
        if self._findings_dir is None:
            return None
        try:
            from skills.provider_route_scorer import ProviderRouteScorer

            recommendation = ProviderRouteScorer(self._findings_dir).recommend_provider(
                workflow=workflow.value,
                requested_provider=requested.provider.value,
            )
            if not recommendation:
                return None
            provider_value = recommendation.get("provider", "")
            if not provider_value or provider_value == requested.provider.value:
                return None
            provider = AgentProvider(provider_value)
            return AgentSelection(
                provider=provider,
                model=recommendation.get("model") or DEFAULT_PROVIDER_MODELS.get(provider, requested.model),
            )
        except Exception:
            return None
