"""Tests for orchestrator/agent_preferences.py — AgentPreferencesManager."""
from __future__ import annotations

import pytest

from orchestrator.agent_preferences import (
    DEFAULT_PROVIDER_MODELS,
    AgentPreferencesManager,
)
from schemas.agent_preferences import (
    AgentPreferences,
    AgentPreferencesView,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
    ProviderReadiness,
)


def _readiness(
    provider: AgentProvider,
    available: bool = True,
    reason: str = "",
) -> ProviderReadiness:
    runtime = "codex_cli" if provider == AgentProvider.CODEX_PRO else "claude_cli"
    return ProviderReadiness(
        provider=provider, available=available, runtime=runtime, reason=reason,
    )


class TestResolveSelection:
    def test_default_provider_when_no_workflow(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(workflow=None)
        assert sel.provider == AgentProvider.CLAUDE_MAX
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.CLAUDE_MAX]
        assert req_model is None

    def test_default_provider_when_workflow_has_no_override(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)
        assert sel.provider == AgentProvider.CLAUDE_MAX
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.CLAUDE_MAX]

    def test_override_provider_for_workflow(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            overrides={
                AgentWorkflow.WFO: AgentSelection(
                    provider=AgentProvider.CODEX_PRO, model="gpt-5.4",
                ),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.WFO)
        assert sel.provider == AgentProvider.CODEX_PRO
        assert sel.model == "gpt-5.4"

    def test_non_overridden_workflow_falls_back_to_default(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
            overrides={
                AgentWorkflow.WFO: AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.DAILY_ANALYSIS)
        assert sel.provider == AgentProvider.CODEX_PRO

    def test_model_override_takes_precedence(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(
            AgentWorkflow.DAILY_ANALYSIS, model_override="opus",
        )
        assert sel.model == "opus"
        assert req_model == "opus"

    def test_blank_model_override_ignored(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(
            AgentWorkflow.DAILY_ANALYSIS, model_override="   ",
        )
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.CLAUDE_MAX]
        assert req_model is None

    def test_none_model_override_ignored(self):
        mgr = AgentPreferencesManager()
        sel, req_model = mgr.resolve_selection(
            AgentWorkflow.DAILY_ANALYSIS, model_override=None,
        )
        assert req_model is None

    def test_model_from_selection_used_when_no_override(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.OPENROUTER, model="custom/model"),
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.TRIAGE)
        assert sel.model == "custom/model"

    def test_default_model_filled_when_selection_has_none(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model=None),
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        sel, _ = mgr.resolve_selection(AgentWorkflow.WEEKLY_ANALYSIS)
        assert sel.model == DEFAULT_PROVIDER_MODELS[AgentProvider.ZAI_CODING_PLAN]


class TestBuildView:
    def test_view_includes_effective_for_all_workflows(self):
        mgr = AgentPreferencesManager()
        view = mgr.build_view()
        assert isinstance(view, AgentPreferencesView)
        assert set(view.effective.keys()) == set(AgentWorkflow)
        for workflow in AgentWorkflow:
            assert view.effective[workflow].provider == AgentProvider.CLAUDE_MAX

    def test_view_reflects_overrides(self):
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            overrides={
                AgentWorkflow.WFO: AgentSelection(provider=AgentProvider.CODEX_PRO),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        view = mgr.build_view()
        assert view.effective[AgentWorkflow.WFO].provider == AgentProvider.CODEX_PRO
        assert view.effective[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.CLAUDE_MAX

    def test_view_includes_provider_statuses_when_resolver_set(self):
        statuses = [_readiness(AgentProvider.CLAUDE_MAX, True)]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        view = mgr.build_view()
        assert len(view.providers) == 1
        assert view.providers[0].provider == AgentProvider.CLAUDE_MAX
        assert view.providers[0].available is True

    def test_view_empty_providers_when_no_resolver(self):
        mgr = AgentPreferencesManager()
        view = mgr.build_view()
        assert view.providers == []

    def test_view_overrides_deep_copied(self):
        prefs = AgentPreferences(
            overrides={
                AgentWorkflow.WFO: AgentSelection(
                    provider=AgentProvider.CODEX_PRO, model="gpt-5.4",
                ),
            },
        )
        mgr = AgentPreferencesManager(preferences=prefs)
        view = mgr.build_view()
        assert AgentWorkflow.WFO in view.overrides
        # Mutating the view should not affect the manager
        view.overrides[AgentWorkflow.WFO].model = "mutated"
        sel, _ = mgr.resolve_selection(AgentWorkflow.WFO)
        assert sel.model == "gpt-5.4"


class TestUnavailableReasons:
    def test_no_resolver_returns_empty(self):
        mgr = AgentPreferencesManager()
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
        )
        assert mgr.unavailable_reasons(prefs) == []

    def test_available_default_no_errors(self):
        statuses = [_readiness(AgentProvider.CLAUDE_MAX, True)]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences()
        assert mgr.unavailable_reasons(prefs) == []

    def test_unavailable_default_returns_error(self):
        statuses = [_readiness(AgentProvider.CODEX_PRO, False, "auth missing")]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
        )
        errors = mgr.unavailable_reasons(prefs)
        assert len(errors) == 1
        assert "codex_pro" in errors[0]
        assert "auth missing" in errors[0]

    def test_unavailable_override_returns_error(self):
        statuses = [
            _readiness(AgentProvider.CLAUDE_MAX, True),
            _readiness(AgentProvider.ZAI_CODING_PLAN, False, "no key"),
        ]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences(
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(
                    provider=AgentProvider.ZAI_CODING_PLAN,
                ),
            },
        )
        errors = mgr.unavailable_reasons(prefs)
        assert len(errors) == 1
        assert "triage" in errors[0]
        assert "zai_coding_plan" in errors[0]

    def test_none_override_skipped(self):
        statuses = [_readiness(AgentProvider.CLAUDE_MAX, True)]
        mgr = AgentPreferencesManager(provider_status_resolver=lambda: statuses)
        prefs = AgentPreferences(
            overrides={AgentWorkflow.WFO: None},
        )
        assert mgr.unavailable_reasons(prefs) == []


class TestGetSetPreferences:
    def test_get_returns_deep_copy(self):
        mgr = AgentPreferencesManager()
        prefs = mgr.get_preferences()
        prefs.default.model = "mutated"
        assert mgr.get_preferences().default.model is None

    def test_set_replaces_preferences(self):
        mgr = AgentPreferencesManager()
        new_prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4"),
        )
        mgr.set_preferences(new_prefs)
        assert mgr.get_preferences().default.provider == AgentProvider.CODEX_PRO

    def test_set_preferences_deep_copies(self):
        mgr = AgentPreferencesManager()
        new_prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO),
        )
        mgr.set_preferences(new_prefs)
        new_prefs.default.model = "mutated-after-set"
        assert mgr.get_preferences().default.model is None

    def test_set_provider_status_resolver(self):
        mgr = AgentPreferencesManager()
        assert mgr.build_view().providers == []
        mgr.set_provider_status_resolver(
            lambda: [_readiness(AgentProvider.CLAUDE_MAX)],
        )
        assert len(mgr.build_view().providers) == 1


class TestModelNormalization:
    def test_whitespace_model_normalized_to_none(self):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="   ")
        assert sel.model is None

    def test_stripped_model_preserved(self):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model="  opus  ")
        assert sel.model == "opus"

    def test_none_model_stays_none(self):
        sel = AgentSelection(provider=AgentProvider.CLAUDE_MAX, model=None)
        assert sel.model is None
