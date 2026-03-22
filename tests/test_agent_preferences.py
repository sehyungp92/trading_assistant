"""Tests for agent preferences: manager, integration (seeding/loading/saving), and API."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.agent_preferences import (
    DEFAULT_PROVIDER_MODELS,
    AgentPreferencesManager,
)
from orchestrator.app import (
    _load_agent_preferences,
    _save_agent_preferences,
    _seed_agent_preferences,
    _selection_from_env,
    create_app,
)
from orchestrator.config import AppConfig
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


# ---------------------------------------------------------------------------
# Integration tests: seeding, loading, saving, env config
# (merged from test_agent_preferences_integration.py)
# ---------------------------------------------------------------------------


class TestSelectionFromEnv:
    def test_valid_provider(self):
        sel = _selection_from_env("claude_max")
        assert sel is not None
        assert sel.provider == AgentProvider.CLAUDE_MAX
        assert sel.model is None

    def test_valid_provider_with_model(self):
        sel = _selection_from_env("codex_pro", "gpt-5.4")
        assert sel is not None
        assert sel.provider == AgentProvider.CODEX_PRO
        assert sel.model == "gpt-5.4"

    def test_empty_string_returns_none(self):
        assert _selection_from_env("") is None
        assert _selection_from_env("  ") is None

    def test_invalid_provider_returns_none(self):
        assert _selection_from_env("invalid_provider") is None

    def test_whitespace_model_normalised_to_none(self):
        sel = _selection_from_env("claude_max", "   ")
        assert sel is not None
        assert sel.model is None

    def test_case_insensitive(self):
        sel = _selection_from_env("CLAUDE_MAX")
        assert sel is not None
        assert sel.provider == AgentProvider.CLAUDE_MAX

    def test_zai_provider(self):
        sel = _selection_from_env("zai_coding_plan", "glm-5")
        assert sel is not None
        assert sel.provider == AgentProvider.ZAI_CODING_PLAN

    def test_openrouter_provider(self):
        sel = _selection_from_env("openrouter", "minimax/minimax-m2.5")
        assert sel is not None
        assert sel.provider == AgentProvider.OPENROUTER


class TestSeedAgentPreferences:
    def test_defaults_to_claude_max(self):
        config = AppConfig()
        prefs = _seed_agent_preferences(config)
        assert prefs.default.provider == AgentProvider.CLAUDE_MAX
        assert len(prefs.overrides) == 0

    def test_custom_default_provider(self):
        config = AppConfig(agent_default_provider="codex_pro", agent_default_model="gpt-5.4")
        prefs = _seed_agent_preferences(config)
        assert prefs.default.provider == AgentProvider.CODEX_PRO
        assert prefs.default.model == "gpt-5.4"

    def test_per_workflow_overrides(self):
        config = AppConfig(
            daily_agent_provider="zai_coding_plan",
            daily_agent_model="glm-5",
            wfo_agent_provider="codex_pro",
            wfo_agent_model="gpt-5.4",
        )
        prefs = _seed_agent_preferences(config)
        assert AgentWorkflow.DAILY_ANALYSIS in prefs.overrides
        assert prefs.overrides[AgentWorkflow.DAILY_ANALYSIS].provider == AgentProvider.ZAI_CODING_PLAN
        assert AgentWorkflow.WFO in prefs.overrides
        assert prefs.overrides[AgentWorkflow.WFO].provider == AgentProvider.CODEX_PRO
        assert AgentWorkflow.WEEKLY_ANALYSIS not in prefs.overrides

    def test_empty_workflow_overrides_not_added(self):
        config = AppConfig(daily_agent_provider="", wfo_agent_provider="")
        prefs = _seed_agent_preferences(config)
        assert len(prefs.overrides) == 0

    def test_invalid_default_falls_back_to_claude_max(self):
        config = AppConfig(agent_default_provider="nonsense")
        prefs = _seed_agent_preferences(config)
        assert prefs.default.provider == AgentProvider.CLAUDE_MAX


class TestLoadSavePreferences:
    def test_load_from_disk(self, tmp_path: Path):
        prefs_path = tmp_path / "agent_preferences.json"
        saved = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.CODEX_PRO, model="gpt-5.4"),
            overrides={
                AgentWorkflow.WFO: AgentSelection(
                    provider=AgentProvider.OPENROUTER, model="minimax/minimax-m2.5",
                ),
            },
        )
        prefs_path.write_text(saved.model_dump_json(indent=2), encoding="utf-8")

        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == AgentProvider.CODEX_PRO
        assert loaded.overrides[AgentWorkflow.WFO].provider == AgentProvider.OPENROUTER

    def test_load_falls_back_to_seed_on_missing(self, tmp_path: Path):
        prefs_path = tmp_path / "missing.json"
        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == AgentProvider.CLAUDE_MAX

    def test_load_falls_back_on_corrupt_file(self, tmp_path: Path):
        prefs_path = tmp_path / "corrupt.json"
        prefs_path.write_text("{bad json", encoding="utf-8")
        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == AgentProvider.CLAUDE_MAX

    def test_save_creates_file(self, tmp_path: Path):
        prefs_path = tmp_path / "data" / "agent_preferences.json"
        prefs = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.ZAI_CODING_PLAN, model="glm-5"),
        )
        _save_agent_preferences(prefs, prefs_path)
        assert prefs_path.exists()
        stored = json.loads(prefs_path.read_text(encoding="utf-8"))
        assert stored["default"]["provider"] == "zai_coding_plan"

    def test_round_trip(self, tmp_path: Path):
        prefs_path = tmp_path / "prefs.json"
        original = AgentPreferences(
            default=AgentSelection(provider=AgentProvider.OPENROUTER, model="custom/model"),
            overrides={
                AgentWorkflow.TRIAGE: AgentSelection(provider=AgentProvider.CLAUDE_MAX),
            },
        )
        _save_agent_preferences(original, prefs_path)
        loaded = _load_agent_preferences(prefs_path, AppConfig())
        assert loaded.default.provider == original.default.provider
        assert loaded.default.model == original.default.model
        assert loaded.overrides[AgentWorkflow.TRIAGE].provider == AgentProvider.CLAUDE_MAX


class TestAppConfigEnvVars:
    def test_from_env_reads_agent_settings(self, monkeypatch):
        monkeypatch.setenv("AGENT_PROVIDER", "codex_pro")
        monkeypatch.setenv("AGENT_MODEL", "gpt-5.4")
        monkeypatch.setenv("DAILY_AGENT_PROVIDER", "zai_coding_plan")
        monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        config = AppConfig.from_env()
        assert config.agent_default_provider == "codex_pro"
        assert config.agent_default_model == "gpt-5.4"
        assert config.daily_agent_provider == "zai_coding_plan"
        assert config.zai_api_key == "zai-test-key"
        assert config.openrouter_api_key == "or-test-key"

    def test_from_env_defaults(self):
        config = AppConfig()
        assert config.claude_command == "claude"
        assert config.codex_command == "codex"
        assert config.zai_api_key == ""
        assert config.openrouter_api_key == ""
        assert config.agent_default_provider == ""


# ---------------------------------------------------------------------------
# API tests: agent provider preference persistence and validation
# (merged from test_agent_preferences_api.py)
# ---------------------------------------------------------------------------


def _set_provider_statuses(app, statuses: dict[AgentProvider, tuple[bool, str, str]]) -> None:
    cache = app.state.agent_runner._auth_checker._provider_status_cache
    cache.clear()
    cache.update({
        provider: ProviderReadiness(
            provider=provider,
            available=available,
            runtime=runtime,
            reason=reason,
        )
        for provider, (available, runtime, reason) in statuses.items()
    })


@pytest.fixture
def app_factory(tmp_path):
    def _create():
        return create_app(db_dir=str(tmp_path), config=AppConfig())

    return _create


class TestAgentPreferencesApi:
    @pytest.mark.asyncio
    async def test_get_returns_default_effective_and_provider_readiness(self, app_factory):
        app = app_factory()
        await app.state.queue.initialize()
        await app.state.registry.initialize()
        _set_provider_statuses(app, {
            AgentProvider.CLAUDE_MAX: (
                False,
                "claude_cli",
                "Claude Max login required: run 'claude auth login'",
            ),
            AgentProvider.CODEX_PRO: (True, "codex_cli", ""),
            AgentProvider.ZAI_CODING_PLAN: (False, "claude_cli", "ZAI_API_KEY is not configured"),
            AgentProvider.OPENROUTER: (False, "claude_cli", "OPENROUTER_API_KEY is not configured"),
        })

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/agent/preferences")

        assert resp.status_code == 200
        data = resp.json()
        assert data["default"]["provider"] == "claude_max"
        assert "daily_analysis" in data["effective"]
        assert "providers" in data
        readiness = {item["provider"]: item for item in data["providers"]}
        assert readiness["claude_max"]["available"] is False
        assert readiness["claude_max"]["reason"] == "Claude Max login required: run 'claude auth login'"
        assert readiness["codex_pro"]["available"] is True
        assert readiness["zai_coding_plan"]["reason"] == "ZAI_API_KEY is not configured"

    @pytest.mark.asyncio
    async def test_put_updates_state_persists_and_applies_override_precedence(self, app_factory, tmp_path):
        app = app_factory()
        await app.state.queue.initialize()
        await app.state.registry.initialize()
        _set_provider_statuses(app, {
            AgentProvider.CLAUDE_MAX: (True, "claude_cli", ""),
            AgentProvider.CODEX_PRO: (True, "codex_cli", ""),
            AgentProvider.ZAI_CODING_PLAN: (True, "claude_cli", ""),
            AgentProvider.OPENROUTER: (True, "claude_cli", ""),
        })

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put("/agent/preferences", json={
                "default": {"provider": "codex_pro", "model": None},
                "overrides": {
                    "daily_analysis": {"provider": "claude_max", "model": "opus"},
                },
            })

        assert resp.status_code == 200
        data = resp.json()
        assert app.state.agent_preferences.default.provider == AgentProvider.CODEX_PRO
        assert data["effective"]["daily_analysis"]["provider"] == "claude_max"
        assert data["effective"]["daily_analysis"]["model"] == "opus"
        assert data["effective"]["weekly_analysis"]["provider"] == "codex_pro"
        assert data["effective"]["weekly_analysis"]["model"] == "gpt-5.4"

        stored = json.loads((tmp_path / "data" / "agent_preferences.json").read_text(encoding="utf-8"))
        assert stored["default"]["provider"] == "codex_pro"
        assert stored["overrides"]["daily_analysis"]["provider"] == "claude_max"

    @pytest.mark.asyncio
    async def test_put_rejects_unavailable_provider(self, app_factory):
        app = app_factory()
        await app.state.queue.initialize()
        await app.state.registry.initialize()
        _set_provider_statuses(app, {
            AgentProvider.CLAUDE_MAX: (True, "claude_cli", ""),
            AgentProvider.CODEX_PRO: (
                False,
                "codex_cli",
                "Command preflight failed for codex: missing executable",
            ),
            AgentProvider.ZAI_CODING_PLAN: (False, "claude_cli", "ZAI_API_KEY is not configured"),
            AgentProvider.OPENROUTER: (False, "claude_cli", "OPENROUTER_API_KEY is not configured"),
        })

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.put("/agent/preferences", json={
                "default": {"provider": "codex_pro", "model": None},
                "overrides": {},
            })

        assert resp.status_code == 400
        assert "unavailable" in resp.json()["detail"]
