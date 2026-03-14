"""Integration tests for agent preferences: seeding, loading, saving, env config."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.app import (
    _load_agent_preferences,
    _save_agent_preferences,
    _seed_agent_preferences,
    _selection_from_env,
)
from orchestrator.config import AppConfig
from schemas.agent_preferences import (
    AgentPreferences,
    AgentProvider,
    AgentSelection,
    AgentWorkflow,
)


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
