"""API tests for agent provider preference persistence and validation."""
from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from orchestrator.app import create_app
from orchestrator.config import AppConfig
from schemas.agent_preferences import AgentProvider, ProviderReadiness


def _set_provider_statuses(app, statuses: dict[AgentProvider, tuple[bool, str, str]]) -> None:
    app.state.agent_runner._provider_status_cache = {
        provider: ProviderReadiness(
            provider=provider,
            available=available,
            runtime=runtime,
            reason=reason,
        )
        for provider, (available, runtime, reason) in statuses.items()
    }


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
