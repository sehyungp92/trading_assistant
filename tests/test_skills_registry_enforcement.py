"""Tests for SkillsRegistry enforcement in AgentRunner (A2)."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from orchestrator.agent_runner import AgentRunner
from orchestrator.app import create_app
from orchestrator.config import AppConfig
from orchestrator.session_store import SessionStore
from orchestrator.skills_registry import SkillsRegistry
from schemas.agent_capabilities import AgentType
from schemas.prompt_package import PromptPackage


@pytest.fixture
def app_with_tmp(tmp_path):
    config = AppConfig(data_dir=str(tmp_path), bot_ids=["bot1"])
    return create_app(db_dir=str(tmp_path), config=config)


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def sample_package():
    return PromptPackage(
        system_prompt="test",
        task_prompt="analyze",
        instructions="do analysis",
    )


def test_registry_instantiated_in_create_app(app_with_tmp):
    assert hasattr(app_with_tmp.state, "skills_registry")
    assert isinstance(app_with_tmp.state.skills_registry, SkillsRegistry)


def test_agent_runner_accepts_skills_registry(tmp_path, session_store):
    registry = SkillsRegistry()
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
    )
    assert runner._skills_registry is registry


@pytest.mark.asyncio
async def test_invocation_with_allowed_agent_type_succeeds(tmp_path, session_store, sample_package):
    registry = SkillsRegistry()
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
        enforce_skills=False,
        claude_path="echo",  # will fail but won't raise PermissionError
    )
    # daily_analysis has generate_report in allowed_actions, so no denial
    result = await runner.invoke(
        agent_type="daily_analysis",
        prompt_package=sample_package,
        run_id="test-run",
    )
    # Should reach invocation (may fail on CLI but not on permission)
    assert result.session_id.startswith("test-run-")


@pytest.mark.asyncio
async def test_invocation_with_forbidden_action_logs_warning(tmp_path, session_store, sample_package, caplog):
    """In soft mode (enforce=False), forbidden agent types log a warning but proceed."""
    from orchestrator.skills_registry import SkillsRegistry
    from schemas.agent_capabilities import AgentCapability, AgentType

    # Create a registry where 'daily_analysis' does NOT have 'generate_report'
    restricted = AgentCapability(
        agent_type=AgentType.DAILY_ANALYSIS,
        allowed_actions=["read_data"],
        forbidden_actions=["generate_report"],
    )
    registry = SkillsRegistry(overrides={AgentType.DAILY_ANALYSIS: restricted})
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
        enforce_skills=False,
        claude_path="echo",
    )

    import logging
    with caplog.at_level(logging.WARNING):
        result = await runner.invoke(
            agent_type="daily_analysis",
            prompt_package=sample_package,
            run_id="test-soft",
        )
    assert "SkillsRegistry denied" in caplog.text
    # But invocation still proceeds
    assert result.session_id.startswith("test-soft-")


@pytest.mark.asyncio
async def test_invocation_with_forbidden_action_raises_in_enforce_mode(tmp_path, session_store, sample_package):
    from schemas.agent_capabilities import AgentCapability, AgentType

    restricted = AgentCapability(
        agent_type=AgentType.DAILY_ANALYSIS,
        allowed_actions=["read_data"],
        forbidden_actions=["generate_report"],
    )
    registry = SkillsRegistry(overrides={AgentType.DAILY_ANALYSIS: restricted})
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
        skills_registry=registry,
        enforce_skills=True,
    )

    with pytest.raises(PermissionError, match="SkillsRegistry denied"):
        await runner.invoke(
            agent_type="daily_analysis",
            prompt_package=sample_package,
            run_id="test-enforce",
        )


def test_missing_registry_skips_enforcement(tmp_path, session_store):
    runner = AgentRunner(
        runs_dir=tmp_path / "runs",
        session_store=session_store,
    )
    assert runner._skills_registry is None


def test_all_seven_agent_types_have_profiles():
    registry = SkillsRegistry()
    for agent_type in AgentType:
        result = registry.check_action(agent_type, "generate_report")
        # Should have a profile (either allowed or denied, not "unknown agent type")
        assert "Unknown agent type" not in result.reason
