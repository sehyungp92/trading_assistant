"""Tests for session history in context (A5)."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from analysis.context_builder import ContextBuilder
from orchestrator.session_store import SessionStore


@pytest.fixture
def memory_dir(tmp_path):
    findings = tmp_path / "findings"
    findings.mkdir()
    policies = tmp_path / "policies" / "v1"
    policies.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def session_store(tmp_path):
    return SessionStore(base_dir=str(tmp_path / "sessions"))


def test_get_recent_sessions_returns_sessions(session_store):
    # Record a session
    session_store.record_invocation(
        session_id="sess-1",
        agent_type="daily_analysis",
        prompt_package={"task": "test"},
        response="Analysis complete for all bots",
        duration_ms=5000,
    )

    result = session_store.get_recent_sessions("daily_analysis", days=7)
    assert len(result) == 1
    assert result[0]["agent_type"] == "daily_analysis"
    assert result[0]["duration_ms"] == 5000
    assert "Analysis complete" in result[0]["response_summary"]


def test_load_session_history_formats_text(memory_dir, session_store):
    session_store.record_invocation(
        session_id="sess-1",
        agent_type="daily_analysis",
        prompt_package={},
        response="Daily report generated",
        duration_ms=3000,
    )

    ctx = ContextBuilder(memory_dir)
    result = ctx.load_session_history(session_store, "daily_analysis")
    assert "Recent daily_analysis sessions" in result
    assert "3000ms" in result


def test_base_package_includes_session_history(memory_dir, session_store):
    session_store.record_invocation(
        session_id="sess-1",
        agent_type="daily_analysis",
        prompt_package={},
        response="Daily report output",
        duration_ms=2000,
    )

    ctx = ContextBuilder(memory_dir)
    pkg = ctx.base_package(session_store=session_store, agent_type="daily_analysis")
    assert "session_history" in pkg.data
    assert "daily_analysis" in pkg.data["session_history"]


def test_empty_session_history_handled_gracefully(memory_dir, session_store):
    ctx = ContextBuilder(memory_dir)
    result = ctx.load_session_history(session_store, "nonexistent_type")
    assert result == ""


def test_session_summary_truncated(memory_dir, session_store):
    long_response = "x" * 1000
    session_store.record_invocation(
        session_id="sess-1",
        agent_type="daily_analysis",
        prompt_package={},
        response=long_response,
        duration_ms=1000,
    )

    ctx = ContextBuilder(memory_dir)
    result = ctx.load_session_history(session_store, "daily_analysis")
    # response_summary is capped at 200 chars, then we take 100 in format
    assert len(result) < 500


def test_filters_by_agent_type(session_store):
    session_store.record_invocation(
        session_id="sess-1",
        agent_type="daily_analysis",
        prompt_package={},
        response="daily",
        duration_ms=1000,
    )
    session_store.record_invocation(
        session_id="sess-2",
        agent_type="weekly_analysis",
        prompt_package={},
        response="weekly",
        duration_ms=2000,
    )

    daily = session_store.get_recent_sessions("daily_analysis", days=7)
    weekly = session_store.get_recent_sessions("weekly_analysis", days=7)

    assert all(s["agent_type"] == "daily_analysis" for s in daily)
    assert all(s["agent_type"] == "weekly_analysis" for s in weekly)
