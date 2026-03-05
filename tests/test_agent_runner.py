"""Tests for orchestrator/agent_runner.py — Claude CLI invocation bridge."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent_runner import AgentRunner, AgentResult
from orchestrator.session_store import SessionStore
from schemas.prompt_package import PromptPackage


@pytest.fixture
def tmp_runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def runner(tmp_runs: Path, session_store: SessionStore) -> AgentRunner:
    return AgentRunner(
        runs_dir=tmp_runs,
        session_store=session_store,
        claude_path="/usr/bin/claude",
        default_model="sonnet",
        default_max_turns=5,
        timeout_seconds=30,
    )


@pytest.fixture
def sample_package() -> PromptPackage:
    return PromptPackage(
        system_prompt="You are a trading analyst.",
        task_prompt="Analyze today's performance.",
        data={"summary": {"pnl": 100}, "risk": {"drawdown": 0.02}},
        instructions="Step 1: Look at PnL.\nStep 2: Check risk.",
    )


def _mock_process(stdout: str = "", stderr: str = "", returncode: int = 0):
    """Create a mock asyncio.Process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(
        stdout.encode("utf-8"),
        stderr.encode("utf-8"),
    ))
    proc.returncode = returncode
    return proc


class TestBuildCommand:
    def test_builds_correct_cli_command(self, runner: AgentRunner, sample_package: PromptPackage):
        cmd = runner._build_command(
            prompt_package=sample_package,
            model="sonnet",
            max_turns=5,
            allowed_tools=None,
        )
        assert cmd[0] == "/usr/bin/claude"
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "Analyze today's performance."
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert "--no-session-persistence" in cmd
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "sonnet"
        assert "--max-turns" in cmd
        assert cmd[cmd.index("--max-turns") + 1] == "5"

    def test_includes_system_prompt_flag(self, runner: AgentRunner, sample_package: PromptPackage):
        cmd = runner._build_command(
            prompt_package=sample_package,
            model="sonnet",
            max_turns=5,
            allowed_tools=None,
        )
        assert "--system-prompt" in cmd
        assert cmd[cmd.index("--system-prompt") + 1] == "You are a trading analyst."

    def test_omits_system_prompt_when_empty(self, runner: AgentRunner):
        pkg = PromptPackage(task_prompt="Do something.")
        cmd = runner._build_command(pkg, model="sonnet", max_turns=3, allowed_tools=None)
        assert "--system-prompt" not in cmd

    def test_includes_allowed_tools(self, runner: AgentRunner, sample_package: PromptPackage):
        cmd = runner._build_command(
            prompt_package=sample_package,
            model="opus",
            max_turns=3,
            allowed_tools=["Read", "Bash", "Grep"],
        )
        assert "--allowed-tools" in cmd
        assert cmd[cmd.index("--allowed-tools") + 1] == "Read,Bash,Grep"


class TestCreateRunDirectory:
    @pytest.mark.asyncio
    async def test_creates_run_directory_and_data_files(
        self, runner: AgentRunner, sample_package: PromptPackage
    ):
        json_output = json.dumps({"result": "Analysis complete.", "cost_usd": 0.05, "duration_ms": 1200})
        mock_proc = _mock_process(stdout=json_output)

        with patch("orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-2026-03-02",
            )

        run_dir = runner._runs_dir / "daily-2026-03-02"
        assert run_dir.exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "risk.json").exists()
        assert (run_dir / "instructions.md").exists()
        assert (run_dir / "system_prompt.md").exists()

        # Verify data file content
        summary_data = json.loads((run_dir / "summary.json").read_text())
        assert summary_data == {"pnl": 100}

    @pytest.mark.asyncio
    async def test_writes_response_file(
        self, runner: AgentRunner, sample_package: PromptPackage
    ):
        json_output = json.dumps({"result": "The analysis shows profit.", "cost_usd": 0.01})
        mock_proc = _mock_process(stdout=json_output)

        with patch("orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-2026-03-02",
            )

        response_path = result.run_dir / "response.md"
        assert response_path.exists()
        assert response_path.read_text() == "The analysis shows profit."


class TestParseOutput:
    @pytest.mark.asyncio
    async def test_parses_json_output(
        self, runner: AgentRunner, sample_package: PromptPackage
    ):
        json_output = json.dumps({
            "result": "Market was bullish today.",
            "cost_usd": 0.123,
            "duration_ms": 5000,
        })
        mock_proc = _mock_process(stdout=json_output)

        with patch("orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-test",
            )

        assert result.success is True
        assert result.response == "Market was bullish today."
        assert result.cost_usd == 0.123

    @pytest.mark.asyncio
    async def test_parses_text_output_fallback(
        self, runner: AgentRunner, sample_package: PromptPackage
    ):
        mock_proc = _mock_process(stdout="Plain text response from Claude.")

        with patch("orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-text",
            )

        assert result.success is True
        assert result.response == "Plain text response from Claude."
        assert result.cost_usd == 0.0


class TestSessionRecording:
    @pytest.mark.asyncio
    async def test_records_session(
        self, runner: AgentRunner, session_store: SessionStore, sample_package: PromptPackage
    ):
        json_output = json.dumps({"result": "Done.", "cost_usd": 0.01})
        mock_proc = _mock_process(stdout=json_output)

        with patch("orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-session-test",
            )

        # Verify session was recorded
        from datetime import datetime, timezone
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sessions = session_store.list_sessions(agent_type="daily_analysis", date=date_str)
        assert len(sessions) >= 1
        assert sessions[0]["agent_type"] == "daily_analysis"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_handles_nonzero_exit(
        self, runner: AgentRunner, sample_package: PromptPackage
    ):
        mock_proc = _mock_process(stdout="", stderr="Error: model unavailable", returncode=1)

        with patch("orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-error",
            )

        assert result.success is False
        assert "Exit code 1" in result.error
        assert "model unavailable" in result.error

    @pytest.mark.asyncio
    async def test_handles_timeout(
        self, runner: AgentRunner, sample_package: PromptPackage
    ):
        async def slow_communicate():
            await asyncio.sleep(100)
            return (b"", b"")

        mock_proc = AsyncMock()
        mock_proc.communicate = slow_communicate
        mock_proc.returncode = 0

        with patch("orchestrator.agent_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner.invoke(
                agent_type="daily_analysis",
                prompt_package=sample_package,
                run_id="daily-timeout",
            )

        assert result.success is False
        assert "Timeout" in result.error
