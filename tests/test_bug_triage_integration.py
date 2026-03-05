# tests/test_bug_triage_integration.py
"""Integration test — full bug triage pipeline end-to-end."""
import json
from pathlib import Path

from schemas.bug_triage import (
    BugSeverity,
    BugComplexity,
    ErrorCategory,
    ErrorEvent,
    TriageOutcome,
    TriageResult,
)
from schemas.pr_review import PRReviewStatus, TradingSafetyCheck
from skills.run_bug_triage import TriageRunner
from skills.failure_log import FailureLog
from skills.pr_review_checker import PRReviewChecker
from analysis.triage_prompt_assembler import TriagePromptAssembler
from analysis.pr_report_builder import PRReportBuilder


class TestFullTriagePipeline:
    """End-to-end: error event → triage → context → prompt → result."""

    def test_critical_crash_flow(self, tmp_path: Path):
        """CRITICAL crash → ALERTED, no auto-fix attempted."""
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot1", error_type="SystemExit",
            message="SIGSEGV in trade loop",
            stack_trace="Traceback:\n  File trade_loop.py:42\nSystemExit: SIGSEGV",
            context={"crash": True},
        )
        result = runner.triage(event)

        assert result.severity == BugSeverity.CRITICAL
        assert result.outcome == TriageOutcome.ALERTED
        # Verify it was logged
        log_lines = (tmp_path / "failure-log.jsonl").read_text().strip().splitlines()
        assert len(log_lines) == 1
        logged = json.loads(log_lines[0])
        assert logged["outcome"] == "alerted"

    def test_high_obvious_fix_flow(self, tmp_path: Path):
        """HIGH + OBVIOUS_FIX → KNOWN_FIX, context built for auto-fix agent."""
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
        )
        event = ErrorEvent(
            bot_id="bot2", error_type="ImportError",
            message="No module named 'ccxt'",
            stack_trace="Traceback:\n  File connector.py:1\nImportError: No module named 'ccxt'",
        )
        result = runner.triage(event, severity_override=BugSeverity.HIGH)

        assert result.severity == BugSeverity.HIGH
        assert result.outcome == TriageOutcome.KNOWN_FIX
        assert result.complexity == BugComplexity.OBVIOUS_FIX

        # Build triage context for the fix agent
        asm = TriagePromptAssembler(memory_dir=tmp_path)
        from skills.triage_context_builder import TriageContextBuilder
        ctx_builder = TriageContextBuilder(source_root=tmp_path)
        ctx = ctx_builder.build(
            event, result.severity,
            result.error_event.category or ErrorCategory.UNKNOWN,
            [],
        )
        prompt_pkg = asm.assemble(ctx, result.severity, result.complexity)
        assert "HIGH" in prompt_pkg.task_prompt
        assert "obvious_fix" in prompt_pkg.task_prompt

    def test_repeated_errors_promote_severity(self, tmp_path: Path):
        """MEDIUM errors repeated >3/hour → promoted to HIGH → NEEDS_INVESTIGATION."""
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=tmp_path / "failure-log.jsonl",
            repeated_error_threshold=3,
        )
        event = ErrorEvent(
            bot_id="bot3", error_type="RuntimeError",
            message="index out of range",
            stack_trace='Traceback:\n  File "algo.py", line 20\nRuntimeError: index out of range',
            source_file="algo.py",
            source_line=20,
        )

        # First triage: MEDIUM
        first = runner.triage(event)
        assert first.severity == BugSeverity.MEDIUM

        # Record 3 errors to trigger promotion
        for _ in range(3):
            runner.record_error(event)

        # Second triage: promoted to HIGH
        second = runner.triage(event)
        assert second.severity == BugSeverity.HIGH

    def test_ralph_loop_rejection_feedback(self, tmp_path: Path):
        """PR rejection → failure log → next triage includes past rejection."""
        log_path = tmp_path / "failure-log.jsonl"
        runner = TriageRunner(
            source_root=tmp_path,
            failure_log_path=log_path,
        )

        # First triage produces a KNOWN_FIX
        event = ErrorEvent(
            bot_id="bot1", error_type="ImportError",
            message="no module pandas",
            stack_trace="...",
        )
        first = runner.triage(event, severity_override=BugSeverity.HIGH)
        assert first.outcome == TriageOutcome.KNOWN_FIX

        # Human rejects the PR
        failure_log = FailureLog(log_path)
        failure_log.record_rejection(first, reason="should use polars, not pandas")

        # Next triage for same error type includes the rejection
        second = runner.triage(event, severity_override=BugSeverity.HIGH)
        assert any("polars" in r for r in second.past_rejections)


class TestPRReviewPipeline:
    """End-to-end: PR → review checker → report builder."""

    def test_safe_pr_passes_review(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*", "*.md"]},
                "requires_approval": {"file_paths": ["skills/*"]},
                "requires_double_approval": {"file_paths": ["orchestrator/*"]},
            }
        })
        result = checker.run_review(
            pr_url="https://github.com/user/repo/pull/42",
            changed_files=["tests/test_foo.py"],
            diff_content="+ def test_new(): assert True",
            ci_passed=True,
        )
        assert result.overall_passed is True

        # Build human-readable report
        md = PRReportBuilder.build_markdown(result)
        assert "PASSED" in md
        assert "pull/42" in md

    def test_dangerous_pr_fails_review(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*"]},
                "requires_approval": {},
                "requires_double_approval": {"file_paths": ["orchestrator/*"]},
            }
        })
        result = checker.run_review(
            pr_url="https://github.com/user/repo/pull/99",
            changed_files=["orchestrator/kill_switch.py"],
            diff_content="- kill_switch_enabled = True\n+ kill_switch_enabled = False",
            ci_passed=True,
        )
        assert result.overall_passed is False

        md = PRReportBuilder.build_markdown(result)
        assert "FAIL" in md or "BLOCKED" in md


class TestFullPipelineCount:
    """Verify the full pipeline produces the expected number of components."""

    def test_all_components_importable(self):
        """Smoke test: all Phase 5 modules can be imported."""
        from schemas import bug_triage  # noqa: F401
        from schemas import pr_review  # noqa: F401
        from skills import severity_classifier  # noqa: F401
        from skills import error_rate_tracker  # noqa: F401
        from skills import bug_complexity_classifier  # noqa: F401
        from skills import failure_log  # noqa: F401
        from skills import triage_context_builder  # noqa: F401
        from skills import pr_review_checker  # noqa: F401
        from skills import run_bug_triage  # noqa: F401
        from analysis import triage_prompt_assembler  # noqa: F401
        from analysis import pr_report_builder  # noqa: F401
