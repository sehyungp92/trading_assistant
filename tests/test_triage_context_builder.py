# tests/test_triage_context_builder.py
"""Tests for triage context builder — assembles stack trace + source + git log."""
import json
from pathlib import Path

from schemas.bug_triage import ErrorEvent, BugSeverity, ErrorCategory
from skills.failure_log import FailureEntry, FailureLog
from skills.triage_context_builder import TriageContextBuilder, TriageContext


class TestTriageContext:
    def test_has_required_fields(self):
        ctx = TriageContext(
            error_event_summary="RuntimeError: division by zero in calc.py:10",
            stack_trace="Traceback...",
            source_snippet="",
            recent_git_log="",
            past_rejections=[],
        )
        assert ctx.error_event_summary != ""
        assert ctx.stack_trace != ""


class TestTriageContextBuilder:
    def test_builds_context_with_source_file(self, tmp_path: Path):
        # Create a fake source file
        src = tmp_path / "calc.py"
        src.write_text("def divide(x, y):\n    return x / y\n")

        e = ErrorEvent(
            bot_id="bot1", error_type="RuntimeError",
            message="division by zero",
            stack_trace='Traceback:\n  File "calc.py", line 2\n    return x / y\nRuntimeError: division by zero',
            source_file="calc.py",
            source_line=2,
        )

        builder = TriageContextBuilder(source_root=tmp_path)
        ctx = builder.build(e, BugSeverity.HIGH, ErrorCategory.UNKNOWN, [])

        assert "RuntimeError" in ctx.error_event_summary
        assert "division by zero" in ctx.error_event_summary
        assert "return x / y" in ctx.source_snippet

    def test_builds_context_without_source_file(self, tmp_path: Path):
        e = ErrorEvent(
            bot_id="bot1", error_type="ConnectionError",
            message="connection lost",
            stack_trace="...",
        )

        builder = TriageContextBuilder(source_root=tmp_path)
        ctx = builder.build(e, BugSeverity.CRITICAL, ErrorCategory.CONNECTION_LOST, [])

        assert ctx.source_snippet == ""
        assert "ConnectionError" in ctx.error_event_summary

    def test_includes_past_rejections(self, tmp_path: Path):
        e = ErrorEvent(
            bot_id="bot1", error_type="ImportError",
            message="no module foo",
            stack_trace="...",
        )

        past = [
            FailureEntry(
                bot_id="bot1", error_type="ImportError", message="no module foo",
                outcome="known_fix", rejection_reason="wrong version",
            ),
        ]

        builder = TriageContextBuilder(source_root=tmp_path)
        ctx = builder.build(e, BugSeverity.HIGH, ErrorCategory.DEPENDENCY, past)

        assert len(ctx.past_rejections) == 1
        assert "wrong version" in ctx.past_rejections[0]

    def test_source_snippet_includes_surrounding_lines(self, tmp_path: Path):
        src = tmp_path / "module.py"
        lines = [f"line {i}\n" for i in range(1, 21)]
        src.write_text("".join(lines))

        e = ErrorEvent(
            bot_id="bot1", error_type="E", message="m",
            stack_trace="...", source_file="module.py", source_line=10,
        )

        builder = TriageContextBuilder(source_root=tmp_path, context_lines=3)
        ctx = builder.build(e, BugSeverity.HIGH, ErrorCategory.UNKNOWN, [])

        # Should include lines 7-13 (3 before, target, 3 after)
        assert "line 7" in ctx.source_snippet
        assert "line 10" in ctx.source_snippet
        assert "line 13" in ctx.source_snippet
