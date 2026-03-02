"""Tests for bug triage schemas."""
from datetime import datetime, timezone

from schemas.bug_triage import (
    BugSeverity,
    BugComplexity,
    TriageOutcome,
    ErrorEvent,
    TriageResult,
    ErrorCategory,
)


class TestBugSeverity:
    def test_all_levels_exist(self):
        assert BugSeverity.CRITICAL == "critical"
        assert BugSeverity.HIGH == "high"
        assert BugSeverity.MEDIUM == "medium"
        assert BugSeverity.LOW == "low"

    def test_ordering(self):
        """CRITICAL is highest severity."""
        ordered = sorted(BugSeverity, key=lambda s: s.rank, reverse=True)
        assert ordered[0] == BugSeverity.CRITICAL
        assert ordered[-1] == BugSeverity.LOW


class TestBugComplexity:
    def test_all_levels_exist(self):
        assert BugComplexity.OBVIOUS_FIX == "obvious_fix"
        assert BugComplexity.SINGLE_FUNCTION == "single_function"
        assert BugComplexity.MULTI_FILE == "multi_file"
        assert BugComplexity.STATE_DEPENDENT == "state_dependent"
        assert BugComplexity.UNKNOWN == "unknown"


class TestTriageOutcome:
    def test_all_outcomes_exist(self):
        assert TriageOutcome.KNOWN_FIX == "known_fix"
        assert TriageOutcome.NEEDS_INVESTIGATION == "needs_investigation"
        assert TriageOutcome.NEEDS_HUMAN == "needs_human"
        assert TriageOutcome.QUEUED_FOR_DAILY == "queued_for_daily"
        assert TriageOutcome.QUEUED_FOR_WEEKLY == "queued_for_weekly"
        assert TriageOutcome.ALERTED == "alerted"


class TestErrorCategory:
    def test_all_categories_exist(self):
        assert ErrorCategory.CRASH == "crash"
        assert ErrorCategory.STUCK_POSITION == "stuck_position"
        assert ErrorCategory.CONNECTION_LOST == "connection_lost"
        assert ErrorCategory.UNEXPECTED_LOSS == "unexpected_loss"
        assert ErrorCategory.REPEATED_ERROR == "repeated_error"
        assert ErrorCategory.API_ERROR == "api_error"
        assert ErrorCategory.CONFIG_ERROR == "config_error"
        assert ErrorCategory.DEPENDENCY == "dependency"
        assert ErrorCategory.WARNING == "warning"
        assert ErrorCategory.DEPRECATION == "deprecation"
        assert ErrorCategory.UNKNOWN == "unknown"


class TestErrorEvent:
    def test_creates_with_required_fields(self):
        e = ErrorEvent(
            bot_id="bot1",
            error_type="RuntimeError",
            message="division by zero",
            stack_trace="Traceback...\n  File main.py:10\nRuntimeError: division by zero",
        )
        assert e.bot_id == "bot1"
        assert e.error_type == "RuntimeError"
        assert e.message == "division by zero"
        assert e.severity is None  # not yet classified
        assert e.category is None
        assert e.source_file == ""
        assert e.source_line == 0

    def test_creates_with_all_fields(self):
        e = ErrorEvent(
            bot_id="bot2",
            error_type="ConnectionError",
            message="cannot reach exchange",
            stack_trace="...",
            source_file="connectors/binance.py",
            source_line=42,
            severity=BugSeverity.CRITICAL,
            category=ErrorCategory.CONNECTION_LOST,
            context={"exchange": "binance", "retry_count": 3},
        )
        assert e.source_file == "connectors/binance.py"
        assert e.source_line == 42
        assert e.severity == BugSeverity.CRITICAL
        assert e.context["retry_count"] == 3

    def test_timestamp_defaults_to_now(self):
        e = ErrorEvent(bot_id="b", error_type="E", message="m", stack_trace="s")
        assert e.timestamp is not None
        assert e.timestamp.tzinfo is not None


class TestTriageResult:
    def test_creates_minimal(self):
        r = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1",
                error_type="RuntimeError",
                message="fail",
                stack_trace="...",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
        )
        assert r.severity == BugSeverity.HIGH
        assert r.outcome == TriageOutcome.KNOWN_FIX
        assert r.suggested_fix == ""
        assert r.github_issue_url == ""
        assert r.pr_url == ""
        assert r.past_rejections == []

    def test_creates_with_context(self):
        r = TriageResult(
            error_event=ErrorEvent(
                bot_id="bot1",
                error_type="ImportError",
                message="no module foo",
                stack_trace="...",
            ),
            severity=BugSeverity.HIGH,
            complexity=BugComplexity.OBVIOUS_FIX,
            outcome=TriageOutcome.KNOWN_FIX,
            suggested_fix="Add foo to requirements.txt",
            affected_files=["requirements.txt"],
            past_rejections=["Previously rejected: wrong version pinned"],
        )
        assert r.suggested_fix == "Add foo to requirements.txt"
        assert len(r.affected_files) == 1
        assert len(r.past_rejections) == 1
