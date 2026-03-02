"""Bug triage schemas — error events, severity, complexity, and triage outcomes.

Used by the deterministic severity classifier and bug complexity classifier
to route errors through the triage pipeline.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BugSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}[self.value]


class BugComplexity(str, Enum):
    OBVIOUS_FIX = "obvious_fix"       # stack trace → obvious fix, dep bumps, config
    SINGLE_FUNCTION = "single_function"  # single-function logic errors
    MULTI_FILE = "multi_file"         # touches multiple files
    STATE_DEPENDENT = "state_dependent"  # timing, exchange edge cases
    UNKNOWN = "unknown"


class TriageOutcome(str, Enum):
    KNOWN_FIX = "known_fix"                   # auto-fix → draft PR
    NEEDS_INVESTIGATION = "needs_investigation"  # create GitHub issue
    NEEDS_HUMAN = "needs_human"               # Telegram alert with summary
    QUEUED_FOR_DAILY = "queued_for_daily"     # MEDIUM severity
    QUEUED_FOR_WEEKLY = "queued_for_weekly"   # LOW severity
    ALERTED = "alerted"                       # CRITICAL → immediate alert


class ErrorCategory(str, Enum):
    CRASH = "crash"
    STUCK_POSITION = "stuck_position"
    CONNECTION_LOST = "connection_lost"
    UNEXPECTED_LOSS = "unexpected_loss"
    REPEATED_ERROR = "repeated_error"
    API_ERROR = "api_error"
    CONFIG_ERROR = "config_error"
    DEPENDENCY = "dependency"
    WARNING = "warning"
    DEPRECATION = "deprecation"
    UNKNOWN = "unknown"


class ErrorEvent(BaseModel):
    """An error event received from a VPS bot sidecar."""

    bot_id: str
    error_type: str             # e.g. "RuntimeError", "ConnectionError"
    message: str                # error message text
    stack_trace: str            # full traceback
    source_file: str = ""       # file where error originated
    source_line: int = 0        # line number
    severity: Optional[BugSeverity] = None    # set by classifier
    category: Optional[ErrorCategory] = None  # set by classifier
    context: dict = {}          # arbitrary context from the bot
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriageResult(BaseModel):
    """The outcome of triaging an error event."""

    error_event: ErrorEvent
    severity: BugSeverity
    complexity: BugComplexity
    outcome: TriageOutcome
    suggested_fix: str = ""
    affected_files: list[str] = []
    github_issue_url: str = ""
    pr_url: str = ""
    past_rejections: list[str] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
