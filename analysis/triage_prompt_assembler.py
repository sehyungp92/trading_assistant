# analysis/triage_prompt_assembler.py
"""Triage prompt assembler — builds context package for the Claude triage agent.

Follows the same pattern as DailyPromptAssembler:
  SYSTEM PROMPT: policies
  TASK PROMPT: triage instructions with severity/complexity
  CONTEXT: stack trace + source + git log + past rejections
"""
from __future__ import annotations

from pathlib import Path

from schemas.bug_triage import BugComplexity, BugSeverity
from skills.triage_context_builder import TriageContext

_TRIAGE_INSTRUCTIONS = """\
1. Analyze the error event summary and severity classification
2. Read the stack trace to identify the root cause
3. If source code is provided, review the code around the error
4. Check past rejections — do NOT repeat previously rejected fixes
5. Based on the complexity assessment:
   - OBVIOUS_FIX: Propose a specific fix with exact file paths and code changes
   - SINGLE_FUNCTION: Identify the root cause, suggest investigation steps
   - MULTI_FILE / STATE_DEPENDENT: Summarize findings, recommend human intervention
6. Output: triage_result.json with outcome, affected_files, and suggested_fix"""


class TriagePromptAssembler:
    """Assembles the full context package for a bug triage agent invocation."""

    def __init__(self, memory_dir: Path) -> None:
        self._memory_dir = memory_dir

    def assemble(
        self,
        context: TriageContext,
        severity: BugSeverity,
        complexity: BugComplexity,
    ) -> dict:
        """Build the complete prompt package."""
        system_prompt = self._build_system_prompt()
        task_prompt = self._build_task_prompt(severity, complexity)
        context_text = self._build_context(context)

        return {
            "system_prompt": system_prompt,
            "task_prompt": task_prompt,
            "context": context_text,
            "instructions": _TRIAGE_INSTRUCTIONS,
        }

    def _build_system_prompt(self) -> str:
        parts: list[str] = []
        policy_dir = self._memory_dir / "policies" / "v1"

        for name in ["agents.md", "trading_rules.md", "soul.md"]:
            path = policy_dir / name
            if path.exists():
                parts.append(f"--- {name} ---\n{path.read_text()}")

        return "\n\n".join(parts)

    def _build_task_prompt(
        self, severity: BugSeverity, complexity: BugComplexity,
    ) -> str:
        return (
            f"Triage this error event.\n"
            f"Severity: {severity.value.upper()}\n"
            f"Complexity: {complexity.value}\n"
            f"Follow the instructions to produce a triage result."
        )

    def _build_context(self, context: TriageContext) -> str:
        sections: list[str] = []

        sections.append(f"## Error Summary\n{context.error_event_summary}")
        sections.append(f"## Stack Trace\n```\n{context.stack_trace}\n```")

        if context.source_snippet:
            sections.append(f"## Source Code\n```python\n{context.source_snippet}\n```")

        if context.recent_git_log:
            sections.append(f"## Recent Git Log\n```\n{context.recent_git_log}\n```")

        if context.past_rejections:
            rejections = "\n".join(f"- {r}" for r in context.past_rejections)
            sections.append(f"## Past Rejections\n{rejections}")

        return "\n\n".join(sections)
