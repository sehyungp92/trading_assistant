# skills/github_pr.py
"""PR builder — creates pull requests against bot repositories using git CLI.

Rules:
- Never force pushes
- Never auto-merges
- Single-concern commits
- Branch naming: ta/suggestion-{id[:8]}-{YYYY-MM-DD}
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from schemas.autonomous_pipeline import (
    BacktestComparison,
    PreflightResult,
    PRRequest,
    PRResult,
    PRReviewStatus,
    ReviewState,
)

logger = logging.getLogger(__name__)


class PRBuilder:
    """Creates PRs against bot repositories using git CLI."""

    def __init__(self, dry_run: bool = False) -> None:
        self._dry_run = dry_run

    async def preflight_check(self, repo_dir: Path) -> PreflightResult:
        """Run pre-flight checks before PR creation.

        Checks:
        1. Clean working tree (warning, non-blocking)
        2. Remote reachable (critical, blocking)
        """
        checks: list[dict] = []
        reasons: list[str] = []
        passed = True

        # Check 1 — Clean tree (warning only)
        code, stdout, _ = await self._run_git(["status", "--porcelain"], repo_dir)
        clean = code == 0 and not stdout.strip()
        checks.append({
            "name": "clean_tree",
            "passed": clean,
            "detail": "clean" if clean else "uncommitted changes present",
        })
        if not clean:
            logger.warning("Pre-flight: working tree is dirty in %s", repo_dir)

        # Check 2 — Remote reachable (critical)
        code, _, stderr = await self._run_git(
            ["ls-remote", "--exit-code", "origin", "HEAD"], repo_dir,
        )
        remote_ok = code == 0
        checks.append({
            "name": "remote_reachable",
            "passed": remote_ok,
            "detail": "ok" if remote_ok else f"unreachable: {stderr.strip()}",
        })
        if not remote_ok:
            passed = False
            reasons.append(f"Remote unreachable: {stderr.strip()}")

        return PreflightResult(passed=passed, checks=checks, reasons=reasons)

    async def check_existing_pr(
        self, branch_name: str, repo_dir: Path,
    ) -> tuple[bool, str | None]:
        """Check if a branch/PR already exists on remote.

        Returns (exists, pr_url_or_None).
        """
        # Check if branch exists on remote
        code, stdout, _ = await self._run_git(
            ["ls-remote", "--heads", "origin", branch_name], repo_dir,
        )
        if code != 0 or not stdout.strip():
            return False, None

        # Branch exists — check for open PR
        code, stdout, _ = await self._run_cmd(
            ["gh", "pr", "list", "--head", branch_name,
             "--state", "open", "--json", "number,url"],
            repo_dir,
        )
        if code == 0 and stdout.strip():
            try:
                import json
                prs = json.loads(stdout)
                if prs:
                    return True, prs[0].get("url", "")
            except Exception:
                pass
        # Branch exists but no open PR
        return True, None

    async def create_pr(self, request: PRRequest) -> PRResult:
        """Create a PR with the given file changes.

        Steps:
        0a. Pre-flight checks
        0b. Check for existing branch/PR (dedup)
        1. git pull to ensure up-to-date
        2. Create branch
        3. Apply file changes
        4. Commit
        5. Push branch
        6. gh pr create
        """
        repo_dir = Path(request.repo_dir)

        if self._dry_run:
            return PRResult(
                success=True,
                pr_url=None,
                branch_name=request.branch_name,
                error="dry-run: skipped git commands",
            )

        # 0a. Pre-flight checks
        preflight = await self.preflight_check(repo_dir)
        if not preflight.passed:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"Pre-flight failed: {'; '.join(preflight.reasons)}",
                preflight=preflight,
            )

        # 0b. Check for existing branch/PR
        exists, existing_url = await self.check_existing_pr(
            request.branch_name, repo_dir,
        )
        if exists and existing_url:
            return PRResult(
                success=True,
                branch_name=request.branch_name,
                existing_pr_url=existing_url,
            )
        if exists and not existing_url:
            return PRResult(
                success=False,
                branch_name=request.branch_name,
                error=f"Branch {request.branch_name} exists on remote but has no open PR (stale state)",
            )

        # 1. Pull latest
        code, _, stderr = await self._run_git(["pull", "--ff-only"], repo_dir)
        if code != 0:
            return PRResult(
                success=False, branch_name=request.branch_name,
                error=f"git pull failed: {stderr}",
            )

        # 2. Create and checkout branch
        code, _, stderr = await self._run_git(
            ["checkout", "-b", request.branch_name], repo_dir,
        )
        if code != 0:
            return PRResult(
                success=False, branch_name=request.branch_name,
                error=f"Branch creation failed: {stderr}",
            )

        # 3. Apply file changes
        for fc in request.file_changes:
            file_path = repo_dir / fc.file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(fc.new_content, encoding="utf-8")

        # 4. Stage and commit
        code, _, stderr = await self._run_git(["add", "."], repo_dir)
        if code != 0:
            return PRResult(
                success=False, branch_name=request.branch_name,
                error=f"git add failed: {stderr}",
            )

        commit_msg = f"trading-assistant: {request.title} (#{request.suggestion_id})"
        code, _, stderr = await self._run_git(
            ["commit", "-m", commit_msg], repo_dir,
        )
        if code != 0:
            return PRResult(
                success=False, branch_name=request.branch_name,
                error=f"git commit failed: {stderr}",
            )

        # 5. Push
        code, _, stderr = await self._run_git(
            ["push", "-u", "origin", request.branch_name], repo_dir,
        )
        if code != 0:
            return PRResult(
                success=False, branch_name=request.branch_name,
                error=f"git push failed: {stderr}",
            )

        # 6. Create PR
        code, stdout, stderr = await self._run_cmd(
            ["gh", "pr", "create",
             "--title", f"[trading-assistant] {request.title}",
             "--body", request.body],
            repo_dir,
        )
        if code != 0:
            return PRResult(
                success=False, branch_name=request.branch_name,
                error=f"gh pr create failed: {stderr}",
            )

        pr_url = stdout.strip()
        return PRResult(
            success=True,
            pr_url=pr_url,
            branch_name=request.branch_name,
        )

    async def _run_git(self, args: list[str], cwd: Path) -> tuple[int, str, str]:
        """Run a git command via asyncio subprocess."""
        return await self._run_cmd(["git"] + args, cwd)

    async def run_gh_command(
        self, args: list[str], cwd: Path, timeout_seconds: int = 60,
    ) -> tuple[int, str, str]:
        """Public API for running gh/git commands with timeout."""
        return await self._run_cmd(args, cwd, timeout_seconds=timeout_seconds)

    async def _run_cmd(
        self, args: list[str], cwd: Path, timeout_seconds: int = 60,
    ) -> tuple[int, str, str]:
        """Run a command via asyncio subprocess with timeout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 1, "", f"Command timed out after {timeout_seconds}s"
            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except Exception as e:
            return 1, "", str(e)

    async def check_pr_reviews(
        self, pr_url: str, repo_dir: Path,
    ) -> PRReviewStatus | None:
        """Check review status of a PR.

        Returns None if the PR number can't be extracted or gh fails.
        """
        pr_number = self._extract_pr_number(pr_url)
        if pr_number is None:
            return None

        code, stdout, _ = await self._run_cmd(
            ["gh", "pr", "view", str(pr_number),
             "--json", "reviews,comments,state"],
            repo_dir,
        )
        if code != 0:
            return None

        try:
            import json
            data = json.loads(stdout)
        except Exception:
            return None

        reviews = data.get("reviews", []) or []
        comments = data.get("comments", []) or []

        # Determine overall state from reviews
        states_seen: set[str] = set()
        reviewers: list[str] = []
        for review in reviews:
            state = review.get("state", "")
            states_seen.add(state)
            author = review.get("author", {}).get("login", "")
            if author and author not in reviewers:
                reviewers.append(author)

        # Priority: CHANGES_REQUESTED > APPROVED > COMMENTED > PENDING
        if "CHANGES_REQUESTED" in states_seen:
            review_state = ReviewState.CHANGES_REQUESTED
        elif "APPROVED" in states_seen:
            review_state = ReviewState.APPROVED
        elif "COMMENTED" in states_seen:
            review_state = ReviewState.COMMENTED
        else:
            review_state = ReviewState.PENDING

        # Collect actionable comments (cap at 5)
        actionable: list[str] = []
        for comment in comments[:5]:
            body = comment.get("body", "").strip()
            if body:
                actionable.append(body[:200])

        needs_attention = (
            review_state == ReviewState.CHANGES_REQUESTED
            or len(actionable) > 0
        )

        return PRReviewStatus(
            pr_number=pr_number,
            pr_url=pr_url,
            review_state=review_state,
            reviewers=reviewers,
            actionable_comments=actionable,
            needs_attention=needs_attention,
        )

    @staticmethod
    def _extract_pr_number(pr_url: str) -> int | None:
        """Extract PR number from a GitHub PR URL."""
        import re
        match = re.search(r"/pull/(\d+)", pr_url)
        if match:
            return int(match.group(1))
        return None

    def format_pr_body(
        self, request: PRRequest, comparison: BacktestComparison | None = None,
    ) -> str:
        """Format PR body with param table, backtest results, rollback instructions."""
        lines: list[str] = []
        lines.append("## Parameter Changes")
        lines.append("")
        lines.append("| Parameter | Current | Proposed |")
        lines.append("|-----------|---------|----------|")
        for fc_data in (request.file_changes if not comparison else []):
            lines.append(f"| {fc_data.file_path} | - | - |")

        if comparison:
            ctx = comparison.context
            lines.append(
                f"| {ctx.param_name} | {ctx.current_value} | {ctx.proposed_value} |"
            )
            lines.append("")
            lines.append("## Backtest Results")
            lines.append("")
            lines.append("| Metric | Baseline | Proposed | Change |")
            lines.append("|--------|----------|----------|--------|")
            lines.append(
                f"| Sharpe | {comparison.baseline.sharpe_ratio:.2f} "
                f"| {comparison.proposed.sharpe_ratio:.2f} "
                f"| {comparison.sharpe_change_pct:+.1f}% |"
            )
            lines.append(
                f"| MaxDD | {comparison.baseline.max_drawdown_pct:.1f}% "
                f"| {comparison.proposed.max_drawdown_pct:.1f}% "
                f"| {comparison.max_dd_change_pct:+.1f}% |"
            )
            lines.append(
                f"| ProfitFactor | {comparison.baseline.profit_factor:.2f} "
                f"| {comparison.proposed.profit_factor:.2f} "
                f"| {comparison.profit_factor_change_pct:+.1f}% |"
            )
            lines.append(
                f"| WinRate | {comparison.baseline.win_rate:.1%} "
                f"| {comparison.proposed.win_rate:.1%} "
                f"| {comparison.win_rate_change_pct:+.1f}% |"
            )
            lines.append("")
            safety = "PASS" if comparison.passes_safety else "FAIL"
            lines.append(f"**Safety Check:** {safety}")
            if comparison.safety_notes:
                for note in comparison.safety_notes:
                    lines.append(f"- {note}")

        lines.append("")
        lines.append("## Rollback")
        lines.append("")
        lines.append("```bash")
        lines.append("git revert <commit-sha>")
        lines.append("```")
        if request.file_changes:
            lines.append("")
            lines.append("## File Changes")
            lines.append("")
            for fc in request.file_changes:
                lines.append(f"### `{fc.file_path}`")
                if fc.diff_preview:
                    lines.append("")
                    lines.append("```diff")
                    lines.append(fc.diff_preview)
                    lines.append("```")
                lines.append("")

        lines.append(f"Suggestion ID: `{request.suggestion_id}`")

        return "\n".join(lines)
