# tests/test_pr_review_schemas.py
"""Tests for PR review schemas."""
from schemas.pr_review import (
    PRReviewStep,
    PRReviewStepResult,
    TradingSafetyCheck,
    TradingSafetyResult,
    PRReviewResult,
    PRReviewStatus,
)


class TestPRReviewStep:
    def test_all_steps_exist(self):
        assert PRReviewStep.CODE_REVIEW == "code_review"
        assert PRReviewStep.CI_CHECK == "ci_check"
        assert PRReviewStep.PERMISSION_GATE == "permission_gate"
        assert PRReviewStep.TRADING_SAFETY == "trading_safety"
        assert PRReviewStep.HUMAN_REVIEW == "human_review"


class TestPRReviewStatus:
    def test_all_statuses_exist(self):
        assert PRReviewStatus.PENDING == "pending"
        assert PRReviewStatus.PASSED == "passed"
        assert PRReviewStatus.FAILED == "failed"
        assert PRReviewStatus.BLOCKED == "blocked"


class TestPRReviewStepResult:
    def test_creates_passed(self):
        r = PRReviewStepResult(
            step=PRReviewStep.CI_CHECK,
            status=PRReviewStatus.PASSED,
            detail="All checks green",
        )
        assert r.step == PRReviewStep.CI_CHECK
        assert r.status == PRReviewStatus.PASSED
        assert r.detail == "All checks green"
        assert r.blocker_reason == ""

    def test_creates_failed_with_blocker(self):
        r = PRReviewStepResult(
            step=PRReviewStep.PERMISSION_GATE,
            status=PRReviewStatus.FAILED,
            detail="File touches risk_limits.py",
            blocker_reason="requires_double_approval for risk_limits.py",
        )
        assert r.blocker_reason != ""


class TestTradingSafetyCheck:
    def test_all_checks_exist(self):
        assert TradingSafetyCheck.POSITION_SIZING_UNCHANGED == "position_sizing_unchanged"
        assert TradingSafetyCheck.RISK_LIMITS_ENFORCED == "risk_limits_enforced"
        assert TradingSafetyCheck.KILL_SWITCH_WORKS == "kill_switch_works"


class TestTradingSafetyResult:
    def test_creates_with_defaults(self):
        r = TradingSafetyResult(
            check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
            passed=True,
        )
        assert r.passed is True
        assert r.detail == ""
        assert r.is_intentional_change is False

    def test_intentional_change(self):
        r = TradingSafetyResult(
            check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
            passed=False,
            detail="Position size formula changed",
            is_intentional_change=True,
        )
        assert r.is_intentional_change is True


class TestPRReviewResult:
    def test_creates_with_all_steps(self):
        steps = [
            PRReviewStepResult(
                step=PRReviewStep.CODE_REVIEW,
                status=PRReviewStatus.PASSED,
            ),
            PRReviewStepResult(
                step=PRReviewStep.CI_CHECK,
                status=PRReviewStatus.PASSED,
            ),
        ]
        safety = [
            TradingSafetyResult(
                check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
                passed=True,
            ),
        ]
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/42",
            steps=steps,
            safety_checks=safety,
        )
        assert r.pr_url.endswith("/42")
        assert len(r.steps) == 2
        assert len(r.safety_checks) == 1

    def test_overall_passed(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/1",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
                PRReviewStepResult(step=PRReviewStep.CI_CHECK, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(check=TradingSafetyCheck.KILL_SWITCH_WORKS, passed=True),
            ],
        )
        assert r.overall_passed is True

    def test_overall_failed_when_step_fails(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/2",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
                PRReviewStepResult(step=PRReviewStep.CI_CHECK, status=PRReviewStatus.FAILED),
            ],
            safety_checks=[],
        )
        assert r.overall_passed is False

    def test_overall_failed_when_safety_fails_and_not_intentional(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/3",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(
                    check=TradingSafetyCheck.RISK_LIMITS_ENFORCED,
                    passed=False,
                    is_intentional_change=False,
                ),
            ],
        )
        assert r.overall_passed is False

    def test_overall_passed_when_safety_fails_but_intentional(self):
        r = PRReviewResult(
            pr_url="https://github.com/user/repo/pull/4",
            steps=[
                PRReviewStepResult(step=PRReviewStep.CODE_REVIEW, status=PRReviewStatus.PASSED),
            ],
            safety_checks=[
                TradingSafetyResult(
                    check=TradingSafetyCheck.POSITION_SIZING_UNCHANGED,
                    passed=False,
                    is_intentional_change=True,
                ),
            ],
        )
        assert r.overall_passed is True
