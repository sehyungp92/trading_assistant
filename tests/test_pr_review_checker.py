# tests/test_pr_review_checker.py
"""Tests for PR review checker — trading-specific safety assertions."""
from schemas.pr_review import (
    PRReviewStep,
    PRReviewStepResult,
    PRReviewStatus,
    TradingSafetyCheck,
    TradingSafetyResult,
    PRReviewResult,
)
from schemas.permissions import PermissionTier
from skills.pr_review_checker import PRReviewChecker


class TestPermissionGateCheck:
    def test_auto_tier_passes(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*", "*.md"]},
                "requires_approval": {"file_paths": ["skills/*"]},
                "requires_double_approval": {"file_paths": ["orchestrator/orchestrator_brain.py"]},
            }
        })
        result = checker.check_permission_gate(["tests/test_foo.py", "README.md"])
        assert result.status == PRReviewStatus.PASSED

    def test_requires_approval_blocks(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*"]},
                "requires_approval": {"file_paths": ["skills/*"]},
                "requires_double_approval": {},
            }
        })
        result = checker.check_permission_gate(["skills/severity_classifier.py"])
        assert result.status == PRReviewStatus.BLOCKED
        assert "requires_approval" in result.blocker_reason.lower()

    def test_double_approval_blocks(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {},
                "requires_approval": {},
                "requires_double_approval": {"file_paths": ["orchestrator/*"]},
            }
        })
        result = checker.check_permission_gate(["orchestrator/brain.py"])
        assert result.status == PRReviewStatus.BLOCKED
        assert "double" in result.blocker_reason.lower()


class TestTradingSafetyChecks:
    def test_position_sizing_unchanged_passes(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["skills/severity_classifier.py"],
            diff_content="+ def classify(self):\n+     return BugSeverity.HIGH",
        )
        pos_check = _find_safety(result, TradingSafetyCheck.POSITION_SIZING_UNCHANGED)
        assert pos_check.passed is True

    def test_position_sizing_flags_when_touched(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["skills/position_sizer.py"],
            diff_content="- position_size = balance * 0.01\n+ position_size = balance * 0.02",
        )
        pos_check = _find_safety(result, TradingSafetyCheck.POSITION_SIZING_UNCHANGED)
        assert pos_check.passed is False

    def test_risk_limits_unchanged_passes(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["tests/test_foo.py"],
            diff_content="+ assert result == True",
        )
        risk_check = _find_safety(result, TradingSafetyCheck.RISK_LIMITS_ENFORCED)
        assert risk_check.passed is True

    def test_risk_limits_flags_when_touched(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["config/risk_limits.yaml"],
            diff_content="- max_drawdown: 0.05\n+ max_drawdown: 0.10",
        )
        risk_check = _find_safety(result, TradingSafetyCheck.RISK_LIMITS_ENFORCED)
        assert risk_check.passed is False

    def test_kill_switch_unchanged_passes(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["tests/test_foo.py"],
            diff_content="+ pass",
        )
        ks_check = _find_safety(result, TradingSafetyCheck.KILL_SWITCH_WORKS)
        assert ks_check.passed is True

    def test_kill_switch_flags_when_touched(self):
        checker = PRReviewChecker(permission_config=_MINIMAL_CONFIG)
        result = checker.check_trading_safety(
            changed_files=["orchestrator/kill_switch.py"],
            diff_content="- if should_kill:\n-     kill()\n+ pass",
        )
        ks_check = _find_safety(result, TradingSafetyCheck.KILL_SWITCH_WORKS)
        assert ks_check.passed is False


class TestRunFullReview:
    def test_full_review_all_pass(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*"]},
                "requires_approval": {},
                "requires_double_approval": {},
            }
        })
        result = checker.run_review(
            pr_url="https://github.com/user/repo/pull/1",
            changed_files=["tests/test_foo.py"],
            diff_content="+ assert True",
            ci_passed=True,
        )
        assert isinstance(result, PRReviewResult)
        assert result.overall_passed is True

    def test_full_review_ci_fails(self):
        checker = PRReviewChecker(permission_config={
            "permission_tiers": {
                "auto": {"file_paths": ["tests/*"]},
                "requires_approval": {},
                "requires_double_approval": {},
            }
        })
        result = checker.run_review(
            pr_url="https://github.com/user/repo/pull/2",
            changed_files=["tests/test_foo.py"],
            diff_content="+ assert True",
            ci_passed=False,
        )
        assert result.overall_passed is False


# --- Helpers ---

_MINIMAL_CONFIG = {
    "permission_tiers": {
        "auto": {"file_paths": ["tests/*", "*.md"]},
        "requires_approval": {},
        "requires_double_approval": {},
    }
}


def _find_safety(
    results: list[TradingSafetyResult], check: TradingSafetyCheck,
) -> TradingSafetyResult:
    for r in results:
        if r.check == check:
            return r
    raise ValueError(f"Safety check {check} not found in results")
