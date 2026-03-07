# tests/test_approval_tracker.py
"""Tests for ApprovalTracker."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from schemas.autonomous_pipeline import ApprovalRequest, ApprovalStatus
from skills.approval_tracker import ApprovalTracker


@pytest.fixture
def tracker(tmp_path: Path) -> ApprovalTracker:
    return ApprovalTracker(tmp_path / "approvals.jsonl")


def _make_request(request_id: str = "r1", **kwargs) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=request_id,
        suggestion_id=kwargs.get("suggestion_id", "s1"),
        bot_id=kwargs.get("bot_id", "bot1"),
        **{k: v for k, v in kwargs.items() if k not in ("suggestion_id", "bot_id")},
    )


class TestApprovalTracker:
    def test_create_and_retrieve(self, tracker: ApprovalTracker):
        req = _make_request("r1")
        tracker.create_request(req)
        found = tracker.get_by_id("r1")
        assert found is not None
        assert found.request_id == "r1"

    def test_approve_transitions_status(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        approved = tracker.approve("r1")
        assert approved.status == ApprovalStatus.APPROVED
        assert approved.resolved_at is not None
        assert approved.resolved_by == "telegram"

    def test_reject_transitions_status(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        rejected = tracker.reject("r1", reason="too risky")
        assert rejected.status == ApprovalStatus.REJECTED
        assert rejected.rejection_reason == "too risky"

    def test_approve_non_pending_raises(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        tracker.approve("r1")
        with pytest.raises(ValueError, match="not PENDING"):
            tracker.approve("r1")

    def test_expire_old_requests(self, tracker: ApprovalTracker):
        old = _make_request("r_old")
        old.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        tracker.create_request(old)
        tracker.create_request(_make_request("r_new"))

        expired = tracker.expire_old(max_age_days=7)
        assert "r_old" in expired
        assert tracker.get_by_id("r_old").status == ApprovalStatus.EXPIRED
        assert tracker.get_by_id("r_new").status == ApprovalStatus.PENDING

    def test_get_pending_only_pending(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        tracker.create_request(_make_request("r2"))
        tracker.approve("r1")
        pending = tracker.get_pending()
        assert len(pending) == 1
        assert pending[0].request_id == "r2"

    def test_set_pr_url(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        tracker.set_pr_url("r1", "https://github.com/user/repo/pull/42")
        assert tracker.get_by_id("r1").pr_url == "https://github.com/user/repo/pull/42"

    def test_deduplication_by_id(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        tracker.create_request(_make_request("r1"))
        all_requests = tracker._load_all()
        assert len(all_requests) == 1

    def test_jsonl_persistence(self, tmp_path: Path):
        path = tmp_path / "approvals.jsonl"
        t1 = ApprovalTracker(path)
        t1.create_request(_make_request("r1"))
        # New tracker instance loads from same file
        t2 = ApprovalTracker(path)
        assert t2.get_by_id("r1") is not None

    def test_get_by_id_unknown(self, tracker: ApprovalTracker):
        assert tracker.get_by_id("nonexistent") is None

    def test_set_message_id(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        tracker.set_message_id("r1", 42)
        found = tracker.get_by_id("r1")
        assert found.message_id == 42

    def test_message_id_persists_across_instances(self, tmp_path: Path):
        path = tmp_path / "approvals.jsonl"
        t1 = ApprovalTracker(path)
        t1.create_request(_make_request("r1"))
        t1.set_message_id("r1", 99)
        t2 = ApprovalTracker(path)
        assert t2.get_by_id("r1").message_id == 99

    def test_message_id_default_none(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        assert tracker.get_by_id("r1").message_id is None

    def test_get_approved_with_prs_filters(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        tracker.create_request(_make_request("r2", suggestion_id="s2"))
        tracker.create_request(_make_request("r3", suggestion_id="s3"))
        tracker.approve("r1")
        tracker.approve("r2")
        tracker.set_pr_url("r1", "https://github.com/x/y/pull/1")
        # r2 approved but no pr_url, r3 still pending
        result = tracker.get_approved_with_prs()
        assert len(result) == 1
        assert result[0].request_id == "r1"

    def test_get_approved_with_prs_empty(self, tracker: ApprovalTracker):
        tracker.create_request(_make_request("r1"))
        assert tracker.get_approved_with_prs() == []

    def test_expire_returns_expired_ids(self, tracker: ApprovalTracker):
        old1 = _make_request("r1")
        old1.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        old2 = _make_request("r2")
        old2.created_at = datetime.now(timezone.utc) - timedelta(days=8)
        tracker.create_request(old1)
        tracker.create_request(old2)
        tracker.create_request(_make_request("r3"))  # recent, not expired

        expired = tracker.expire_old(max_age_days=7)
        assert set(expired) == {"r1", "r2"}
        assert tracker.get_by_id("r3").status == ApprovalStatus.PENDING
