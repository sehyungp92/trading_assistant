"""Tests for JSONL locking (Task 8).

These trackers use threading.Lock to protect concurrent JSONL writes.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from skills.approval_tracker import ApprovalTracker
from skills.deployment_monitor import DeploymentMonitor
from skills.experiment_manager import ExperimentManager


class TestLockExists:
    """All three components have threading.Lock instances."""

    def test_deployment_monitor_has_lock(self, tmp_path):
        monitor = DeploymentMonitor(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        assert hasattr(monitor, "_lock")
        assert isinstance(monitor._lock, type(threading.Lock()))

    def test_approval_tracker_has_lock(self, tmp_path):
        tracker = ApprovalTracker(storage_path=tmp_path / "approvals.jsonl")
        assert hasattr(tracker, "_lock")
        assert isinstance(tracker._lock, type(threading.Lock()))

    def test_experiment_manager_has_lock(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        mgr = ExperimentManager(findings_dir=findings)
        assert hasattr(mgr, "_lock")
        assert isinstance(mgr._lock, type(threading.Lock()))


class TestLockUsable:
    """Lock-protected methods work correctly under concurrent access."""

    def test_deployment_monitor_concurrent_create(self, tmp_path):
        (tmp_path / "findings").mkdir()
        (tmp_path / "curated").mkdir()
        monitor = DeploymentMonitor(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        monitor.create_deployment(
            deployment_id="dep1", approval_request_id="req1",
            pr_url="https://github.com/u/r/pull/1",
            bot_id="bot1", param_changes=[],
        )
        assert monitor.get_by_id("dep1") is not None

    def test_approval_tracker_concurrent_create(self, tmp_path):
        tracker = ApprovalTracker(storage_path=tmp_path / "approvals.jsonl")
        from schemas.autonomous_pipeline import ApprovalRequest
        req = ApprovalRequest(
            request_id="r1", suggestion_id="s1", bot_id="bot1",
            param_changes=[],
        )
        tracker.create_request(req)
        assert len(tracker.get_pending()) == 1

    def test_experiment_manager_concurrent_create(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        mgr = ExperimentManager(findings_dir=findings)
        from schemas.experiments import ExperimentConfig, ExperimentVariant
        config = ExperimentConfig(
            experiment_id="exp-1", bot_id="bot1", title="Test",
            variants=[
                ExperimentVariant(name="c", params={}, allocation_pct=50),
                ExperimentVariant(name="t", params={}, allocation_pct=50),
            ],
        )
        mgr.create_experiment(config)
        assert mgr.get_by_id("exp-1") is not None

    def test_suggestion_tracker_has_lock(self, tmp_path):
        from skills.suggestion_tracker import SuggestionTracker
        tracker = SuggestionTracker(store_dir=tmp_path / "findings")
        assert hasattr(tracker, "_lock")
        assert isinstance(tracker._lock, type(threading.Lock()))

    def test_concurrent_writes_dont_corrupt(self, tmp_path):
        """Multiple threads writing simultaneously should not corrupt data."""
        tracker = ApprovalTracker(storage_path=tmp_path / "approvals.jsonl")
        from schemas.autonomous_pipeline import ApprovalRequest
        errors = []

        def create_request(i: int) -> None:
            try:
                req = ApprovalRequest(
                    request_id=f"r{i}", suggestion_id=f"s{i}", bot_id="bot1",
                    param_changes=[],
                )
                tracker.create_request(req)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_request, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(tracker.get_pending()) == 10
