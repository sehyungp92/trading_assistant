"""Tests for async JSONL locking (Task 8)."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from skills.approval_tracker import ApprovalTracker
from skills.deployment_monitor import DeploymentMonitor
from skills.experiment_manager import ExperimentManager


class TestLockExists:
    """All three components have asyncio.Lock instances."""

    def test_deployment_monitor_has_lock(self, tmp_path):
        monitor = DeploymentMonitor(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        assert hasattr(monitor, "_lock")
        assert isinstance(monitor._lock, asyncio.Lock)

    def test_approval_tracker_has_lock(self, tmp_path):
        tracker = ApprovalTracker(storage_path=tmp_path / "approvals.jsonl")
        assert hasattr(tracker, "_lock")
        assert isinstance(tracker._lock, asyncio.Lock)

    def test_experiment_manager_has_lock(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        mgr = ExperimentManager(findings_dir=findings)
        assert hasattr(mgr, "_lock")
        assert isinstance(mgr._lock, asyncio.Lock)


class TestLockUsable:
    """Locks can be acquired and released properly."""

    async def test_deployment_monitor_lock_acquirable(self, tmp_path):
        (tmp_path / "findings").mkdir()
        (tmp_path / "curated").mkdir()
        monitor = DeploymentMonitor(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        async with monitor._lock:
            monitor.create_deployment(
                deployment_id="dep1", approval_request_id="req1",
                pr_url="https://github.com/u/r/pull/1",
                bot_id="bot1", param_changes=[],
            )
        assert monitor.get_by_id("dep1") is not None

    async def test_approval_tracker_lock_acquirable(self, tmp_path):
        tracker = ApprovalTracker(storage_path=tmp_path / "approvals.jsonl")
        from schemas.autonomous_pipeline import ApprovalRequest
        req = ApprovalRequest(
            request_id="r1", suggestion_id="s1", bot_id="bot1",
            param_changes=[],
        )
        async with tracker._lock:
            tracker.create_request(req)
        assert len(tracker.get_pending()) == 1

    async def test_experiment_manager_lock_acquirable(self, tmp_path):
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
        async with mgr._lock:
            mgr.create_experiment(config)
        assert mgr.get_by_id("exp-1") is not None
