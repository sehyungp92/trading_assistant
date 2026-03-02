from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.scheduler import SchedulerConfig, create_scheduler_jobs


class TestSchedulerConfig:
    def test_default_config(self):
        config = SchedulerConfig()
        assert config.monitoring_interval_minutes == 10
        assert config.worker_interval_seconds == 60
        assert config.relay_poll_interval_seconds == 300

    def test_custom_config(self):
        config = SchedulerConfig(
            monitoring_interval_minutes=5,
            worker_interval_seconds=30,
        )
        assert config.monitoring_interval_minutes == 5
        assert config.worker_interval_seconds == 30


class TestCreateSchedulerJobs:
    def test_creates_expected_jobs(self):
        config = SchedulerConfig()
        worker_fn = AsyncMock()
        monitoring_fn = AsyncMock()
        relay_fn = AsyncMock()

        jobs = create_scheduler_jobs(
            config=config,
            worker_fn=worker_fn,
            monitoring_fn=monitoring_fn,
            relay_fn=relay_fn,
        )

        assert len(jobs) == 3
        job_names = {j["name"] for j in jobs}
        assert "worker" in job_names
        assert "monitoring" in job_names
        assert "relay_poll" in job_names
