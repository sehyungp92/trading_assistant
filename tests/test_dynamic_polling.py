# tests/test_dynamic_polling.py
"""Tests for adaptive VPS receiver polling."""
import pytest
from unittest.mock import AsyncMock

from orchestrator.adapters.vps_receiver import VPSReceiver


class TestAdaptivePolling:
    def test_initial_interval_is_default(self):
        queue = AsyncMock()
        receiver = VPSReceiver(
            relay_url="http://relay:8080",
            local_queue=queue,
            min_poll_seconds=10,
            max_poll_seconds=300,
        )
        assert receiver.current_poll_interval == 300  # starts at max (quiet)

    def test_interval_shrinks_after_events_received(self):
        queue = AsyncMock()
        receiver = VPSReceiver(
            relay_url="http://relay:8080",
            local_queue=queue,
            min_poll_seconds=10,
            max_poll_seconds=300,
        )
        receiver.adapt_interval(events_received=5)
        assert receiver.current_poll_interval == 10  # shrink to min

    def test_interval_grows_when_no_events(self):
        queue = AsyncMock()
        receiver = VPSReceiver(
            relay_url="http://relay:8080",
            local_queue=queue,
            min_poll_seconds=10,
            max_poll_seconds=300,
        )
        receiver.adapt_interval(events_received=5)  # shrink first
        assert receiver.current_poll_interval == 10
        receiver.adapt_interval(events_received=0)  # no events
        assert receiver.current_poll_interval > 10  # should grow

    def test_interval_capped_at_max(self):
        queue = AsyncMock()
        receiver = VPSReceiver(
            relay_url="http://relay:8080",
            local_queue=queue,
            min_poll_seconds=10,
            max_poll_seconds=60,
        )
        # Many empty polls
        for _ in range(20):
            receiver.adapt_interval(events_received=0)
        assert receiver.current_poll_interval <= 60
