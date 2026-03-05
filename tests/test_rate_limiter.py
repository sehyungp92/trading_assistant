"""Tests for relay rate limiting (M6)."""
from __future__ import annotations
import pytest
from relay.rate_limiter import SlidingWindowRateLimiter

class TestSlidingWindowRateLimiter:
    def test_allows_within_limit(self):
        limiter = SlidingWindowRateLimiter(max_requests_per_minute=5, max_requests_per_hour=100)
        for i in range(5):
            assert limiter.is_allowed("bot-a") is True

    def test_rejects_over_minute_limit(self):
        limiter = SlidingWindowRateLimiter(max_requests_per_minute=3, max_requests_per_hour=100)
        for i in range(3):
            limiter.is_allowed("bot-a")
        assert limiter.is_allowed("bot-a") is False

    def test_per_key_isolation(self):
        limiter = SlidingWindowRateLimiter(max_requests_per_minute=2, max_requests_per_hour=100)
        limiter.is_allowed("bot-a")
        limiter.is_allowed("bot-a")
        assert limiter.is_allowed("bot-a") is False
        assert limiter.is_allowed("bot-b") is True  # Different key

    def test_get_usage(self):
        limiter = SlidingWindowRateLimiter(max_requests_per_minute=10, max_requests_per_hour=100)
        for _ in range(5):
            limiter.is_allowed("bot-a")
        usage = limiter.get_usage("bot-a")
        assert usage["minute"]["used"] == 5
        assert usage["minute"]["limit"] == 10
        assert usage["hour"]["used"] == 5

    def test_empty_usage(self):
        limiter = SlidingWindowRateLimiter()
        usage = limiter.get_usage("unknown")
        assert usage["minute"]["used"] == 0
        assert usage["hour"]["used"] == 0
