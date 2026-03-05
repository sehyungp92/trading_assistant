from __future__ import annotations
import time
from collections import defaultdict

class SlidingWindowRateLimiter:
    """Per-key sliding window rate limiter."""

    def __init__(self, max_requests_per_minute: int = 60, max_requests_per_hour: int = 1000) -> None:
        self._max_per_minute = max_requests_per_minute
        self._max_per_hour = max_requests_per_hour
        self._minute_windows: dict[str, list[float]] = defaultdict(list)
        self._hour_windows: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if a request from `key` is allowed. Records the request if allowed."""
        now = time.monotonic()

        # Clean old entries
        minute_cutoff = now - 60
        hour_cutoff = now - 3600
        self._minute_windows[key] = [t for t in self._minute_windows[key] if t > minute_cutoff]
        self._hour_windows[key] = [t for t in self._hour_windows[key] if t > hour_cutoff]

        # Check limits
        if len(self._minute_windows[key]) >= self._max_per_minute:
            return False
        if len(self._hour_windows[key]) >= self._max_per_hour:
            return False

        # Record request
        self._minute_windows[key].append(now)
        self._hour_windows[key].append(now)
        return True

    def get_usage(self, key: str) -> dict:
        """Get current usage for a key."""
        now = time.monotonic()
        minute_cutoff = now - 60
        hour_cutoff = now - 3600
        minute_count = len([t for t in self._minute_windows.get(key, []) if t > minute_cutoff])
        hour_count = len([t for t in self._hour_windows.get(key, []) if t > hour_cutoff])
        return {
            "minute": {"used": minute_count, "limit": self._max_per_minute},
            "hour": {"used": hour_count, "limit": self._max_per_hour},
        }
