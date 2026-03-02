"""HMAC-SHA256 signature computation and verification for event payloads."""

import hashlib
import hmac as _hmac


def compute_hmac(body: str, secret: str) -> str:
    """Compute HMAC-SHA256 hex digest for a request body."""
    return _hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


def verify_hmac(body: str, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 signature. Uses constant-time comparison."""
    expected = compute_hmac(body, secret)
    return _hmac.compare_digest(expected, signature)
