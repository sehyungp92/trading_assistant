# skills/_outcome_utils.py
"""Shared outcome evaluation utilities.

Centralises is_positive / is_conclusive logic so that learning_ledger,
suggestion_scorer, and convergence_tracker stay consistent.
"""
from __future__ import annotations

_INCONCLUSIVE_VERDICTS = frozenset({"inconclusive", "insufficient_data"})


def is_positive_outcome(outcome: dict) -> bool:
    """Check if an outcome is positive, preferring verdict field."""
    verdict = outcome.get("verdict", "")
    if verdict:
        return verdict == "positive"
    # Legacy fallback: check pnl_delta or pnl_delta_7d
    delta = outcome.get("pnl_delta")
    if delta is None:
        delta = outcome.get("pnl_delta_7d", 0)
    try:
        return float(delta) > 0
    except (TypeError, ValueError):
        return False


def is_conclusive_outcome(outcome: dict) -> bool:
    """Returns False for inconclusive/insufficient_data verdicts.

    These outcomes should be excluded from both numerator AND denominator
    when computing win rates, since the learning system added these
    verdicts specifically for ambiguous outcomes.
    """
    verdict = (outcome.get("verdict") or "").lower()
    return verdict not in _INCONCLUSIVE_VERDICTS
