# schemas/objective_weights.py
"""Shared objective weights for ground truth composite scoring.

Single source of truth for the 6-component performance weighting used by
GroundTruthComputer (absolute z-score composite) and ParameterSearcher
(relative improvement composite). Both systems MUST derive their weights
from these constants.

Weights aligned with soul.md priorities:
  expected_total_r (30%): annualized net PnL — primary optimization target
  calmar (20%): preferred risk-adjusted metric; net profit / max drawdown
  profit_factor (15%): gross_wins / gross_losses quality ratio
  expectancy (15%): win_rate × (avg_win / avg_loss) — the expectancy equation
  inverse_drawdown (10%): hard constraint emphasis
  process_quality (10%): process over outcomes; anti-gaming safeguard

Scoring contexts (intentional divergence):
  - GroundTruthComputer: Absolute evaluation via z-scores against the bot's
    own 90-day history, output normalized to [0, 1]. A score of 0.5 means
    "performing at historical average". Used for bot health monitoring and
    outcome measurement.
  - ParameterSearcher: Relative candidate comparison via improvement ratios
    against a baseline simulation. Baseline always scores 1.0; candidates
    above/below are proportionally better/worse. Used for parameter candidate
    ranking and routing (APPROVE / EXPERIMENT / DISCARD).
  - ParamOptimizer (WFO): Single-metric optimization (Sharpe, Sortino, etc.)
    configured per bot — does not use the composite weights at all.

These systems intentionally use different scales because they answer different
questions: "how healthy is this bot?" vs "is this candidate better than current?"
vs "what parameter maximizes Sharpe?". Scores from different systems MUST NOT be
compared directly. Use ratio_to_unit_scale() when a [0, 1] normalization of
ParameterSearcher output is needed for display or logging alongside ground truth.
"""
from __future__ import annotations

import math


# Full 6-component weights (sum = 1.0)
W_EXPECTED_R: float = 0.30
W_CALMAR: float = 0.20
W_PROFIT_FACTOR: float = 0.15
W_EXPECTANCY: float = 0.15
W_INV_DRAWDOWN: float = 0.10
W_PROCESS: float = 0.10

# Pre-computed 5-component weights excluding process_quality,
# renormalized to the remaining 90%. Used by ParameterSearcher where
# process quality cannot be simulated from trade data alone.
# Derivation: each weight / 0.90
_RENORM = 1.0 - W_PROCESS
W_EXPECTED_R_NO_PROCESS: float = round(W_EXPECTED_R / _RENORM, 3)  # 0.333
W_CALMAR_NO_PROCESS: float = round(W_CALMAR / _RENORM, 3)          # 0.222
W_PROFIT_FACTOR_NO_PROCESS: float = round(W_PROFIT_FACTOR / _RENORM, 3)  # 0.167
W_EXPECTANCY_NO_PROCESS: float = round(W_EXPECTANCY / _RENORM, 3)  # 0.167
W_INV_DRAWDOWN_NO_PROCESS: float = round(W_INV_DRAWDOWN / _RENORM, 3)  # 0.111


def ratio_to_unit_scale(ratio: float, sensitivity: float = 5.0) -> float:
    """Map a ratio-based composite score to [0, 1] for cross-system comparability.

    Uses a logistic function centered on 1.0 (the baseline ratio):
      - ratio = 1.0  →  0.5  (neutral, matches GroundTruthComputer midpoint)
      - ratio > 1.0  →  (0.5, 1.0)  (improvement)
      - ratio < 1.0  →  (0.0, 0.5)  (degradation)

    Args:
        ratio: Raw ratio-based composite (baseline = 1.0).
        sensitivity: Controls how quickly the output saturates toward 0 or 1.
            Default 5.0 maps ±20% improvement to roughly [0.27, 0.73].
    """
    return 1.0 / (1.0 + math.exp(-sensitivity * (ratio - 1.0)))
