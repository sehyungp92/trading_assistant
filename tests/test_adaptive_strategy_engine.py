"""Tests for adaptive strategy engine integration with ThresholdLearner."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from analysis.strategy_engine import StrategyEngine
from schemas.detection_context import DetectionContext, ThresholdProfile, ThresholdRecord
from schemas.weekly_metrics import BotWeeklySummary, FilterWeeklySummary
from skills.threshold_learner import ThresholdLearner


def _make_bot_summary(
    bot_id: str = "bot_a",
    avg_win: float = 100.0,
    avg_loss: float = -20.0,
    win_rate: float = 0.55,
    total_trades: int = 50,
) -> BotWeeklySummary:
    win_count = int(total_trades * win_rate)
    loss_count = total_trades - win_count
    return BotWeeklySummary(
        week_start="2026-01-01",
        week_end="2026-01-07",
        bot_id=bot_id,
        total_trades=total_trades,
        win_count=win_count,
        loss_count=loss_count,
        avg_win=avg_win,
        avg_loss=avg_loss,
        net_pnl=(avg_win * win_rate * total_trades) + (avg_loss * (1 - win_rate) * total_trades),
    )


def _make_filter_summary(
    filter_name: str = "volatility_filter",
    net_impact_pnl: float = -50.0,
    total_blocks: int = 10,
    confidence: float = 0.7,
) -> FilterWeeklySummary:
    return FilterWeeklySummary(
        bot_id="bot_a",
        filter_name=filter_name,
        total_blocks=total_blocks,
        blocks_that_would_have_won=3,
        blocks_that_would_have_lost=7,
        net_impact_pnl=net_impact_pnl,
        confidence=confidence,
    )


class TestWithoutLearner:
    """Without ThresholdLearner — backward compatibility."""

    def test_no_learner_uses_default_tight_stop(self):
        """Without learner, detectors use default thresholds."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        # loss/win ratio = 20/100 = 0.2, below default 0.3 -> should trigger
        summary = _make_bot_summary(avg_win=100.0, avg_loss=-20.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.3  # default

    def test_no_learner_uses_default_alpha_decay(self):
        """Without learner, alpha_decay uses its parameter default."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0,
            decay_threshold=0.3,
        )
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.3

    def test_no_learner_uses_default_filter_cost(self):
        """Without learner, filter cost uses its default threshold."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        f = _make_filter_summary(net_impact_pnl=-50.0)
        result = engine.analyze_filters("bot_a", [f])
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.0  # default


class TestWithLearner:
    """With ThresholdLearner providing learned thresholds."""

    def _make_learner(self, thresholds: dict[str, float]) -> MagicMock:
        """Create a mock ThresholdLearner with given threshold responses."""
        learner = MagicMock(spec=ThresholdLearner)

        def get_threshold(detector_name, threshold_name, bot_id, default):
            key = f"{detector_name}:{threshold_name}"
            return thresholds.get(key, default)

        learner.get_threshold.side_effect = get_threshold
        return learner

    def test_learned_alpha_decay_threshold_used(self):
        """With learner providing a learned threshold for alpha_decay, uses learned value."""
        learner = self._make_learner({"alpha_decay:decay_threshold": 0.15})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # decay_ratio = (1.0 - 0.8) / 1.0 = 0.2
        # default 0.3 would NOT trigger (0.2 < 0.3), but learned 0.15 WILL trigger (0.2 >= 0.15)
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.8,
            rolling_sharpe_60d=0.9,
            rolling_sharpe_90d=1.0,
            decay_threshold=0.3,
        )
        assert len(result) == 1
        assert result[0].detection_context.threshold_value == 0.15

    def test_learned_filter_cost_per_bot(self):
        """With learner providing learned threshold for filter_cost, uses per-bot value."""
        learner = self._make_learner({"filter_cost:filter_cost_threshold": -100.0})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # Net impact -50, default threshold 0.0 -> triggers (-50 < 0.0)
        # But learned threshold -100.0 -> does NOT trigger (-50 is NOT < -100.0)
        f = _make_filter_summary(net_impact_pnl=-50.0)
        result = engine.analyze_filters("bot_a", [f])
        assert len(result) == 0  # filter NOT triggered because learned threshold is lower

    def test_learned_threshold_more_sensitive_more_suggestions(self):
        """Learned threshold more sensitive (lower) produces more suggestions."""
        learner = self._make_learner({"alpha_decay:decay_threshold": 0.05})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # decay_ratio = (1.0 - 0.9) / 1.0 = 0.1
        # default 0.3 would NOT trigger, but learned 0.05 triggers
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.9,
            rolling_sharpe_60d=0.95,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 1

    def test_learned_threshold_more_conservative_fewer_suggestions(self):
        """Learned threshold more conservative (higher) produces fewer suggestions."""
        learner = self._make_learner({"alpha_decay:decay_threshold": 0.9})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # decay_ratio = (1.0 - 0.5) / 1.0 = 0.5
        # default 0.3 WOULD trigger, but learned 0.9 does NOT (0.5 < 0.9)
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.7,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 0

    def test_build_report_passes_learner_through(self):
        """build_report passes threshold_learner to underlying detectors."""
        learner = self._make_learner({"tight_stop:tight_stop_ratio": 0.5})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # loss/win ratio = 40/100 = 0.4
        # default 0.3 would trigger (0.4 >= 0.3 is False => 0.4 < 0.3 is False, so no trigger)
        # Wait: condition is loss_win_ratio < threshold. 0.4 < 0.3 is False. So default does NOT trigger.
        # With learned 0.5: 0.4 < 0.5 is True, so it WILL trigger.
        summary = _make_bot_summary(avg_win=100.0, avg_loss=-40.0)
        report = engine.build_report({"bot_a": summary})
        assert len(report.suggestions) == 1
        assert report.suggestions[0].detection_context.threshold_value == 0.5

    def test_detection_context_records_effective_threshold(self):
        """detection_context records the effective (learned) threshold, not the default."""
        learner = self._make_learner({"alpha_decay:decay_threshold": 0.1})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.7,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 1
        ctx = result[0].detection_context
        assert ctx is not None
        assert ctx.threshold_value == 0.1  # learned, NOT the default 0.3
        assert ctx.detector_name == "alpha_decay"
        assert ctx.threshold_name == "decay_threshold"

    def test_mixed_some_learned_some_default(self):
        """Mixed: some detectors have learned thresholds, others use defaults."""
        # Only alpha_decay has a learned threshold
        learner = self._make_learner({"alpha_decay:decay_threshold": 0.1})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )

        # Alpha decay — uses learned 0.1
        alpha_result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.8,
            rolling_sharpe_60d=0.9,
            rolling_sharpe_90d=1.0,  # decay_ratio = 0.2 >= 0.1 -> triggers
        )
        assert len(alpha_result) == 1
        assert alpha_result[0].detection_context.threshold_value == 0.1

        # Signal decay — no learned value, uses default 0.2
        signal_result = engine.detect_signal_decay(
            "bot_a",
            signal_outcome_correlation_30d=0.3,
            signal_outcome_correlation_90d=0.6,  # drop = 0.3 >= 0.2 -> triggers
        )
        assert len(signal_result) == 1
        assert signal_result[0].detection_context.threshold_value == 0.2  # default

    def test_feature_flag_off_no_learner(self):
        """Feature flag off: StrategyEngine constructed without learner."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        assert engine._threshold_learner is None

        # Should still work with default thresholds
        summary = _make_bot_summary(avg_win=100.0, avg_loss=-20.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 1


@pytest.mark.parametrize("detector_name,method_name,kwargs,threshold_key", [
    (
        "alpha_decay", "detect_alpha_decay",
        {"bot_id": "bot_a", "rolling_sharpe_30d": 0.5, "rolling_sharpe_60d": 0.7, "rolling_sharpe_90d": 1.0},
        "alpha_decay:decay_threshold",
    ),
    (
        "signal_decay", "detect_signal_decay",
        {"bot_id": "bot_a", "signal_outcome_correlation_30d": 0.3, "signal_outcome_correlation_90d": 0.6},
        "signal_decay:decay_threshold",
    ),
    (
        "drawdown_concentration", "detect_drawdown_patterns",
        {"bot_id": "bot_a", "largest_single_loss_pct": 10.0, "max_drawdown_pct": 15.0, "avg_loss_pct": 2.0},
        "drawdown_concentration:concentration_threshold",
    ),
    (
        "position_sizing", "detect_position_sizing_issues",
        {"bot_id": "bot_a", "avg_win_pct": 1.0, "avg_loss_pct": 2.0, "win_rate": 0.6},
        "position_sizing:loss_win_ratio_threshold",
    ),
])
def test_parametrized_detectors_use_learned_threshold(
    detector_name, method_name, kwargs, threshold_key,
):
    """Parametrized test: multiple detectors use _get_threshold correctly."""
    learner = MagicMock(spec=ThresholdLearner)
    learner.get_threshold.return_value = 999.0  # very high threshold

    engine = StrategyEngine(
        week_start="2026-01-01", week_end="2026-01-07",
        threshold_learner=learner,
    )

    method = getattr(engine, method_name)
    result = method(**kwargs)

    # The learner should have been called
    learner.get_threshold.assert_called()
    # At least one call should have used the detector_name
    calls = learner.get_threshold.call_args_list
    detector_calls = [c for c in calls if c[0][0] == detector_name]
    assert len(detector_calls) > 0, f"Expected _get_threshold to be called for {detector_name}"
