"""Tests for strategy engine thresholds — default and learned via ThresholdLearner."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from analysis.strategy_engine import StrategyEngine
from schemas.weekly_metrics import BotWeeklySummary, FilterWeeklySummary


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


class TestDefaultThresholds:
    """StrategyEngine uses hardcoded default thresholds."""

    def test_default_tight_stop(self):
        """Detectors use default thresholds."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        # loss/win ratio = 20/100 = 0.2, below default 0.3 -> should trigger
        summary = _make_bot_summary(avg_win=100.0, avg_loss=-20.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.3  # default

    def test_default_alpha_decay(self):
        """Alpha_decay uses its parameter default."""
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

    def test_default_filter_cost(self):
        """Filter cost uses its default threshold."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        f = _make_filter_summary(net_impact_pnl=-50.0)
        result = engine.analyze_filters("bot_a", [f])
        assert len(result) == 1
        assert result[0].detection_context is not None
        assert result[0].detection_context.threshold_value == 0.0  # default

    def test_threshold_learner_attribute_defaults_to_none(self):
        """StrategyEngine has _threshold_learner attribute, defaulting to None."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        assert hasattr(engine, "_threshold_learner")
        assert engine._threshold_learner is None

    def test_detection_context_records_default_threshold(self):
        """detection_context records the default threshold value."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.7,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 1
        ctx = result[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "alpha_decay"
        assert ctx.threshold_name == "decay_threshold"

    def test_mixed_detectors_all_use_defaults(self):
        """Multiple detectors all use their respective default thresholds."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")

        # Alpha decay — uses default
        alpha_result = engine.detect_alpha_decay(
            "bot_a",
            rolling_sharpe_30d=0.8,
            rolling_sharpe_60d=0.9,
            rolling_sharpe_90d=1.0,  # decay_ratio = 0.2
        )

        # Signal decay — uses default 0.2
        signal_result = engine.detect_signal_decay(
            "bot_a",
            signal_outcome_correlation_30d=0.3,
            signal_outcome_correlation_90d=0.6,  # drop = 0.3 >= 0.2 -> triggers
        )
        assert len(signal_result) == 1
        assert signal_result[0].detection_context.threshold_value == 0.2  # default

    def test_build_report_uses_defaults(self):
        """build_report uses default thresholds."""
        engine = StrategyEngine(week_start="2026-01-01", week_end="2026-01-07")
        # loss/win ratio = 20/100 = 0.2, below default 0.3 -> triggers
        summary = _make_bot_summary(avg_win=100.0, avg_loss=-20.0)
        report = engine.build_report({"bot_a": summary})
        assert len(report.suggestions) == 1
        assert report.suggestions[0].detection_context.threshold_value == 0.3  # default


class TestWithThresholdLearner:
    """StrategyEngine uses learned thresholds from ThresholdLearner."""

    def _make_learner(self, overrides: dict[str, float] | None = None) -> MagicMock:
        """Create a mock ThresholdLearner that returns overrides or falls through to default."""
        learner = MagicMock()
        _overrides = overrides or {}

        def _get(detector_name, threshold_name, bot_id, default):
            key = f"{detector_name}:{threshold_name}:{bot_id}"
            return _overrides.get(key, default)

        learner.get_threshold.side_effect = _get
        return learner

    def test_with_threshold_learner_uses_learned_values(self):
        """Engine uses learned threshold when learner provides one."""
        # Learned tight_stop_ratio = 0.15 (lower than default 0.3)
        learner = self._make_learner({"tight_stop:tight_stop_ratio:bot_a": 0.15})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # loss/win ratio = 20/100 = 0.2 — above learned 0.15, so should NOT trigger
        summary = _make_bot_summary(bot_id="bot_a", avg_win=100.0, avg_loss=-20.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 0  # 0.2 >= 0.15, no suggestion

    def test_learned_threshold_triggers_detection(self):
        """Learned threshold can make a detector MORE sensitive."""
        # Learned tight_stop_ratio = 0.5 (higher than default 0.3)
        learner = self._make_learner({"tight_stop:tight_stop_ratio:bot_a": 0.5})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # loss/win ratio = 40/100 = 0.4 — below learned 0.5, triggers
        summary = _make_bot_summary(bot_id="bot_a", avg_win=100.0, avg_loss=-40.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 1
        assert result[0].detection_context.threshold_value == 0.5  # learned

    def test_without_learner_backward_compat(self):
        """None learner returns defaults — identical to old behavior."""
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=None,
        )
        # loss/win ratio = 20/100 = 0.2, below default 0.3 -> triggers
        summary = _make_bot_summary(bot_id="bot_a", avg_win=100.0, avg_loss=-20.0)
        result = engine.analyze_parameters(summary)
        assert len(result) == 1
        assert result[0].detection_context.threshold_value == 0.3  # default

    def test_learner_affects_alpha_decay(self):
        """Learner overrides alpha_decay threshold."""
        learner = self._make_learner({"alpha_decay:decay_threshold:bot_a": 0.6})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        # decay_ratio = (1.0 - 0.5)/1.0 = 0.5, below learned 0.6 -> no trigger
        result = engine.detect_alpha_decay(
            "bot_a", rolling_sharpe_30d=0.5, rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 0

    def test_learner_affects_filter_cost(self):
        """Learner overrides filter_cost_threshold."""
        learner = self._make_learner({"filter_cost:filter_cost_threshold:bot_a": -100.0})
        engine = StrategyEngine(
            week_start="2026-01-01", week_end="2026-01-07",
            threshold_learner=learner,
        )
        f = _make_filter_summary(net_impact_pnl=-50.0)
        # -50 > -100 (learned threshold), so no trigger
        result = engine.analyze_filters("bot_a", [f])
        assert len(result) == 0
