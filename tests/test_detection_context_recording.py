# tests/test_detection_context_recording.py
"""Tests for detection_context recording on StrategySuggestion.

Verifies that all strategy engine detectors populate DetectionContext
with correct detector_name, threshold_name, threshold_value, observed_value,
and that margin is computed correctly.
"""
from schemas.detection_context import DetectionContext
from schemas.strategy_suggestions import SuggestionTier, StrategySuggestion
from schemas.weekly_metrics import (
    BotWeeklySummary,
    CorrelationSummary,
    FilterWeeklySummary,
    RegimePerformanceTrend,
)
from analysis.strategy_engine import StrategyEngine


def _make_engine() -> StrategyEngine:
    return StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")


class TestAnalyzeParametersDetectionContext:
    def test_tight_stop_has_detection_context(self):
        """analyze_parameters should set detection_context with detector_name='tight_stop'."""
        engine = _make_engine()
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=20,
            loss_count=30,
            net_pnl=-100.0,
            avg_win=200.0,
            avg_loss=-30.0,  # loss_win_ratio = 30/200 = 0.15
        )
        suggestions = engine.analyze_parameters(summary)
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "tight_stop"
        assert ctx.bot_id == "bot1"
        assert ctx.threshold_name == "tight_stop_ratio"
        assert ctx.threshold_value == 0.3
        assert ctx.observed_value == 30.0 / 200.0


class TestAnalyzeFiltersDetectionContext:
    def test_filter_cost_has_detection_context(self):
        """analyze_filters should set detection_context with detector_name='filter_cost'."""
        engine = _make_engine()
        fs = FilterWeeklySummary(
            bot_id="bot3",
            filter_name="volume_filter",
            total_blocks=47,
            net_impact_pnl=-180.0,
        )
        suggestions = engine.analyze_filters("bot3", [fs])
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "filter_cost"
        assert ctx.bot_id == "bot3"
        assert ctx.threshold_name == "filter_cost_threshold"
        assert ctx.threshold_value == 0.0
        assert ctx.observed_value == -180.0


class TestDetectAlphaDecayDetectionContext:
    def test_alpha_decay_has_detection_context(self):
        """detect_alpha_decay should set detection_context with detector_name='alpha_decay'."""
        engine = _make_engine()
        # 90d=1.0, 30d=0.5 → decay_ratio = (1.0-0.5)/1.0 = 0.5 > 0.3 threshold
        suggestions = engine.detect_alpha_decay(
            bot_id="bot1",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0,
            decay_threshold=0.3,
        )
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "alpha_decay"
        assert ctx.threshold_name == "decay_threshold"
        assert ctx.threshold_value == 0.3
        assert ctx.observed_value == 0.5  # decay_ratio


class TestDetectExitTimingDetectionContext:
    def test_exit_timing_has_detection_context(self):
        """detect_exit_timing_issues should set detection_context with detector_name='exit_timing'."""
        engine = _make_engine()
        suggestions = engine.detect_exit_timing_issues(
            bot_id="bot1",
            avg_exit_efficiency=0.3,
            premature_exit_pct=0.6,
            efficiency_threshold=0.5,
        )
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "exit_timing"
        assert ctx.bot_id == "bot1"
        assert ctx.threshold_name == "efficiency_threshold"
        assert ctx.threshold_value == 0.5
        assert ctx.observed_value == 0.3


class TestNoDetectionContextWhenNoSuggestion:
    def test_no_suggestion_means_no_context(self):
        """When thresholds are not exceeded, no suggestions or contexts are produced."""
        engine = _make_engine()
        # Balanced stop → no suggestion
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=28,
            loss_count=22,
            net_pnl=300.0,
            avg_win=80.0,
            avg_loss=-50.0,  # ratio = 50/80 = 0.625 > 0.3
        )
        assert len(engine.analyze_parameters(summary)) == 0

        # Beneficial filter → no suggestion
        fs = FilterWeeklySummary(
            bot_id="bot1",
            filter_name="spread_filter",
            total_blocks=10,
            net_impact_pnl=200.0,
        )
        assert len(engine.analyze_filters("bot1", [fs])) == 0

        # No alpha decay
        assert len(engine.detect_alpha_decay("bot1", 0.9, 0.8, 1.0, decay_threshold=0.3)) == 0


class TestAllDetectorsProduceDetectionContext:
    def test_signal_decay_has_context(self):
        engine = _make_engine()
        suggestions = engine.detect_signal_decay(
            bot_id="bot1",
            signal_outcome_correlation_30d=0.2,
            signal_outcome_correlation_90d=0.6,
            decay_threshold=0.2,
        )
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "signal_decay"
        assert abs(ctx.observed_value - 0.4) < 1e-9  # drop = 0.6 - 0.2

    def test_correlation_breakdown_has_context(self):
        engine = _make_engine()
        corr = CorrelationSummary(
            bot_a="botA",
            bot_b="botB",
            rolling_30d_correlation=0.85,
            same_direction_pct=0.7,
        )
        suggestions = engine.detect_correlation_breakdown([corr], threshold=0.7)
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "correlation"
        assert ctx.bot_id == "botA+botB"
        assert ctx.threshold_value == 0.7
        assert ctx.observed_value == 0.85

    def test_regime_loss_has_context(self):
        engine = StrategyEngine(
            week_start="2026-02-23", week_end="2026-03-01",
            regime_min_weeks=3,
        )
        trend = RegimePerformanceTrend(
            bot_id="bot1",
            regime="bear",
            weekly_pnl=[-100, -200, -50],
        )
        suggestions = engine.analyze_regime_fit("bot1", [trend])
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "regime_loss"
        assert ctx.threshold_value == 3.0
        assert ctx.observed_value == 3.0  # all 3 weeks losing

    def test_drawdown_concentration_has_context(self):
        engine = _make_engine()
        suggestions = engine.detect_drawdown_patterns(
            bot_id="bot1",
            largest_single_loss_pct=9.0,
            max_drawdown_pct=12.0,
            avg_loss_pct=2.0,
            concentration_threshold=3.0,
        )
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "drawdown_concentration"
        assert ctx.observed_value == 4.5  # 9.0 / 2.0

    def test_position_sizing_has_context(self):
        engine = _make_engine()
        suggestions = engine.detect_position_sizing_issues(
            bot_id="bot1",
            avg_win_pct=1.0,
            avg_loss_pct=2.0,
            win_rate=0.6,
            loss_win_ratio_threshold=1.5,
        )
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "position_sizing"
        assert ctx.observed_value == 2.0  # 2.0 / 1.0

    def test_component_signal_decay_has_context(self):
        engine = _make_engine()
        signal_health = {
            "components": [
                {"component_name": "rsi", "stability": 0.1, "win_correlation": 0.01, "trade_count": 20},
            ]
        }
        suggestions = engine.detect_component_signal_decay(
            bot_id="bot1",
            signal_health_data=signal_health,
            stability_threshold=0.3,
        )
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "component_signal_decay"
        assert ctx.observed_value == 0.1  # min stability of degraded components

    def test_time_of_day_has_context(self):
        engine = _make_engine()

        class HourlyBucket:
            def __init__(self, hour, trade_count, pnl, win_rate):
                self.hour = hour
                self.trade_count = trade_count
                self.pnl = pnl
                self.win_rate = win_rate

        bucket = HourlyBucket(hour=3, trade_count=15, pnl=-200, win_rate=0.2)
        suggestions = engine.detect_time_of_day_patterns(
            bot_id="bot1",
            hourly_buckets=[bucket],
            loss_threshold=0.35,
        )
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "time_of_day"
        assert ctx.threshold_value == 0.35
        assert ctx.observed_value == 0.2

    def test_filter_interactions_redundant_has_context(self):
        engine = _make_engine()
        interactions = [
            {
                "filter_a": "fA",
                "filter_b": "fB",
                "interaction_type": "redundant",
                "redundancy_score": 0.8,
                "recommendation": "Remove fB",
            }
        ]
        suggestions = engine.detect_filter_interactions("bot1", interactions)
        # Should have at least the redundant suggestion
        redundant = [s for s in suggestions if "Redundant" in s.title]
        assert len(redundant) == 1
        ctx = redundant[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "filter_interactions"
        assert ctx.observed_value == 0.8

    def test_factor_decay_has_context(self):
        engine = _make_engine()
        factor_data = [
            {
                "factor_name": "momentum",
                "win_rate_trend": "degrading",
                "below_threshold": True,
                "rolling_30d_win_rate": 0.35,
                "days_of_data": 30,
            }
        ]
        suggestions = engine.detect_factor_correlation_decay("bot1", factor_data)
        assert len(suggestions) == 1
        ctx = suggestions[0].detection_context
        assert ctx is not None
        assert ctx.detector_name == "factor_decay"
        assert ctx.observed_value == 0.35


class TestDetectionContextMargin:
    def test_margin_equals_abs_difference(self):
        """Verify margin = abs(observed - threshold) in the detection context."""
        engine = _make_engine()
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=20,
            loss_count=30,
            net_pnl=-100.0,
            avg_win=200.0,
            avg_loss=-30.0,  # ratio = 0.15
        )
        suggestions = engine.analyze_parameters(summary)
        ctx = suggestions[0].detection_context
        assert ctx is not None
        expected_margin = abs(0.15 - 0.3)
        assert abs(ctx.margin - expected_margin) < 1e-9


class TestBackwardCompatDetectionContextOptional:
    def test_detection_context_defaults_to_none(self):
        """Creating a StrategySuggestion without detection_context defaults to None."""
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER,
            title="Test",
            description="Test description",
        )
        assert s.detection_context is None

    def test_serialization_round_trip(self):
        """StrategySuggestion with detection_context survives JSON round-trip."""
        ctx = DetectionContext(
            detector_name="test",
            bot_id="bot1",
            threshold_name="t",
            threshold_value=0.5,
            observed_value=0.3,
        )
        s = StrategySuggestion(
            tier=SuggestionTier.PARAMETER,
            title="Test",
            description="desc",
            detection_context=ctx,
        )
        data = s.model_dump()
        restored = StrategySuggestion.model_validate(data)
        assert restored.detection_context is not None
        assert restored.detection_context.detector_name == "test"
        assert restored.detection_context.margin == abs(0.3 - 0.5)
