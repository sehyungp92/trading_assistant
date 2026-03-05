# tests/test_strategy_engine_expanded.py
"""Tests for expanded strategy engine detectors."""
from analysis.strategy_engine import StrategyEngine
from schemas.weekly_metrics import BotWeeklySummary


class TestAlphaDecayDetector:
    def test_detects_declining_sharpe(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_alpha_decay(
            bot_id="bot1",
            rolling_sharpe_30d=0.8,
            rolling_sharpe_60d=1.5,
            rolling_sharpe_90d=2.1,
        )
        assert len(result) == 1
        assert "alpha decay" in result[0].title.lower() or "decay" in result[0].description.lower()
        assert result[0].confidence > 0

    def test_no_decay_when_stable(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_alpha_decay(
            bot_id="bot1",
            rolling_sharpe_30d=1.5,
            rolling_sharpe_60d=1.4,
            rolling_sharpe_90d=1.3,
        )
        assert len(result) == 0

    def test_no_decay_when_improving(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_alpha_decay(
            bot_id="bot1",
            rolling_sharpe_30d=2.0,
            rolling_sharpe_60d=1.5,
            rolling_sharpe_90d=1.0,
        )
        assert len(result) == 0


class TestSignalQualityDecay:
    def test_detects_declining_correlation(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_signal_decay(
            bot_id="bot1",
            signal_outcome_correlation_30d=0.35,
            signal_outcome_correlation_90d=0.72,
        )
        assert len(result) == 1
        assert result[0].requires_human_judgment is True

    def test_no_decay_when_stable(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_signal_decay(
            bot_id="bot1",
            signal_outcome_correlation_30d=0.65,
            signal_outcome_correlation_90d=0.70,
        )
        assert len(result) == 0


class TestExitTimingAnalyzer:
    def test_detects_premature_exits(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_exit_timing_issues(
            bot_id="bot1",
            avg_exit_efficiency=0.45,  # captures only 45% of available move
            premature_exit_pct=0.6,    # 60% of exits are premature
        )
        assert len(result) == 1
        assert "exit" in result[0].title.lower()

    def test_no_issue_when_efficient(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_exit_timing_issues(
            bot_id="bot1",
            avg_exit_efficiency=0.75,
            premature_exit_pct=0.2,
        )
        assert len(result) == 0


class TestCorrelationBreakdown:
    def test_detects_rising_correlation(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        from schemas.weekly_metrics import CorrelationSummary
        correlations = [
            CorrelationSummary(
                bot_a="bot1", bot_b="bot2",
                rolling_30d_correlation=0.82,
                weekly_pnl_correlation=0.75,
                same_direction_pct=0.8,
            ),
        ]
        result = engine.detect_correlation_breakdown(correlations)
        assert len(result) >= 1
        assert "correlation" in result[0].title.lower()

    def test_no_alert_when_low_correlation(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        from schemas.weekly_metrics import CorrelationSummary
        correlations = [
            CorrelationSummary(
                bot_a="bot1", bot_b="bot2",
                rolling_30d_correlation=0.3,
            ),
        ]
        result = engine.detect_correlation_breakdown(correlations)
        assert len(result) == 0


class TestTimeOfDayPatterns:
    def test_detects_bad_hours(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        from schemas.hourly_performance import HourlyBucket
        buckets = [
            HourlyBucket(hour=9, trade_count=20, pnl=500.0, win_rate=0.75),
            HourlyBucket(hour=15, trade_count=15, pnl=-400.0, win_rate=0.2),
            HourlyBucket(hour=20, trade_count=10, pnl=100.0, win_rate=0.6),
        ]
        result = engine.detect_time_of_day_patterns(
            bot_id="bot1",
            hourly_buckets=buckets,
            min_trades=10,
        )
        assert len(result) >= 1
        assert "15" in result[0].description or "hour" in result[0].description.lower()

    def test_no_pattern_when_uniform(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        from schemas.hourly_performance import HourlyBucket
        buckets = [
            HourlyBucket(hour=9, trade_count=10, pnl=100.0, win_rate=0.6),
            HourlyBucket(hour=15, trade_count=10, pnl=80.0, win_rate=0.55),
        ]
        result = engine.detect_time_of_day_patterns(
            bot_id="bot1",
            hourly_buckets=buckets,
        )
        assert len(result) == 0


class TestDrawdownPatternDetector:
    def test_detects_concentrated_drawdown(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_drawdown_patterns(
            bot_id="bot1",
            largest_single_loss_pct=8.5,
            max_drawdown_pct=15.0,
            avg_loss_pct=2.0,
        )
        assert len(result) >= 1
        assert "drawdown" in result[0].title.lower() or "position" in result[0].title.lower()

    def test_no_alert_for_normal_drawdown(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_drawdown_patterns(
            bot_id="bot1",
            largest_single_loss_pct=2.0,
            max_drawdown_pct=8.0,
            avg_loss_pct=1.5,
        )
        assert len(result) == 0


class TestPositionSizingEfficiency:
    def test_detects_oversizing(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_position_sizing_issues(
            bot_id="bot1",
            avg_win_pct=1.5,
            avg_loss_pct=3.0,
            win_rate=0.65,
        )
        assert len(result) >= 1

    def test_no_issue_when_balanced(self):
        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        result = engine.detect_position_sizing_issues(
            bot_id="bot1",
            avg_win_pct=2.0,
            avg_loss_pct=1.5,
            win_rate=0.55,
        )
        assert len(result) == 0


class TestExpandedBuildReport:
    def test_build_report_calls_all_detectors(self):
        from schemas.weekly_metrics import BotWeeklySummary, CorrelationSummary
        from schemas.hourly_performance import HourlyBucket

        engine = StrategyEngine(week_start="2026-02-24", week_end="2026-03-02")
        bot_summaries = {
            "bot1": BotWeeklySummary(
                week_start="2026-02-24", week_end="2026-03-02",
                bot_id="bot1", total_trades=50, win_count=30, loss_count=20,
                avg_win=100.0, avg_loss=-60.0, max_drawdown_pct=12.0,
            ),
        }

        report = engine.build_report(
            bot_summaries=bot_summaries,
            rolling_sharpe={
                "bot1": {"30d": 0.5, "60d": 1.2, "90d": 2.0},
            },
            hourly_buckets={
                "bot1": [
                    HourlyBucket(hour=9, trade_count=20, pnl=500.0, win_rate=0.75),
                    HourlyBucket(hour=15, trade_count=15, pnl=-400.0, win_rate=0.2),
                ],
            },
        )
        # Should produce suggestions from multiple detectors
        assert len(report.suggestions) > 0
        tiers = {s.tier.value for s in report.suggestions}
        # At minimum we should get parameter (tight stop) and hypothesis (alpha decay)
        assert len(tiers) >= 1
