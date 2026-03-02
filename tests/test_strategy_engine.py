# tests/test_strategy_engine.py
"""Tests for the 4-tier strategy refinement engine."""
from schemas.strategy_suggestions import SuggestionTier, StrategySuggestion, RefinementReport
from schemas.weekly_metrics import (
    BotWeeklySummary,
    FilterWeeklySummary,
    ProcessQualityTrend,
    RegimePerformanceTrend,
)
from analysis.strategy_engine import StrategyEngine


class TestParameterSuggestions:
    def test_detects_tight_stop_loss(self):
        """If a bot's avg_loss is small relative to avg_win (< 0.3 ratio),
        stops may be too tight, cutting winners short."""
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=20,
            loss_count=30,
            net_pnl=-100.0,
            avg_win=200.0,
            avg_loss=-30.0,  # losses are tiny → stops too tight
        )
        suggestions = engine.analyze_parameters(summary)
        assert any(
            s.tier == SuggestionTier.PARAMETER and "stop" in s.title.lower()
            for s in suggestions
        )

    def test_no_suggestion_when_balanced(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=28,
            loss_count=22,
            net_pnl=300.0,
            avg_win=80.0,
            avg_loss=-50.0,
        )
        suggestions = engine.analyze_parameters(summary)
        stop_suggestions = [s for s in suggestions if "stop" in s.title.lower()]
        assert len(stop_suggestions) == 0


class TestFilterSuggestions:
    def test_detects_costly_filter(self):
        """If a filter's net impact is negative (cost > saved), suggest relaxing."""
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        filter_summary = FilterWeeklySummary(
            bot_id="bot3",
            filter_name="volume_filter",
            total_blocks=47,
            blocks_that_would_have_won=31,
            blocks_that_would_have_lost=16,
            net_impact_pnl=-180.0,
        )
        suggestions = engine.analyze_filters("bot3", [filter_summary])
        assert len(suggestions) == 1
        assert suggestions[0].tier == SuggestionTier.FILTER
        assert "volume_filter" in suggestions[0].title

    def test_no_suggestion_for_beneficial_filter(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        filter_summary = FilterWeeklySummary(
            bot_id="bot1",
            filter_name="spread_filter",
            total_blocks=10,
            net_impact_pnl=200.0,  # filter saves more than it costs
        )
        suggestions = engine.analyze_filters("bot1", [filter_summary])
        assert len(suggestions) == 0


class TestStrategyVariantSuggestions:
    def test_detects_regime_mismatch(self):
        """If a bot loses consistently in one regime, suggest a regime gate."""
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        trend = RegimePerformanceTrend(
            bot_id="bot1",
            regime="ranging",
            weekly_pnl=[-50.0, -80.0, -40.0, -60.0],  # consistent losses
            weekly_win_rate=[0.3, 0.25, 0.35, 0.28],
            weekly_trade_count=[10, 12, 8, 11],
        )
        suggestions = engine.analyze_regime_fit("bot1", [trend])
        assert len(suggestions) >= 1
        assert suggestions[0].tier == SuggestionTier.STRATEGY_VARIANT
        assert suggestions[0].requires_human_judgment is True

    def test_no_suggestion_for_profitable_regime(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        trend = RegimePerformanceTrend(
            bot_id="bot1",
            regime="trending_up",
            weekly_pnl=[100.0, 120.0, 80.0, 150.0],
            weekly_win_rate=[0.7, 0.75, 0.65, 0.8],
            weekly_trade_count=[10, 12, 8, 11],
        )
        suggestions = engine.analyze_regime_fit("bot1", [trend])
        assert len(suggestions) == 0


class TestRefinementReport:
    def test_builds_full_report(self):
        engine = StrategyEngine(week_start="2026-02-23", week_end="2026-03-01")
        summary = BotWeeklySummary(
            week_start="2026-02-23",
            week_end="2026-03-01",
            bot_id="bot1",
            total_trades=50,
            win_count=20,
            loss_count=30,
            net_pnl=-100.0,
            avg_win=200.0,
            avg_loss=-30.0,
        )
        filter_summaries = [
            FilterWeeklySummary(
                bot_id="bot1",
                filter_name="volume_filter",
                total_blocks=20,
                net_impact_pnl=-100.0,
            )
        ]
        regime_trends = [
            RegimePerformanceTrend(
                bot_id="bot1",
                regime="ranging",
                weekly_pnl=[-50.0, -80.0, -40.0, -60.0],
                weekly_win_rate=[0.3, 0.25, 0.35, 0.28],
                weekly_trade_count=[10, 12, 8, 11],
            )
        ]
        report = engine.build_report(
            bot_summaries={"bot1": summary},
            filter_summaries={"bot1": filter_summaries},
            regime_trends={"bot1": regime_trends},
        )
        assert isinstance(report, RefinementReport)
        assert len(report.suggestions) > 0
