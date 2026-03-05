# tests/test_filter_sensitivity.py
"""Tests for filter sensitivity analysis."""
from datetime import datetime, timezone

from schemas.events import MissedOpportunityEvent
from schemas.filter_sensitivity import FilterSensitivityCurve, FilterSensitivityReport


def _make_missed(pair, blocked_by, outcome_24h, margin_pct=None):
    return MissedOpportunityEvent(
        bot_id="bot1", pair=pair, signal="momentum",
        blocked_by=blocked_by, outcome_24h=outcome_24h,
        confidence=0.8, assumption_tags=[],
        margin_pct=margin_pct,
    )


class TestFilterSensitivitySchema:
    def test_curve_model(self):
        curve = FilterSensitivityCurve(
            filter_name="volume_filter", bot_id="bot1",
            current_block_count=20, current_net_impact=-500.0,
        )
        assert curve.filter_name == "volume_filter"

    def test_report_model(self):
        report = FilterSensitivityReport(
            bot_id="bot1", date="2026-03-01", curves=[],
        )
        assert report.curves == []


class TestFilterSensitivityAnalyzer:
    def test_computes_net_impact_per_filter(self):
        from skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer

        missed = [
            _make_missed("BTC/USDT", "volume_filter", 500.0),
            _make_missed("ETH/USDT", "volume_filter", -200.0),
            _make_missed("BTC/USDT", "rsi_filter", 100.0),
        ]
        analyzer = FilterSensitivityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.analyze(missed)
        assert isinstance(report, FilterSensitivityReport)
        vol = next(c for c in report.curves if c.filter_name == "volume_filter")
        assert vol.current_block_count == 2
        # Net impact: removing filter would add 500-200 = 300
        assert vol.current_net_impact == 300.0

    def test_breakeven_identified(self):
        from skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer

        # Filter blocks 5 trades: 3 would have been losers, 2 winners
        missed = [
            _make_missed("BTC", "vol_filter", -100.0),
            _make_missed("ETH", "vol_filter", -150.0),
            _make_missed("SOL", "vol_filter", -80.0),
            _make_missed("BTC", "vol_filter", 200.0),
            _make_missed("ETH", "vol_filter", 300.0),
        ]
        analyzer = FilterSensitivityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.analyze(missed)
        vol = next(c for c in report.curves if c.filter_name == "vol_filter")
        # Net: -100-150-80+200+300 = 170 (filter costs money)
        assert vol.current_net_impact == 170.0
        assert vol.recommendation is not None

    def test_no_missed_produces_empty_report(self):
        from skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer

        analyzer = FilterSensitivityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.analyze([])
        assert report.curves == []

    def test_with_margin_pct_data(self):
        from skills.filter_sensitivity_analyzer import FilterSensitivityAnalyzer

        missed = [
            _make_missed("BTC", "vol_filter", 500.0, margin_pct=5.0),   # close to threshold
            _make_missed("ETH", "vol_filter", -200.0, margin_pct=25.0),  # far from threshold
            _make_missed("SOL", "vol_filter", 300.0, margin_pct=8.0),   # moderately close
        ]
        analyzer = FilterSensitivityAnalyzer(bot_id="bot1", date="2026-03-01")
        report = analyzer.analyze(missed)
        vol = next(c for c in report.curves if c.filter_name == "vol_filter")
        # Should have sensitivity points when margin data available
        assert len(vol.sensitivity_points) > 0
