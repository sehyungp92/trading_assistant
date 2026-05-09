"""Integration tests for crypto trader support.

Tests cross-cutting integration of:
- DailyMetricsBuilder: crypto-specific curated files (funding, grade, confluence, leverage)
- DailyPromptAssembler: crypto perpetual supplement instructions
- ResponseValidator: crypto safety gates (leverage cap, funding threshold, risk_pct)
- Config: strategy profile loading and bot timezone three-field parsing
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pytest

from analysis.prompt_assembler import DailyPromptAssembler
from analysis.response_validator import ResponseValidator
from orchestrator.config import _parse_bot_timezones
from orchestrator.strategy_registry_loader import load_strategy_registry
from schemas.agent_response import AgentSuggestion, ParsedAnalysis
from schemas.events import TradeEvent
from schemas.strategy_profile import (
    StrategyArchetype,
    StrategyProfile,
    StrategyRegistry,
)
from skills.build_daily_metrics import DailyMetricsBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_crypto_trade(**overrides) -> TradeEvent:
    """Create a crypto TradeEvent with sensible defaults and optional overrides."""
    defaults = dict(
        trade_id="t001",
        bot_id="crypto_trader",
        strategy_id="MomentumPullback_M15",
        pair="BTCUSDT",
        side="LONG",
        entry_time=datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 5, 1, 14, 0, 0, tzinfo=timezone.utc),
        entry_price=50000.0,
        exit_price=50500.0,
        position_size=0.1,
        pnl=50.0,
        pnl_pct=1.0,
        exit_reason="TP1",
    )
    defaults.update(overrides)
    return TradeEvent(**defaults)


def _make_stock_trade(**overrides) -> TradeEvent:
    """Create a stock TradeEvent with no crypto-specific fields."""
    defaults = dict(
        trade_id="s001",
        bot_id="stock_trader",
        strategy_id="ORB_v1",
        pair="AAPL",
        side="LONG",
        entry_time=datetime(2026, 5, 1, 14, 30, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 5, 1, 15, 30, 0, tzinfo=timezone.utc),
        entry_price=180.0,
        exit_price=182.0,
        position_size=100.0,
        pnl=200.0,
        pnl_pct=1.11,
        exit_reason="TRAILING",
    )
    defaults.update(overrides)
    return TradeEvent(**defaults)


def _crypto_registry() -> StrategyRegistry:
    """Build a minimal StrategyRegistry containing one crypto perpetual strategy."""
    return StrategyRegistry(strategies={
        "MomentumPullback_M15": StrategyProfile(
            display_name="Momentum Pullback M15",
            bot_id="crypto_trader",
            family="crypto",
            archetype=StrategyArchetype.MOMENTUM_PULLBACK_CRYPTO,
            asset_class="crypto_perpetual",
        ),
    })


def _stock_registry() -> StrategyRegistry:
    """Build a minimal StrategyRegistry with only stock strategies."""
    return StrategyRegistry(strategies={
        "ORB_v1": StrategyProfile(
            display_name="Opening Range Breakout",
            bot_id="stock_trader",
            family="stock",
            archetype=StrategyArchetype.OPENING_RANGE_BREAKOUT,
            asset_class="us_equity",
        ),
    })


# ---------------------------------------------------------------------------
# 1. Daily Metrics — crypto curated file builders
# ---------------------------------------------------------------------------

class TestDailyMetricsFundingAnalysis:
    """TradeEvents with funding_paid produce funding_analysis.json via write_curated."""

    def test_daily_metrics_funding_analysis(self, tmp_path: Path):
        trades = [
            _make_crypto_trade(trade_id="t1", funding_paid=-15.0, pnl=500.0),
            _make_crypto_trade(trade_id="t2", funding_paid=-5.0, pnl=-100.0, side="SHORT"),
        ]
        builder = DailyMetricsBuilder(date="2026-05-01", bot_id="crypto_trader")
        builder.write_curated(trades=trades, missed=[], base_dir=tmp_path)

        funding_path = tmp_path / "2026-05-01" / "crypto_trader" / "funding_analysis.json"
        assert funding_path.exists(), "funding_analysis.json should be created for trades with funding_paid"

        data = json.loads(funding_path.read_text())
        assert data["total_funding_paid"] == pytest.approx(-20.0)
        assert "per_direction" in data
        assert "per_symbol" in data


class TestDailyMetricsGradeAnalysis:
    """TradeEvents with setup_grade produce grade_analysis.json."""

    def test_daily_metrics_grade_analysis(self, tmp_path: Path):
        trades = [
            _make_crypto_trade(trade_id="t1", setup_grade="A", pnl=500.0),
            _make_crypto_trade(trade_id="t2", setup_grade="B", pnl=-100.0),
            _make_crypto_trade(trade_id="t3", setup_grade="A", pnl=300.0),
        ]
        builder = DailyMetricsBuilder(date="2026-05-01", bot_id="crypto_trader")
        builder.write_curated(trades=trades, missed=[], base_dir=tmp_path)

        grade_path = tmp_path / "2026-05-01" / "crypto_trader" / "grade_analysis.json"
        assert grade_path.exists(), "grade_analysis.json should be created for trades with setup_grade"

        data = json.loads(grade_path.read_text())
        assert "per_grade" in data
        assert "A" in data["per_grade"]
        assert "B" in data["per_grade"]
        assert "grade_expectancy_gap" in data
        # A avg_pnl=400, B avg_pnl=-100 → gap=500
        assert data["grade_expectancy_gap"] == pytest.approx(500.0)


class TestDailyMetricsConfluenceAnalysis:
    """TradeEvents with confluences produce confluence_analysis.json."""

    def test_daily_metrics_confluence_analysis(self, tmp_path: Path):
        trades = [
            _make_crypto_trade(
                trade_id="t1", confluences=["ema_alignment", "volume_surge"], pnl=500.0,
            ),
            _make_crypto_trade(
                trade_id="t2", confluences=["ema_alignment"], pnl=-100.0,
            ),
        ]
        builder = DailyMetricsBuilder(date="2026-05-01", bot_id="crypto_trader")
        builder.write_curated(trades=trades, missed=[], base_dir=tmp_path)

        confluence_path = tmp_path / "2026-05-01" / "crypto_trader" / "confluence_analysis.json"
        assert confluence_path.exists(), "confluence_analysis.json should be created for trades with confluences"

        data = json.loads(confluence_path.read_text())
        assert "by_count" in data
        assert "by_factor" in data
        # ema_alignment appears in both trades
        assert "ema_alignment" in data["by_factor"]


class TestDailyMetricsLeverageAnalysis:
    """TradeEvents with sizing_inputs containing leverage produce leverage_analysis.json."""

    def test_daily_metrics_leverage_analysis(self, tmp_path: Path):
        trades = [
            _make_crypto_trade(trade_id="t1", sizing_inputs={"leverage": 5.0}, pnl=500.0),
            _make_crypto_trade(trade_id="t2", sizing_inputs={"leverage": 3.0}, pnl=-100.0),
        ]
        builder = DailyMetricsBuilder(date="2026-05-01", bot_id="crypto_trader")
        builder.write_curated(trades=trades, missed=[], base_dir=tmp_path)

        leverage_path = tmp_path / "2026-05-01" / "crypto_trader" / "leverage_analysis.json"
        assert leverage_path.exists(), "leverage_analysis.json should be created for trades with leverage"

        data = json.loads(leverage_path.read_text())
        assert "avg_leverage" in data
        assert data["avg_leverage"] == pytest.approx(4.0)


class TestDailyMetricsNoCryptoFields:
    """Stock TradeEvents (no crypto fields) should NOT produce crypto-specific files."""

    def test_daily_metrics_no_crypto_fields(self, tmp_path: Path):
        trades = [
            _make_stock_trade(trade_id="s1"),
            _make_stock_trade(trade_id="s2"),
        ]
        builder = DailyMetricsBuilder(date="2026-05-01", bot_id="stock_trader")
        builder.write_curated(trades=trades, missed=[], base_dir=tmp_path)

        output_dir = tmp_path / "2026-05-01" / "stock_trader"
        assert not (output_dir / "funding_analysis.json").exists(), \
            "funding_analysis.json should NOT be created for stock trades"
        assert not (output_dir / "grade_analysis.json").exists(), \
            "grade_analysis.json should NOT be created for stock trades"
        assert not (output_dir / "confluence_analysis.json").exists(), \
            "confluence_analysis.json should NOT be created for stock trades"
        assert not (output_dir / "leverage_analysis.json").exists(), \
            "leverage_analysis.json should NOT be created for stock trades"


# ---------------------------------------------------------------------------
# 2. Prompt Assembler — crypto supplement
# ---------------------------------------------------------------------------

@dataclass
class _MockTriageEvent:
    event_type: str = "anomaly"
    bot_id: str = "crypto_trader"
    severity: float = 0.8
    description: str = "High funding cost eroding edge"


@dataclass
class _MockTriageReport:
    routine_summary: str = "Crypto trader had 5 trades, 3 winners."
    significant_events: list = None
    focus_questions: list = None

    def __post_init__(self):
        if self.significant_events is None:
            self.significant_events = [_MockTriageEvent()]
        if self.focus_questions is None:
            self.focus_questions = ["Is funding cost eroding the edge?"]


class TestPromptAssemblerCryptoSupplement:
    """Crypto bot should receive supplemental CRYPTO PERPETUAL ANALYSIS instructions."""

    def test_prompt_assembler_crypto_supplement(self, tmp_path: Path):
        registry = _crypto_registry()
        assembler = DailyPromptAssembler(
            date="2026-05-01",
            bots=["crypto_trader"],
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            strategy_registry=registry,
        )
        # Crypto supplement is appended only when a triage report is provided
        triage = _MockTriageReport()
        instructions = assembler._build_instructions(triage_report=triage)
        assert "CRYPTO PERPETUAL ANALYSIS" in instructions
        assert "funding" in instructions.lower()


class TestPromptAssemblerStockNoSupplement:
    """Stock bot should NOT receive crypto-specific instructions."""

    def test_prompt_assembler_stock_no_supplement(self, tmp_path: Path):
        registry = _stock_registry()
        assembler = DailyPromptAssembler(
            date="2026-05-01",
            bots=["stock_trader"],
            curated_dir=tmp_path / "curated",
            memory_dir=tmp_path / "memory",
            strategy_registry=registry,
        )
        instructions = assembler._build_instructions(triage_report=None)
        assert "CRYPTO PERPETUAL ANALYSIS" not in instructions


# ---------------------------------------------------------------------------
# 3. Validator — crypto safety gates
# ---------------------------------------------------------------------------

class TestValidatorLeverageCapGate:
    """Leverage cap increase should be blocked without 60d evidence and 0.9 confidence."""

    def test_validator_leverage_cap_gate(self):
        registry = _crypto_registry()
        validator = ResponseValidator(strategy_registry=registry)

        suggestion = AgentSuggestion(
            suggestion_id="#lev1",
            bot_id="crypto_trader",
            category="position_sizing",
            title="Increase max leverage for majors",
            confidence=0.5,
            target_param="max_leverage_major",
            proposed_value=8.0,
            evidence_summary="Based on 30 days of data showing stable drawdown",
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])
        result = validator.validate(parsed)

        assert len(result.blocked_suggestions) >= 1
        blocked_reasons = [b.reason for b in result.blocked_suggestions]
        assert any("leverage" in r.lower() or "confidence" in r.lower() or "60" in r
                    for r in blocked_reasons), \
            f"Expected leverage gate block, got: {blocked_reasons}"


class TestValidatorFundingThresholdGate:
    """Funding_extreme loosening requires evidence of 'blocked' or 'profitable' trades."""

    def test_validator_funding_threshold_gate(self):
        registry = _crypto_registry()
        validator = ResponseValidator(strategy_registry=registry)

        suggestion = AgentSuggestion(
            suggestion_id="#fund1",
            bot_id="crypto_trader",
            category="filter_threshold",
            title="Loosen funding extreme threshold",
            confidence=0.7,
            target_param="funding_extreme",
            proposed_value=0.10,  # loosening from 0.05 to 0.10
            evidence_summary="Recent data shows moderate funding does not hurt returns",
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])

        # Gate blocks funding_extreme suggestions that lack "blocked"/"profitable"
        # evidence keywords, regardless of direction
        result = validator.validate(parsed)

        assert len(result.blocked_suggestions) >= 1
        blocked_reasons = [b.reason for b in result.blocked_suggestions]
        assert any("funding" in r.lower() or "blocked" in r.lower() or "profitable" in r.lower()
                    for r in blocked_reasons), \
            f"Expected funding gate block, got: {blocked_reasons}"


class TestValidatorCryptoRiskPctSafety:
    """Crypto risk_pct change with low confidence should be blocked."""

    def test_validator_crypto_risk_pct_safety(self):
        registry = _crypto_registry()
        validator = ResponseValidator(strategy_registry=registry)

        suggestion = AgentSuggestion(
            suggestion_id="#risk1",
            bot_id="crypto_trader",
            category="position_sizing",
            title="Increase risk_pct for BTC",
            confidence=0.5,
            target_param="risk_pct_a",
            proposed_value=0.02,
            evidence_summary="BTC has low volatility recently",
        )
        parsed = ParsedAnalysis(suggestions=[suggestion])
        result = validator.validate(parsed)

        assert len(result.blocked_suggestions) >= 1
        blocked_reasons = [b.reason for b in result.blocked_suggestions]
        assert any("risk_pct" in r.lower() or "confidence" in r.lower() or "leverage" in r.lower()
                    for r in blocked_reasons), \
            f"Expected crypto risk_pct block, got: {blocked_reasons}"


# ---------------------------------------------------------------------------
# 4. Config — strategy profiles and bot timezone parsing
# ---------------------------------------------------------------------------

class TestStrategyProfilesCryptoLoad:
    """All 3 crypto strategies should parse from strategy_profiles.yaml."""

    def test_strategy_profiles_crypto_load(self):
        profiles_path = Path(__file__).resolve().parent.parent / "data" / "strategy_profiles.yaml"
        registry = load_strategy_registry(profiles_path)

        # Verify all 3 crypto strategies exist
        assert "MomentumPullback_M15" in registry.strategies
        assert "InstitutionalAnchor_H1" in registry.strategies
        assert "VolumeProfileBreakout_M30" in registry.strategies

        # Verify MomentumPullback_M15 details
        mp = registry.strategies["MomentumPullback_M15"]
        assert mp.asset_class == "crypto_perpetual"
        assert mp.archetype == StrategyArchetype.MOMENTUM_PULLBACK_CRYPTO
        assert mp.bot_id == "crypto_trader"
        assert mp.family == "crypto"

        # Verify InstitutionalAnchor_H1 details
        ia = registry.strategies["InstitutionalAnchor_H1"]
        assert ia.asset_class == "crypto_perpetual"
        assert ia.archetype == StrategyArchetype.INSTITUTIONAL_ANCHOR
        assert ia.bot_id == "crypto_trader"

        # Verify VolumeProfileBreakout_M30 details
        vpb = registry.strategies["VolumeProfileBreakout_M30"]
        assert vpb.asset_class == "crypto_perpetual"
        assert vpb.archetype == StrategyArchetype.VOLUME_PROFILE_BREAKOUT
        assert vpb.bot_id == "crypto_trader"


class TestParseBotTimezonesThreeField:
    """_parse_bot_timezones with three-field format produces correct BotConfig."""

    def test_parse_bot_timezones_three_field(self):
        result = _parse_bot_timezones("crypto_trader:UTC:00:00", ["crypto_trader"])
        assert "crypto_trader" in result
        config = result["crypto_trader"]
        assert config.timezone == "UTC"
        assert config.market_close_local == "00:00"
        assert config.bot_id == "crypto_trader"
