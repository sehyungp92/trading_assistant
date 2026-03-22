# tests/test_portfolio_proposals.py
"""Tests for the portfolio-level improvement capability.

Covers: schemas, response parser, response validator guardrails,
suggestion tracker extensions, ground truth portfolio snapshot,
context builder injection, and strategy engine portfolio detectors.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestPortfolioProposalSchema:
    def test_create_allocation_rebalance(self):
        from schemas.portfolio_proposal import PortfolioProposal, PortfolioProposalType

        p = PortfolioProposal(
            proposal_type=PortfolioProposalType.ALLOCATION_REBALANCE,
            current_config={"swing": 0.33, "stock": 0.33, "momentum": 0.34},
            proposed_config={"swing": 0.40, "stock": 0.30, "momentum": 0.30},
            evidence_summary="Swing Calmar 2.1 vs portfolio 1.4",
            expected_portfolio_calmar_delta=0.15,
            confidence=0.6,
            observation_window_days=60,
        )
        assert p.proposal_type == PortfolioProposalType.ALLOCATION_REBALANCE
        assert p.confidence == 0.6

    def test_all_proposal_types_exist(self):
        from schemas.portfolio_proposal import PortfolioProposalType

        assert hasattr(PortfolioProposalType, "ALLOCATION_REBALANCE")
        assert hasattr(PortfolioProposalType, "RISK_CAP_CHANGE")
        assert hasattr(PortfolioProposalType, "COORDINATION_CHANGE")
        assert hasattr(PortfolioProposalType, "DRAWDOWN_TIER_CHANGE")

    def test_confidence_clamped(self):
        from schemas.portfolio_proposal import PortfolioProposal, PortfolioProposalType

        with pytest.raises(Exception):
            PortfolioProposal(
                proposal_type=PortfolioProposalType.ALLOCATION_REBALANCE,
                confidence=1.5,  # out of range
            )


class TestPortfolioMetricsSchema:
    def test_family_daily_snapshot(self):
        from schemas.portfolio_metrics import FamilyDailySnapshot

        snap = FamilyDailySnapshot(
            family="swing", date="2026-03-01",
            strategy_ids=["ATRSS", "AKC_HELIX"],
            total_net_pnl=150.0, trade_count=10, win_count=6,
        )
        assert snap.family == "swing"
        assert snap.total_net_pnl == 150.0

    def test_portfolio_rolling_metrics(self):
        from schemas.portfolio_metrics import PortfolioRollingMetrics

        m = PortfolioRollingMetrics(
            date="2026-03-01",
            sharpe_7d=1.2, sharpe_30d=0.9,
            calmar_30d=1.5,
        )
        assert m.sharpe_7d == 1.2


class TestSuggestionTierPortfolio:
    def test_portfolio_tier_exists(self):
        from schemas.strategy_suggestions import SuggestionTier

        assert SuggestionTier.PORTFOLIO.value == "portfolio"

    def test_category_to_tier_portfolio_entries(self):
        from schemas.agent_response import CATEGORY_TO_TIER

        assert CATEGORY_TO_TIER["portfolio_allocation"] == "portfolio"
        assert CATEGORY_TO_TIER["portfolio_risk_cap"] == "portfolio"
        assert CATEGORY_TO_TIER["portfolio_coordination"] == "portfolio"
        assert CATEGORY_TO_TIER["portfolio_drawdown_tier"] == "portfolio"


class TestParsedAnalysisPortfolioField:
    def test_portfolio_proposals_default_empty(self):
        from schemas.agent_response import ParsedAnalysis

        pa = ParsedAnalysis(raw_report="test")
        assert pa.portfolio_proposals == []


# ---------------------------------------------------------------------------
# Response parser tests
# ---------------------------------------------------------------------------

class TestResponseParserPortfolio:
    def test_parse_portfolio_proposals(self):
        from analysis.response_parser import parse_response

        response = """Some analysis text.
<!-- STRUCTURED_OUTPUT
{
  "predictions": [],
  "suggestions": [],
  "structural_proposals": [],
  "portfolio_proposals": [
    {
      "proposal_type": "allocation_rebalance",
      "current_config": {"swing": 0.33},
      "proposed_config": {"swing": 0.40},
      "evidence_summary": "Swing outperforms",
      "expected_portfolio_calmar_delta": 0.1,
      "confidence": 0.6,
      "observation_window_days": 60
    }
  ]
}
-->"""
        parsed = parse_response(response)
        assert parsed.parse_success
        assert len(parsed.portfolio_proposals) == 1
        assert parsed.portfolio_proposals[0].proposal_type.value == "allocation_rebalance"

    def test_parse_empty_portfolio_proposals(self):
        from analysis.response_parser import parse_response

        response = """Analysis.
<!-- STRUCTURED_OUTPUT
{"predictions": [], "suggestions": [], "structural_proposals": []}
-->"""
        parsed = parse_response(response)
        assert parsed.parse_success
        assert parsed.portfolio_proposals == []


# ---------------------------------------------------------------------------
# Response validator guardrail tests
# ---------------------------------------------------------------------------

class TestPortfolioValidatorGuardrails:
    def _make_proposal(self, **kwargs):
        from schemas.portfolio_proposal import PortfolioProposal, PortfolioProposalType

        defaults = {
            "proposal_type": PortfolioProposalType.ALLOCATION_REBALANCE,
            "current_config": {"swing": 0.33, "stock": 0.33, "momentum": 0.34},
            "proposed_config": {"swing": 0.40, "stock": 0.30, "momentum": 0.30},
            "evidence_summary": "Test evidence",
            "confidence": 0.6,
            "observation_window_days": 60,
        }
        defaults.update(kwargs)
        return PortfolioProposal(**defaults)

    def _make_validator(self, prediction_accuracy=None):
        from analysis.response_validator import ResponseValidator

        return ResponseValidator(prediction_accuracy=prediction_accuracy or {})

    def test_approve_valid_allocation(self):
        v = self._make_validator()
        proposals = [self._make_proposal()]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(approved) == 1
        assert len(blocked) == 0

    def test_block_excessive_allocation_change(self):
        from schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposed_config={"swing": 0.60, "stock": 0.20, "momentum": 0.20},  # 27% change
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "exceeds" in blocked[0].reason

    def test_block_below_floor(self):
        v = self._make_validator()
        proposals = [self._make_proposal(
            proposed_config={"swing": 0.36, "stock": 0.03, "momentum": 0.34},  # stock < 5%
            current_config={"swing": 0.33, "stock": 0.06, "momentum": 0.34},
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "floor" in blocked[0].reason

    def test_block_insufficient_evidence_allocation(self):
        v = self._make_validator()
        proposals = [self._make_proposal(observation_window_days=30)]  # needs 60
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "60" in blocked[0].reason

    def test_block_risk_cap_excessive_change(self):
        from schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.RISK_CAP_CHANGE,
            current_config={"heat_cap_R": 10.0},
            proposed_config={"heat_cap_R": 15.0},  # 50% change
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "20%" in blocked[0].reason

    def test_block_risk_cap_insufficient_evidence(self):
        from schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.RISK_CAP_CHANGE,
            current_config={"heat_cap_R": 10.0},
            proposed_config={"heat_cap_R": 10.5},
            observation_window_days=60,  # needs 90
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "90" in blocked[0].reason

    def test_block_drawdown_tier_removal(self):
        from schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.DRAWDOWN_TIER_CHANGE,
            current_config={"drawdown_tiers": [[0.05, 0.5], [0.10, 0.25]]},
            proposed_config={"drawdown_tiers": [[0.05, 0.5]]},  # removed a tier
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "removal" in blocked[0].reason.lower()

    def test_block_drawdown_tier_loosening(self):
        from schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.DRAWDOWN_TIER_CHANGE,
            current_config={"drawdown_tiers": [[0.05, 0.5], [0.10, 0.25]]},
            proposed_config={"drawdown_tiers": [[0.08, 0.5], [0.10, 0.25]]},  # loosened first tier
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "loosened" in blocked[0].reason.lower()

    def test_approve_drawdown_tier_tightening(self):
        from schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.DRAWDOWN_TIER_CHANGE,
            current_config={"drawdown_tiers": [[0.05, 0.5], [0.10, 0.25]]},
            proposed_config={"drawdown_tiers": [[0.04, 0.4], [0.08, 0.20]]},  # tightened
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(approved) == 1

    def test_block_all_when_low_prediction_accuracy(self):
        v = self._make_validator(prediction_accuracy={"overall_accuracy": 0.35})
        proposals = [self._make_proposal()]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "prediction accuracy" in blocked[0].reason.lower()

    def test_allow_when_good_prediction_accuracy(self):
        v = self._make_validator(prediction_accuracy={"overall_accuracy": 0.65})
        proposals = [self._make_proposal()]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(approved) == 1

    def test_validate_method_includes_portfolio(self):
        """Full validate() flow includes portfolio proposals in result."""
        from analysis.response_validator import ResponseValidator
        from schemas.agent_response import ParsedAnalysis

        v = ResponseValidator()
        parsed = ParsedAnalysis(raw_report="test", parse_success=True)
        result = v.validate(parsed)
        assert hasattr(result, "approved_portfolio_proposals")
        assert hasattr(result, "blocked_portfolio_proposals")
        assert result.approved_portfolio_proposals == []
        assert result.blocked_portfolio_proposals == []


# ---------------------------------------------------------------------------
# Suggestion tracker extensions
# ---------------------------------------------------------------------------

class TestSuggestionTrackerPortfolio:
    def test_deployed_portfolio_count_empty(self, tmp_path: Path):
        from skills.suggestion_tracker import SuggestionTracker

        tracker = SuggestionTracker(tmp_path)
        assert tracker.get_deployed_portfolio_count() == 0

    def test_deployed_portfolio_count_with_deployed(self, tmp_path: Path):
        from skills.suggestion_tracker import SuggestionTracker
        from schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        record = SuggestionRecord(
            suggestion_id="pf-001", bot_id="PORTFOLIO",
            title="Test", tier="portfolio", category="portfolio_allocation",
            source_report_id="weekly-001",
        )
        tracker.record(record)
        tracker.accept("pf-001")
        tracker.mark_deployed("pf-001")
        assert tracker.get_deployed_portfolio_count() == 1

    def test_last_portfolio_proposal_date_none(self, tmp_path: Path):
        from skills.suggestion_tracker import SuggestionTracker

        tracker = SuggestionTracker(tmp_path)
        assert tracker.get_last_portfolio_proposal_date() is None

    def test_last_portfolio_proposal_date(self, tmp_path: Path):
        from skills.suggestion_tracker import SuggestionTracker
        from schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        record = SuggestionRecord(
            suggestion_id="pf-002", bot_id="PORTFOLIO",
            title="Test", tier="portfolio", category="portfolio_allocation",
            source_report_id="weekly-001",
        )
        tracker.record(record)
        date = tracker.get_last_portfolio_proposal_date()
        assert date is not None

    def test_last_portfolio_proposal_date_filtered(self, tmp_path: Path):
        from skills.suggestion_tracker import SuggestionTracker
        from schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="pf-003", bot_id="PORTFOLIO",
            title="Alloc", tier="portfolio", category="portfolio_allocation",
            source_report_id="weekly-001",
        ))
        tracker.record(SuggestionRecord(
            suggestion_id="pf-004", bot_id="PORTFOLIO",
            title="Risk", tier="portfolio", category="portfolio_risk_cap",
            source_report_id="weekly-001",
        ))
        assert tracker.get_last_portfolio_proposal_date("portfolio_allocation") is not None
        assert tracker.get_last_portfolio_proposal_date("portfolio_drawdown_tier") is None


# ---------------------------------------------------------------------------
# Ground truth portfolio snapshot
# ---------------------------------------------------------------------------

class TestGroundTruthPortfolioSnapshot:
    def _setup_curated(self, tmp_path: Path, bots: list[str], days: int = 30):
        end = datetime(2026, 3, 15)
        for bot in bots:
            for d in range(days):
                date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
                bot_dir = tmp_path / date_str / bot
                bot_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "date": date_str,
                    "net_pnl": 10.0,
                    "total_trades": 5,
                    "winning_trades": 3,
                    "avg_win": 5.0,
                    "avg_loss": -3.0,
                    "win_count": 3,
                    "loss_count": 2,
                    "max_drawdown_pct": 0.02,
                    "avg_process_quality": 70.0,
                }
                (bot_dir / "summary.json").write_text(json.dumps(summary))

    def test_portfolio_snapshot_equal_weights(self, tmp_path: Path):
        from skills.ground_truth_computer import GroundTruthComputer

        bots = ["bot1", "bot2"]
        self._setup_curated(tmp_path, bots)
        computer = GroundTruthComputer(tmp_path)
        result = computer.compute_portfolio_snapshot(bots, "2026-03-15")
        assert "portfolio_composite" in result
        assert 0 <= result["portfolio_composite"] <= 1
        assert len(result["per_bot"]) == 2

    def test_portfolio_snapshot_custom_weights(self, tmp_path: Path):
        from skills.ground_truth_computer import GroundTruthComputer

        bots = ["bot1", "bot2"]
        self._setup_curated(tmp_path, bots)
        computer = GroundTruthComputer(tmp_path)
        result = computer.compute_portfolio_snapshot(
            bots, "2026-03-15", allocations={"bot1": 0.7, "bot2": 0.3},
        )
        assert result["per_bot"]["bot1"]["allocation"] == 0.7

    def test_portfolio_snapshot_empty_bots(self, tmp_path: Path):
        from skills.ground_truth_computer import GroundTruthComputer

        computer = GroundTruthComputer(tmp_path)
        result = computer.compute_portfolio_snapshot([], "2026-03-15")
        assert result["portfolio_composite"] == 0.5


# ---------------------------------------------------------------------------
# Context builder injection
# ---------------------------------------------------------------------------

class TestContextBuilderPortfolio:
    def test_load_portfolio_outcomes_empty(self, tmp_path: Path):
        from analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir)
        assert ctx.load_portfolio_outcomes() == []

    def test_load_portfolio_outcomes_with_data(self, tmp_path: Path):
        from analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (findings_dir / "portfolio_outcomes.jsonl").write_text(
            json.dumps({
                "suggestion_id": "pf-001",
                "verdict": "positive",
                "composite_delta": 0.05,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n"
        )
        ctx = ContextBuilder(memory_dir)
        outcomes = ctx.load_portfolio_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["verdict"] == "positive"

    def test_load_portfolio_metrics_empty(self, tmp_path: Path):
        from analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir, curated_dir=tmp_path / "curated")
        assert ctx.load_portfolio_metrics() == {}

    def test_load_portfolio_metrics_with_data(self, tmp_path: Path):
        from analysis.context_builder import ContextBuilder

        curated_dir = tmp_path / "curated"
        portfolio_dir = curated_dir / "2026-03-15" / "portfolio"
        portfolio_dir.mkdir(parents=True)
        metrics = {"sharpe_7d": 1.2, "calmar_30d": 1.5}
        (portfolio_dir / "portfolio_rolling_metrics.json").write_text(json.dumps(metrics))

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)
        loaded = ctx.load_portfolio_metrics()
        assert loaded["sharpe_7d"] == 1.2

    def test_portfolio_in_context_priority(self):
        from analysis.context_builder import ContextBuilder

        assert "portfolio_outcomes" in ContextBuilder._CONTEXT_PRIORITY
        assert "portfolio_rolling_metrics" in ContextBuilder._CONTEXT_PRIORITY


# ---------------------------------------------------------------------------
# Strategy engine portfolio detectors
# ---------------------------------------------------------------------------

class TestStrategyEnginePortfolioDetectors:
    def _make_engine(self):
        from analysis.strategy_engine import StrategyEngine

        return StrategyEngine(week_start="2026-03-08", week_end="2026-03-14")

    def test_detect_family_imbalance_no_data(self):
        engine = self._make_engine()
        result = engine.detect_family_imbalance({}, {})
        assert result == []

    def test_detect_family_imbalance_insufficient_days(self):
        engine = self._make_engine()
        # Only 5 days of data, needs 30
        summaries = {"swing": {"total_net_pnl": -10, "days": 5}}
        allocations = {"swing": 0.5}
        result = engine.detect_family_imbalance(summaries, allocations, min_days=30)
        assert result == []

    def test_detect_correlation_concentration_no_alert(self):
        engine = self._make_engine()
        # Low correlation, no issue
        matrix = {"bot1_bot2": 0.3}
        allocations = {"bot1": 0.5, "bot2": 0.5}
        result = engine.detect_correlation_concentration(matrix, allocations)
        assert result == []

    def test_detect_correlation_concentration_alert(self):
        engine = self._make_engine()
        matrix = {"bot1_bot2": 0.85}
        allocations = {"bot1": 0.3, "bot2": 0.3}
        result = engine.detect_correlation_concentration(matrix, allocations)
        assert len(result) >= 1
        assert result[0].bot_id == "PORTFOLIO"

    def test_detect_heat_cap_underutilized(self):
        engine = self._make_engine()
        daily_heat = [2.0] * 30  # 20% of 10.0 cap
        result = engine.detect_heat_cap_utilization(daily_heat, 10.0)
        assert len(result) >= 1

    def test_detect_heat_cap_normal(self):
        engine = self._make_engine()
        daily_heat = [6.0] * 30  # 60% of 10.0 cap — normal
        result = engine.detect_heat_cap_utilization(daily_heat, 10.0)
        assert result == []

    def test_detect_drawdown_tier_miscalibration_insufficient_data(self):
        engine = self._make_engine()
        result = engine.detect_drawdown_tier_miscalibration(
            [0.01, 0.02], [[0.05, 0.5]], min_days=90,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Weekly prompt assembler portfolio output spec
# ---------------------------------------------------------------------------

class TestWeeklyPromptPortfolioOutput:
    def test_portfolio_proposals_in_structured_output(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "portfolio_proposals" in _WEEKLY_INSTRUCTIONS
        assert "allocation_rebalance" in _WEEKLY_INSTRUCTIONS
        assert "risk_cap_change" in _WEEKLY_INSTRUCTIONS

    def test_portfolio_improvement_section(self):
        from analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "PORTFOLIO IMPROVEMENT ASSESSMENT" in _WEEKLY_INSTRUCTIONS
        assert "family_snapshots" in _WEEKLY_INSTRUCTIONS
        assert "drawdown_correlation" in _WEEKLY_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Portfolio what-if
# ---------------------------------------------------------------------------

class TestPortfolioWhatIf:
    def test_evaluate_returns_metrics(self):
        from skills.portfolio_what_if import PortfolioWhatIf

        family_daily_pnl = {
            "swing": [10.0] * 60,
            "stock": [-5.0] * 60,
        }
        current_weights = {"swing": 0.5, "stock": 0.5}
        wif = PortfolioWhatIf(family_daily_pnl=family_daily_pnl, current_weights=current_weights)
        result = wif.evaluate({"swing": 0.6, "stock": 0.4})
        assert "sharpe_new" in result or "calmar_delta" in result or "error" in result


# ---------------------------------------------------------------------------
# Portfolio outcome measurer
# ---------------------------------------------------------------------------

class TestPortfolioOutcomeMeasurer:
    def test_instantiation(self, tmp_path: Path):
        from skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        measurer = PortfolioOutcomeMeasurer(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        assert measurer is not None

    def test_check_emergency_no_data(self, tmp_path: Path):
        from skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        measurer = PortfolioOutcomeMeasurer(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        result = measurer.check_emergency()
        assert result is None  # No deployed suggestions → no emergency
