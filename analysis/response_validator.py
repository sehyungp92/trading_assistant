# analysis/response_validator.py
"""Response validator — enforces constraints on agent suggestions and predictions.

Strips blocked suggestions, adjusts confidence based on calibration and track record,
and appends Validator Notes to the report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from schemas.agent_response import AgentPrediction, AgentSuggestion, ParsedAnalysis, StructuralProposal
from schemas.suggestion_scoring import CategoryScorecard

logger = logging.getLogger(__name__)


@dataclass
class BlockedSuggestion:
    suggestion: AgentSuggestion
    reason: str


@dataclass
class ValidationResult:
    approved_suggestions: list[AgentSuggestion] = field(default_factory=list)
    blocked_suggestions: list[BlockedSuggestion] = field(default_factory=list)
    approved_predictions: list[AgentPrediction] = field(default_factory=list)
    approved_structural_proposals: list[StructuralProposal] = field(default_factory=list)
    blocked_structural_proposals: list[StructuralProposal] = field(default_factory=list)
    validator_notes: str = ""


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class ResponseValidator:
    def __init__(
        self,
        rejected_suggestions: list[dict] | None = None,
        forecast_meta: dict | None = None,
        category_scorecard: CategoryScorecard | None = None,
        hypothesis_track_record: dict | None = None,
    ) -> None:
        self._rejected = rejected_suggestions or []
        self._forecast_meta = forecast_meta or {}
        self._scorecard = category_scorecard or CategoryScorecard()
        self._hypothesis_track_record = hypothesis_track_record or {}

    def validate(self, parsed: ParsedAnalysis) -> ValidationResult:
        """Validate a parsed analysis, filtering suggestions and adjusting confidence."""
        approved: list[AgentSuggestion] = []
        blocked: list[BlockedSuggestion] = []
        notes_lines: list[str] = []

        for suggestion in parsed.suggestions:
            # 1. Rejection check — fuzzy match against rejected suggestions
            rejection_match = self._check_rejection(suggestion)
            if rejection_match:
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=f"Similar to rejected suggestion: {rejection_match}",
                ))
                continue

            # 2. Category track record check
            score = self._scorecard.get_score(suggestion.bot_id, suggestion.category)
            if score and score.sample_size >= 5 and score.win_rate < 0.3:
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=f"Poor category track record: {suggestion.category} win_rate={score.win_rate:.0%} (n={score.sample_size})",
                ))
                continue

            # 2b. Simplicity criterion — marginal track record not worth implementation cost
            if (
                score
                and score.sample_size >= 3
                and score.win_rate < 0.5
                and score.avg_pnl_delta < 0.001
            ):
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=(
                        f"Blocked: marginal track record in {suggestion.category} "
                        f"(win_rate={score.win_rate:.0%}, avg_pnl={score.avg_pnl_delta:.4f}) "
                        f"— not worth the implementation cost"
                    ),
                ))
                continue

            # 2c. Low-confidence structural suggestions blocked
            if (
                suggestion.category == "structural"
                and suggestion.confidence < 0.4
            ):
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=(
                        "Blocked: low-confidence structural changes carry "
                        "high risk for uncertain benefit"
                    ),
                ))
                continue

            # 3. Calibration adjustment
            adjusted = self._apply_calibration(suggestion)
            approved.append(adjusted)

        # Adjust prediction confidence based on calibration
        approved_predictions = [
            self._adjust_prediction_confidence(p) for p in parsed.predictions
        ]

        # Validate structural proposals
        approved_proposals, blocked_proposals = (
            self._validate_structural_proposals(parsed.structural_proposals)
        )

        # Build validator notes
        if blocked:
            notes_lines.append(f"**{len(blocked)} suggestion(s) blocked:**")
            for b in blocked:
                notes_lines.append(f"- \"{b.suggestion.title}\" — {b.reason}")

        calibration_adj = self._forecast_meta.get("calibration_adjustment", 0)
        rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
        if rolling_acc and rolling_acc < 0.5:
            notes_lines.append(
                f"\n**Calibration warning:** Rolling 4-week accuracy is {rolling_acc:.0%}. "
                f"Confidence scores have been adjusted downward."
            )

        if blocked_proposals:
            notes_lines.append(f"\n**{len(blocked_proposals)} structural proposal(s) blocked (missing acceptance criteria):**")
            for p in blocked_proposals:
                notes_lines.append(f"- \"{p.title}\"")

        return ValidationResult(
            approved_suggestions=approved,
            blocked_suggestions=blocked,
            approved_predictions=approved_predictions,
            approved_structural_proposals=approved_proposals,
            blocked_structural_proposals=blocked_proposals,
            validator_notes="\n".join(notes_lines),
        )

    def _check_rejection(self, suggestion: AgentSuggestion) -> str | None:
        """Check if a suggestion is too similar to a rejected one."""
        for rejected in self._rejected:
            rej_bot = rejected.get("bot_id", "")
            rej_title = rejected.get("title", "")
            rej_tier = rejected.get("tier", "")

            # Must be same bot
            if rej_bot and suggestion.bot_id and rej_bot != suggestion.bot_id:
                continue

            # Check title similarity
            if _jaccard_similarity(suggestion.title, rej_title) >= 0.6:
                return rej_title

        return None

    def _apply_calibration(self, suggestion: AgentSuggestion) -> AgentSuggestion:
        """Apply calibration-based confidence adjustments.

        When empirical calibration buckets are available, uses bucket-matched
        observed accuracy to blend confidence. Falls back to heuristic when
        no bucket data exists.
        """
        confidence = suggestion.confidence

        # Try empirical bucket-based adjustment first
        bucket_adjusted = self._bucket_adjust_confidence(confidence)
        if bucket_adjusted is not None:
            confidence = bucket_adjusted
        else:
            # Heuristic fallback
            rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
            rolling_factor = rolling_acc if (rolling_acc and rolling_acc < 0.5) else 1.0
            score = self._scorecard.get_score(suggestion.bot_id, suggestion.category)
            category_factor = score.confidence_multiplier if (score and score.confidence_multiplier < 1.0) else 1.0
            confidence *= min(rolling_factor, category_factor)

        calibration = self._forecast_meta.get("calibration_adjustment", 0)
        if calibration and calibration < -0.2:
            confidence = min(confidence, 0.6)

        # Still apply category factor even when using bucket adjustment
        if bucket_adjusted is not None:
            score = self._scorecard.get_score(suggestion.bot_id, suggestion.category)
            if score and score.confidence_multiplier < 1.0:
                confidence *= score.confidence_multiplier

        confidence = max(0.0, min(1.0, round(confidence, 3)))

        return suggestion.model_copy(update={"confidence": confidence})

    def _adjust_prediction_confidence(self, prediction: AgentPrediction) -> AgentPrediction:
        """Adjust prediction confidence based on forecast meta."""
        confidence = prediction.confidence

        # Try empirical bucket-based adjustment first
        bucket_adjusted = self._bucket_adjust_confidence(confidence)
        if bucket_adjusted is not None:
            confidence = bucket_adjusted
        else:
            rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
            if rolling_acc and rolling_acc < 0.5:
                confidence *= rolling_acc

        calibration = self._forecast_meta.get("calibration_adjustment", 0)
        if calibration and calibration < -0.2:
            confidence = min(confidence, 0.6)

        confidence = max(0.0, min(1.0, round(confidence, 3)))
        return prediction.model_copy(update={"confidence": confidence})

    def _validate_structural_proposals(
        self,
        proposals: list[StructuralProposal],
    ) -> tuple[list[StructuralProposal], list[StructuralProposal]]:
        """Validate structural proposals have acceptance criteria.

        Also blocks proposals linked to hypotheses with non-positive effectiveness.
        Returns (approved, blocked).
        """
        approved: list[StructuralProposal] = []
        blocked_list: list[StructuralProposal] = []

        for proposal in proposals:
            # Check hypothesis track record
            hyp_id = proposal.hypothesis_id
            if hyp_id and self._hypothesis_track_record:
                hyp_data = self._hypothesis_track_record.get(hyp_id)
                if hyp_data:
                    effectiveness = hyp_data.get("effectiveness", 0.0)
                    status = hyp_data.get("status", "")
                    if effectiveness <= 0 or status == "retired":
                        blocked_list.append(proposal)
                        logger.info(
                            "Blocked structural proposal '%s': hypothesis %s has "
                            "effectiveness=%.3f, status=%s",
                            proposal.title, hyp_id, effectiveness, status,
                        )
                        continue

            # Must have at least one criterion with a 'metric' field
            criteria = proposal.acceptance_criteria
            has_valid_criteria = any(
                isinstance(c, dict) and c.get("metric")
                for c in criteria
            )
            if not has_valid_criteria:
                blocked_list.append(proposal)
            else:
                approved.append(proposal)

        return approved, blocked_list

    def _bucket_adjust_confidence(self, confidence: float) -> float | None:
        """Adjust confidence using empirical calibration buckets.

        Returns adjusted confidence, or None if no bucket data available.
        """
        buckets = self._forecast_meta.get("calibration_buckets", [])
        if not buckets:
            return None

        # Find matching bucket
        for bucket in buckets:
            lo = bucket.get("bucket_lower", 0)
            hi = bucket.get("bucket_upper", 1)
            count = bucket.get("prediction_count", 0)
            in_range = (lo <= confidence < hi) if hi < 1.0 else (lo <= confidence <= hi)
            if in_range and count >= 5:
                gap = bucket.get("gap", 0)
                if abs(gap) > 0.1:
                    observed = bucket.get("observed_accuracy", confidence)
                    # Blend: 70% original, 30% observed accuracy
                    return confidence * 0.7 + observed * 0.3
                return confidence  # Bucket matched, gap small — preserve original

        return None  # No reliable bucket matched
