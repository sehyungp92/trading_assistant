# analysis/response_validator.py
"""Response validator — enforces constraints on agent suggestions and predictions.

Strips blocked suggestions, adjusts confidence based on calibration and track record,
and appends Validator Notes to the report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from schemas.agent_response import AgentPrediction, AgentSuggestion, ParsedAnalysis
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
    ) -> None:
        self._rejected = rejected_suggestions or []
        self._forecast_meta = forecast_meta or {}
        self._scorecard = category_scorecard or CategoryScorecard()

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

            # 3. Calibration adjustment
            adjusted = self._apply_calibration(suggestion)
            approved.append(adjusted)

        # Adjust prediction confidence based on calibration
        approved_predictions = [
            self._adjust_prediction_confidence(p) for p in parsed.predictions
        ]

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

        return ValidationResult(
            approved_suggestions=approved,
            blocked_suggestions=blocked,
            approved_predictions=approved_predictions,
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
        """Apply calibration-based confidence adjustments."""
        confidence = suggestion.confidence
        rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
        calibration = self._forecast_meta.get("calibration_adjustment", 0)

        # Use min() of the two factors instead of compounding both
        rolling_factor = rolling_acc if (rolling_acc and rolling_acc < 0.5) else 1.0
        score = self._scorecard.get_score(suggestion.bot_id, suggestion.category)
        category_factor = score.confidence_multiplier if (score and score.confidence_multiplier < 1.0) else 1.0
        confidence *= min(rolling_factor, category_factor)

        if calibration and calibration < -0.2:
            confidence = min(confidence, 0.6)

        confidence = max(0.0, min(1.0, round(confidence, 3)))

        return suggestion.model_copy(update={"confidence": confidence})

    def _adjust_prediction_confidence(self, prediction: AgentPrediction) -> AgentPrediction:
        """Adjust prediction confidence based on forecast meta."""
        confidence = prediction.confidence
        rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
        calibration = self._forecast_meta.get("calibration_adjustment", 0)

        if rolling_acc and rolling_acc < 0.5:
            confidence *= rolling_acc

        if calibration and calibration < -0.2:
            confidence = min(confidence, 0.6)

        confidence = max(0.0, min(1.0, round(confidence, 3)))
        return prediction.model_copy(update={"confidence": confidence})
