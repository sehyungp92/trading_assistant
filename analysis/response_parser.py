# analysis/response_parser.py
"""Response parser — extracts structured data from an agent markdown response.

Looks for a <!-- STRUCTURED_OUTPUT ... --> block in the response and parses
the JSON content into a ParsedAnalysis model.
"""
from __future__ import annotations

import json
import logging
import re

from schemas.agent_response import (
    AgentPrediction,
    AgentSuggestion,
    ParsedAnalysis,
    StructuralProposal,
)

logger = logging.getLogger(__name__)


def _safe_parse_list(items: list, model_class: type) -> list:
    """Parse each item individually, skipping invalid ones for partial recovery."""
    result = []
    for i, item in enumerate(items):
        try:
            result.append(model_class(**item) if isinstance(item, dict) else item)
        except Exception as exc:
            logger.warning("Skipping invalid item %d in %s: %s", i, model_class.__name__, exc)
    return result

_BLOCK_PATTERN = re.compile(
    r"<!--\s*STRUCTURED_OUTPUT\s*\n(.*?)\n\s*-->",
    re.DOTALL,
)
# Fallback: some models (GPT, GLM) may emit a ```json fence instead of
# an HTML comment block.  Accept both for multi-LLM provider parity.
_JSON_FENCE_PATTERN = re.compile(r"```json\s*\n(.*?)\n\s*```", re.DOTALL)


def _extract_structured_json(response: str) -> dict | None:
    """Try comment-block first, then fenced JSON.  Returns parsed dict or None."""
    for pattern in (_BLOCK_PATTERN, _JSON_FENCE_PATTERN):
        match = pattern.search(response)
        if not match:
            continue
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse structured output JSON: %s", exc)
    return None


# Keys that identify a JSON object as a suggestion vs prediction vs structural proposal
_SUGGESTION_KEYS = {"suggestion_id", "category", "expected_impact"}
_PREDICTION_KEYS = {"metric", "direction", "confidence", "timeframe_days"}
_STRUCTURAL_KEYS = {"reversibility", "estimated_complexity", "acceptance_criteria"}
_PORTFOLIO_KEYS = {"proposal_type", "proposed_config", "expected_portfolio_calmar_delta"}

# Match top-level JSON objects in markdown text (non-greedy, brace-balanced)
_INLINE_JSON_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_from_markdown(response: str) -> dict | None:
    """Fallback: extract inline JSON objects that look like suggestions/predictions.

    Scans the response for JSON objects containing known schema keys and
    assembles them into a partial structured output dict. Conservative:
    only extracts when keys match expected schemas to avoid false positives.

    Also handles bare top-level wrappers (``{"predictions": [...], ...}``)
    emitted without comment-block or fence markers.
    """
    suggestions: list[dict] = []
    predictions: list[dict] = []
    structural_proposals: list[dict] = []
    portfolio_proposals: list[dict] = []

    # Top-level wrapper keys that indicate the LLM emitted the expected
    # structure but forgot the comment-block / fence markers.
    _WRAPPER_KEYS = {"predictions", "suggestions", "structural_proposals", "portfolio_proposals"}

    for match in _INLINE_JSON_PATTERN.finditer(response):
        try:
            obj = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue

        keys = set(obj.keys())

        # Check for top-level wrapper first (e.g. bare {"predictions": [...], "suggestions": [...]})
        if keys & _WRAPPER_KEYS and any(isinstance(obj.get(k), list) for k in _WRAPPER_KEYS):
            obj["_fallback_extraction"] = True
            return obj

        # Must have bot_id to be a valid structured item
        if "bot_id" not in keys and "proposal_type" not in keys:
            continue

        if keys & _SUGGESTION_KEYS and "title" in keys:
            suggestions.append(obj)
        elif keys & _PREDICTION_KEYS and "bot_id" in keys:
            predictions.append(obj)
        elif keys & _STRUCTURAL_KEYS and "title" in keys:
            structural_proposals.append(obj)
        elif keys & _PORTFOLIO_KEYS:
            portfolio_proposals.append(obj)

    if not (suggestions or predictions or structural_proposals or portfolio_proposals):
        return None

    result: dict = {"_fallback_extraction": True}
    if predictions:
        result["predictions"] = predictions
    if suggestions:
        result["suggestions"] = suggestions
    if structural_proposals:
        result["structural_proposals"] = structural_proposals
    if portfolio_proposals:
        result["portfolio_proposals"] = portfolio_proposals
    return result


def parse_response(response: str) -> ParsedAnalysis:
    """Parse an agent response, extracting the structured output block.

    Recognises both ``<!-- STRUCTURED_OUTPUT -->`` comment blocks (preferred)
    and ````json`` fenced code blocks (fallback for non-Claude providers).

    If no block is found or JSON is malformed, returns ParsedAnalysis with
    parse_success=False and empty lists. The raw_report is always preserved.
    """
    if not response:
        return ParsedAnalysis(parse_success=False, raw_report="")

    data = _extract_structured_json(response)
    fallback_used = False
    if data is None:
        data = _extract_from_markdown(response)
        if data is None:
            return ParsedAnalysis(parse_success=False, raw_report=response)
        fallback_used = True
        logger.info("Used fallback markdown extraction for structured output")

    predictions = _safe_parse_list(data.get("predictions", []), AgentPrediction)
    suggestions = _safe_parse_list(data.get("suggestions", []), AgentSuggestion)
    structural_proposals = _safe_parse_list(data.get("structural_proposals", []), StructuralProposal)

    # Parse portfolio proposals if present
    portfolio_proposals: list = []
    raw_portfolio = data.get("portfolio_proposals", [])
    if raw_portfolio:
        try:
            from schemas.portfolio_proposal import PortfolioProposal
            portfolio_proposals = _safe_parse_list(raw_portfolio, PortfolioProposal)
        except Exception as exc:
            logger.warning("Failed to parse portfolio_proposals: %s", exc)

    return ParsedAnalysis(
        predictions=predictions,
        suggestions=suggestions,
        structural_proposals=structural_proposals,
        portfolio_proposals=portfolio_proposals,
        raw_report=response,
        parse_success=True,
        fallback_used=fallback_used,
        raw_structured=data,
    )
