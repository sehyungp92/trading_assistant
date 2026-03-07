# analysis/response_parser.py
"""Response parser — extracts structured data from Claude's markdown response.

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


def parse_response(response: str) -> ParsedAnalysis:
    """Parse a Claude response, extracting the structured output block.

    If no block is found or JSON is malformed, returns ParsedAnalysis with
    parse_success=False and empty lists. The raw_report is always preserved.
    """
    if not response:
        return ParsedAnalysis(parse_success=False, raw_report="")

    match = _BLOCK_PATTERN.search(response)
    if not match:
        return ParsedAnalysis(parse_success=False, raw_report=response)

    json_text = match.group(1).strip()
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse structured output JSON: %s", exc)
        return ParsedAnalysis(parse_success=False, raw_report=response)

    predictions = _safe_parse_list(data.get("predictions", []), AgentPrediction)
    suggestions = _safe_parse_list(data.get("suggestions", []), AgentSuggestion)
    structural_proposals = _safe_parse_list(data.get("structural_proposals", []), StructuralProposal)

    return ParsedAnalysis(
        predictions=predictions,
        suggestions=suggestions,
        structural_proposals=structural_proposals,
        raw_report=response,
        parse_success=True,
    )
