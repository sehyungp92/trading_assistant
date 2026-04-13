# skills/learning_card_store.py
"""JSONL-backed persistence for LearningCards with factory methods.

Provides:
  - load/save of card index from ``memory/findings/learning_cards.jsonl``
  - Factory methods to create cards from existing synthesis artifacts
  - Bulk conversion from existing JSONL stores (corrections, outcomes, etc.)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from schemas.learning_card import (
    CardType,
    LearningCard,
    LearningCardIndex,
)

logger = logging.getLogger(__name__)

_DEFAULT_FILENAME = "learning_cards.jsonl"


class LearningCardStore:
    """JSONL-backed store for learning cards."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir
        self._path = findings_dir / _DEFAULT_FILENAME
        self._index: LearningCardIndex | None = None

    @property
    def path(self) -> Path:
        return self._path

    def load(self, *, force: bool = False) -> LearningCardIndex:
        """Load cards from JSONL, caching in memory."""
        if self._index is not None and not force:
            return self._index
        index = LearningCardIndex()
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    try:
                        card = LearningCard.model_validate_json(line)
                        index.add(card)
                    except Exception:
                        logger.debug("Skipping malformed learning card line")
        self._index = index
        return index

    def save(self, index: LearningCardIndex | None = None) -> None:
        """Write all cards back to JSONL."""
        idx = index or self._index or LearningCardIndex()
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            for card in idx.cards:
                f.write(card.model_dump_json() + "\n")
        self._index = idx

    def add_card(self, card: LearningCard) -> None:
        """Add a card and persist."""
        index = self.load()
        index.add(card)
        self.save(index)

    def ranked_for_prompt(
        self,
        limit: int = 20,
        bot_id: str = "",
        workflow: str = "",
        tags: list[str] | None = None,
    ) -> list[LearningCard]:
        """Return top-ranked active cards for prompt injection."""
        index = self.load()
        return index.ranked(
            limit=limit,
            query_bot_id=bot_id,
            query_workflow=workflow,
            query_tags=tags,
        )

    # -- Factory methods for creating cards from existing artifacts --

    @staticmethod
    def _extract_provenance(entry: dict) -> dict:
        """Extract common provenance fields from a JSONL entry."""
        prov: dict = {}
        if entry.get("source_workflow"):
            prov["source_workflow"] = entry["source_workflow"]
        if entry.get("source_run_id") or entry.get("run_id"):
            prov["source_run_id"] = entry.get("source_run_id") or entry.get("run_id", "")
        for ts_key in ("created_at", "timestamp", "recorded_at"):
            if entry.get(ts_key):
                try:
                    from datetime import datetime
                    prov["created_at"] = datetime.fromisoformat(entry[ts_key])
                except (ValueError, TypeError):
                    pass
                break
        return prov

    @staticmethod
    def card_from_correction(entry: dict) -> LearningCard:
        """Create a card from a corrections.jsonl entry."""
        return LearningCard(
            card_type=CardType.CORRECTION,
            source_id=entry.get("id", entry.get("correction_id", "")),
            bot_id=entry.get("bot_id", ""),
            title=entry.get("summary", entry.get("description", "Correction"))[:120],
            content=entry.get("description", entry.get("summary", "")),
            tags=entry.get("categories", []),
            confidence=0.8,
            impact_score=0.3,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_outcome(entry: dict) -> LearningCard:
        """Create a card from an outcomes.jsonl entry."""
        verdict = entry.get("verdict", "unknown")
        impact = 0.5 if verdict == "positive" else (-0.3 if verdict == "negative" else 0.0)
        return LearningCard(
            card_type=CardType.OUTCOME,
            source_id=entry.get("suggestion_id", ""),
            bot_id=entry.get("bot_id", ""),
            title=f"Outcome: {entry.get('category', 'unknown')} — {verdict}",
            content=entry.get("explanation", ""),
            evidence_summary=entry.get("evidence_base", ""),
            tags=[entry.get("category", "")],
            confidence=0.7 if entry.get("measurement_quality", "").lower() in ("high", "medium") else 0.3,
            impact_score=impact,
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_hypothesis(entry: dict) -> LearningCard:
        """Create a card from a hypothesis record."""
        effectiveness = entry.get("effectiveness", 0.0)
        return LearningCard(
            card_type=CardType.HYPOTHESIS,
            source_id=entry.get("hypothesis_id", entry.get("id", "")),
            bot_id=entry.get("bot_id", ""),
            title=entry.get("title", entry.get("symptom", "Hypothesis"))[:120],
            content=entry.get("description", entry.get("solution", "")),
            tags=[entry.get("category", "")],
            confidence=min(max(effectiveness, 0.0), 1.0),
            impact_score=effectiveness,
            observation_count=entry.get("proposal_count", 1),
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_discovery(entry: dict) -> LearningCard:
        """Create a card from a discoveries.jsonl entry."""
        return LearningCard(
            card_type=CardType.DISCOVERY,
            source_id=entry.get("discovery_id", entry.get("id", "")),
            bot_id=entry.get("bot_id", ""),
            title=entry.get("title", entry.get("pattern", "Discovery"))[:120],
            content=entry.get("description", entry.get("details", "")),
            evidence_summary=entry.get("evidence", ""),
            tags=entry.get("tags", []),
            confidence=entry.get("confidence", 0.5),
            **LearningCardStore._extract_provenance(entry),
        )

    @staticmethod
    def card_from_validator_block(entry: dict) -> LearningCard:
        """Create a 'do not repeat' card from a validation_patterns entry."""
        return LearningCard(
            card_type=CardType.VALIDATOR_BLOCK,
            source_id=entry.get("pattern_id", entry.get("category", "")),
            bot_id=entry.get("bot_id", ""),
            title=f"BLOCKED: {entry.get('category', 'unknown')} — {entry.get('reason', 'blocked')}",
            content=entry.get("reason", entry.get("description", "")),
            tags=[entry.get("category", "")],
            confidence=0.9,
            impact_score=-0.5,
            observation_count=entry.get("block_count", 1),
            **LearningCardStore._extract_provenance(entry),
        )

    def ingest_from_existing(self) -> int:
        """Bulk-create cards from existing JSONL stores.

        Scans corrections, outcomes, and discoveries JSONL files and creates
        cards for entries not already in the index.  Returns count of new cards.
        """
        index = self.load()
        existing_ids = {c.card_id for c in index.cards}
        new_count = 0

        sources: list[tuple[str, CardType]] = [
            ("corrections.jsonl", CardType.CORRECTION),
            ("outcomes.jsonl", CardType.OUTCOME),
            ("discoveries.jsonl", CardType.DISCOVERY),
        ]

        factory_map = {
            CardType.CORRECTION: self.card_from_correction,
            CardType.OUTCOME: self.card_from_outcome,
            CardType.DISCOVERY: self.card_from_discovery,
        }

        for filename, card_type in sources:
            path = self._findings_dir / filename
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    card = factory_map[card_type](entry)
                    if card.card_id not in existing_ids:
                        index.add(card)
                        existing_ids.add(card.card_id)
                        new_count += 1
                except Exception:
                    continue

        if new_count:
            self.save(index)
            logger.info("Ingested %d new learning cards from existing stores", new_count)

        return new_count
