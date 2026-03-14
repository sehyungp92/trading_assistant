# skills/transfer_proposal_builder.py
"""TransferProposalBuilder — identifies cross-bot pattern transfer candidates.

For each VALIDATED/IMPLEMENTED pattern, finds eligible target bots and scores compatibility.
Also measures outcomes of completed transfers and feeds back into scoring.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from schemas.pattern_library import PatternEntry, PatternStatus
from schemas.transfer_proposals import TransferOutcome, TransferProposal

logger = logging.getLogger(__name__)


class TransferProposalBuilder:
    def __init__(
        self,
        pattern_library,  # PatternLibrary instance
        curated_dir: Path,
        bots: list[str],
        findings_dir: Path | None = None,
    ) -> None:
        self._library = pattern_library
        self._curated_dir = curated_dir
        self._bots = bots
        self._findings_dir = findings_dir
        self._outcomes_path = (findings_dir / "transfer_outcomes.jsonl") if findings_dir else None

    def build_proposals(self) -> list[TransferProposal]:
        """Build transfer proposals for active patterns (PROPOSED, VALIDATED, IMPLEMENTED)."""
        active = self._library.load_active()
        transferable = active  # load_active() already excludes REJECTED

        # Load track record for scoring adjustment
        track_record = self.compute_transfer_track_record()

        proposals: list[TransferProposal] = []
        for pattern in transferable:
            eligible = self._find_eligible_bots(pattern)
            for bot_id in eligible:
                score = self._compute_compatibility(pattern, bot_id)

                # Adjust score based on track record
                pattern_track = track_record.get(pattern.pattern_id, {})
                if pattern_track:
                    success_rate = pattern_track.get("success_rate", 0.5)
                    if success_rate > 0.5:
                        score = min(1.0, score + 0.1)
                    elif success_rate < 0.3 and pattern_track.get("total", 0) >= 2:
                        score = max(0.0, score - 0.2)

                proposals.append(TransferProposal(
                    pattern_id=pattern.pattern_id,
                    source_bot=pattern.source_bot,
                    target_bot=bot_id,
                    pattern_title=pattern.title,
                    category=pattern.category.value if hasattr(pattern.category, "value") else str(pattern.category),
                    compatibility_score=round(score, 2),
                    rationale=f"Pattern '{pattern.title}' validated on {pattern.source_bot}, "
                              f"compatibility score {score:.2f} for {bot_id}.",
                ))

        # Sort by score descending
        proposals.sort(key=lambda p: p.compatibility_score, reverse=True)
        return proposals

    def measure_transfer_outcomes(self, curated_dir: Path | None = None) -> list[TransferOutcome]:
        """Measure outcomes of transferred patterns by comparing before/after metrics.

        For each TRANSFERRED pattern (target_bots populated), compares the target
        bot's 7-day metrics before and after the transfer date.
        """
        curated = curated_dir or self._curated_dir
        active = self._library.load_active()
        transferred = [
            p for p in active
            if p.target_bots and p.status in (PatternStatus.VALIDATED, PatternStatus.IMPLEMENTED)
        ]

        outcomes: list[TransferOutcome] = []
        existing = self._load_outcomes()
        measured_pairs = {(o.get("pattern_id"), o.get("target_bot")) for o in existing}

        for pattern in transferred:
            transfer_date = pattern.validated_at or (
                pattern.updated_at.strftime("%Y-%m-%d")
                if hasattr(pattern.updated_at, "strftime") else ""
            )
            for target_bot in pattern.target_bots:
                if (pattern.pattern_id, target_bot) in measured_pairs:
                    continue

                before = self._load_bot_metrics(target_bot, transfer_date, curated, before=True)
                after = self._load_bot_metrics(target_bot, transfer_date, curated, before=False)

                if before is None or after is None:
                    continue

                pnl_delta = after.get("pnl", 0) - before.get("pnl", 0)
                wr_delta = after.get("win_rate", 0) - before.get("win_rate", 0)

                if pnl_delta > 0 and wr_delta >= 0:
                    verdict = "positive"
                elif pnl_delta < 0:
                    verdict = "negative"
                else:
                    verdict = "neutral"

                outcome = TransferOutcome(
                    pattern_id=pattern.pattern_id,
                    source_bot=pattern.source_bot,
                    target_bot=target_bot,
                    transferred_at=transfer_date,
                    pnl_delta_7d=round(pnl_delta, 4),
                    win_rate_delta_7d=round(wr_delta, 4),
                    verdict=verdict,
                )
                outcomes.append(outcome)

        # Persist
        if outcomes and self._outcomes_path:
            self._outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._outcomes_path, "a", encoding="utf-8") as f:
                for o in outcomes:
                    f.write(json.dumps(o.model_dump(mode="json"), default=str) + "\n")

        return outcomes

    def compute_transfer_track_record(self) -> dict[str, dict]:
        """Compute per-pattern success rates across all measured transfers."""
        outcomes = self._load_outcomes()
        if not outcomes:
            return {}

        by_pattern: dict[str, list[dict]] = defaultdict(list)
        for o in outcomes:
            by_pattern[o.get("pattern_id", "")].append(o)

        track_record: dict[str, dict] = {}
        for pattern_id, pattern_outcomes in by_pattern.items():
            positive = sum(1 for o in pattern_outcomes if o.get("verdict") == "positive")
            total = len(pattern_outcomes)
            track_record[pattern_id] = {
                "total": total,
                "positive": positive,
                "negative": sum(1 for o in pattern_outcomes if o.get("verdict") == "negative"),
                "success_rate": round(positive / total, 3) if total > 0 else 0.0,
            }

        return track_record

    def create_from_reasoning(
        self, reasoning: dict, source_bot: str,
    ) -> list[str]:
        """Create transfer proposals from a transferable outcome reasoning.

        Args:
            reasoning: Outcome reasoning dict with transferable=True.
            source_bot: The bot where the original suggestion was applied.

        Returns:
            List of target bot IDs that received proposals.
        """
        mechanism = reasoning.get("mechanism", "")
        suggestion_id = reasoning.get("suggestion_id", "")
        title = reasoning.get("title", mechanism[:80] if mechanism else "Transferable pattern")

        import hashlib as _hashlib
        pattern_id = _hashlib.sha256(
            f"reasoning:{suggestion_id}:{source_bot}".encode()
        ).hexdigest()[:12]

        targets: list[str] = []
        for bot_id in self._bots:
            if bot_id == source_bot:
                continue
            try:
                proposal = TransferProposal(
                    pattern_id=pattern_id,
                    source_bot=source_bot,
                    target_bot=bot_id,
                    pattern_title=title,
                    category=reasoning.get("category", "structural"),
                    compatibility_score=0.5,  # default — no pattern entry to score against
                    rationale=(
                        f"Outcome reasoning for suggestion {suggestion_id} identified "
                        f"transferable mechanism: {mechanism[:200]}"
                    ),
                )
                # Persist proposal
                if self._findings_dir:
                    proposals_path = self._findings_dir / "transfer_proposals.jsonl"
                    proposals_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(proposals_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(proposal.model_dump(mode="json"), default=str) + "\n")
                targets.append(bot_id)
            except Exception:
                logger.warning("Failed to create transfer proposal for %s → %s", source_bot, bot_id)
        return targets

    @staticmethod
    def load_track_record_from_file(findings_dir: Path) -> dict[str, dict]:
        """Load transfer track record directly from outcomes file, no builder needed."""
        outcomes_path = findings_dir / "transfer_outcomes.jsonl"
        if not outcomes_path.exists():
            return {}
        records: list[dict] = []
        for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not records:
            return {}
        by_pattern: dict[str, list[dict]] = defaultdict(list)
        for o in records:
            by_pattern[o.get("pattern_id", "")].append(o)
        track_record: dict[str, dict] = {}
        for pattern_id, pattern_outcomes in by_pattern.items():
            positive = sum(1 for o in pattern_outcomes if o.get("verdict") == "positive")
            total = len(pattern_outcomes)
            track_record[pattern_id] = {
                "total": total,
                "positive": positive,
                "negative": sum(1 for o in pattern_outcomes if o.get("verdict") == "negative"),
                "success_rate": round(positive / total, 3) if total > 0 else 0.0,
            }
        return track_record

    def _load_outcomes(self) -> list[dict]:
        if not self._outcomes_path or not self._outcomes_path.exists():
            return []
        records: list[dict] = []
        for line in self._outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def _find_eligible_bots(self, pattern: PatternEntry) -> list[str]:
        """Find bots eligible for receiving a pattern transfer."""
        already_targeted = set(pattern.target_bots or [])
        source = pattern.source_bot

        return [
            bot for bot in self._bots
            if bot != source and bot not in already_targeted
        ]

    def _compute_compatibility(self, pattern: PatternEntry, target_bot: str) -> float:
        """Score compatibility between a pattern and a target bot."""
        source_regimes = self._load_regime_distribution(pattern.source_bot)
        target_regimes = self._load_regime_distribution(target_bot)

        if not source_regimes or not target_regimes:
            return 0.5

        source_keys = set(source_regimes.keys())
        target_keys = set(target_regimes.keys())
        if not source_keys or not target_keys:
            return 0.5

        intersection = source_keys & target_keys
        union = source_keys | target_keys
        regime_overlap = len(intersection) / len(union) if union else 0.0

        overlap_weight = 0.0
        for regime in intersection:
            overlap_weight += min(
                source_regimes.get(regime, 0),
                target_regimes.get(regime, 0),
            )

        return min(1.0, regime_overlap * 0.5 + overlap_weight * 0.5)

    def _load_regime_distribution(self, bot_id: str) -> dict[str, float]:
        """Load the most recent regime_analysis.json for a bot."""
        from datetime import datetime, timedelta, timezone

        today = datetime.now(timezone.utc)
        for days_back in range(7):
            date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
            path = self._curated_dir / date_str / bot_id / "regime_analysis.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data.get("regime_trade_count"), dict):
                        counts = {
                            str(regime): int(count or 0)
                            for regime, count in data["regime_trade_count"].items()
                        }
                    else:
                        counts = {
                            str(regime): int(
                                (value.get("trade_count", 1) if isinstance(value, dict) else 1) or 0
                            )
                            for regime, value in data.items()
                        }
                    total = sum(counts.values())
                    if total > 0:
                        return {regime: count / total for regime, count in counts.items() if count > 0}
                except (json.JSONDecodeError, OSError):
                    pass
        return {}

    def _load_bot_metrics(
        self, bot_id: str, date: str, curated_dir: Path, before: bool,
    ) -> dict | None:
        """Load aggregated bot metrics for 7 days before or after a date."""
        from datetime import datetime, timedelta

        if not date:
            return None

        try:
            base = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return None

        if before:
            dates = [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
        else:
            dates = [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]

        pnl_total = 0.0
        wr_values: list[float] = []
        found = 0

        for d in dates:
            summary_path = curated_dir / d / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    pnl_total += data.get("net_pnl", 0)
                    wr = data.get("win_rate")
                    if wr is not None:
                        wr_values.append(float(wr))
                    found += 1
                except (json.JSONDecodeError, OSError, ValueError):
                    pass

        if found < 3:
            return None

        return {
            "pnl": pnl_total,
            "win_rate": sum(wr_values) / len(wr_values) if wr_values else 0.0,
        }
