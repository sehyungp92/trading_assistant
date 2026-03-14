# skills/learning_cycle.py
"""LearningCycle — autonomous weekly learning loop.

Autoresearch's infinite loop adapted for trading:
measure → synthesize → keep/discard → recalibrate → propose next.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from schemas.learning_ledger import LearningLedgerEntry

logger = logging.getLogger(__name__)


class LearningCycle:
    """Autonomous weekly learning cycle.

    Weekly sequence: ground truth → synthesis → hypothesis lifecycle →
    category recalibration → experiment selection → ledger record.
    """

    def __init__(
        self,
        curated_dir: Path,
        memory_dir: Path,
        runs_dir: Path,
        bots: list[str],
        suggestion_tracker=None,
        hypothesis_library=None,
        experiment_tracker=None,
        prediction_tracker=None,
        calibration_tracker=None,
    ) -> None:
        self._curated_dir = curated_dir
        self._memory_dir = memory_dir
        self._runs_dir = runs_dir
        self._bots = bots
        self._suggestion_tracker = suggestion_tracker
        self._hypothesis_library = hypothesis_library
        self._experiment_tracker = experiment_tracker
        self._prediction_tracker = prediction_tracker
        self._calibration_tracker = calibration_tracker

    async def run(
        self, week_start: str, week_end: str,
    ) -> LearningLedgerEntry:
        """Execute the full weekly learning cycle.

        1. Compute ground truth snapshots (start and end of week)
        2. Build retrospective synthesis (keep/discard verdicts)
        3. Update hypothesis lifecycle from verdicts
        4. Recalibrate suggestion categories
        5. Select next experiments (max 2 per bot)
        6. Record to learning ledger
        """
        from skills.ground_truth_computer import GroundTruthComputer
        from skills.learning_ledger import LearningLedger
        from skills.retrospective_builder import RetrospectiveBuilder
        from skills.suggestion_scorer import SuggestionScorer

        findings_dir = self._memory_dir / "findings"
        computer = GroundTruthComputer(self._curated_dir)
        ledger = LearningLedger(findings_dir)

        # 1. Ground truth snapshots
        gt_start = computer.compute_all_bots(self._bots, week_start)
        gt_end = computer.compute_all_bots(self._bots, week_end)

        composite_delta: dict[str, float] = {}
        for bot_id in self._bots:
            start_score = gt_start.get(bot_id)
            end_score = gt_end.get(bot_id)
            if start_score and end_score:
                composite_delta[bot_id] = round(
                    end_score.composite_score - start_score.composite_score, 4,
                )

        # 2. Retrospective synthesis
        retro = RetrospectiveBuilder(
            runs_dir=self._runs_dir,
            curated_dir=self._curated_dir,
            memory_dir=self._memory_dir,
        )
        synthesis = retro.build_synthesis(week_start, week_end)

        # 3. Update hypothesis lifecycle from verdicts
        if self._hypothesis_library:
            for item, positive in [
                *((i, True) for i in synthesis.what_worked),
                *((i, False) for i in synthesis.what_failed),
            ]:
                sid = getattr(item, "suggestion_id", "") or ""
                if not sid:
                    continue
                try:
                    self._hypothesis_library.record_outcome(sid, positive=positive)
                except Exception:
                    logger.debug("No hypothesis for suggestion %s", sid)

        # 4. Recalibrate suggestion categories
        scorer = SuggestionScorer(findings_dir)
        if synthesis.discard:
            scorer.apply_recalibration(synthesis.discard)
            logger.info(
                "Recalibrated %d categories", len(synthesis.discard),
            )

        # 5. Select next experiments (max 2 per bot) and record them
        selected_experiments = self._select_next_experiments()
        self._record_selected_experiments(selected_experiments)

        # 6. Count activity for ledger
        suggestions_proposed, suggestions_accepted, suggestions_implemented = (
            self._count_suggestions(week_start, week_end)
        )
        experiments_concluded = self._count_experiments(week_start, week_end)
        discoveries_found = self._count_jsonl_in_range(
            findings_dir / "discoveries.jsonl",
            week_start, week_end,
            ts_keys=("discovered_at", "timestamp"),
        )

        # Net improvement: majority of bots improved (not just sum > 0)
        if composite_delta:
            improved = sum(1 for d in composite_delta.values() if d > 0)
            net_improvement = improved > len(composite_delta) / 2
        else:
            net_improvement = False

        entry = ledger.record_week(
            week_start=week_start,
            week_end=week_end,
            bots=self._bots,
            gt_start=gt_start,
            gt_end=gt_end,
            composite_delta=composite_delta,
            net_improvement=net_improvement,
            suggestions_proposed=suggestions_proposed,
            suggestions_accepted=suggestions_accepted,
            suggestions_implemented=suggestions_implemented,
            experiments_concluded=experiments_concluded,
            discoveries_found=discoveries_found,
            what_worked=[item.title for item in synthesis.what_worked],
            what_failed=[item.title for item in synthesis.what_failed],
            lessons_for_next_week=synthesis.lessons,
        )

        # Log calibration reliability summary (informational only)
        if self._calibration_tracker:
            self._log_calibration_summary(week_start)

        logger.info(
            "Learning cycle complete: net_improvement=%s, composite_deltas=%s",
            net_improvement, composite_delta,
        )

        return entry

    def _log_calibration_summary(self, week_start: str) -> None:
        """Log per-(bot, category) backtest reliability summary."""
        import json
        summaries: list[dict] = []
        cal_path = self._memory_dir / "findings" / "backtest_calibration.jsonl"
        if not cal_path.exists():
            return
        # Collect unique (bot, category) pairs
        seen: set[tuple[str, str]] = set()
        for line in cal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec.get("bot_id", ""), rec.get("param_category", ""))
                seen.add(key)
            except Exception:
                pass
        for bot_id, category in sorted(seen):
            reliability, n = self._calibration_tracker.get_reliability(bot_id, category)
            if n > 0:
                summaries.append({
                    "bot_id": bot_id,
                    "category": category,
                    "reliability": round(reliability, 2),
                    "sample_count": n,
                })
        if summaries:
            summary_path = self._memory_dir / "findings" / "calibration_summary.jsonl"
            try:
                with summary_path.open("w", encoding="utf-8") as f:
                    for s in summaries:
                        f.write(json.dumps(s) + "\n")
                logger.info("Calibration summary: %d entries written", len(summaries))
            except Exception:
                logger.exception("Failed to write calibration summary")

    # ── Counting helpers ──

    def _count_suggestions(
        self, week_start: str, week_end: str,
    ) -> tuple[int, int, int]:
        """Count suggestions proposed/accepted/implemented in the week."""
        proposed = accepted = implemented = 0
        if not self._suggestion_tracker:
            return proposed, accepted, implemented
        try:
            for s in self._suggestion_tracker.load_all():
                ts = s.get("timestamp", "") or s.get("created_at", "")
                if not ts or not (week_start <= ts[:10] <= week_end):
                    continue
                proposed += 1
                status = s.get("status", "")
                if status in ("accepted", "merged", "deployed", "implemented", "measured"):
                    accepted += 1
                if status in ("implemented", "measured"):
                    implemented += 1
        except Exception:
            logger.warning("Failed to count suggestions for ledger")
        return proposed, accepted, implemented

    def _count_experiments(self, week_start: str, week_end: str) -> int:
        """Count experiments concluded in the week."""
        if not self._experiment_tracker:
            return 0
        try:
            loader = getattr(
                self._experiment_tracker, "load_all",
                getattr(self._experiment_tracker, "_load_all", None),
            )
            if not loader:
                return 0
            count = 0
            for exp in loader():
                concluded = str(getattr(exp, "concluded_at", "") or "")
                if concluded and week_start <= concluded[:10] <= week_end:
                    count += 1
            return count
        except Exception:
            logger.warning("Failed to count experiments for ledger")
            return 0

    @staticmethod
    def _count_jsonl_in_range(
        path: Path, week_start: str, week_end: str,
        *, ts_keys: tuple[str, ...] = ("timestamp",),
    ) -> int:
        """Count JSONL entries whose timestamp falls in [week_start, week_end]."""
        if not path.exists():
            return 0
        count = 0
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = ""
                for k in ts_keys:
                    ts = d.get(k, "") or ""
                    if ts:
                        break
                if ts and len(ts) >= 10 and week_start <= ts[:10] <= week_end:
                    count += 1
        except Exception:
            logger.warning("Failed to count entries in %s", path.name)
        return count

    # ── Experiment selection ──

    def _select_next_experiments(self) -> list[dict]:
        """Select next experiments: max 2 per bot, positive-effectiveness only.

        Skips hypotheses in discarded categories.
        """
        if not self._hypothesis_library or not self._experiment_tracker:
            return []

        # Load discarded categories
        discarded = self._load_discarded_categories()

        try:
            hypotheses = self._hypothesis_library.get_active()
        except Exception:
            return []

        hypotheses.sort(
            key=lambda h: getattr(h, "effectiveness", 0.0), reverse=True,
        )

        # Count already-active experiments per bot
        bot_exp_count: dict[str, int] = {}
        for exp in self._experiment_tracker.get_active_experiments():
            bid = getattr(exp, "bot_id", "") or ""
            if bid:
                bot_exp_count[bid] = bot_exp_count.get(bid, 0) + 1

        selected: list[dict] = []
        for hyp in hypotheses:
            effectiveness = getattr(hyp, "effectiveness", 0.0)
            if effectiveness <= 0:
                continue

            category = getattr(hyp, "category", "")
            hyp_id = getattr(hyp, "hypothesis_id", "") or getattr(hyp, "id", "")

            for bot_id in self._bots:
                if bot_exp_count.get(bot_id, 0) >= 2:
                    continue
                if (bot_id, category) in discarded:
                    continue

                selected.append({
                    "hypothesis_id": hyp_id,
                    "bot_id": bot_id,
                    "category": category,
                    "effectiveness": effectiveness,
                })
                bot_exp_count[bot_id] = bot_exp_count.get(bot_id, 0) + 1

        if selected:
            logger.info("Selected %d next experiments", len(selected))
        return selected

    def _record_selected_experiments(self, selected: list[dict]) -> None:
        """Record selected experiments via experiment_tracker."""
        if not selected or not self._experiment_tracker:
            return
        from schemas.structural_experiment import ExperimentRecord
        import hashlib

        for exp_dict in selected:
            experiment_id = hashlib.sha256(
                f"{exp_dict['hypothesis_id']}:{exp_dict['bot_id']}".encode(),
            ).hexdigest()[:12]
            record = ExperimentRecord(
                experiment_id=experiment_id,
                bot_id=exp_dict["bot_id"],
                title=f"Hypothesis test: {exp_dict.get('category', 'unknown')}",
                hypothesis_id=exp_dict.get("hypothesis_id"),
            )
            try:
                self._experiment_tracker.record_experiment(record)
                logger.info(
                    "Recorded experiment %s for hypothesis %s on bot %s",
                    experiment_id, exp_dict["hypothesis_id"], exp_dict["bot_id"],
                )
            except Exception:
                logger.warning(
                    "Failed to record experiment for hypothesis %s",
                    exp_dict.get("hypothesis_id"),
                )

    def _load_discarded_categories(self) -> set[tuple[str, str]]:
        """Load (bot_id, category) pairs with confidence_multiplier <= 0.3."""
        discarded: set[tuple[str, str]] = set()
        overrides_path = self._memory_dir / "findings" / "category_overrides.jsonl"
        if not overrides_path.exists():
            return discarded
        try:
            for line in overrides_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("confidence_multiplier", 1.0) <= 0.3:
                    discarded.add((
                        entry.get("bot_id", ""),
                        entry.get("category", ""),
                    ))
        except Exception:
            logger.warning("Failed to load category overrides")
        return discarded
