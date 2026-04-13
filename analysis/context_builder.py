# analysis/context_builder.py
"""Generic context builder — DRY policy and corrections loading for all prompt assemblers."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from schemas.memory import MemoryIndex
from schemas.prompt_package import PromptPackage

_POLICY_FILES = ["agent.md", "trading_rules.md", "soul.md"]
_FINDINGS_MAX_AGE_DAYS = 90
_FINDINGS_MAX_ENTRIES = 50


def _parse_timestamp(entry: dict) -> datetime | None:
    """Try to parse a timestamp from common fields."""
    for key in ("timestamp", "created_at", "recorded_at", "date"):
        val = entry.get(key)
        if val and isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                pass
    return None


def _filter_by_bot(entries: list[dict], bot_id: str) -> list[dict]:
    """Filter entries by bot_id. Keeps entries that match or have no bot_id field."""
    if not bot_id:
        return entries
    result = []
    for entry in entries:
        entry_bot = entry.get("bot_id", "") or entry.get("target_id", "")
        # Keep entries that match the bot_id or are bot-agnostic (no bot_id field)
        if not entry_bot or bot_id in entry_bot:
            result.append(entry)
    return result


def _apply_temporal_window(
    entries: list[dict],
    max_age_days: int = _FINDINGS_MAX_AGE_DAYS,
    max_entries: int = _FINDINGS_MAX_ENTRIES,
) -> list[dict]:
    """Sort by recency, exclude entries older than max_age_days, cap at max_entries."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)

    # Separate entries with and without timestamps
    with_ts: list[tuple[datetime, dict]] = []
    without_ts: list[dict] = []

    for entry in entries:
        ts = _parse_timestamp(entry)
        if ts is not None:
            # Make timezone-aware if naive
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                with_ts.append((ts, entry))
        else:
            without_ts.append(entry)

    # Sort by exponential decay score (half-life = 2 weeks, most influential first)
    half_life = 14.0

    def _decay_score(ts: datetime) -> float:
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        return 2.0 ** (-age_days / half_life)

    with_ts.sort(key=lambda x: _decay_score(x[0]), reverse=True)
    result = [e for _, e in with_ts] + without_ts

    return result[:max_entries]


class ContextBuilder:
    """Loads shared context (policies, corrections, metadata) used by all assemblers."""

    def __init__(self, memory_dir: Path, curated_dir: Path | None = None) -> None:
        self._memory_dir = memory_dir
        self._curated_dir = curated_dir

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    def build_system_prompt(self) -> str:
        """Load policy files from memory/policies/v1/ into a system prompt."""
        parts: list[str] = []
        policy_dir = self._memory_dir / "policies" / "v1"
        for name in _POLICY_FILES:
            path = policy_dir / name
            if path.exists():
                parts.append(f"--- {name} ---\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

    def load_corrections(self, bot_id: str = "") -> list[dict]:
        """Load manual corrections from findings/corrections.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        If bot_id is provided, only returns corrections relevant to that bot.
        """
        corrections_path = self._memory_dir / "findings" / "corrections.jsonl"
        if not corrections_path.exists():
            return []
        corrections: list[dict] = []
        for line in corrections_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                corrections.append(json.loads(line))
        filtered = _filter_by_bot(corrections, bot_id) if bot_id else corrections
        return _apply_temporal_window(filtered)

    def load_failure_log(self, bot_id: str = "") -> list[dict]:
        """Load failure log entries from findings/failure-log.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        If bot_id is provided, only returns entries relevant to that bot.
        """
        path = self._memory_dir / "findings" / "failure-log.jsonl"
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))
        filtered = _filter_by_bot(entries, bot_id) if bot_id else entries
        return _apply_temporal_window(filtered)

    def load_rejected_suggestions(self) -> list[dict]:
        """Load rejected suggestions from findings/suggestions.jsonl."""
        path = self._memory_dir / "findings" / "suggestions.jsonl"
        if not path.exists():
            return []
        rejected: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "rejected":
                    rejected.append(rec)
        return rejected

    _QUALITY_RANK = {"high": 3, "medium": 2, "low": 1, "insufficient": 0}

    def load_outcome_measurements(
        self, min_quality: str = "medium",
    ) -> tuple[list[dict], list[dict]]:
        """Load outcome measurements from findings/outcomes.jsonl.

        Deduplicates by suggestion_id (last-write-wins) to prevent
        double-counting from legacy dual-write patterns.

        Filters by measurement_quality: only HIGH/MEDIUM (by default) are
        returned as reliable outcomes. LOW/INSUFFICIENT entries are returned
        separately as low-quality outcomes for spurious_outcomes injection.

        Args:
            min_quality: Minimum quality tier to include ("high", "medium",
                "low", "insufficient"). Defaults to "medium".

        Returns:
            Tuple of (reliable_outcomes, low_quality_outcomes).
            Entries without a measurement_quality field are included in
            reliable_outcomes for backward compatibility.
        """
        path = self._memory_dir / "findings" / "outcomes.jsonl"
        if not path.exists():
            return [], []
        seen: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                sid = entry.get("suggestion_id", "")
                if sid:
                    seen[sid] = entry
                else:
                    seen[id(entry)] = entry  # type: ignore[assignment]

        min_rank = self._QUALITY_RANK.get(min_quality.lower(), 2)
        reliable: list[dict] = []
        low_quality: list[dict] = []
        for entry in seen.values():
            quality = (entry.get("measurement_quality") or "").lower()
            if not quality:
                # No quality field — include for backward compat
                reliable.append(entry)
            elif self._QUALITY_RANK.get(quality, 2) >= min_rank:
                reliable.append(entry)
            else:
                low_quality.append(entry)
        return reliable, low_quality

    def load_allocation_history(self) -> list[dict]:
        """Load allocation history from findings/allocation_history.jsonl.

        Applies temporal decay: sorted by recency, capped at 90 days / 50 entries.
        """
        path = self._memory_dir / "findings" / "allocation_history.jsonl"
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))
        return _apply_temporal_window(entries)

    def list_policy_files(self) -> list[str]:
        """List paths to included policy files (for context_files tracking)."""
        files: list[str] = []
        policy_dir = self._memory_dir / "policies" / "v1"
        for name in _POLICY_FILES:
            path = policy_dir / name
            if path.exists():
                files.append(str(path))
        return files

    def runtime_metadata(self, bot_configs: dict | None = None) -> dict:
        """Return runtime metadata for the prompt package.

        Args:
            bot_configs: Optional dict of ``{bot_id: BotConfig}`` to include
                per-bot timezone information in the metadata.
        """
        now = datetime.now(timezone.utc)
        meta: dict = {
            "assembled_at": now.isoformat(),
            "timezone": "UTC",
        }
        if bot_configs:
            meta["bot_timezones"] = {
                bid: cfg.timezone if hasattr(cfg, "timezone") else "UTC"
                for bid, cfg in bot_configs.items()
            }
        return meta

    @staticmethod
    def check_data_availability(
        index: MemoryIndex | None, bot_id: str, date: str,
    ) -> dict:
        """Check if curated data exists for a bot on a given date.

        Returns dict with: has_curated (bool), available_dates (list[str]).
        If index is None, returns unknown state.
        """
        if index is None:
            return {"has_curated": None, "available_dates": []}

        bot_dates = index.curated_dates_by_bot.get(bot_id, [])
        return {
            "has_curated": date in bot_dates,
            "available_dates": bot_dates,
        }

    def load_session_history(self, session_store, agent_type: str, days: int = 7) -> str:
        """Load recent session summaries as formatted text.

        Args:
            session_store: SessionStore instance.
            agent_type: Type of agent to load history for.
            days: Number of days to look back.

        Returns:
            Formatted string summarizing recent sessions, or empty string.
        """
        try:
            sessions = session_store.get_recent_sessions(agent_type, days=days)
        except Exception:
            return ""
        if not sessions:
            return ""

        formatted_lines = [f"Recent {agent_type} sessions (last {days} days):"]
        for s in sessions[:20]:  # cap to avoid context bloat
            details: list[str] = []
            provider = s.get("provider")
            effective_model = s.get("effective_model")
            if provider and effective_model:
                details.append(f"{provider}/{effective_model}")
            elif provider:
                details.append(str(provider))

            duration = s.get("duration_ms", 0)
            details.append(f"{duration}ms")

            first_output_ms = s.get("first_output_ms")
            if isinstance(first_output_ms, int) and first_output_ms > 0:
                details.append(f"first {first_output_ms}ms")

            tool_call_count = s.get("tool_call_count")
            if isinstance(tool_call_count, int) and tool_call_count > 0:
                details.append(f"tools {tool_call_count}")

            stream_event_count = s.get("stream_event_count")
            if isinstance(stream_event_count, int) and stream_event_count > 0:
                details.append(f"stream {stream_event_count}")

            auth_mode = s.get("auth_mode")
            if auth_mode:
                details.append(str(auth_mode))

            summary = s.get("response_summary", "")[:100]
            formatted_lines.append(
                f"- {s.get('date', '?')}: {', '.join(details)} -- {summary}"
            )
        return "\n".join(formatted_lines)

    def load_pattern_library(self, bot_id: str = "") -> list[dict]:
        """Load cross-bot pattern library entries.

        If bot_id is provided, only returns patterns relevant to that bot.
        """
        try:
            from skills.pattern_library import PatternLibrary

            lib = PatternLibrary(self._memory_dir / "findings")
            if bot_id:
                entries = lib.load_for_bot(bot_id)
            else:
                entries = lib.load_active()
            return [e.model_dump(mode="json") for e in entries]
        except Exception:
            return []

    def load_contradictions(
        self, date: str, bots: list[str], curated_dir: Path,
    ) -> list[dict]:
        """Load temporal contradictions across recent daily reports.

        Returns list of ContradictionItem dicts for prompt injection.
        """
        try:
            from skills.contradiction_detector import ContradictionDetector

            detector = ContradictionDetector(
                date=date, bots=bots, curated_dir=curated_dir,
            )
            report = detector.detect()
            return [item.model_dump(mode="json") for item in report.items]
        except Exception:
            return []

    def load_signal_factor_history(
        self, bot_id: str, date: str, findings_dir: Path,
    ) -> dict:
        """Load rolling signal factor analysis for a bot.

        Returns SignalFactorRollingReport as dict, or empty dict if insufficient data.
        """
        try:
            from skills.signal_factor_tracker import SignalFactorTracker

            tracker = SignalFactorTracker(findings_dir)
            report = tracker.compute_rolling(bot_id, date)
            if not report.factors:
                return {}
            return report.model_dump(mode="json")
        except Exception:
            return {}

    def load_correction_patterns(self) -> list[dict]:
        """Load extracted correction patterns from findings/correction_patterns.jsonl."""
        path = self._memory_dir / "findings" / "correction_patterns.jsonl"
        if not path.exists():
            return []
        patterns: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    patterns.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return patterns

    def load_forecast_meta(self) -> dict:
        """Load forecast meta-analysis from findings/forecast_history.jsonl.

        When prediction verdicts are available, includes empirical calibration
        buckets, ECE, and Brier score. Also includes directional bias analysis.
        """
        try:
            from skills.forecast_tracker import ForecastTracker

            tracker = ForecastTracker(self._memory_dir / "findings")
            records = tracker.load_all()
            if not records:
                return {}

            # Load prediction verdicts for empirical calibration
            verdicts = self._load_prediction_verdicts()

            # Compute directional bias from prediction tracker
            dir_bias: dict[str, dict] = {}
            try:
                from skills.prediction_tracker import PredictionTracker as _PT
                if self._curated_dir:
                    pt = _PT(self._memory_dir / "findings")
                    dir_bias = pt.compute_directional_bias(self._curated_dir)
            except Exception:
                pass

            meta = tracker.compute_meta_analysis(
                prediction_verdicts=verdicts if verdicts else None,
                directional_bias=dir_bias if dir_bias else None,
            )
            return meta.model_dump(mode="json")
        except Exception:
            return {}

    def _load_prediction_verdicts(self) -> list:
        """Load prediction verdicts from the prediction tracker."""
        try:
            from skills.prediction_tracker import PredictionTracker

            tracker = PredictionTracker(self._memory_dir / "findings")
            path = self._memory_dir / "findings" / "predictions.jsonl"
            if not path.exists():
                return []
            predictions = tracker.load_predictions()
            if not predictions or not self._curated_dir:
                return []
            evaluation = tracker.evaluate_predictions(predictions, self._curated_dir)
            return evaluation.verdicts if evaluation else []
        except Exception:
            return []

    def load_active_suggestions(self) -> list[dict]:
        """Load non-rejected suggestions from findings/suggestions.jsonl.

        Returns suggestions with unresolved status,
        applying temporal window (90d, 30-entry cap).
        """
        path = self._memory_dir / "findings" / "suggestions.jsonl"
        if not path.exists():
            return []
        active: list[dict] = []
        active_statuses = {
            "proposed",
            "accepted",
            "merged",
            "deployed",
        }
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                rec = json.loads(line)
                status = rec.get("status", "")
                if status in active_statuses:
                    active.append(rec)
        return _apply_temporal_window(active, max_entries=30)

    def load_category_scorecard(self) -> dict:
        """Load category-level suggestion success rates."""
        try:
            from skills.suggestion_scorer import SuggestionScorer

            scorer = SuggestionScorer(self._memory_dir / "findings")
            scorecard = scorer.compute_scorecard()
            if scorecard.scores:
                return scorecard.model_dump(mode="json")
        except Exception:
            pass
        return {}

    def load_optimization_allocation(self) -> dict:
        """Load per-category value analysis for optimization direction guidance."""
        try:
            from skills.suggestion_scorer import SuggestionScorer

            scorer = SuggestionScorer(self._memory_dir / "findings")
            value_map = scorer.compute_category_value_map()
            if value_map and len(value_map) > 1:  # more than just _recommendations
                return value_map
        except Exception:
            pass
        return {}

    def load_regime_stratified_scores(self) -> dict | None:
        """Load category win rates stratified by macro regime."""
        try:
            from skills.suggestion_scorer import SuggestionScorer
            scorer = SuggestionScorer(self._memory_dir / "findings")
            scores = scorer.compute_regime_stratified_scores()
            return scores if scores else None
        except Exception:
            return None

    def load_search_signal_summary(self) -> dict:
        """Load search signal approve/discard summary from search_signals.jsonl."""
        path = self._memory_dir / "findings" / "search_signals.jsonl"
        if not path.exists():
            return {}
        from collections import defaultdict
        counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"approve": 0, "discard": 0}
        )
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                bot_id = rec.get("bot_id", "")
                category = rec.get("category", "")
                key = (bot_id, category)
                if rec.get("positive"):
                    counts[key]["approve"] += 1
                else:
                    counts[key]["discard"] += 1
        except (OSError, json.JSONDecodeError):
            return {}

        if not counts:
            return {}

        summary: dict[str, dict] = {}
        for (bot_id, category), c in counts.items():
            total = c["approve"] + c["discard"]
            summary[f"{bot_id}:{category}"] = {
                "approve_count": c["approve"],
                "discard_count": c["discard"],
                "approve_rate": round(c["approve"] / total, 3) if total > 0 else 0.0,
            }
        return summary

    def load_prediction_accuracy(self) -> dict:
        """Load per-metric prediction accuracy from the prediction tracker.

        When curated_dir is available, computes real accuracy by evaluating predictions
        against actual curated data. Otherwise returns prediction count metadata.
        """
        try:
            from skills.prediction_tracker import PredictionTracker

            tracker = PredictionTracker(self._memory_dir / "findings")
            if not (self._memory_dir / "findings" / "predictions.jsonl").exists():
                return {}
            predictions = tracker.load_predictions()
            if not predictions:
                return {}

            # When curated_dir available, compute real per-metric accuracy
            if self._curated_dir and self._curated_dir.exists():
                accuracy_by_metric = tracker.get_accuracy_by_metric(self._curated_dir)
                if accuracy_by_metric:
                    return {
                        "has_predictions": True,
                        "count": len(predictions),
                        "accuracy_by_metric": accuracy_by_metric,
                    }

            return {"has_predictions": True, "count": len(predictions)}
        except Exception:
            return {}

    def load_hypothesis_track_record(self) -> dict:
        """Load hypothesis effectiveness scores for prompt injection."""
        try:
            from skills.hypothesis_library import HypothesisLibrary

            lib = HypothesisLibrary(self._memory_dir / "findings")
            track = lib.get_track_record()
            if track:
                return track
        except Exception:
            pass
        return {}

    def load_transfer_track_record(self) -> dict:
        """Load transfer outcome success rates for prompt injection."""
        try:
            from skills.transfer_proposal_builder import TransferProposalBuilder

            return TransferProposalBuilder.load_track_record_from_file(
                self._memory_dir / "findings",
            )
        except Exception:
            return {}

    def load_experiment_track_record(self) -> dict:
        """Load structural experiment pass/fail track record."""
        try:
            from skills.structural_experiment_tracker import StructuralExperimentTracker

            tracker = StructuralExperimentTracker(self._memory_dir / "findings")
            record = tracker.compute_track_record()
            if record.get("total", 0) > 0:
                return record
        except Exception:
            pass
        return {}

    def load_recalibrations(self) -> list[dict]:
        """Load causal recalibrations from findings/recalibrations.jsonl.

        Returns recalibrations with bot_id, category, revised_confidence,
        and lessons_learned, filtered by temporal window (90d, 30-entry cap).
        """
        path = self._memory_dir / "findings" / "recalibrations.jsonl"
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return _apply_temporal_window(entries, max_entries=30)

    def load_outcome_reasonings(self) -> list[dict]:
        """Load causal outcome reasonings from findings/outcome_reasonings.jsonl.

        Returns recent reasonings with lessons learned, mechanisms, and
        transferability assessments for injection into prompts.
        """
        path = self._memory_dir / "findings" / "outcome_reasonings.jsonl"
        if not path.exists():
            return []
        reasonings: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    reasonings.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return _apply_temporal_window(reasonings, max_entries=20)

    def load_discoveries(self) -> list[dict]:
        """Load discoveries from findings/discoveries.jsonl."""
        path = self._memory_dir / "findings" / "discoveries.jsonl"
        if not path.exists():
            return []
        discoveries: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    discoveries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return _apply_temporal_window(discoveries, max_entries=20)

    def load_active_experiments(self) -> list[dict]:
        """Load active structural experiments for prompt injection."""
        try:
            from skills.structural_experiment_tracker import StructuralExperimentTracker

            tracker = StructuralExperimentTracker(self._memory_dir / "findings")
            active = tracker.get_active_experiments()
            return [e.model_dump(mode="json") for e in active]
        except Exception:
            return []

    def load_reliability_summary(self) -> dict:
        """Load reliability tracking summary from findings."""
        try:
            from skills.reliability_tracker import ReliabilityTracker

            tracker = ReliabilityTracker(self._memory_dir / "findings")
            summary = tracker.compute_summary()
            if summary.scorecards_by_class:
                return summary.model_dump(mode="json")
        except Exception:
            pass
        return {}

    def load_validation_patterns(self) -> dict:
        """Load aggregated validation patterns from findings/validation_log.jsonl.

        Groups blocked suggestions by category over the last 30 days.
        Returns summary dict: {"category": {"blocked_count": N, "common_reasons": [...]}}
        """
        path = self._memory_dir / "findings" / "validation_log.jsonl"
        if not path.exists():
            return {}
        try:
            from collections import defaultdict

            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            category_blocks: dict[str, list[str]] = defaultdict(list)

            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                ts = entry.get("timestamp", "")
                if ts:
                    try:
                        entry_time = datetime.fromisoformat(ts)
                        if entry_time.tzinfo is None:
                            entry_time = entry_time.replace(tzinfo=timezone.utc)
                        if entry_time < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass
                for detail in entry.get("blocked_details", []):
                    bot_id = detail.get("bot_id", "")
                    reason = detail.get("reason", "")
                    # Infer category from reason or title
                    title = detail.get("title", "").lower()
                    for keyword, cat in [
                        ("exit", "exit_timing"), ("filter", "filter_threshold"),
                        ("stop", "stop_loss"), ("signal", "signal"),
                        ("regime", "regime_gate"), ("sizing", "position_sizing"),
                    ]:
                        if keyword in title:
                            category_blocks[cat].append(reason)
                            break
                    else:
                        category_blocks["other"].append(reason)

            if not category_blocks:
                return {}

            result: dict = {}
            for cat, reasons in category_blocks.items():
                # Deduplicate and count
                unique_reasons = list(set(reasons))[:5]
                result[cat] = {
                    "blocked_count": len(reasons),
                    "common_reasons": unique_reasons,
                }
            return result
        except Exception:
            return {}

    def load_threshold_profile(self) -> dict:
        """Load learned threshold profiles from findings/learned_thresholds.jsonl.

        Returns a dict with per-bot threshold data when available.
        """
        path = self._memory_dir / "findings" / "learned_thresholds.jsonl"
        if not path.exists():
            return {}
        try:
            profiles: list[dict] = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    profiles.append(json.loads(line))
            if profiles:
                return {"profiles": profiles, "count": len(profiles)}
        except Exception:
            pass
        return {}

    def load_ground_truth_trend(self) -> dict:
        """Load ground truth composite score trend from learning_ledger.jsonl.

        Returns last 12 weeks of composite scores per bot, recent lessons,
        and curated analysis notes (deduplicated, relevance-decayed, outcome-boosted).
        """
        try:
            from skills.learning_ledger import LearningLedger

            ledger = LearningLedger(self._memory_dir / "findings")
            trend = ledger.get_trend(weeks=12)
            lessons = ledger.get_lessons(weeks=4)
            curated_notes = ledger.get_curated_notes(max_notes=30)
            if not trend and not lessons and not curated_notes:
                return {}
            result: dict = {}
            if trend:
                result["composite_trend"] = trend
            if lessons:
                result["recent_lessons"] = lessons
            if curated_notes:
                result["curated_analysis_notes"] = curated_notes
            try:
                latest = ledger.get_latest()
                if latest:
                    result["net_improvement"] = latest.net_improvement
                    result["composite_delta"] = latest.composite_delta
            except Exception:
                # Graceful: latest may fail if ledger entries have incomplete GT data
                # Still return trend + lessons
                pass
            return result
        except Exception:
            return {}

    def load_cycle_effectiveness(self) -> list[dict]:
        """Load last 8 cycle effectiveness entries from learning_ledger.jsonl."""
        path = self._memory_dir / "findings" / "learning_ledger.jsonl"
        if not path.exists():
            return []
        entries: list[dict] = []
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        except Exception:
            return []
        entries.sort(key=lambda e: e.get("week_start", ""))
        recent = entries[-8:]
        result = []
        for e in recent:
            ce = e.get("cycle_effectiveness", 0.0)
            if ce > 0 or e.get("suggestions_proposed", 0) > 0:
                result.append({
                    "week": e.get("week_start", ""),
                    "effectiveness": ce,
                    "net_improvement": e.get("net_improvement", False),
                    "suggestions_proposed": e.get("suggestions_proposed", 0),
                    "suggestions_implemented": e.get("suggestions_implemented", 0),
                })
        return result

    def load_suggestion_quality_trend(self, value_map: dict | None = None) -> dict:
        """Load suggestion quality trend from SuggestionScorer."""
        try:
            from skills.suggestion_scorer import SuggestionScorer
            scorer = SuggestionScorer(self._memory_dir / "findings")
            return scorer.compute_suggestion_quality_trend(value_map=value_map)
        except Exception:
            return {}

    def load_convergence_report(self) -> dict:
        """Load convergence report synthesising learning loop health."""
        try:
            from skills.convergence_tracker import ConvergenceTracker

            tracker = ConvergenceTracker(self._memory_dir / "findings")
            report = tracker.compute_report(weeks=12)
            # Only include if we have real data (not all insufficient_data)
            if report.overall_status.value == "insufficient_data":
                return {}
            return report.model_dump(mode="json")
        except Exception:
            return {}

    def load_instrumentation_readiness(self, bots: list[str]) -> dict:
        """Load per-bot instrumentation readiness scorecards."""
        if not self._curated_dir or not bots:
            return {}
        try:
            from skills.instrumentation_scorer import InstrumentationScorer

            scorer = InstrumentationScorer(self._curated_dir, lookback_days=30)
            reports = scorer.score_all_bots(bots)
            return {
                bot_id: report.model_dump(mode="json")
                for bot_id, report in reports.items()
                if report.days_with_data > 0
            }
        except Exception:
            return {}

    def load_retrospective_synthesis(self) -> dict:
        """Load most recent retrospective synthesis from findings."""
        path = self._memory_dir / "findings" / "retrospective_synthesis.jsonl"
        if not path.exists():
            return {}
        try:
            entries: list[dict] = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
            if entries:
                return entries[-1]  # most recent
        except Exception:
            pass
        return {}

    def load_spurious_outcomes(self) -> list[dict]:
        """Load outcomes determined to be spurious (not genuinely caused by the suggestion)."""
        path = self._memory_dir / "findings" / "spurious_outcomes.jsonl"
        if not path.exists():
            return []
        try:
            entries: list[dict] = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
            return entries
        except Exception:
            return []

    def load_strategy_ideas(self) -> list[dict]:
        """Load strategy ideas from findings/strategy_ideas.jsonl.

        Returns active (non-retired) strategy ideas with temporal window.
        """
        path = self._memory_dir / "findings" / "strategy_ideas.jsonl"
        if not path.exists():
            return []
        try:
            ideas: list[dict] = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("status", "proposed") != "retired":
                        ideas.append(entry)
            return _apply_temporal_window(ideas, max_entries=10)
        except Exception:
            return []

    def load_portfolio_outcomes(self) -> list[dict]:
        """Load portfolio-level suggestion outcomes from findings/portfolio_outcomes.jsonl.

        Returns recent portfolio change outcomes with verdicts and composite deltas.
        """
        path = self._memory_dir / "findings" / "portfolio_outcomes.jsonl"
        if not path.exists():
            return []
        try:
            entries: list[dict] = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
            return _apply_temporal_window(entries, max_entries=20)
        except Exception:
            return []

    def load_portfolio_metrics(self) -> dict:
        """Load latest portfolio rolling metrics from curated data.

        Returns the most recent portfolio_rolling_metrics.json if available.
        """
        if not self._curated_dir:
            return {}
        try:
            # Find most recent date directory with portfolio metrics
            portfolio_dirs = sorted(
                self._curated_dir.glob("*/portfolio/portfolio_rolling_metrics.json"),
                reverse=True,
            )
            if portfolio_dirs:
                return json.loads(portfolio_dirs[0].read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def load_consolidated_patterns(self) -> str:
        """Load patterns_consolidated.md if it exists."""
        path = self._memory_dir / "findings" / "patterns_consolidated.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def load_search_reports(self, bot_id: str = "", lookback_n: int = 5) -> list[dict]:
        """Recent parameter search reports: what params were explored, routing decisions."""
        path = self._memory_dir / "findings" / "search_reports.jsonl"
        if not path.exists():
            return []
        try:
            reports: list[dict] = []
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if bot_id and entry.get("bot_id") != bot_id:
                    continue
                # Strip large candidate arrays, keep summary fields
                reports.append({
                    "suggestion_id": entry.get("suggestion_id"),
                    "bot_id": entry.get("bot_id"),
                    "param_name": entry.get("param_name"),
                    "routing": entry.get("routing"),
                    "best_value": entry.get("best_value"),
                    "discard_reason": entry.get("discard_reason", ""),
                    "exploration_summary": entry.get("exploration_summary", ""),
                    "searched_at": entry.get("searched_at", ""),
                })
            return reports[-lookback_n:]
        except Exception:
            return []

    def load_backtest_reliability(self, bot_id: str = "") -> dict[str, float]:
        """Per-category backtest reliability ratios (categories with n >= 3)."""
        path = self._memory_dir / "findings" / "backtest_calibration.jsonl"
        if not path.exists():
            return {}
        try:
            from collections import defaultdict
            correct: dict[str, int] = defaultdict(int)
            total: dict[str, int] = defaultdict(int)
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if bot_id and entry.get("bot_id") != bot_id:
                    continue
                cat = entry.get("param_category", "")
                if entry.get("prediction_correct") is not None:
                    total[cat] += 1
                    if entry.get("prediction_correct"):
                        correct[cat] += 1
            return {
                cat: round(correct[cat] / total[cat], 2)
                for cat in total
                if total[cat] >= 3
            }
        except Exception:
            return {}

    def build_self_assessment(
        self,
        forecast_meta: dict | None = None,
        category_scorecard: dict | None = None,
        correction_patterns: list[dict] | None = None,
        recalibrations: list[dict] | None = None,
    ) -> str:
        """Synthesize a plain-text self-assessment from multiple learning signals.

        Combines directional biases, calibration state, category strengths/weaknesses,
        recurring corrections, and causal lessons into a narrative summary.
        Returns empty string if fewer than 2 signals are available.

        When called from base_package(), pre-loaded data is passed to avoid
        duplicate I/O. When called standalone, loads data on demand.
        """
        if forecast_meta is None:
            forecast_meta = self.load_forecast_meta()
        if category_scorecard is None:
            category_scorecard = self.load_category_scorecard()
        if correction_patterns is None:
            correction_patterns = self.load_correction_patterns()
        if recalibrations is None:
            recalibrations = self.load_recalibrations()

        signals: list[str] = []

        # 1. Directional biases from forecast meta
        dir_bias = forecast_meta.get("directional_bias", {})
        if dir_bias:
            bias_lines = []
            for metric, info in dir_bias.items():
                bias = info.get("bias", "balanced")
                if bias != "balanced":
                    mag = info.get("bias_magnitude", 0)
                    bias_lines.append(
                        f"  - {metric}: {bias} (magnitude {mag:.2f})"
                    )
            if bias_lines:
                signals.append(
                    "Directional biases:\n" + "\n".join(bias_lines)
                )

        # 2. Calibration state
        ece = forecast_meta.get("expected_calibration_error")
        if ece is not None:
            cal_adj = forecast_meta.get("calibration_adjustment", 0)
            if cal_adj < -0.1:
                direction = "overconfident"
            elif cal_adj > 0.1:
                direction = "underconfident"
            else:
                direction = "reasonably calibrated"
            signals.append(f"Calibration: ECE={ece:.3f}, {direction}")

        # 3. Category strengths/weaknesses from scorecard
        scores = category_scorecard.get("scores", [])
        if scores:
            strong = []
            weak = []
            for s in scores:
                wr = s.get("win_rate", 0)
                n = s.get("sample_size", 0)
                if n < 3:
                    continue
                label = f"{s.get('bot_id', '?')}/{s.get('category', '?')} ({wr:.0%}, n={n})"
                if wr >= 0.6:
                    strong.append(label)
                elif wr < 0.4:
                    weak.append(label)
            if strong:
                signals.append("Strong categories: " + ", ".join(strong[:5]))
            if weak:
                signals.append("Weak categories (avoid or justify): " + ", ".join(weak[:5]))

        # 4. Recurring corrections (top 3 by count)
        if correction_patterns:
            sorted_patterns = sorted(correction_patterns, key=lambda p: p.get("count", 0), reverse=True)
            top = sorted_patterns[:3]
            lines = [
                f"  - {p.get('description', '?')} (count={p.get('count', 0)})"
                for p in top
            ]
            signals.append("Recurring corrections:\n" + "\n".join(lines))

        # 5. Causal lessons from recalibrations
        if recalibrations:
            all_lessons: list[str] = []
            seen: set[str] = set()
            for r in recalibrations:
                for lesson in r.get("lessons_learned", []):
                    if lesson and lesson not in seen:
                        seen.add(lesson)
                        all_lessons.append(lesson)
            if all_lessons:
                signals.append(
                    "Causal lessons learned:\n"
                    + "\n".join(f"  - {l}" for l in all_lessons[:5])
                )

        if len(signals) < 2:
            return ""

        return "SELF-ASSESSMENT (auto-synthesized from learning data):\n\n" + "\n\n".join(signals)

    def load_macro_regime_context(self) -> dict:
        """Load latest macro regime state from curated portfolio data.

        Looks for macro_regime_analysis.json in the most recent curated portfolio dir.
        """
        if not self._curated_dir:
            return {}
        try:
            # Find most recent date dir with portfolio data
            curated = Path(self._curated_dir)
            if not curated.exists():
                return {}
            date_dirs = sorted(
                [d for d in curated.iterdir() if d.is_dir() and not d.name.startswith(".")],
                reverse=True,
            )
            for date_dir in date_dirs[:7]:  # check last 7 days
                regime_file = date_dir / "portfolio" / "macro_regime_analysis.json"
                if regime_file.exists():
                    data = json.loads(regime_file.read_text(encoding="utf-8"))
                    if data:
                        return data
        except Exception:
            pass
        return {}

    def load_regime_config_history(self) -> list[dict]:
        """Load rolling regime config from recent curated bot dirs.

        Collects applied_regime_config.json from the last 30 days of curated data.
        """
        if not self._curated_dir:
            return []
        try:
            curated = Path(self._curated_dir)
            if not curated.exists():
                return []
            date_dirs = sorted(
                [d for d in curated.iterdir() if d.is_dir() and not d.name.startswith(".")],
                reverse=True,
            )
            history: list[dict] = []
            for date_dir in date_dirs[:30]:
                for bot_dir in date_dir.iterdir():
                    if not bot_dir.is_dir() or bot_dir.name == "portfolio":
                        continue
                    config_file = bot_dir / "applied_regime_config.json"
                    if config_file.exists():
                        data = json.loads(config_file.read_text(encoding="utf-8"))
                        if data:
                            history.append({
                                "date": date_dir.name,
                                "bot_id": bot_dir.name,
                                **data,
                            })
            return history
        except Exception:
            return []

    # Priority order for context items (highest value first).
    # Items not in this list get lowest priority.
    _CONTEXT_PRIORITY: list[str] = [
        # Core context — always include when available
        "ground_truth_trend",
        "portfolio_outcomes",
        "portfolio_rolling_metrics",
        "macro_regime_context",
        "self_assessment",
        "convergence_report",
        "strategy_profiles",
        "archetype_expectations",
        "coordination_rules",
        "portfolio_risk_config",
        "last_week_synthesis",
        # Learning signals — high value for improvement
        "active_suggestions",
        "rejected_suggestions",
        "category_scorecard",
        "regime_stratified_scores",
        "prediction_accuracy_by_metric",
        "outcome_measurements",
        "forecast_meta_analysis",
        "correction_patterns",
        "validation_patterns",
        "active_experiments",
        "backtest_reliability",
        "regime_config_history",
        "transfer_track_record",
        "cycle_effectiveness_trend",
        "suggestion_quality_trend",
        "optimization_allocation",
        "search_signal_summary",
        "search_reports",
        "hypothesis_track_record",
        "discoveries",
        "strategy_ideas",
        # Lower-priority learning context
        "outcome_reasonings",
        "recalibrations",
        "threshold_profile",
        "experiment_track_record",
        "consolidated_patterns",
        "spurious_outcomes",
        "pattern_library",
        "failure_log",
        "reliability_summary",
        "instrumentation_readiness",
        "allocation_history",
        "session_history",
    ]

    def base_package(
        self,
        session_store=None,
        agent_type: str = "",
        bot_configs: dict | None = None,
        context_budget_items: int = 15,
        strategy_registry=None,
    ) -> PromptPackage:
        """Build a PromptPackage pre-filled with system prompt, corrections, and metadata.

        Args:
            session_store: Optional SessionStore for loading session history.
            agent_type: Agent type for session history filtering.
            bot_configs: Optional ``{bot_id: BotConfig}`` for timezone metadata.
        """
        failure_log = self.load_failure_log()
        rejected_suggestions = self.load_rejected_suggestions()
        outcome_measurements, low_quality_outcomes = self.load_outcome_measurements()
        allocation_history = self.load_allocation_history()
        consolidated_patterns = self.load_consolidated_patterns()
        data: dict = {}
        if failure_log:
            data["failure_log"] = failure_log
        if rejected_suggestions:
            data["rejected_suggestions"] = rejected_suggestions
        if outcome_measurements:
            data["outcome_measurements"] = outcome_measurements
        if allocation_history:
            data["allocation_history"] = allocation_history
        if consolidated_patterns:
            data["consolidated_patterns"] = consolidated_patterns
        pattern_library = self.load_pattern_library()
        if pattern_library:
            data["pattern_library"] = pattern_library
        correction_patterns = self.load_correction_patterns()
        if correction_patterns:
            data["correction_patterns"] = correction_patterns
        forecast_meta = self.load_forecast_meta()
        if forecast_meta:
            data["forecast_meta_analysis"] = forecast_meta
        active_suggestions = self.load_active_suggestions()
        if active_suggestions:
            data["active_suggestions"] = active_suggestions
        category_scorecard = self.load_category_scorecard()
        if category_scorecard:
            data["category_scorecard"] = category_scorecard
        regime_stratified_scores = self.load_regime_stratified_scores()
        if regime_stratified_scores:
            data["regime_stratified_scores"] = regime_stratified_scores
        prediction_accuracy = self.load_prediction_accuracy()
        if prediction_accuracy:
            data["prediction_accuracy_by_metric"] = prediction_accuracy
        hypothesis_track_record = self.load_hypothesis_track_record()
        if hypothesis_track_record:
            data["hypothesis_track_record"] = hypothesis_track_record
        transfer_track_record = self.load_transfer_track_record()
        if transfer_track_record:
            data["transfer_track_record"] = transfer_track_record
        validation_patterns = self.load_validation_patterns()
        if validation_patterns:
            data["validation_patterns"] = validation_patterns
        threshold_profile = self.load_threshold_profile()
        if threshold_profile:
            data["threshold_profile"] = threshold_profile
        reliability_summary = self.load_reliability_summary()
        if reliability_summary:
            data["reliability_summary"] = reliability_summary
        experiment_track_record = self.load_experiment_track_record()
        if experiment_track_record:
            data["experiment_track_record"] = experiment_track_record
        active_experiments = self.load_active_experiments()
        if active_experiments:
            data["active_experiments"] = active_experiments
        outcome_reasonings = self.load_outcome_reasonings()
        if outcome_reasonings:
            data["outcome_reasonings"] = outcome_reasonings
        recalibrations = self.load_recalibrations()
        if recalibrations:
            data["recalibrations"] = recalibrations
        discoveries = self.load_discoveries()
        if discoveries:
            data["discoveries"] = discoveries
        optimization_allocation = self.load_optimization_allocation()
        if optimization_allocation:
            data["optimization_allocation"] = optimization_allocation
        search_signal_summary = self.load_search_signal_summary()
        if search_signal_summary:
            data["search_signal_summary"] = search_signal_summary
        ground_truth_trend = self.load_ground_truth_trend()
        if ground_truth_trend:
            data["ground_truth_trend"] = ground_truth_trend
        self_assessment = self.build_self_assessment(
            forecast_meta=forecast_meta,
            category_scorecard=category_scorecard,
            correction_patterns=correction_patterns,
            recalibrations=recalibrations,
        )
        if self_assessment:
            data["self_assessment"] = self_assessment
        convergence_report = self.load_convergence_report()
        if convergence_report:
            data["convergence_report"] = convergence_report
        if bot_configs:
            instrumentation = self.load_instrumentation_readiness(
                list(bot_configs.keys()),
            )
            if instrumentation:
                data["instrumentation_readiness"] = instrumentation
        cycle_effectiveness = self.load_cycle_effectiveness()
        if cycle_effectiveness:
            data["cycle_effectiveness_trend"] = cycle_effectiveness
        suggestion_quality_trend = self.load_suggestion_quality_trend(
            value_map=optimization_allocation,
        )
        if suggestion_quality_trend:
            data["suggestion_quality_trend"] = suggestion_quality_trend
        retrospective_synthesis = self.load_retrospective_synthesis()
        if retrospective_synthesis:
            data["last_week_synthesis"] = retrospective_synthesis
        spurious_outcomes = self.load_spurious_outcomes()
        # Merge low-quality outcome measurements into spurious_outcomes
        all_spurious = spurious_outcomes + low_quality_outcomes
        if all_spurious:
            data["spurious_outcomes"] = all_spurious
        strategy_ideas = self.load_strategy_ideas()
        if strategy_ideas:
            data["strategy_ideas"] = strategy_ideas
        search_reports = self.load_search_reports()
        if search_reports:
            data["search_reports"] = search_reports
        backtest_reliability = self.load_backtest_reliability()
        if backtest_reliability:
            data["backtest_reliability"] = backtest_reliability
        portfolio_outcomes = self.load_portfolio_outcomes()
        if portfolio_outcomes:
            data["portfolio_outcomes"] = portfolio_outcomes
        portfolio_metrics = self.load_portfolio_metrics()
        if portfolio_metrics:
            data["portfolio_rolling_metrics"] = portfolio_metrics
        macro_regime = self.load_macro_regime_context()
        if macro_regime:
            data["macro_regime_context"] = macro_regime
        regime_config_history = self.load_regime_config_history()
        if regime_config_history:
            data["regime_config_history"] = regime_config_history
        if session_store and agent_type:
            session_history = self.load_session_history(session_store, agent_type)
            if session_history:
                data["session_history"] = session_history

        # Inject strategy registry data if available
        if strategy_registry and getattr(strategy_registry, "strategies", None):
            data["strategy_profiles"] = {
                sid: profile.model_dump(mode="json", exclude_unset=True)
                for sid, profile in strategy_registry.strategies.items()
            }
            if strategy_registry.coordination.signals or strategy_registry.coordination.cooldown_pairs:
                data["coordination_rules"] = strategy_registry.coordination.model_dump(mode="json")
            if strategy_registry.archetype_expectations:
                data["archetype_expectations"] = {
                    k: v.model_dump(mode="json")
                    for k, v in strategy_registry.archetype_expectations.items()
                }
            if strategy_registry.portfolio.heat_cap_R > 0:
                data["portfolio_risk_config"] = strategy_registry.portfolio.model_dump(mode="json")

        # Apply context budget — adaptive: expand default budget up to 25 when
        # many items have data, but never exceed an explicitly set budget.
        # The default of 15 auto-expands; explicit low values are respected.
        total_available = len(data)
        if context_budget_items == 15:
            # Default budget — allow adaptive expansion
            effective_budget = max(context_budget_items, min(total_available, 25))
        else:
            effective_budget = context_budget_items

        omitted_keys: list[str] = []
        if total_available > effective_budget:
            priority_set = set(self._CONTEXT_PRIORITY)
            # Split into prioritized and unprioritized
            prioritized = {
                k: v for k, v in data.items() if k in priority_set
            }
            unprioritized = {
                k: v for k, v in data.items() if k not in priority_set
            }
            # Sort prioritized by their order in _CONTEXT_PRIORITY
            priority_order = {
                k: i for i, k in enumerate(self._CONTEXT_PRIORITY)
            }
            sorted_keys = sorted(
                prioritized.keys(),
                key=lambda k: priority_order.get(k, 999),
            )
            # Take top N from prioritized, fill remainder with unprioritized
            budget_keys = sorted_keys[:effective_budget]
            remaining = effective_budget - len(budget_keys)
            if remaining > 0:
                budget_keys.extend(list(unprioritized.keys())[:remaining])
            dropped = set(data.keys()) - set(budget_keys)
            omitted_keys = sorted(dropped)
            if dropped:
                logger.warning(
                    "Context budget: dropped %d low-priority items: %s",
                    len(dropped), omitted_keys,
                )
            data = {k: data[k] for k in budget_keys if k in data}

        metadata = self.runtime_metadata(bot_configs=bot_configs)
        metadata["_context_budget_manifest"] = {
            "included": sorted(data.keys()),
            "omitted": omitted_keys,
            "total_available": total_available,
        }

        return PromptPackage(
            system_prompt=self.build_system_prompt(),
            corrections=self.load_corrections(),
            context_files=self.list_policy_files(),
            metadata=metadata,
            data=data,
        )
