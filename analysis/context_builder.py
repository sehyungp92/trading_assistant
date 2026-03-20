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

_POLICY_FILES = ["agents.md", "trading_rules.md", "soul.md"]
_FINDINGS_MAX_AGE_DAYS = 90
_FINDINGS_MAX_ENTRIES = 50


def _parse_timestamp(entry: dict) -> datetime | None:
    """Try to parse a timestamp from common fields."""
    for key in ("timestamp", "created_at", "date"):
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

    def load_outcome_measurements(self) -> list[dict]:
        """Load outcome measurements from findings/outcomes.jsonl.

        Deduplicates by suggestion_id (last-write-wins) to prevent
        double-counting from legacy dual-write patterns.
        """
        path = self._memory_dir / "findings" / "outcomes.jsonl"
        if not path.exists():
            return []
        seen: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                sid = entry.get("suggestion_id", "")
                if sid:
                    seen[sid] = entry
                else:
                    seen[id(entry)] = entry  # type: ignore[assignment]
        return list(seen.values())

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
        buckets, ECE, and Brier score.
        """
        try:
            from skills.forecast_tracker import ForecastTracker

            tracker = ForecastTracker(self._memory_dir / "findings")
            records = tracker.load_all()
            if not records:
                return {}

            # Load prediction verdicts for empirical calibration
            verdicts = self._load_prediction_verdicts()
            meta = tracker.compute_meta_analysis(
                prediction_verdicts=verdicts if verdicts else None,
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

        Returns last 12 weeks of composite scores per bot, plus recent lessons.
        """
        try:
            from skills.learning_ledger import LearningLedger

            ledger = LearningLedger(self._memory_dir / "findings")
            trend = ledger.get_trend(weeks=12)
            lessons = ledger.get_lessons(weeks=4)
            if not trend and not lessons:
                return {}
            result: dict = {}
            if trend:
                result["composite_trend"] = trend
            if lessons:
                result["recent_lessons"] = lessons
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

    # Priority order for context items (highest value first).
    # Items not in this list get lowest priority.
    _CONTEXT_PRIORITY: list[str] = [
        "ground_truth_trend",
        "last_week_synthesis",
        "active_suggestions",
        "category_scorecard",
        "prediction_accuracy_by_metric",
        "outcome_measurements",
        "forecast_meta_analysis",
        "active_experiments",
        "backtest_reliability",
        "search_reports",
        "hypothesis_track_record",
        "discoveries",
        "strategy_ideas",
        "outcome_reasonings",
        "spurious_outcomes",
        "pattern_library",
        "reliability_summary",
    ]

    def base_package(
        self,
        session_store=None,
        agent_type: str = "",
        bot_configs: dict | None = None,
        context_budget_items: int = 15,
    ) -> PromptPackage:
        """Build a PromptPackage pre-filled with system prompt, corrections, and metadata.

        Args:
            session_store: Optional SessionStore for loading session history.
            agent_type: Agent type for session history filtering.
            bot_configs: Optional ``{bot_id: BotConfig}`` for timezone metadata.
        """
        failure_log = self.load_failure_log()
        rejected_suggestions = self.load_rejected_suggestions()
        outcome_measurements = self.load_outcome_measurements()
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
        discoveries = self.load_discoveries()
        if discoveries:
            data["discoveries"] = discoveries
        ground_truth_trend = self.load_ground_truth_trend()
        if ground_truth_trend:
            data["ground_truth_trend"] = ground_truth_trend
        retrospective_synthesis = self.load_retrospective_synthesis()
        if retrospective_synthesis:
            data["last_week_synthesis"] = retrospective_synthesis
        spurious_outcomes = self.load_spurious_outcomes()
        if spurious_outcomes:
            data["spurious_outcomes"] = spurious_outcomes
        strategy_ideas = self.load_strategy_ideas()
        if strategy_ideas:
            data["strategy_ideas"] = strategy_ideas
        search_reports = self.load_search_reports()
        if search_reports:
            data["search_reports"] = search_reports
        backtest_reliability = self.load_backtest_reliability()
        if backtest_reliability:
            data["backtest_reliability"] = backtest_reliability
        if session_store and agent_type:
            session_history = self.load_session_history(session_store, agent_type)
            if session_history:
                data["session_history"] = session_history

        # Apply context budget — keep highest-priority items within limit
        if len(data) > context_budget_items:
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
            budget_keys = sorted_keys[:context_budget_items]
            remaining = context_budget_items - len(budget_keys)
            if remaining > 0:
                budget_keys.extend(list(unprioritized.keys())[:remaining])
            dropped = set(data.keys()) - set(budget_keys)
            if dropped:
                logger.warning(
                    "Context budget: dropped %d low-priority items: %s",
                    len(dropped), sorted(dropped),
                )
            data = {k: data[k] for k in budget_keys if k in data}

        return PromptPackage(
            system_prompt=self.build_system_prompt(),
            corrections=self.load_corrections(),
            context_files=self.list_policy_files(),
            metadata=self.runtime_metadata(bot_configs=bot_configs),
            data=data,
        )
