"""Weekly provider scoring and safe learned-routing recommendations."""
from __future__ import annotations

import json
from pathlib import Path


class ProviderRouteScorer:
    """Compute workflow-specific provider scores from attributed learning data."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = Path(findings_dir)
        self._path = self._findings_dir / "provider_route_scores.jsonl"

    def recompute(self) -> list[dict]:
        suggestions = self._load_suggestions_by_id()
        stats: dict[tuple[str, str, str], dict] = {}

        def _bucket(workflow: str, provider: str, model: str) -> dict:
            key = (workflow, provider, model)
            return stats.setdefault(key, {
                "validation": [],
                "validation_weight": [],
                "outcomes": [],
                "outcome_weight": [],
                "calibration": [],
            })

        for entry in self._load_jsonl("validation_log.jsonl"):
            workflow = str(entry.get("agent_type", "") or "").strip()
            provider = str(entry.get("provider", "") or "").strip()
            model = str(entry.get("model", "") or "").strip()
            total = int(entry.get("approved_count", 0) + entry.get("blocked_count", 0))
            if not workflow or not provider or total <= 0:
                continue
            bucket = _bucket(workflow, provider, model)
            bucket["validation"].append(entry.get("approved_count", 0) / total)
            bucket["validation_weight"].append(total)

        for entry in self._load_jsonl("outcomes.jsonl"):
            suggestion_id = entry.get("suggestion_id", "")
            suggestion = suggestions.get(suggestion_id, {})
            detection = suggestion.get("detection_context") or {}
            if not isinstance(detection, dict):
                detection = {}

            provider = str(entry.get("source_provider", "") or detection.get("source_provider", "")).strip()
            model = str(entry.get("source_model", "") or detection.get("source_model", "")).strip()
            workflow = self._infer_workflow(entry.get("source_run_id", "") or suggestion.get("source_report_id", ""))
            verdict = str(entry.get("verdict", "") or "").strip()
            if not workflow or not provider or verdict not in {"positive", "negative"}:
                continue

            quality = str(entry.get("measurement_quality", "") or "").lower()
            weight = 1.0 if quality == "high" else 0.7 if quality == "medium" else 0.4
            bucket = _bucket(workflow, provider, model)
            bucket["outcomes"].append(1.0 if verdict == "positive" else 0.0)
            bucket["outcome_weight"].append(weight)

        for entry in self._load_jsonl("recalibrations.jsonl"):
            suggestion_id = entry.get("suggestion_id", entry.get("id", ""))
            suggestion = suggestions.get(suggestion_id, {})
            detection = suggestion.get("detection_context") or {}
            if not isinstance(detection, dict):
                detection = {}

            provider = str(detection.get("source_provider", "") or "").strip()
            model = str(detection.get("source_model", "") or "").strip()
            workflow = self._infer_workflow(entry.get("source_run_id", "") or suggestion.get("source_report_id", ""))
            if not workflow or not provider:
                continue

            revised = float(entry.get("revised_confidence", 0.0))
            original = float(entry.get("original_confidence", entry.get("confidence", 0.5)))
            bucket = _bucket(workflow, provider, model)
            bucket["calibration"].append(max(0.0, 1.0 - abs(revised - original)))

        scores: list[dict] = []
        for (workflow, provider, model), bucket in sorted(stats.items()):
            validation_score = self._weighted_mean(bucket["validation"], bucket["validation_weight"])
            outcome_score = self._weighted_mean(bucket["outcomes"], bucket["outcome_weight"])
            calibration_score = self._mean(bucket["calibration"])

            components: list[tuple[float, float]] = []
            if validation_score is not None:
                components.append((0.4, validation_score))
            if outcome_score is not None:
                components.append((0.4, outcome_score))
            if calibration_score is not None:
                components.append((0.2, calibration_score))
            if not components:
                continue

            weight_total = sum(weight for weight, _ in components)
            composite = sum(weight * value for weight, value in components) / weight_total
            sample_count = len(bucket["validation"]) + len(bucket["outcomes"]) + len(bucket["calibration"])
            scores.append({
                "workflow": workflow,
                "provider": provider,
                "model": model,
                "composite_score": round(composite, 4),
                "validation_pass_rate": round(validation_score, 4) if validation_score is not None else None,
                "outcome_quality": round(outcome_score, 4) if outcome_score is not None else None,
                "calibration_accuracy": round(calibration_score, 4) if calibration_score is not None else None,
                "sample_count": sample_count,
            })

        self._write_scores(scores)
        return scores

    def load_scores(self) -> list[dict]:
        return self._load_jsonl(self._path.name)

    def recommend_provider(
        self,
        workflow: str,
        requested_provider: str = "",
        *,
        min_samples: int = 5,
        min_score_gap: float = 0.08,
    ) -> dict | None:
        scores = [
            score for score in self.load_scores()
            if score.get("workflow") == workflow and score.get("sample_count", 0) >= min_samples
        ]
        if not scores:
            return None

        scores.sort(key=lambda item: (item.get("composite_score", 0.0), item.get("sample_count", 0)), reverse=True)
        best = scores[0]
        if not requested_provider or best.get("provider") == requested_provider:
            return best

        requested = next((score for score in scores if score.get("provider") == requested_provider), None)
        if requested is None:
            return None
        if best.get("composite_score", 0.0) < requested.get("composite_score", 0.0) + min_score_gap:
            return None
        return best

    def _write_scores(self, scores: list[dict]) -> None:
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for score in scores:
                handle.write(json.dumps(score) + "\n")

    def _load_jsonl(self, filename: str) -> list[dict]:
        path = self._findings_dir / filename
        if not path.exists():
            return []
        result: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def _load_suggestions_by_id(self) -> dict[str, dict]:
        return {
            entry.get("suggestion_id", ""): entry
            for entry in self._load_jsonl("suggestions.jsonl")
            if entry.get("suggestion_id")
        }

    @staticmethod
    def _infer_workflow(run_id: str) -> str:
        lower = str(run_id or "").lower()
        mapping = {
            "daily": "daily_analysis",
            "weekly": "weekly_analysis",
            "wfo": "wfo",
            "triage": "triage",
            "discovery": "discovery_analysis",
            "reasoning": "outcome_reasoning",
        }
        for prefix, workflow in mapping.items():
            if lower.startswith(prefix):
                return workflow
        return ""

    @staticmethod
    def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
        if not values or not weights or len(values) != len(weights):
            return None
        total_weight = sum(weights)
        if total_weight <= 0:
            return None
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)
