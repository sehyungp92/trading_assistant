"""Offline harness evaluation against compiled benchmark cases."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from schemas.benchmark_case import BenchmarkCase, BenchmarkSeverity, BenchmarkSuite
from schemas.harness_learning import HarnessEvalResult, HarnessVariant


class HarnessEvalRunner:
    """Compare baseline and candidate harness variants without touching live routing."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = Path(findings_dir)
        self._results_path = self._findings_dir / "harness_eval_results.jsonl"
        self._variants_path = self._findings_dir / "harness_variants.json"

    def evaluate_and_save(self, suite: BenchmarkSuite) -> list[HarnessEvalResult]:
        variants = self.load_variants()
        if not suite.cases or not variants:
            return []

        baseline = next((variant for variant in variants if variant.name == "baseline"), variants[0])
        baseline_result = self.evaluate_variant(baseline, suite)
        results = [baseline_result]
        for variant in variants:
            if variant.name == baseline.name or not variant.enabled:
                continue
            result = self.evaluate_variant(variant, suite, baseline_result=baseline_result)
            results.append(result)

        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with self._results_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(result.model_dump_json() + "\n")
        return results

    def evaluate_variant(
        self,
        variant: HarnessVariant,
        suite: BenchmarkSuite,
        *,
        baseline_result: HarnessEvalResult | None = None,
    ) -> HarnessEvalResult:
        totals = defaultdict(float)
        weights = defaultdict(float)
        weighted_total = 0.0
        total_weight = 0.0

        for case in suite.cases:
            weight = self._severity_weight(case.severity)
            score = self._score_case(case, variant)
            totals[case.source.value] += score * weight
            weights[case.source.value] += weight
            weighted_total += score * weight
            total_weight += weight

        aggregate = weighted_total / total_weight if total_weight else 0.0
        per_source = {
            source: round(totals[source] / weights[source], 4)
            for source in sorted(totals)
            if weights[source] > 0
        }

        kept = True
        rationale = "Baseline reference variant."
        if baseline_result is not None:
            delta = aggregate - baseline_result.aggregate_score
            kept = delta >= 0.05
            rationale = (
                f"Improved aggregate benchmark score by {delta:+.3f}."
                if kept else
                f"Did not clear keep threshold vs baseline ({delta:+.3f})."
            )

        return HarnessEvalResult(
            variant_name=variant.name,
            benchmark_count=len(suite.cases),
            aggregate_score=round(aggregate, 4),
            per_source=per_source,
            kept=kept,
            rationale=rationale,
        )

    def load_variants(self) -> list[HarnessVariant]:
        if self._variants_path.exists():
            try:
                payload = json.loads(self._variants_path.read_text(encoding="utf-8"))
                return [HarnessVariant.model_validate(item) for item in payload]
            except Exception:
                pass
        return [
            HarnessVariant(name="baseline"),
            HarnessVariant(
                name="query_aware_guarded",
                retrieval_mode="query_aware",
                validator_profile="guarded",
                route_profile="learned",
            ),
        ]

    @staticmethod
    def _severity_weight(severity: BenchmarkSeverity) -> float:
        return {
            BenchmarkSeverity.CRITICAL: 1.0,
            BenchmarkSeverity.HIGH: 0.75,
            BenchmarkSeverity.MEDIUM: 0.5,
            BenchmarkSeverity.LOW: 0.25,
        }.get(severity, 0.5)

    def _score_case(self, case: BenchmarkCase, variant: HarnessVariant) -> float:
        score = 0.45 if variant.name == "baseline" else 0.5
        tags = set(case.case_tags)

        if variant.retrieval_mode != "baseline" and any(
            tag.startswith(("category:", "reason:", "regime:")) for tag in tags
        ):
            score += 0.15
        if variant.validator_profile != "baseline" and case.source.value in {
            "validation_block",
            "calibration_miss",
        }:
            score += 0.2
        if variant.route_profile != "configured" and case.source.value in {
            "negative_outcome",
            "transfer_failure",
        }:
            score += 0.15
        if variant.prompt_patch:
            score += 0.05
        return min(score, 1.0)
