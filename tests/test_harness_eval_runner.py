from __future__ import annotations

from pathlib import Path

from schemas.benchmark_case import BenchmarkCase, BenchmarkSeverity, BenchmarkSource, BenchmarkSuite
from skills.harness_eval_runner import HarnessEvalRunner


def test_candidate_variant_is_kept_when_it_scores_above_baseline(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    runner = HarnessEvalRunner(findings)
    suite = BenchmarkSuite(cases=[
        BenchmarkCase(
            case_id="a",
            source=BenchmarkSource.VALIDATION_BLOCK,
            source_id="validation:a",
            severity=BenchmarkSeverity.HIGH,
            case_tags=["category:parameter", "reason:duplicate_idea"],
        ),
        BenchmarkCase(
            case_id="b",
            source=BenchmarkSource.NEGATIVE_OUTCOME,
            source_id="outcome:b",
            severity=BenchmarkSeverity.CRITICAL,
            case_tags=["category:parameter"],
        ),
    ])

    results = runner.evaluate_and_save(suite)

    baseline = next(result for result in results if result.variant_name == "baseline")
    candidate = next(result for result in results if result.variant_name == "query_aware_guarded")
    assert candidate.aggregate_score > baseline.aggregate_score
    assert candidate.kept is True
