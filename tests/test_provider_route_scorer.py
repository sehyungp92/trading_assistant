from __future__ import annotations

import json
from pathlib import Path

from skills.provider_route_scorer import ProviderRouteScorer


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_recompute_scores_from_validation_outcomes_and_recalibration(tmp_path: Path):
    findings = tmp_path / "memory" / "findings"
    _write_jsonl(findings / "suggestions.jsonl", [{
        "suggestion_id": "s1",
        "source_report_id": "daily-2026-04-20",
        "detection_context": {
            "source_provider": "codex_pro",
            "source_model": "gpt-5.4",
        },
    }])
    _write_jsonl(findings / "validation_log.jsonl", [{
        "agent_type": "daily_analysis",
        "provider": "codex_pro",
        "model": "gpt-5.4",
        "approved_count": 4,
        "blocked_count": 1,
    }])
    _write_jsonl(findings / "outcomes.jsonl", [{
        "suggestion_id": "s1",
        "source_run_id": "daily-2026-04-20",
        "source_provider": "codex_pro",
        "source_model": "gpt-5.4",
        "verdict": "positive",
        "measurement_quality": "high",
    }])
    _write_jsonl(findings / "recalibrations.jsonl", [{
        "suggestion_id": "s1",
        "revised_confidence": 0.6,
        "original_confidence": 0.7,
    }])

    scores = ProviderRouteScorer(findings).recompute()

    assert len(scores) == 1
    assert scores[0]["workflow"] == "daily_analysis"
    assert scores[0]["provider"] == "codex_pro"
    assert scores[0]["sample_count"] == 3
    assert scores[0]["composite_score"] > 0.7
