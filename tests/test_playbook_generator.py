from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from schemas.generated_playbook import GeneratedPlaybook
from skills.playbook_generator import PlaybookGenerator


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_generates_playbook_only_for_repeated_evidence(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    findings = memory_dir / "findings"
    now = datetime.now(timezone.utc)
    _write_jsonl(findings / "benchmark_cases.jsonl", [
        {
            "case_id": "a",
            "case_tags": ["category:parameter", "reason:duplicate_idea"],
            "agent_type": "daily_analysis",
            "created_at": (now - timedelta(days=3)).isoformat(),
        },
        {
            "case_id": "b",
            "case_tags": ["category:parameter", "reason:duplicate_idea"],
            "agent_type": "daily_analysis",
            "created_at": (now - timedelta(days=2)).isoformat(),
        },
        {
            "case_id": "c",
            "case_tags": ["category:parameter", "reason:duplicate_idea"],
            "agent_type": "daily_analysis",
            "created_at": (now - timedelta(days=1)).isoformat(),
        },
    ])

    playbooks = PlaybookGenerator(memory_dir).generate()

    assert len(playbooks) == 1
    playbook = playbooks[0]
    assert playbook.status.value == "active"
    assert len(playbook.evidence_refs) == 3
    assert "approval gates" in playbook.provenance or playbook.required_evidence
    manifest = memory_dir / "playbooks" / "generated" / "playbooks.jsonl"
    assert manifest.exists()


def test_cleans_up_stale_generated_markdown(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    generated_dir = memory_dir / "playbooks" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    stale = generated_dir / "stale-playbook.md"
    stale.write_text("old", encoding="utf-8")

    findings = memory_dir / "findings"
    now = datetime.now(timezone.utc)
    _write_jsonl(findings / "benchmark_cases.jsonl", [
        {
            "case_id": "a",
            "case_tags": ["category:parameter", "reason:duplicate_idea"],
            "agent_type": "daily_analysis",
            "created_at": (now - timedelta(days=3)).isoformat(),
        },
        {
            "case_id": "b",
            "case_tags": ["category:parameter", "reason:duplicate_idea"],
            "agent_type": "daily_analysis",
            "created_at": (now - timedelta(days=2)).isoformat(),
        },
        {
            "case_id": "c",
            "case_tags": ["category:parameter", "reason:duplicate_idea"],
            "agent_type": "daily_analysis",
            "created_at": (now - timedelta(days=1)).isoformat(),
        },
    ])

    PlaybookGenerator(memory_dir).generate()

    assert not stale.exists()


def test_playbook_match_requires_workflow_when_workflow_is_specified():
    playbook = GeneratedPlaybook(
        workflow="weekly_analysis",
        title="Investigate recurring parameter issues",
        trigger_tags=["category:parameter"],
        evidence_refs=["benchmark:a", "benchmark:b", "benchmark:c"],
    )

    assert playbook.match_score("daily_analysis", ["category:parameter"]) == 0.0
