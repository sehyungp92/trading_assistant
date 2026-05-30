from __future__ import annotations

import json
from pathlib import Path

from schemas.backtest_artifacts import BacktestArtifactIndex
from schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from skills.monthly_model_review_runner import MonthlyModelReviewRunner


def test_existing_model_review_preserves_invocation_attribution(tmp_path: Path) -> None:
    review_path = tmp_path / "model_review.json"
    review_path.write_text("{}", encoding="utf-8")
    (tmp_path / "model_review_invocation.json").write_text(
        json.dumps({
            "provider": "codex_pro",
            "model": "gpt-5.4",
            "runtime": "codex_cli",
        }),
        encoding="utf-8",
    )

    result = MonthlyModelReviewRunner().run(
        monthly_result=MonthlyValidationResult(
            run_id="monthly-bot1-strat1-2026-04",
            run_month="2026-04",
            bot_id="bot1",
            strategy_id="strat1",
            status=MonthlyValidationStatus.KEEP,
        ),
        artifact_index=BacktestArtifactIndex(
            run_id="monthly-bot1-strat1-2026-04",
            artifact_root=str(tmp_path),
        ),
        artifact_root=tmp_path,
        existing_review_path=review_path,
    )

    assert result.provider == "codex_pro"
    assert result.model == "gpt-5.4"
    assert result.runtime == "codex_cli"
