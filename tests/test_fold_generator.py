# tests/test_fold_generator.py
"""Tests for WFO fold generator."""
from schemas.wfo_config import WFOConfig, WFOMethod, ParameterSpace
from schemas.wfo_results import FoldDefinition
from skills.fold_generator import FoldGenerator


def _make_config(
    method: WFOMethod = WFOMethod.ANCHORED,
    is_days: int = 180,
    oos_days: int = 30,
    step_days: int = 30,
    min_folds: int = 6,
) -> WFOConfig:
    return WFOConfig(
        bot_id="bot1",
        method=method,
        in_sample_days=is_days,
        out_of_sample_days=oos_days,
        step_days=step_days,
        min_folds=min_folds,
        parameter_space=ParameterSpace(bot_id="bot1", parameters=[]),
    )


class TestAnchoredFolds:
    def test_generates_correct_number_of_folds(self):
        cfg = _make_config(is_days=180, oos_days=30, step_days=30, min_folds=1)
        gen = FoldGenerator(cfg)
        # Data: 2025-01-01 to 2026-03-01 = 424 days
        # Fold 0: IS 0-180, OOS 180-210 → need 210 days
        # Fold 1: IS 0-210, OOS 210-240 → need 240 days
        # ...keeps going until IS_end + OOS > data_end
        folds = gen.generate("2025-01-01", "2026-03-01")
        assert len(folds) >= 6

    def test_is_always_starts_at_data_start(self):
        cfg = _make_config()
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-03-01")
        for f in folds:
            assert f.is_start == "2025-01-01"

    def test_is_end_grows_by_step(self):
        cfg = _make_config(is_days=60, oos_days=30, step_days=30, min_folds=1)
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2025-12-31")
        if len(folds) >= 2:
            from datetime import datetime

            end_0 = datetime.strptime(folds[0].is_end, "%Y-%m-%d")
            end_1 = datetime.strptime(folds[1].is_end, "%Y-%m-%d")
            assert (end_1 - end_0).days == 30

    def test_oos_immediately_follows_is(self):
        cfg = _make_config()
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-03-01")
        for f in folds:
            assert f.oos_start == f.is_end

    def test_no_oos_past_data_end(self):
        cfg = _make_config()
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-03-01")
        for f in folds:
            assert f.oos_end <= "2026-03-01"

    def test_returns_empty_if_insufficient_data(self):
        cfg = _make_config(is_days=180, oos_days=30, min_folds=6)
        gen = FoldGenerator(cfg)
        # Only 100 days of data — not enough for IS=180 + OOS=30
        folds = gen.generate("2025-01-01", "2025-04-11")
        assert len(folds) == 0

    def test_fold_numbering_sequential(self):
        cfg = _make_config(min_folds=1)
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-03-01")
        for i, f in enumerate(folds):
            assert f.fold_number == i


class TestRollingFolds:
    def test_is_start_advances_by_step(self):
        cfg = _make_config(method=WFOMethod.ROLLING, is_days=90, oos_days=30, step_days=30, min_folds=1)
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-01-01")
        if len(folds) >= 2:
            from datetime import datetime

            start_0 = datetime.strptime(folds[0].is_start, "%Y-%m-%d")
            start_1 = datetime.strptime(folds[1].is_start, "%Y-%m-%d")
            assert (start_1 - start_0).days == 30

    def test_is_length_constant(self):
        cfg = _make_config(method=WFOMethod.ROLLING, is_days=90, oos_days=30, step_days=30, min_folds=1)
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-01-01")
        from datetime import datetime

        for f in folds:
            is_start = datetime.strptime(f.is_start, "%Y-%m-%d")
            is_end = datetime.strptime(f.is_end, "%Y-%m-%d")
            assert (is_end - is_start).days == 90

    def test_no_oos_past_data_end(self):
        cfg = _make_config(method=WFOMethod.ROLLING, min_folds=1)
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-03-01")
        for f in folds:
            assert f.oos_end <= "2026-03-01"


class TestMinFoldsEnforcement:
    def test_returns_empty_below_min_folds(self):
        cfg = _make_config(is_days=180, oos_days=30, step_days=30, min_folds=100)
        gen = FoldGenerator(cfg)
        folds = gen.generate("2025-01-01", "2026-03-01")
        assert len(folds) == 0  # can't generate 100 folds from ~1 year of data
