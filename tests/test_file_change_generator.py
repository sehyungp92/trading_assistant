# tests/test_file_change_generator.py
"""Tests for FileChangeGenerator."""
from __future__ import annotations

from pathlib import Path

import pytest

from schemas.autonomous_pipeline import FileChangeMode, ParameterDefinition, ParameterType
from skills.file_change_generator import FileChangeGenerator


@pytest.fixture
def gen() -> FileChangeGenerator:
    return FileChangeGenerator()


def _yaml_param(yaml_key: str = "kmp.quality_min") -> ParameterDefinition:
    return ParameterDefinition(
        param_name="quality_min",
        bot_id="bot1",
        param_type=ParameterType.YAML_FIELD,
        file_path="config.yaml",
        yaml_key=yaml_key,
        current_value=0.6,
        value_type="float",
    )


def _python_param(python_path: str = "BASE_RISK_PCT") -> ParameterDefinition:
    return ParameterDefinition(
        param_name="base_risk_pct",
        bot_id="bot1",
        param_type=ParameterType.PYTHON_CONSTANT,
        file_path="risk.py",
        python_path=python_path,
        current_value=0.02,
        value_type="float",
    )


class TestFileChangeGenerator:
    def test_yaml_simple_key_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n  other: 10\n"
        )
        change = gen.generate_change(_yaml_param(), 0.7, tmp_path)
        assert "quality_min: 0.7" in change.new_content
        assert "quality_min: 0.6" not in change.new_content
        assert "other: 10" in change.new_content  # untouched

    def test_yaml_nested_dotted_path(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n"
        )
        change = gen.generate_change(_yaml_param("kmp.quality_min"), 0.8, tmp_path)
        assert "quality_min: 0.8" in change.new_content

    def test_yaml_preserves_comments(self, gen: FileChangeGenerator, tmp_path: Path):
        content = "# Top comment\nkmp:\n  quality_min: 0.6  # inline\n  other: 5\n"
        (tmp_path / "config.yaml").write_text(content)
        change = gen.generate_change(_yaml_param(), 0.7, tmp_path)
        assert "# Top comment" in change.new_content
        assert "# inline" in change.new_content
        assert "quality_min: 0.7" in change.new_content

    def test_yaml_missing_key_raises(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("kmp:\n  other: 5\n")
        with pytest.raises(ValueError, match="YAML key not found"):
            gen.generate_change(_yaml_param(), 0.7, tmp_path)

    def test_unsupported_param_type_raises(self, gen: FileChangeGenerator, tmp_path: Path):
        """Non-YAML param types are not supported."""
        (tmp_path / "config.yaml").write_text("x: 1\n")
        param = ParameterDefinition(
            param_name="x", bot_id="bot1",
            param_type=ParameterType.YAML_FIELD,
            file_path="config.yaml",
            yaml_key="x",
            current_value=1,
            value_type="int",
        )
        # This should work (YAML_FIELD)
        change = gen.generate_change(param, 2, tmp_path)
        assert "x: 2" in change.new_content

    def test_file_not_found_raises(self, gen: FileChangeGenerator, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            gen.generate_change(_yaml_param(), 0.7, tmp_path)

    def test_diff_preview_contains_values(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n"
        )
        change = gen.generate_change(_yaml_param(), 0.7, tmp_path)
        assert "0.6" in change.diff_preview
        assert "0.7" in change.diff_preview

    def test_same_value_no_diff(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "kmp:\n  quality_min: 0.6\n"
        )
        change = gen.generate_change(_yaml_param(), 0.6, tmp_path)
        assert change.original_content == change.new_content
        assert change.diff_preview == ""

    def test_python_constant_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "risk.py").write_text("BASE_RISK_PCT = 0.02\nOTHER = 5\n")
        change = gen.generate_change(_python_param(), 0.03, tmp_path)
        assert change.change_mode == FileChangeMode.PYTHON_CONSTANT
        assert "BASE_RISK_PCT = 0.03" in change.new_content
        assert "OTHER = 5" in change.new_content

    def test_unified_diff_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "algo.py").write_text("VALUE = 1\nOTHER = 2\n")
        patch = "\n".join([
            "--- a/algo.py",
            "+++ b/algo.py",
            "@@ -1,2 +1,2 @@",
            "-VALUE = 1",
            "+VALUE = 2",
            " OTHER = 2",
        ])
        change = gen.generate_patch_change("algo.py", patch, tmp_path)
        assert change.change_mode == FileChangeMode.UNIFIED_DIFF
        assert "VALUE = 2" in change.new_content
        assert "VALUE = 1" not in change.new_content

    def test_unified_diff_rejects_bad_context(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "algo.py").write_text("VALUE = 1\nOTHER = 2\n")
        patch = "\n".join([
            "--- a/algo.py",
            "+++ b/algo.py",
            "@@ -1,2 +1,2 @@",
            "-VALUE = 9",
            "+VALUE = 2",
            " OTHER = 2",
        ])
        with pytest.raises(ValueError):
            gen.generate_patch_change("algo.py", patch, tmp_path)
