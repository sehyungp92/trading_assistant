# tests/test_file_change_generator.py
"""Tests for FileChangeGenerator."""
from __future__ import annotations

from pathlib import Path

import pytest

from schemas.autonomous_pipeline import ParameterDefinition, ParameterType
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


def _python_param(python_path: str = "BASE_RISK_PCT", value_type: str = "float") -> ParameterDefinition:
    return ParameterDefinition(
        param_name="BASE_RISK_PCT",
        bot_id="bot1",
        param_type=ParameterType.PYTHON_CONSTANT,
        file_path="config/risk.py",
        python_path=python_path,
        current_value=0.02,
        value_type=value_type,
    )


def _dataclass_param() -> ParameterDefinition:
    return ParameterDefinition(
        param_name="threshold",
        bot_id="bot1",
        param_type=ParameterType.DATACLASS_FIELD,
        file_path="config.py",
        python_path="Config.threshold",
        current_value=0.5,
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

    def test_python_float_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "risk.py").write_text("BASE_RISK_PCT = 0.02\n")
        change = gen.generate_change(_python_param(), 0.03, tmp_path)
        assert "BASE_RISK_PCT = 0.03" in change.new_content

    def test_python_int_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "risk.py").write_text("MAX_POS = 5\n")
        param = ParameterDefinition(
            param_name="MAX_POS", bot_id="bot1",
            param_type=ParameterType.PYTHON_CONSTANT,
            file_path="config/risk.py",
            python_path="MAX_POS", current_value=5, value_type="int",
        )
        change = gen.generate_change(param, 8, tmp_path)
        assert "MAX_POS = 8" in change.new_content

    def test_python_bool_change(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "risk.py").write_text("ENABLED = True\n")
        param = ParameterDefinition(
            param_name="ENABLED", bot_id="bot1",
            param_type=ParameterType.PYTHON_CONSTANT,
            file_path="config/risk.py",
            python_path="ENABLED", current_value=True, value_type="bool",
        )
        change = gen.generate_change(param, False, tmp_path)
        assert "ENABLED = False" in change.new_content

    def test_python_with_type_annotation(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "risk.py").write_text("BASE_RISK_PCT: float = 0.02\n")
        change = gen.generate_change(_python_param(), 0.03, tmp_path)
        assert "BASE_RISK_PCT: float = 0.03" in change.new_content

    def test_dataclass_field(self, gen: FileChangeGenerator, tmp_path: Path):
        (tmp_path / "config.py").write_text(
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Config:\n"
            "    threshold: float = 0.5\n"
            "    name: str = 'default'\n"
        )
        change = gen.generate_change(_dataclass_param(), 0.8, tmp_path)
        assert "threshold: float = 0.8" in change.new_content
        assert "name: str = 'default'" in change.new_content

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
