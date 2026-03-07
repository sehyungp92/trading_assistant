# skills/file_change_generator.py
"""File change generator — modifies config files for parameter changes.

Supports three config formats:
- YAML fields (dotted key paths)
- Python constants (NAME = value or NAME: type = value)
- Dataclass fields (field_name: type = default)
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from schemas.autonomous_pipeline import FileChange, ParameterDefinition, ParameterType


class FileChangeGenerator:
    """Generates file modifications for parameter changes."""

    def generate_change(
        self,
        param: ParameterDefinition,
        new_value: Any,
        repo_dir: Path,
    ) -> FileChange:
        """Read file, apply change, return FileChange with diff."""
        repo_dir = Path(repo_dir)
        full_path = repo_dir / param.file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")

        original = full_path.read_text(encoding="utf-8")

        if param.param_type == ParameterType.YAML_FIELD:
            modified = self._modify_yaml(original, param.yaml_key, new_value)
        elif param.param_type == ParameterType.PYTHON_CONSTANT:
            var_name = param.python_path.split(".")[-1] if param.python_path else param.param_name
            modified = self._modify_python_constant(original, var_name, new_value, param.value_type)
        elif param.param_type == ParameterType.DATACLASS_FIELD:
            parts = (param.python_path or param.param_name).rsplit(".", 1)
            field_name = parts[-1] if len(parts) == 1 else parts[1]
            modified = self._modify_dataclass_field(original, field_name, new_value, param.value_type)
        else:
            raise ValueError(f"Unsupported param_type: {param.param_type}")

        diff = "\n".join(difflib.unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile=f"a/{param.file_path}",
            tofile=f"b/{param.file_path}",
            lineterm="",
        ))

        return FileChange(
            file_path=param.file_path,
            original_content=original,
            new_content=modified,
            diff_preview=diff,
        )

    def _modify_yaml(self, content: str, yaml_key: str, new_value: Any) -> str:
        """Modify a YAML value by dotted key path, preserving formatting."""
        keys = yaml_key.split(".")
        lines = content.split("\n")
        result = self._set_yaml_value(lines, keys, 0, 0, new_value)
        return "\n".join(result)

    def _set_yaml_value(
        self, lines: list[str], keys: list[str], key_idx: int,
        min_indent: int, new_value: Any,
    ) -> list[str]:
        """Recursively find and replace YAML value at key path."""
        target_key = keys[key_idx]
        is_last = key_idx == len(keys) - 1

        result = list(lines)
        for i, line in enumerate(result):
            stripped = line.lstrip()
            if stripped.startswith("#") or not stripped:
                continue
            indent = len(line) - len(stripped)
            if indent < min_indent:
                continue

            # Check if this line has the target key
            match = re.match(r"^(\s*)(\w[\w\-]*)\s*:\s*(.*)", line)
            if match and match.group(2) == target_key and indent >= min_indent:
                if is_last:
                    # Replace value
                    prefix = match.group(1) + match.group(2) + ": "
                    comment = ""
                    rest = match.group(3)
                    # Preserve inline comment
                    comment_match = re.search(r"\s+#\s+.*$", rest)
                    if comment_match:
                        comment = comment_match.group()
                        rest = rest[:comment_match.start()]
                    result[i] = prefix + self._format_yaml_value(new_value) + comment
                    return result
                else:
                    # Go deeper — find children at indent+2
                    return result[:i + 1] + list(
                        self._set_yaml_value(
                            result[i + 1:], keys, key_idx + 1, indent + 2, new_value
                        )
                    )
        return result

    @staticmethod
    def _format_yaml_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return str(value)
        if isinstance(value, int):
            return str(value)
        return str(value)

    def _modify_python_constant(
        self, content: str, var_name: str, new_value: Any, value_type: str = "float",
    ) -> str:
        """Replace a Python constant assignment."""
        formatted = self._format_python_value(new_value, value_type)
        # Match: VAR_NAME = value  or  VAR_NAME: type = value
        pattern = re.compile(
            rf"^(\s*)({re.escape(var_name)})\s*(?::\s*\w+\s*)?=\s*(.*)$",
            re.MULTILINE,
        )
        match = pattern.search(content)
        if match:
            indent = match.group(1)
            # Preserve type annotation if present
            orig_line = match.group(0)
            annotation_match = re.match(
                rf"^(\s*{re.escape(var_name)})\s*(:\s*\w+\s*)?=\s*(.*)$",
                orig_line,
            )
            if annotation_match and annotation_match.group(2):
                replacement = f"{indent}{var_name}{annotation_match.group(2)}= {formatted}"
            else:
                replacement = f"{indent}{var_name} = {formatted}"
            # Preserve inline comment
            rest = match.group(3)
            comment_match = re.search(r"\s+#\s+.*$", rest)
            if comment_match:
                replacement += comment_match.group()
            return pattern.sub(replacement, content, count=1)
        return content

    def _modify_dataclass_field(
        self, content: str, field_name: str, new_value: Any, value_type: str = "float",
    ) -> str:
        """Replace a dataclass field default value."""
        formatted = self._format_python_value(new_value, value_type)
        pattern = re.compile(
            rf"^(\s*)({re.escape(field_name)})\s*:\s*(\w+)\s*=\s*(.*)$",
            re.MULTILINE,
        )
        match = pattern.search(content)
        if match:
            indent = match.group(1)
            type_ann = match.group(3)
            rest = match.group(4)
            comment = ""
            comment_match = re.search(r"\s+#\s+.*$", rest)
            if comment_match:
                comment = comment_match.group()
            return pattern.sub(
                f"{indent}{field_name}: {type_ann} = {formatted}{comment}",
                content,
                count=1,
            )
        return content

    @staticmethod
    def _format_python_value(value: Any, value_type: str = "float") -> str:
        if value_type == "bool":
            return "True" if value else "False"
        if value_type == "int":
            return str(int(value))
        if value_type == "float":
            return str(float(value))
        if value_type == "str":
            return repr(str(value))
        return repr(value)
