"""Generate validated file changes for bot repository updates."""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from schemas.autonomous_pipeline import (
    FileChange,
    FileChangeMode,
    ParameterDefinition,
    ParameterType,
)


class FileChangeGenerator:
    """Generates repo file mutations for parameters and agent-produced patches."""

    def generate_change(
        self,
        param: ParameterDefinition,
        new_value: Any,
        repo_dir: Path,
    ) -> FileChange:
        """Read a file, apply a parameter change, and return a FileChange."""
        repo_dir = Path(repo_dir)
        full_path = repo_dir / param.file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")

        original = full_path.read_text(encoding="utf-8")
        if param.param_type == ParameterType.YAML_FIELD:
            modified = self._modify_yaml(original, param.yaml_key or "", new_value)
            change_mode = FileChangeMode.YAML_FIELD
            metadata = {"yaml_key": param.yaml_key or ""}
        elif param.param_type == ParameterType.PYTHON_CONSTANT:
            modified = self._modify_python_constant(
                original, param.python_path or "", new_value,
            )
            change_mode = FileChangeMode.PYTHON_CONSTANT
            metadata = {"python_path": param.python_path or ""}
        else:
            raise ValueError(f"Unsupported param_type: {param.param_type}")

        return self._build_change(
            file_path=param.file_path,
            original=original,
            modified=modified,
            change_mode=change_mode,
            metadata=metadata,
        )

    def generate_patch_change(
        self,
        file_path: str,
        patch: str,
        repo_dir: Path,
    ) -> FileChange:
        """Apply a validated unified diff and return the resulting FileChange."""
        repo_dir = Path(repo_dir)
        full_path = repo_dir / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"Patch target not found: {full_path}")

        original = full_path.read_text(encoding="utf-8")
        modified = self._apply_unified_diff(
            original=original,
            patch=patch,
            expected_file_path=file_path,
        )
        return self._build_change(
            file_path=file_path,
            original=original,
            modified=modified,
            change_mode=FileChangeMode.UNIFIED_DIFF,
            metadata={},
            patch=patch,
        )

    def _build_change(
        self,
        file_path: str,
        original: str,
        modified: str,
        change_mode: FileChangeMode,
        metadata: dict[str, Any],
        patch: str = "",
    ) -> FileChange:
        diff = "\n".join(difflib.unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))
        return FileChange(
            file_path=file_path,
            original_content=original,
            new_content=modified,
            change_mode=change_mode,
            metadata=metadata,
            patch=patch,
            diff_preview=diff,
        )

    def _modify_yaml(self, content: str, yaml_key: str, new_value: Any) -> str:
        keys = yaml_key.split(".")
        lines = content.split("\n")
        result, changed = self._set_yaml_value(lines, keys, 0, 0, new_value)
        if not changed:
            raise ValueError(f"YAML key not found: {yaml_key}")
        return "\n".join(result)

    def _set_yaml_value(
        self,
        lines: list[str],
        keys: list[str],
        key_idx: int,
        min_indent: int,
        new_value: Any,
    ) -> tuple[list[str], bool]:
        target_key = keys[key_idx]
        is_last = key_idx == len(keys) - 1
        result = list(lines)

        for idx, line in enumerate(result):
            stripped = line.lstrip()
            if stripped.startswith("#") or not stripped:
                continue
            indent = len(line) - len(stripped)
            if indent < min_indent:
                continue

            match = re.match(r"^(\s*)([\w\-]+)\s*:\s*(.*)", line)
            if not match or match.group(2) != target_key or indent < min_indent:
                continue

            if is_last:
                prefix = match.group(1) + match.group(2) + ": "
                comment = ""
                rest = match.group(3)
                comment_match = re.search(r"\s+#\s+.*$", rest)
                if comment_match:
                    comment = comment_match.group()
                result[idx] = prefix + self._format_yaml_value(new_value) + comment
                return result, True

            nested_result, nested_changed = self._set_yaml_value(
                result[idx + 1:],
                keys,
                key_idx + 1,
                indent + 2,
                new_value,
            )
            if nested_changed:
                return result[:idx + 1] + list(nested_result), True
            return result, False
        return result, False

    def _modify_python_constant(
        self,
        content: str,
        python_path: str,
        new_value: Any,
    ) -> str:
        pattern = re.compile(
            rf"^(?P<indent>\s*){re.escape(python_path)}\s*=\s*(?P<value>.+?)\s*$",
            re.MULTILINE,
        )

        def repl(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return f"{indent}{python_path} = {self._format_python_value(new_value)}"

        modified, count = pattern.subn(repl, content, count=1)
        if count == 0:
            raise ValueError(f"Python constant not found: {python_path}")
        return modified

    def _apply_unified_diff(
        self,
        original: str,
        patch: str,
        expected_file_path: str,
    ) -> str:
        lines = patch.splitlines()
        if len(lines) < 3:
            raise ValueError("Unified diff is too short")

        header_index = self._find_diff_header(lines)
        if header_index is None:
            raise ValueError("Unified diff headers not found")

        old_path = self._strip_prefix(lines[header_index], "--- ")
        new_path = self._strip_prefix(lines[header_index + 1], "+++ ")
        expected_variants = {
            expected_file_path,
            f"a/{expected_file_path}",
            f"b/{expected_file_path}",
        }
        if old_path not in expected_variants or new_path not in expected_variants:
            raise ValueError(
                f"Patch paths do not match expected file {expected_file_path}: "
                f"{old_path} -> {new_path}",
            )

        source_lines = original.splitlines()
        result: list[str] = []
        cursor = 0
        idx = header_index + 2

        while idx < len(lines):
            line = lines[idx]
            if not line:
                idx += 1
                continue
            if not line.startswith("@@"):
                raise ValueError(f"Expected hunk header, got: {line}")

            match = re.match(
                r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
                r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@",
                line,
            )
            if not match:
                raise ValueError(f"Malformed hunk header: {line}")

            old_start = int(match.group("old_start"))
            old_index = max(old_start - 1, 0)
            result.extend(source_lines[cursor:old_index])
            cursor = old_index
            idx += 1

            while idx < len(lines):
                hunk_line = lines[idx]
                if hunk_line.startswith("@@"):
                    break
                if hunk_line.startswith("\\"):
                    idx += 1
                    continue

                prefix = hunk_line[:1]
                payload = hunk_line[1:]
                if prefix == " ":
                    self._expect_source_line(source_lines, cursor, payload)
                    result.append(payload)
                    cursor += 1
                elif prefix == "-":
                    self._expect_source_line(source_lines, cursor, payload)
                    cursor += 1
                elif prefix == "+":
                    result.append(payload)
                else:
                    raise ValueError(f"Unexpected unified diff line: {hunk_line}")
                idx += 1

        result.extend(source_lines[cursor:])
        rebuilt = "\n".join(result)
        if original.endswith("\n"):
            rebuilt += "\n"
        return rebuilt

    @staticmethod
    def _find_diff_header(lines: list[str]) -> int | None:
        for idx in range(len(lines) - 1):
            if lines[idx].startswith("--- ") and lines[idx + 1].startswith("+++ "):
                return idx
        return None

    @staticmethod
    def _strip_prefix(line: str, prefix: str) -> str:
        if not line.startswith(prefix):
            raise ValueError(f"Expected line prefix {prefix!r}: {line}")
        return line[len(prefix):].strip()

    @staticmethod
    def _expect_source_line(source_lines: list[str], index: int, payload: str) -> None:
        if index >= len(source_lines):
            raise ValueError("Unified diff references lines past end of file")
        if source_lines[index] != payload:
            raise ValueError(
                f"Unified diff context mismatch at source line {index + 1}: "
                f"expected {source_lines[index]!r}, got {payload!r}",
            )

    @staticmethod
    def _format_yaml_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _format_python_value(value: Any) -> str:
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)
