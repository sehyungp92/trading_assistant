# skills/config_registry.py
"""Config registry — loads bot parameter profiles from YAML config files.

Each bot has a YAML file defining its tunable parameters, file paths,
valid ranges, and safety flags. The registry resolves suggestion text
to matching parameter definitions.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from schemas.autonomous_pipeline import (
    BotConfigProfile,
    ParameterDefinition,
    ParameterType,
)

logger = logging.getLogger(__name__)


class ConfigRegistry:
    """Loads and queries bot configuration profiles."""

    def __init__(self, config_dir: Path) -> None:
        self._profiles: dict[str, BotConfigProfile] = {}
        self._load_profiles(config_dir)

    def _load_profiles(self, config_dir: Path) -> None:
        config_dir = Path(config_dir)
        if not config_dir.exists():
            logger.warning("Config directory %s does not exist", config_dir)
            return
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
                if not data or not isinstance(data, dict):
                    continue
                bot_id = data.get("bot_id", yaml_file.stem)
                params = []
                for p in data.get("parameters", []):
                    p["bot_id"] = bot_id
                    params.append(ParameterDefinition(**p))
                profile = BotConfigProfile(
                    bot_id=bot_id,
                    repo_url=data.get("repo_url", ""),
                    repo_dir=data.get("repo_dir", ""),
                    default_branch=data.get("default_branch", "main"),
                    allowed_edit_paths=data.get("allowed_edit_paths", []) or [],
                    structural_context_paths=data.get("structural_context_paths", []) or [],
                    verification_commands=data.get("verification_commands", []) or [],
                    parameters=params,
                    strategies=data.get("strategies", []),
                )
                self._profiles[bot_id] = profile
                logger.info("Loaded config profile for %s (%d params)", bot_id, len(params))
            except Exception:
                logger.exception("Failed to load config from %s", yaml_file)

    def get_profile(self, bot_id: str) -> BotConfigProfile | None:
        return self._profiles.get(bot_id)

    def get_parameter(self, bot_id: str, param_name: str) -> ParameterDefinition | None:
        profile = self._profiles.get(bot_id)
        if profile is None:
            return None
        return profile.get_parameter(param_name)

    def find_parameters_by_category(self, bot_id: str, category: str) -> list[ParameterDefinition]:
        profile = self._profiles.get(bot_id)
        if profile is None:
            return []
        return profile.get_parameters_by_category(category)

    def resolve_suggestion_to_params(self, suggestion) -> list[ParameterDefinition]:
        """Match a suggestion to parameter definitions by category and keyword.

        Args:
            suggestion: SuggestionRecord (dict or model with bot_id, category, title, description)
        """
        bot_id = suggestion.get("bot_id") if isinstance(suggestion, dict) else getattr(suggestion, "bot_id", "")
        category = suggestion.get("category") if isinstance(suggestion, dict) else getattr(suggestion, "category", "")
        title = suggestion.get("title") if isinstance(suggestion, dict) else getattr(suggestion, "title", "")
        description = suggestion.get("description") if isinstance(suggestion, dict) else getattr(suggestion, "description", "")

        profile = self._profiles.get(bot_id)
        if profile is None:
            return []

        # Match by category first
        category_matches = profile.get_parameters_by_category(category)

        # If we have category matches, filter by keyword match in title/description
        text = f"{title} {description}".lower()
        if category_matches:
            keyword_matches = [
                p for p in category_matches
                if self._keyword_match(p.param_name, text)
            ]
            return keyword_matches if keyword_matches else category_matches

        # Fallback: keyword match across all params
        return [
            p for p in profile.parameters
            if self._keyword_match(p.param_name, text)
        ]

    @staticmethod
    def _keyword_match(param_name: str, text: str) -> bool:
        """Check if a parameter name appears in the suggestion text."""
        # Convert param_name to searchable variants
        variants = {param_name.lower()}
        # Add space-separated version (quality_min_threshold -> quality min threshold)
        variants.add(param_name.lower().replace("_", " "))
        # Add individual words
        words = param_name.lower().split("_")
        return any(v in text for v in variants) or all(w in text for w in words if len(w) > 2)

    def validate_value(self, param: ParameterDefinition, value: Any) -> tuple[bool, str | None]:
        """Validate a proposed value against parameter constraints."""
        if param.valid_range is not None:
            try:
                fval = float(value)
            except (TypeError, ValueError):
                return False, f"Cannot convert {value!r} to float for range check"
            if fval < param.valid_range[0] or fval > param.valid_range[1]:
                return False, f"Value {fval} outside range [{param.valid_range[0]}, {param.valid_range[1]}]"
        if param.valid_values is not None:
            if value not in param.valid_values:
                return False, f"Value {value!r} not in allowed values {param.valid_values}"
        return True, None

    def list_bot_ids(self) -> list[str]:
        return list(self._profiles.keys())
