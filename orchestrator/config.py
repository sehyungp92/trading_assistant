"""Application configuration reads from environment variables and local .env."""

from __future__ import annotations

import logging
import os
from json import JSONDecodeError
from pathlib import Path

from pydantic import BaseModel

from schemas.bot_config import BotConfig
from schemas.strategy_profile import StrategyRegistry

logger = logging.getLogger(__name__)


def _default_dotenv_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv_defaults(dotenv_path: str | Path | None = None) -> dict[str, str]:
    """Load .env pairs with normal precedence semantics.

    Values read from disk are defaults only; existing process environment
    variables still win because callers overlay ``os.environ`` on top.
    """
    path = Path(dotenv_path) if dotenv_path is not None else _default_dotenv_path()
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Could not read .env from %s", path)
        return values

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            logger.warning("Ignoring malformed .env line in %s: %r", path, raw_line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _parse_command_args(raw: str, env_name: str) -> list[str]:
    """Parse a JSON-array command-args env var into a list of strings."""
    if not raw.strip():
        return []
    try:
        import json

        parsed = json.loads(raw)
    except JSONDecodeError:
        logger.warning("Ignoring invalid %s JSON: %r", env_name, raw)
        return []

    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        logger.warning("Ignoring %s because it must be a JSON array of strings", env_name)
        return []
    return parsed


def _parse_bot_timezones(raw: str, bot_ids: list[str]) -> dict[str, BotConfig]:
    """Parse BOT_TIMEZONES env var into BotConfig dict.

    Format: ``bot_id:Asia/Seoul,bot_id2:US/Eastern``.
    Bots in ``bot_ids`` but not in the mapping get default UTC config.
    """
    configs: dict[str, BotConfig] = {}

    if raw:
        for pair in raw.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                logger.warning("Ignoring malformed BOT_TIMEZONES entry: %r", pair)
                continue
            bot_id, tz = pair.split(":", 1)
            bot_id = bot_id.strip()
            tz = tz.strip()
            if not bot_id or not tz:
                continue
            try:
                configs[bot_id] = BotConfig(bot_id=bot_id, timezone=tz)
            except Exception as exc:
                logger.warning("Invalid timezone config for %s: %s", bot_id, exc)

    for bid in bot_ids:
        if bid not in configs:
            configs[bid] = BotConfig(bot_id=bid)

    return configs


class AppConfig(BaseModel):
    """Configuration loaded from environment variables."""

    bot_ids: list[str] = []
    bot_configs: dict[str, BotConfig] = {}
    claude_command: str = "claude"
    claude_command_args: list[str] = []
    codex_command: str = "codex"
    codex_command_args: list[str] = []
    zai_api_key: str = ""
    openrouter_api_key: str = ""
    agent_default_provider: str = ""
    agent_default_model: str = ""
    daily_agent_provider: str = ""
    daily_agent_model: str = ""
    weekly_agent_provider: str = ""
    weekly_agent_model: str = ""
    wfo_agent_provider: str = ""
    wfo_agent_model: str = ""
    triage_agent_provider: str = ""
    triage_agent_model: str = ""
    relay_url: str = ""
    relay_hmac_secret: str = ""
    relay_api_key: str = ""
    orchestrator_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_bot_token: str = ""
    discord_channel_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    email_from: str = ""
    email_to: str = ""
    data_dir: str = "."
    log_level: str = "INFO"
    bot_config_dir: str = "data/bot_configs"
    bot_repo_dir: str = "."
    bot_repo_cache_dir: str = "runs/repo_cache"
    github_token: str = ""
    strategy_registry: StrategyRegistry = StrategyRegistry()
    autonomous_enabled: bool = False
    adaptive_thresholds_enabled: bool = False
    deployment_monitoring_enabled: bool = False
    ab_testing_enabled: bool = False

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> AppConfig:
        """Build config from process env layered over local .env defaults."""
        env = _load_dotenv_defaults(dotenv_path)
        env.update(os.environ)

        bot_ids_raw = env.get("BOT_IDS", "")
        bot_ids = [b.strip() for b in bot_ids_raw.split(",") if b.strip()] if bot_ids_raw else []

        bot_configs = _parse_bot_timezones(env.get("BOT_TIMEZONES", ""), bot_ids)

        from orchestrator.strategy_registry_loader import load_strategy_registry

        strategy_profiles_path = env.get("STRATEGY_PROFILES_PATH", "")
        strategy_registry = load_strategy_registry(
            Path(strategy_profiles_path) if strategy_profiles_path else None
        )

        return cls(
            bot_ids=bot_ids,
            bot_configs=bot_configs,
            strategy_registry=strategy_registry,
            claude_command=env.get("CLAUDE_COMMAND", "claude"),
            claude_command_args=_parse_command_args(
                env.get("CLAUDE_COMMAND_ARGS", ""),
                "CLAUDE_COMMAND_ARGS",
            ),
            codex_command=env.get("CODEX_COMMAND", "codex"),
            codex_command_args=_parse_command_args(
                env.get("CODEX_COMMAND_ARGS", ""),
                "CODEX_COMMAND_ARGS",
            ),
            zai_api_key=env.get("ZAI_API_KEY", ""),
            openrouter_api_key=env.get("OPENROUTER_API_KEY", ""),
            agent_default_provider=env.get("AGENT_PROVIDER", ""),
            agent_default_model=env.get("AGENT_MODEL", ""),
            daily_agent_provider=env.get("DAILY_AGENT_PROVIDER", ""),
            daily_agent_model=env.get("DAILY_AGENT_MODEL", ""),
            weekly_agent_provider=env.get("WEEKLY_AGENT_PROVIDER", ""),
            weekly_agent_model=env.get("WEEKLY_AGENT_MODEL", ""),
            wfo_agent_provider=env.get("WFO_AGENT_PROVIDER", ""),
            wfo_agent_model=env.get("WFO_AGENT_MODEL", ""),
            triage_agent_provider=env.get("TRIAGE_AGENT_PROVIDER", ""),
            triage_agent_model=env.get("TRIAGE_AGENT_MODEL", ""),
            relay_url=env.get("RELAY_URL", ""),
            relay_hmac_secret=env.get("RELAY_HMAC_SECRET", ""),
            relay_api_key=env.get("RELAY_API_KEY", ""),
            orchestrator_api_key=env.get("ORCHESTRATOR_API_KEY", ""),
            telegram_bot_token=env.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=env.get("TELEGRAM_CHAT_ID", ""),
            discord_bot_token=env.get("DISCORD_BOT_TOKEN", ""),
            discord_channel_id=env.get("DISCORD_CHANNEL_ID", ""),
            smtp_host=env.get("SMTP_HOST", ""),
            smtp_port=int(env.get("SMTP_PORT", "587")),
            smtp_user=env.get("SMTP_USER", ""),
            smtp_pass=env.get("SMTP_PASS", ""),
            email_from=env.get("EMAIL_FROM", ""),
            email_to=env.get("EMAIL_TO", ""),
            data_dir=env.get("DATA_DIR", "."),
            log_level=env.get("LOG_LEVEL", "INFO"),
            bot_config_dir=env.get("BOT_CONFIG_DIR", "data/bot_configs"),
            bot_repo_dir=env.get("BOT_REPO_DIR", "."),
            bot_repo_cache_dir=env.get("BOT_REPO_CACHE_DIR", "runs/repo_cache"),
            github_token=env.get("GITHUB_TOKEN", ""),
            autonomous_enabled=env.get("AUTONOMOUS_ENABLED", "false").lower() in ("true", "1", "yes"),
            adaptive_thresholds_enabled=env.get("ADAPTIVE_THRESHOLDS_ENABLED", "false").lower() in ("true", "1", "yes"),
            deployment_monitoring_enabled=env.get("DEPLOYMENT_MONITORING_ENABLED", "false").lower() in ("true", "1", "yes"),
            ab_testing_enabled=env.get("AB_TESTING_ENABLED", "false").lower() in ("true", "1", "yes"),
        )
