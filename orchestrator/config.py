"""Application configuration — reads from environment variables."""
from __future__ import annotations

import logging
import os

from pydantic import BaseModel

from schemas.bot_config import BotConfig

logger = logging.getLogger(__name__)


def _parse_bot_timezones(raw: str, bot_ids: list[str]) -> dict[str, BotConfig]:
    """Parse BOT_TIMEZONES env var into BotConfig dict.

    Format: ``bot_id:Asia/Seoul,bot_id2:US/Eastern``
    Bots in bot_ids but not in the mapping get default UTC config.
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

    # Ensure every bot_id has a config (default UTC)
    for bid in bot_ids:
        if bid not in configs:
            configs[bid] = BotConfig(bot_id=bid)

    return configs


class AppConfig(BaseModel):
    """Configuration loaded from environment variables."""

    bot_ids: list[str] = []
    bot_configs: dict[str, BotConfig] = {}
    relay_url: str = ""
    relay_hmac_secret: str = ""
    relay_api_key: str = ""
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
    autonomous_enabled: bool = False
    adaptive_thresholds_enabled: bool = False
    deployment_monitoring_enabled: bool = False
    ab_testing_enabled: bool = False

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build config from environment variables."""
        bot_ids_raw = os.environ.get("BOT_IDS", "")
        bot_ids = [b.strip() for b in bot_ids_raw.split(",") if b.strip()] if bot_ids_raw else []

        bot_configs = _parse_bot_timezones(
            os.environ.get("BOT_TIMEZONES", ""), bot_ids,
        )

        return cls(
            bot_ids=bot_ids,
            bot_configs=bot_configs,
            relay_url=os.environ.get("RELAY_URL", ""),
            relay_hmac_secret=os.environ.get("RELAY_HMAC_SECRET", ""),
            relay_api_key=os.environ.get("RELAY_API_KEY", ""),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            discord_bot_token=os.environ.get("DISCORD_BOT_TOKEN", ""),
            discord_channel_id=os.environ.get("DISCORD_CHANNEL_ID", ""),
            smtp_host=os.environ.get("SMTP_HOST", ""),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER", ""),
            smtp_pass=os.environ.get("SMTP_PASS", ""),
            email_from=os.environ.get("EMAIL_FROM", ""),
            email_to=os.environ.get("EMAIL_TO", ""),
            data_dir=os.environ.get("DATA_DIR", "."),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            bot_config_dir=os.environ.get("BOT_CONFIG_DIR", "data/bot_configs"),
            bot_repo_dir=os.environ.get("BOT_REPO_DIR", "."),
            autonomous_enabled=os.environ.get("AUTONOMOUS_ENABLED", "false").lower() in ("true", "1", "yes"),
            adaptive_thresholds_enabled=os.environ.get("ADAPTIVE_THRESHOLDS_ENABLED", "false").lower() in ("true", "1", "yes"),
            deployment_monitoring_enabled=os.environ.get("DEPLOYMENT_MONITORING_ENABLED", "false").lower() in ("true", "1", "yes"),
            ab_testing_enabled=os.environ.get("AB_TESTING_ENABLED", "false").lower() in ("true", "1", "yes"),
        )
