"""Application configuration — reads from environment variables."""
from __future__ import annotations

import os

from pydantic import BaseModel


class AppConfig(BaseModel):
    """Configuration loaded from environment variables."""

    bot_ids: list[str] = []
    relay_url: str = ""
    relay_hmac_secret: str = ""
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

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build config from environment variables."""
        bot_ids_raw = os.environ.get("BOT_IDS", "")
        bot_ids = [b.strip() for b in bot_ids_raw.split(",") if b.strip()] if bot_ids_raw else []

        return cls(
            bot_ids=bot_ids,
            relay_url=os.environ.get("RELAY_URL", ""),
            relay_hmac_secret=os.environ.get("RELAY_HMAC_SECRET", ""),
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
        )
