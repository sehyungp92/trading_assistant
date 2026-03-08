"""Startup catch-up — detect and enqueue missed scheduled jobs."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from schemas.bot_config import BotConfig

logger = logging.getLogger(__name__)


class StartupCatchup:
    """Reads run_history.jsonl to determine if scheduled jobs were missed.

    Catches up all missed days (up to ``max_catchup_days``) in chronological
    order, so multi-day laptop-off periods are fully recovered.
    """

    def __init__(
        self,
        run_history_path: Path,
        bot_configs: dict[str, BotConfig] | None = None,
        max_catchup_days: int = 7,
    ) -> None:
        self._history_path = Path(run_history_path)
        self._bot_configs = bot_configs or {}
        self._max_catchup_days = max_catchup_days
        self._runs = self._load_history()

    def _load_history(self) -> list[dict]:
        if not self._history_path.exists():
            return []
        runs = []
        try:
            for line in self._history_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.warning("Could not read run history at %s", self._history_path)
        return runs

    def _last_run_date(self, handler_name: str) -> str | None:
        """Return the most recent date string for a given handler."""
        matches = [
            r for r in self._runs
            if r.get("handler") == handler_name
            and r.get("status") in ("completed", "skipped", "running")
        ]
        if not matches:
            return None
        # Sort by started_at descending
        matches.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        # Extract date from run_id (e.g., "daily-2026-03-08" -> "2026-03-08")
        run_id = matches[0].get("run_id", "")
        parts = run_id.split("-", 1)
        if len(parts) >= 2:
            return parts[1]
        return None

    def needs_daily_catchup(self) -> list[dict]:
        """Check if daily analysis needs catching up for any bot group.

        Returns list of ``{"bots": [...], "date": "YYYY-MM-DD"}`` dicts
        for each missed day per group, oldest first.  Up to
        ``max_catchup_days`` entries per group.
        """
        from orchestrator.tz_utils import bot_trading_date

        catchups: list[dict] = []
        last_daily = self._last_run_date("daily_analysis")

        if self._bot_configs:
            from orchestrator.tz_utils import group_bots_by_analysis_time
            groups = group_bots_by_analysis_time(self._bot_configs)

            for _time_key, bot_list in groups.items():
                tz = self._bot_configs[bot_list[0]].timezone
                yesterday = bot_trading_date(
                    tz, datetime.now(timezone.utc) - timedelta(days=1),
                )
                if last_daily is None:
                    # No history — catch up yesterday only
                    catchups.append({"bots": bot_list, "date": yesterday})
                elif last_daily < yesterday:
                    # Iterate each missed day from last_daily+1 through yesterday
                    from datetime import date as date_type
                    start = date_type.fromisoformat(last_daily) + timedelta(days=1)
                    end = date_type.fromisoformat(yesterday)
                    count = 0
                    d = start
                    while d <= end and count < self._max_catchup_days:
                        catchups.append({
                            "bots": bot_list,
                            "date": d.isoformat(),
                        })
                        d += timedelta(days=1)
                        count += 1
        else:
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            if last_daily is None:
                catchups.append({"date": yesterday})
            elif last_daily < yesterday:
                from datetime import date as date_type
                start = date_type.fromisoformat(last_daily) + timedelta(days=1)
                end = date_type.fromisoformat(yesterday)
                count = 0
                d = start
                while d <= end and count < self._max_catchup_days:
                    catchups.append({"date": d.isoformat()})
                    d += timedelta(days=1)
                    count += 1

        return catchups

    def needs_weekly_catchup(self) -> bool:
        """Check if weekly analysis was missed this week."""
        last_weekly = self._last_run_date("weekly_analysis")
        if last_weekly is None:
            return True

        now = datetime.now(timezone.utc)
        # Find the most recent Sunday
        days_since_sunday = (now.weekday() + 1) % 7
        last_sunday = (now - timedelta(days=days_since_sunday)).strftime("%Y-%m-%d")

        return last_weekly < last_sunday

    def build_catchup_events(self) -> list[dict]:
        """Build event dicts to enqueue for missed jobs."""
        events: list[dict] = []

        for catchup in self.needs_daily_catchup():
            date = catchup["date"]
            bots = catchup.get("bots")
            payload = {"date": date}
            if bots:
                payload["bots"] = bots
            events.append({
                "event_type": "daily_analysis_trigger",
                "bot_id": "system",
                "event_id": f"catchup-daily-{date}",
                "payload": json.dumps(payload),
            })

        if self.needs_weekly_catchup():
            events.append({
                "event_type": "weekly_summary_trigger",
                "bot_id": "system",
                "event_id": f"catchup-weekly-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                "payload": "{}",
            })

        return events
