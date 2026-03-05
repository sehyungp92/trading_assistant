"""Cross-bot portfolio risk computation — daily, no LLM calls.

Computes exposure, concentration, and crowding alerts.
If crowding alerts trigger, the daily report leads with them.
"""
from __future__ import annotations

from collections import defaultdict

from schemas.daily_metrics import BotDailySummary
from schemas.portfolio_risk import PortfolioRiskCard, CrowdingAlert


class PortfolioRiskComputer:
    """Computes a PortfolioRiskCard from bot summaries and position details."""

    def __init__(
        self,
        date: str,
        bot_summaries: list[BotDailySummary],
        position_details: dict[str, list[dict]],
        max_single_symbol_pct: float = 50.0,
        max_total_exposure_pct: float = 80.0,
        correlation_threshold: float = 0.7,
    ) -> None:
        self.date = date
        self.bot_summaries = bot_summaries
        self.position_details = position_details  # bot_id → [{symbol, direction, exposure_pct}]
        self.max_single_symbol_pct = max_single_symbol_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.correlation_threshold = correlation_threshold

    def compute(self) -> PortfolioRiskCard:
        """Build the full risk card."""
        exposure_by_symbol = self._compute_exposure_by_symbol()
        exposure_by_direction = self._compute_exposure_by_direction()
        total_exposure = sum(exposure_by_symbol.values())
        concentration_score = self._compute_concentration_score(exposure_by_symbol, total_exposure)
        alerts = self._detect_alerts(exposure_by_symbol, exposure_by_direction, total_exposure)

        return PortfolioRiskCard(
            date=self.date,
            total_exposure_pct=total_exposure,
            exposure_by_symbol=exposure_by_symbol,
            exposure_by_direction=exposure_by_direction,
            concentration_score=concentration_score,
            crowding_alerts=alerts,
        )

    def _compute_exposure_by_symbol(self) -> dict[str, float]:
        by_symbol: dict[str, float] = defaultdict(float)
        for positions in self.position_details.values():
            for pos in positions:
                by_symbol[pos["symbol"]] += pos["exposure_pct"]
        return dict(by_symbol)

    def _compute_exposure_by_direction(self) -> dict[str, float]:
        by_dir: dict[str, float] = defaultdict(float)
        for positions in self.position_details.values():
            for pos in positions:
                by_dir[pos["direction"]] += pos["exposure_pct"]
        return dict(by_dir)

    def _compute_concentration_score(
        self, exposure_by_symbol: dict[str, float], total_exposure: float
    ) -> float:
        """Herfindahl-Hirschman Index normalized to 0–100. Higher = more concentrated."""
        if total_exposure == 0:
            return 0.0
        shares = [exp / total_exposure for exp in exposure_by_symbol.values()]
        hhi = sum(s * s for s in shares)
        # HHI ranges from 1/N (perfectly diversified) to 1.0 (single symbol)
        # Normalize: 0 = perfectly diversified, 100 = single symbol
        n = len(shares)
        if n <= 1:
            return 100.0
        min_hhi = 1.0 / n
        return ((hhi - min_hhi) / (1.0 - min_hhi)) * 100.0

    def _detect_alerts(
        self,
        exposure_by_symbol: dict[str, float],
        exposure_by_direction: dict[str, float],
        total_exposure: float,
    ) -> list[CrowdingAlert]:
        alerts: list[CrowdingAlert] = []

        # Single symbol concentration
        for symbol, exp in exposure_by_symbol.items():
            if exp > self.max_single_symbol_pct:
                alerts.append(CrowdingAlert(
                    alert_type="single_symbol_concentration",
                    description=f"{symbol} > {self.max_single_symbol_pct}% of total exposure ({exp:.1f}%)",
                    severity="medium",
                    symbol=symbol,
                    exposure_pct=exp,
                ))

        # Total exposure too high
        if total_exposure > self.max_total_exposure_pct:
            alerts.append(CrowdingAlert(
                alert_type="total_exposure",
                description=f"Total exposure {total_exposure:.1f}% exceeds {self.max_total_exposure_pct}%",
                severity="high",
            ))

        # All bots on same side of same symbol
        symbol_direction_bots: dict[tuple[str, str], list[str]] = defaultdict(list)
        for bot_id, positions in self.position_details.items():
            for pos in positions:
                key = (pos["symbol"], pos["direction"])
                symbol_direction_bots[key].append(bot_id)

        for (symbol, direction), bots in symbol_direction_bots.items():
            if len(bots) >= 3:  # 3+ bots on same side of same asset
                alerts.append(CrowdingAlert(
                    alert_type="same_side",
                    description=f"{len(bots)} bots all {direction} on {symbol}",
                    severity="high",
                    bots_involved=bots,
                    symbol=symbol,
                ))

        return alerts
