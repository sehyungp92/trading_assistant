# tests/test_drawdown_analysis.py
"""Tests for drawdown analysis schemas."""
from schemas.drawdown_analysis import DrawdownEpisode, DrawdownAttribution


class TestDrawdownEpisode:
    def test_creates_episode(self):
        ep = DrawdownEpisode(
            bot_id="bot1",
            start_date="2026-02-20",
            end_date="2026-02-25",
            peak_pnl=5000.0,
            trough_pnl=4200.0,
            drawdown_pct=16.0,
            trade_count=12,
            duration_days=5,
        )
        assert ep.drawdown_pct == 16.0
        assert ep.duration_days == 5

    def test_recovery_flag(self):
        ep = DrawdownEpisode(
            bot_id="bot1",
            start_date="2026-02-20",
            end_date="2026-02-25",
            peak_pnl=5000.0,
            trough_pnl=4200.0,
            recovered=True,
            recovery_date="2026-02-28",
        )
        assert ep.recovered is True


class TestDrawdownAttribution:
    def test_creates_attribution(self):
        attr = DrawdownAttribution(
            bot_id="bot1",
            date="2026-03-01",
            episodes=[
                DrawdownEpisode(
                    bot_id="bot1",
                    start_date="2026-02-20",
                    end_date="2026-02-25",
                    peak_pnl=5000.0,
                    trough_pnl=4200.0,
                    drawdown_pct=16.0,
                    trade_count=12,
                    duration_days=5,
                ),
            ],
            top_contributing_root_causes={"regime_mismatch": 5, "weak_signal": 3},
            largest_single_loss_pct=4.2,
        )
        assert len(attr.episodes) == 1
        assert attr.top_contributing_root_causes["regime_mismatch"] == 5

    def test_max_drawdown_property(self):
        attr = DrawdownAttribution(
            bot_id="bot1",
            date="2026-03-01",
            episodes=[
                DrawdownEpisode(bot_id="bot1", start_date="a", end_date="b",
                                peak_pnl=100, trough_pnl=90, drawdown_pct=10.0),
                DrawdownEpisode(bot_id="bot1", start_date="c", end_date="d",
                                peak_pnl=200, trough_pnl=150, drawdown_pct=25.0),
            ],
        )
        assert attr.max_drawdown_pct == 25.0

    def test_empty_episodes(self):
        attr = DrawdownAttribution(bot_id="bot1", date="2026-03-01")
        assert attr.max_drawdown_pct == 0.0
