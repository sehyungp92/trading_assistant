# Trading Rules — v1

## Constraints
- Maximum 5 actionable suggestions per daily report
- Maximum 5 actionable suggestions per weekly report, ranked by expected Calmar impact
- All suggestions must be specific, testable, and quantified with dollar impact
- Never recommend increasing position size without WFO validation
- Missed opportunity calculations must disclose simulation assumptions
- No suggestion should be based on fewer than 30 trades or 30 days of data

## Analysis Standards
- Always distinguish process errors from normal losses using root cause taxonomy
- Report process quality scores alongside PnL
- Include regime context for every trade analysis
- Cross-bot correlation must be checked daily
- Slippage analysis must compare spread_at_entry to expected fill

## Position Sizing Rules
- unit_risk_pct changes require WFO validation before implementation
- Daily stop levels are per-strategy hard limits — never recommend removing them
- Heat cap (total simultaneous risk) must not exceed per-bot maximum
- Size multiplier adjustments (e.g., direction filter boost) must be backed by >50 trade sample

## Per-Bot Risk Budgets
- Each bot has a maximum drawdown threshold defined in its simulation policy
- Portfolio-level max drawdown is the sum-weighted max of individual bot drawdowns
- Crowding alerts trigger when HHI concentration exceeds 0.4 or correlation >0.7
- Any single bot exceeding 60% of its max drawdown triggers a CRITICAL alert

## Regime-Specific Behavior by Strategy Archetype

### Trend-Following (ATRSS)
- Expected to profit in trending regimes, underperform in ranging — underperformance
  in ranging is NORMAL, not a signal to change parameters
- Alpha decay threshold is looser (high variance by nature)
- Do NOT suggest tightening stops to improve win rate — this destroys the fat-tail payoff

### Multi-TF Momentum (AKC_Helix_v40)
- Trend-sensitive but uses multiple timeframe confirmation — higher variance than pure trend
- Performs in trending and volatile regimes, underperforms in ranging
- Alpha decay threshold is loose like trend-following (high variance by nature)

### Breakout / Box Breakout (SWING_BREAKOUT_V3, NQDTC_v2.1)
- Expected to profit after compression, underperform in persistent trends
- High false-breakout rate is inherent — focus on cost-per-attempt, not win rate
- Time-of-day patterns are critical for intraday variants

### VWAP Pullback (VdubusNQ_v4)
- Intraday strategy — time-of-day analysis is relevant
- Expected to profit in trending regimes via pullback entries near VWAP
- Exit timing is a primary lever — tighter efficiency threshold than swing strategies

### Pullback (S5_PB, S5_DUAL)
- Expected to perform in moderate trends, fail in strong reversals
- Entry timing is the primary lever — analyze entry latency vs regime speed

### Divergence Swing (AKC_HELIX)
- Wider stops (SL=1.5R, TP=2.5R) are by design — do not suggest tightening
- Performs across regimes but with higher variance

### Intraday Momentum (IARIC_v1, ALCB_v1)
- Time-of-day analysis is HIGH relevance — opening hour vs midday vs close
- Slippage is the primary drag (3-5 bps) — analyze fill quality aggressively

### Opening Range Breakout (US_ORB_v1)
- First 30 minutes only — time window is non-negotiable
- Wider slippage (5 bps) in early session is expected, not a problem

### k_stock_trader — Korean Equity (KMP, KPR, PCIM, NULRIMOK)
- Separate VPS, separate capital pool (KRW), Asia/Seoul timezone — no cross-market coordination
- LONG-only strategies on KRX (Korea Exchange)
- KMP (momentum breakout): time-of-day is critical (09:15-14:30 KST), spread/slippage in KRW is a key drag
  Preferred regime: trending_up only — underperformance in ranging is EXPECTED
- KPR (VWAP pullback MEAN-REVERSION): prefers ranging/volatile markets, ADVERSE to trending
  VWAP depth and time-of-day late multiplier are the primary levers — do NOT evaluate like
  US VWAP pullback strategies which are trend-following
- PCIM (event-driven influencer signal): prefers trending_up/ranging, adverse to volatile
  Entry cutoff 10:00 KST, YouTube influencer signal quality drives performance
- NULRIMOK (flow following, swing/multi-day): leader tier quality and flow persistence are primary
  Highly dependent on institutional order flow — prefers trending_up/ranging, adverse to volatile
- Risk budget is independent: heat_cap_R=6.0 and daily_stop_R=3.0 are NOT shared with US portfolio
- Do NOT suggest cross-market transfers between k_stock and US strategies — different market
  microstructure, currency, and regulatory environment

## Inter-Strategy Coordination Rules
- ATRSS entry triggers AKC_HELIX stop tightening (same symbol) — evaluate if this
  fires correctly and whether the tightening threshold is appropriate
- ATRSS position triggers AKC_HELIX size boost (same symbol, same direction, 1.25x)
- AKC_Helix_v40 and NQDTC_v2.1 have 120-min cooldown (09:45-11:30 session) — check
  if cooldown prevented good setups or correctly avoided whipsaws
- VdubusNQ_v4 direction filter on NQDTC_v2.1 (agree=1.5x, oppose=0x) — evaluate
  signal agreement rate and quality of filtered trades
- Stock strategies: 50% size reduction on symbol collision between IARIC/ORB/ALCB

## General Regime Rules
- Regime mismatches in root causes are actionable only if persistent (>5 occurrences in 30 days)

## Actionable Suggestion Thresholds
- PARAMETER tier: must show >10% improvement in target metric over >50 trades, survive WFO robustness test
- FILTER tier: must show filter cost (blocked good trades) exceeding filter benefit (avoided bad trades) by >20%
- STRATEGY_VARIANT tier: must have theoretical justification + >30 day backtest showing Calmar improvement
- HYPOTHESIS tier: must include specific test plan with success/failure criteria before implementation

## Escalation Criteria
- CRITICAL (immediate Telegram): drawdown >80% of limit, error rate >3/hour, heartbeat missing >2 hours
- HIGH (next analysis cycle): process quality <50 for >3 consecutive trades, new error type, crowding alert
- NORMAL (daily report): parameter suggestions, filter adjustments, regime observations
- LOW (weekly report only): structural hypotheses, allocation rebalancing, long-term trend observations

## Portfolio-Level Change Constraints
- Maximum 15% allocation change per family per cycle (monthly cadence)
- Minimum 5% allocation floor per family — never reduce below this
- heat_cap_R changes limited to +/-10% per quarterly review, requires 90+ days of evidence
- Drawdown tiers can only be NARROWED, never loosened or removed
- Daily/weekly stop levels can only be TIGHTENED, never loosened
- Maximum 1 DEPLOYED portfolio change at a time (for clean attribution)
- Allocation changes require 60+ days of evidence; risk/drawdown changes require 90+ days
- All portfolio proposals must cite specific family metrics and projected portfolio Calmar impact
- Emergency reversals (portfolio composite drops >10% within 14 days) bypass cadence gates

## What NOT to Suggest
- Changes based on single-day anomalies (wait for pattern confirmation)
- Suggestions previously rejected by the user (check rejected_suggestions context)
- Parameter changes outside the strategy's tested range without WFO validation
- Removing filters without quantifying the cost of false positives they currently block
- Strategy additions or removals without portfolio-level impact analysis
