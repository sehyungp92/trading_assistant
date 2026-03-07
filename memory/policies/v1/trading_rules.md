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

## Regime-Specific Behavior
- Trending regime: trend-following strategies should show positive expectancy; if not, flag for review
- Ranging regime: mean-reversion strategies should outperform; trend strategies may underperform (normal)
- Volatile regime: all strategies may show wider drawdowns — compare to regime-adjusted benchmarks
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

## What NOT to Suggest
- Changes based on single-day anomalies (wait for pattern confirmation)
- Suggestions previously rejected by the user (check rejected_suggestions context)
- Parameter changes outside the strategy's tested range without WFO validation
- Removing filters without quantifying the cost of false positives they currently block
- Strategy additions or removals without portfolio-level impact analysis
