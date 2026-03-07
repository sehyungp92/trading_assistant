# WFO Pipeline Skill

## Purpose
Run walk-forward optimization to validate parameter changes. Produces statistically robust parameter recommendations with cost sensitivity and regime stability checks.

## Trigger
Weekly/monthly cron, or spawned by weekly analysis when parameter-tier suggestions need validation.

## Pipeline
1. **Config Load** — WFO config YAML defines parameter space, fold strategy, cost model
2. **Fold Generation** — anchored or rolling window folds (train/test splits)
3. **Leakage Detection** — feature audit + label forward-fill checks
4. **Backtest Simulation** — trade replay with parameter filtering per fold
5. **Parameter Optimization** — grid search over parameter space
6. **Robustness Testing** — neighborhood stability + regime stability + safety flags
7. **Cost Sensitivity** — test result fragility at higher transaction costs
8. **Report Build** — markdown summary + structured JSON output
9. **Claude Review** — 10-step validation framework (confirmatory, not exploratory)
10. **Decision** — ADOPT / TEST_FURTHER / REJECT

## Fold Strategies
- **Anchored**: expanding window (train grows, test fixed) — more stable
- **Rolling**: fixed-size sliding window — more responsive to regime changes

## Cost Model
- Fixed costs (commissions)
- Spread costs (bid-ask)
- Empirical slippage (from slippage_stats.json)

## Input Data
- WFO config YAML (parameter_space, fold_config, cost_model)
- Historical trades from curated data (trades.jsonl, missed.jsonl)

## Output
- wfo_report.json (optimization results, cost sensitivity, robustness scores, safety flags)
- wfo_report.md (human-readable summary)
- wfo_analysis.md (Claude's review and decision)

## Quality Criteria
- In-sample vs out-of-sample gap >30% is a red flag for overfitting
- Parameter stability across folds — jumping parameters suggest noise
- Regime robustness — parameters must work across different market conditions
- Cost model must use realistic assumptions
- Never recommend implementing parameters not tested out-of-sample
- Consider if parameter space is too narrow (missing better regions)
