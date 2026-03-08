# Trading Assistant

An agent system that monitors multiple trading bots across VPSes, automatically triages errors, and continuously analyzes trading performance to propose parameter, structural, and portfolio-level improvements — closing the loop between observation and action over time.

## What It Does

### Monitors bot health and catches problems early

- Ingests every trade, missed opportunity, daily snapshot, and error event from each bot in real-time via a relay service
- Classifies error severity deterministically (CRITICAL/HIGH/MEDIUM/LOW) and routes by complexity — obvious fixes get a Claude-generated diagnosis and draft PR, complex issues get flagged for human investigation
- Tracks error rates per bot with sliding-window spike detection (>3/hour auto-promotes severity)
- Monitors bot heartbeats and alerts on gaps (>2h warning, >4h critical)
- Runs morning and evening proactive scans for unusual losses (>2σ from 30-day mean) and repeated error patterns

### Analyzes trading performance across multiple dimensions

Every day, raw event data is transformed into a curated analysis package per bot covering:

- **Trade quality**: winners, losers, process failures (quality score < 60), and notable missed opportunities
- **Regime analysis**: PnL, win rate, and trade count broken down by market regime
- **Filter effectiveness**: per-filter block count, saved PnL, and missed PnL
- **Time-of-day patterns**: hourly PnL and win rate distributions
- **Slippage**: per-symbol and per-hour spread distributions (mean, median, p75, p95)
- **Exit efficiency**: premature exit rates by exit reason and regime
- **Signal attribution**: per signal factor win rate, PnL contribution, and trend
- **Drawdown episodes**: equity curve segmentation with duration, root cause attribution, and recovery tracking
- **Root cause distribution**: across a 21-type controlled taxonomy

Claude Code is invoked daily and weekly to interpret this data, producing reports delivered via Telegram, Discord, or email.

### Optimizes parameters through walk-forward optimization

The WFO pipeline runs grid search over configured parameter spaces with:

- Anchored or rolling fold generation on historical trade data
- Out-of-sample validation per fold with consensus parameter selection
- Cost sensitivity testing at multiple slippage multipliers
- Neighborhood stability testing (nearby parameters should also perform well)
- Regime stability testing (profitable across market regime types)
- Temporal leakage auditing
- Safety flag generation and a final recommendation of ADOPT, TEST_FURTHER, or REJECT

When parameters pass, the autonomous pipeline backtests the specific change, validates safety, and sends a Telegram approval card. On human approval, it creates a GitHub PR with the parameter file changes. After merge, deployment monitoring tracks the bot for regressions and auto-creates a rollback PR if performance degrades.

### Proposes structural and portfolio-level improvements

Beyond parameter tuning, the strategy engine runs 11 detectors looking for higher-level issues:

- Alpha decay and signal decay (declining signal-to-outcome correlation)
- Exit timing issues (premature exits based on MFE/exit efficiency)
- Correlation breakdown between bots
- Position sizing mismatches (size vs. outcome)
- Filter interaction effects (multi-filter combinations)
- Factor correlation drift
- Microstructure issues (fill quality, adverse selection)

These findings feed into Claude's weekly analysis, which produces structural suggestions (e.g., "add a regime-aware exit rule", "split this strategy into two variants"). Portfolio-level analysis computes cross-bot exposure concentration, crowding alerts, and risk-parity allocation recommendations with Calmar ratio tilt. Cross-bot transfer proposals identify validated patterns from one bot that may apply to others, scored by regime distribution overlap and historical transfer success rates.

Structural suggestions surface in reports for human consideration — they are never auto-implemented.

### Learns from its own suggestions over time

Every suggestion the system makes is tracked through a full lifecycle:

```
Strategy Engine → suggestion with ID → SuggestionTracker
         ↓
User approves/rejects via Telegram → lifecycle update
         ↓
AutoOutcomeMeasurer (weekly) → 7-day pre/post performance comparison → outcomes.jsonl
         ↓
Per-category win rate scorecard → blocks suggestion types with poor track records
         ↓
Prediction accuracy tracking → confidence calibration on future analyses
         ↓
All of the above injected into the next analysis prompt
```

The system won't re-suggest rejected ideas. Categories with low success rates get their suggestions stripped before delivery. Confidence levels are capped based on historical prediction accuracy. Hypotheses that accumulate rejections and negative outcomes are auto-retired.

## Architecture

```
VPS Bots → Sidecar → Relay VPS → POST /events → EventQueue (SQLite)
                                                       ↓
                                              OrchestratorBrain
                                            (classify → create task)
                                                       ↓
                                                    Worker
                                              (pick task → run agent)
                                                       ↓
                                                  AgentRunner
                                          (assemble context → Claude CLI)
                                                       ↓
                                                runs/<id>/outputs/
                                                       ↓
                                          Telegram / Discord / Email
```

### Package Layout

```
orchestrator/   — FastAPI app, brain, worker, scheduler, event queue, handlers
analysis/       — prompt assemblers, strategy engine, context builders, response validation
skills/         — data pipelines, metrics builders, simulation runners, trackers
comms/          — Telegram, Discord, Email adapters + dispatcher + renderers
schemas/        — Pydantic v2 models for all data contracts
memory/         — policies/ (human-edited) + findings/ (system-written)
tests/          — pytest, asyncio_mode=auto, ~2000 tests
```

## Quick Start

**Prerequisites:** Python 3.12+

```bash
# Clone and install
git clone <repo-url>
cd trading-assistant
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your credentials (bot tokens, relay URL, etc.)

# Run tests
pytest tests/

# Start the orchestrator
uvicorn orchestrator.app:app --reload
```

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `BOT_IDS` | Yes | Comma-separated bot identifiers |
| `RELAY_URL` | Yes | Relay VPS endpoint URL |
| `RELAY_HMAC_SECRET` | Yes | HMAC-SHA256 secret for relay auth |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token for notifications |
| `TELEGRAM_CHAT_ID` | No | Telegram chat for reports |
| `DISCORD_BOT_TOKEN` | No | Discord bot token |
| `DISCORD_CHANNEL_ID` | No | Discord channel for reports |
| `SMTP_HOST/PORT/USER/PASS` | No | Email delivery configuration |
| `DATA_DIR` | No | Base data directory (default: `.`) |

See `orchestrator/config.py` for all settings.

## Scheduled Tasks

| Task | Schedule | What It Does |
|------|----------|--------------|
| Daily Analysis | 22:30 UTC | Per-bot daily report with quality gate |
| Weekly Summary | Sunday | Cross-bot review, strategy suggestions, structural proposals |
| WFO | Weekly/monthly | Walk-forward parameter optimization |
| Proactive Scanner | Morning + evening | Anomaly detection, heartbeat monitoring |
| Outcome Measurement | Sunday 10:00 UTC | Measures implemented suggestions against pre/post performance |
| Memory Consolidation | Sunday 09:00 UTC | Aggregates findings, generates hypothesis candidates |
| Transfer Outcome Tracking | Sunday 10:30 UTC | Measures cross-bot pattern transfer results |

Bug triage runs on-demand when HIGH+ severity errors arrive. Claude Code is invoked per-task (not always-running) — zero cost when idle.

## Human-in-the-Loop

The system is read-only with respect to bots — it never sends commands to them. All changes flow through GitHub PRs that require human action to merge:

- **Parameter changes**: backtested and safety-checked automatically, but require explicit Telegram approval before a PR is even created
- **Structural suggestions**: surface in reports only, never auto-implemented
- **Bug fix PRs**: generated for obvious fixes, always require human review
- **Rollback PRs**: created automatically on regression detection, still require human merge
- **Portfolio allocation**: recommendations only — reallocating capital is a manual action

Three-tier permission gates (auto / requires_approval / requires_double_approval) enforce this. Trading logic changes always require approval.

## Commands

```bash
# Start the orchestrator
uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000

# Manual task triggers
python -m orchestrator.scheduler --run-now daily_report
python -m orchestrator.scheduler --run-now weekly_summary

# Build curated data for a date
python skills/build_daily_metrics.py --date 2026-03-01

# Build portfolio risk card
python skills/compute_portfolio_risk.py --date 2026-03-01

# Run tests
pytest tests/
pytest tests/ -k "integration"
pytest tests/test_handlers.py -v
```

## Tech Stack

- **Python 3.12**, FastAPI, uvicorn, APScheduler
- **SQLite** for event queue + task tracking
- **Claude Code CLI** invoked per-task
- **Pydantic v2** for all data contracts
- **Telegram Bot API**, discord.py, SMTP for notifications
- **HMAC-SHA256** for relay authentication
