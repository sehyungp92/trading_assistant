# Trading Assistant

A local orchestrator that ingests structured event data from multiple trading
bots on remote VPSes, runs analysis via configurable agent runtimes (Claude CLI,
Codex CLI, or API providers), and delivers reports/actions via Telegram, Discord,
and email.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/               # 3154 tests, all should pass
uvicorn orchestrator.app:app --reload   # start orchestrator on :8000
```

## Architecture

```
orchestrator/   — brain, worker, scheduler, event queue, handlers, config
  db/           — SQLite schema + queue (idempotent dedup by event_id)
analysis/       — prompt assemblers, strategy engine, context builders, response parser/validator
skills/         — data pipelines (daily metrics, WFO, backtesting, ground truth, parameter search, etc.)
comms/          — Telegram, Discord, Email adapters + dispatcher + renderers + message bus
schemas/        — all Pydantic v2 models
memory/         — policies/ (human-edited, versioned) + findings/ (system-written, time-scoped)
  skills/       — markdown instructions per agent type (daily_analysis, weekly_summary, etc.)
tests/          — pytest, asyncio_mode=auto
docs/           — architecture docs, plans, assessments
_references/    — reference codebases (openclaw, thepopebot, trading bots)
```

## Tech Stack

- **Python 3.12+**, FastAPI, uvicorn, APScheduler
- **SQLite** for queue + task tracking (no Postgres/Redis needed)
- **Agent runtimes** (Claude CLI, Codex CLI) invoked on-demand per task (not a daemon)
- **Telegram Bot API** (long polling), **discord.py**, **imaplib/smtplib**
- **Pydantic v2** for all schemas, **pandas** for data processing

## Key Patterns

- All prompt assemblers return `PromptPackage` (schemas/prompt_package.py), not raw dicts
- Channel adapters extend `BaseChannel` ABC (comms/base_channel.py) for lifecycle + retry
- Brain routing is deterministic — no LLM calls for event routing
- Event dedup via SHA-256 `event_id`
- Three-tier permissions: auto / requires_approval / requires_double_approval
- SQLite for event queue + task registry (single-user system)
- HMAC-SHA256 for relay auth

## Key Design Decisions

1. **Claude Code is invoked per-task, not always-running.** The worker writes a run folder with context files, invokes `claude` CLI, reads outputs. Cost = zero when idle.
2. **Every event has a deterministic `event_id`** (SHA256 hash of bot_id + timestamp + type + payload_key, truncated to 16 chars). Deduplication enforced at relay, gateway, and queue layers.
3. **Memory is split: policies (stable, versioned) vs findings (mutable, time-scoped).** The system never modifies policies autonomously. Findings are additive and time-stamped.
4. **Permission gates enforce human-in-the-loop.** Three tiers defined in `memory/policies/v1/permission_gates.md`. Trading logic changes always require approval.
5. **All inbound messages are untrusted.** `input_sanitizer.py` strips prompt injection patterns before any message reaches an agent.
6. **Reports have a Definition of Done.** `schemas/report_checklist.py` must pass before delivery. Incomplete reports are flagged.
7. **Instrumentation failures must not affect trading.** This system is read-only with respect to bots — it consumes their data, never sends commands to them.
8. **Agent invocations are stateless.** All context must be assembled in the run folder. Claude Code has no memory between invocations.

## Configuration

Copy `.env.example` to `.env` and set your credentials.
Bot IDs: comma-separated in `BOT_IDS` env var.
See `orchestrator/config.py` for all settings.

### Per-Bot Timezones

Set `BOT_TIMEZONES` to map bots to IANA timezones:
```
BOT_TIMEZONES=k_stock_trader:Asia/Seoul,us_bot:US/Eastern
```
Unlisted bots default to UTC. This controls:
- Daily analysis trigger time (market close + delay per bot group)
- Hourly performance bucketing (local hours, not UTC)
- Slippage hour bucketing
- Morning/evening scan date boundaries

Key files: `schemas/bot_config.py`, `orchestrator/tz_utils.py`, `orchestrator/catchup.py`.

### Auto-Start (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1
```

Missed scheduled jobs (laptop off) are caught up on startup via `orchestrator/catchup.py`.
APScheduler `misfire_grace_time` also allows late-firing (12h for daily, 48h for weekly/WFO).

## Event Flow

```
VPS Bot → Relay VPS → (poll) → EventQueue → Brain → Worker → Handler → Agent Runtime → Notification
```

## Event Schema From Bots

All bots emit these event types (JSONL, via sidecar):

- **TradeEvent** — entry + exit with signal, regime, filters, process quality score, root causes
- **MissedOpportunityEvent** — blocked signals with simulation policy + backfilled outcomes
- **DailySnapshot** — end-of-day aggregate per bot
- **ErrorEvent** — exceptions with stack traces

Every event includes `EventMetadata`: event_id, bot_id, exchange_timestamp, local_timestamp, clock_skew_ms, data_source_id, bar_id.

Root causes use a controlled taxonomy: `regime_mismatch`, `weak_signal`, `strong_signal`, `late_entry`, `early_exit`, `premature_stop`, `slippage_spike`, `good_execution`, `filter_blocked_good`, `filter_saved_bad`, `risk_cap_hit`, `data_gap`, `order_reject`, `latency_spike`, `correlation_crowding`, `funding_adverse`, `funding_favorable`, `regime_aligned`, `normal_loss`, `normal_win`, `exceptional_win`.

## Agent Types + Context Assembly

When invoking an agent for a task, `agent_runner.py` assembles:

```
SYSTEM: memory/policies/v1/agent.md + soul.md + trading_rules.md
CONTEXT: relevant findings (corrections, patterns) from last 30 days
TASK: task-specific prompt from memory/skills/<skill>.md
DATA: files from data/curated/<date>/<bot>/
OUTPUT DIR: runs/<task_id>/
```

| Agent | Trigger | Key Inputs | Key Outputs |
|-------|---------|-----------|-------------|
| Daily Analysis | cron per timezone | curated daily data, portfolio risk card | daily_report.md, report_checklist |
| Weekly Summary | Sunday cron | 7 daily reports, 30d rolling metrics | weekly_report.md |
| Strategy Refinement | weekly report flags issue | 90d bot data filtered by regime | refinement_proposal.md |
| WFO | weekly/monthly cron | historical data + wfo_config.yaml | wfo_report.md, draft PR |
| Bug Triage | HIGH error event | stack trace + source files | diagnosis + optional fix PR |
| Comms | after any report | report markdown | formatted Telegram/Discord/email |

## Multi-LLM Configuration

The system supports multiple agent runtime providers. Set via env vars or
the `/api/agent-preferences` endpoint.

| Provider | Runtime | Env Vars |
|----------|---------|----------|
| `claude_max` (default) | Claude CLI | Claude Max subscription |
| `codex_pro` | Codex CLI | ChatGPT Plus/Pro login |
| `zai_coding_plan` | Claude CLI (redirected) | `ZAI_API_KEY` |
| `openrouter` | Claude CLI (redirected) | `OPENROUTER_API_KEY` |

Key files: `orchestrator/agent_runner.py`, `orchestrator/agent_preferences.py`,
`orchestrator/provider_auth.py`, `orchestrator/invocation_builder.py`,
`schemas/agent_preferences.py`.

Features:
- Per-workflow provider overrides (`AgentPreferences.overrides`)
- Automatic fallback chains with cooldown (`orchestrator/provider_cooldown.py`)
- Per-workflow tuning: timeout, max_turns, allowed_tools (`WorkflowTuning`)
- Cost tracking per invocation (`orchestrator/cost_tracker.py` → `data/cost_log.jsonl`)

## Feedback Loop Flow

```
Strategy Engine → StrategySuggestion → _record_suggestions() → SuggestionTracker (suggestions.jsonl)
                                                                        ↓
User: "approve suggestion #abc123"  →  FeedbackHandler.parse()  →  SUGGESTION_ACCEPT
                                                                        ↓
                                     SuggestionTracker.implement()  →  status = IMPLEMENTED
                                                                        ↓
AutoOutcomeMeasurer (Sun 10:00 UTC)  →  measures IMPLEMENTED  →  outcomes.jsonl
                                                                        ↓
ForecastTracker.record_week()  →  forecast_history.jsonl  →  meta-analysis (calibration)
                                                                        ↓
ContextBuilder.base_package()  →  outcome_measurements + forecast_meta_analysis in next prompt
```

Key files:
- `skills/suggestion_tracker.py` — JSONL-backed lifecycle (proposed → accepted → implemented → measured)
- `skills/forecast_tracker.py` — rolling accuracy meta-analysis and confidence calibration
- `skills/hypothesis_library.py` — adaptive JSONL-backed hypothesis catalog with lifecycle tracking
- `skills/transfer_proposal_builder.py` — cross-bot pattern transfer with outcome measurement
- `skills/prediction_tracker.py` — structured prediction recording and evaluation
- `skills/suggestion_scorer.py` — per-category success rates from outcomes
- `analysis/response_parser.py` — extracts structured JSON from Claude's markdown responses
- `analysis/response_validator.py` — strips blocked suggestions, enforces calibration constraints

## Commands

```bash
# Start gateway
uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000

# Run worker (separate process or background)
python -m orchestrator.worker

# Manual task trigger
python -m orchestrator.scheduler --run-now daily_report
python -m orchestrator.scheduler --run-now weekly_summary

# All tests
pytest tests/
pytest tests/test_handlers.py -v        # handler tests
pytest tests/ -k "integration"          # integration tests only
```
