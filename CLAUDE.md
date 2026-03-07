# Trading Assistant

Personal trading assistant agent system — orchestrates analysis of multiple
trading bots across VPSes.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/               # 1576+ tests, all should pass
uvicorn orchestrator.app:app --reload   # start orchestrator on :8000
```

## Architecture

```
orchestrator/   — brain, worker, scheduler, event queue, handlers, config
analysis/       — prompt assemblers, strategy engine, context builders
skills/         — data pipelines (daily metrics, WFO, bug triage, proactive scanner)
comms/          — Telegram, Discord, Email adapters + dispatcher + renderers
relay/          — VPS event buffer service (deployed separately on a VPS)
schemas/        — all Pydantic models
memory/         — policies/ (human-edited) + findings/ (system-written)
tests/          — pytest, asyncio_mode=auto
```

## Key Patterns

- All prompt assemblers return `PromptPackage` (schemas/prompt_package.py), not raw dicts
- Channel adapters extend `BaseChannel` ABC (comms/base_channel.py) for lifecycle + retry
- Brain routing is deterministic — no LLM calls for event routing
- Event dedup via SHA-256 `event_id`
- Three-tier permissions: auto / requires_approval / requires_double_approval
- SQLite for event queue + task registry (single-user system)
- HMAC-SHA256 for relay auth

## Configuration

Copy `.env.example` to `.env` and set your credentials.
Bot IDs: comma-separated in `BOT_IDS` env var.
See `orchestrator/config.py` for all settings.

## Event Flow

```
VPS Bot → Relay VPS → (poll) → EventQueue → Brain → Worker → Handler → Claude CLI → Notification
```

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

## Test Commands

```bash
pytest tests/                           # all tests
pytest tests/test_handlers.py -v        # handler tests
pytest tests/ -k "integration"          # integration tests only
```
