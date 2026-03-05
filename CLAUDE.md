# Trading Assistant

Personal trading assistant agent system — orchestrates analysis of multiple
trading bots across VPSes.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/               # 740+ tests, all should pass
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

## Test Commands

```bash
pytest tests/                           # all tests
pytest tests/test_handlers.py -v        # handler tests
pytest tests/ -k "integration"          # integration tests only
```
