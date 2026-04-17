# Trading Assistant

An agent system that monitors multiple trading bots across VPSes, automatically triages errors, and continuously analyses trading performance to propose parameter, structural, and portfolio-level improvements — closing the loop between observation and action over time.

## What It Does

### Monitors bot health and catches problems early

- Ingests every trade, missed opportunity, daily snapshot, and error event from each bot in real-time via a relay service
- Classifies error severity deterministically (CRITICAL/HIGH/MEDIUM/LOW) and routes by complexity — obvious fixes get a Claude-generated diagnosis and draft PR, complex issues get flagged for human investigation
- Tracks error rates per bot with sliding-window spike detection (>3/hour auto-promotes severity)
- Monitors bot heartbeats and alerts on gaps (>2h warning, >4h critical)
- Runs morning and evening proactive scans for unusual losses (>2σ from 30-day mean) and repeated error patterns

### Analyses trading performance across multiple dimensions

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

The agent runtime can be switched per workflow between Claude Max, Codex Pro, Z.AI Coding Plan, and OpenRouter-backed Claude-compatible models, while reports still flow through Telegram, Discord, or email.

### Optimises parameters through walk-forward optimisation

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

Beyond parameter tuning, the strategy engine runs 19 detectors across four levels:

- **Signal quality** (4): alpha decay, signal decay, component signal decay, factor correlation decay
- **Execution quality** (4): exit timing issues, microstructure issues (fill quality, adverse selection), stress entry patterns, position sizing mismatches
- **Regime awareness** (3): regime config effectiveness, regime transition cost, time-of-day patterns
- **Portfolio structure** (5): family imbalance, correlation concentration, drawdown tier miscalibration, coordination gaps, heat cap utilisation
- **Pattern detection** (3): drawdown patterns, filter interaction effects, correlation breakdown

These findings feed into the configured weekly analysis provider, which produces structural suggestions (e.g., "add a regime-aware exit rule", "split this strategy into two variants"). Portfolio-level analysis computes cross-bot exposure concentration, crowding alerts, and risk-parity allocation recommendations with Calmar ratio tilt. Cross-bot transfer proposals identify validated patterns from one bot that may apply to others, scored by regime distribution overlap and historical transfer success rates.

Structural suggestions surface in reports for human consideration — they are never auto-implemented.

### Learns from its own suggestions over time

Every suggestion the system makes is tracked through a full lifecycle:

```
                         Strategy Engine (19 detectors)
                                    ↓
                    suggestion with ID → SuggestionTracker
                          ↓                       ↓
                   parameter change          structural proposal
                          ↓                       ↓
                   ParameterSearcher         6 validation gates
                   (grid search, backtest,   (hypothesis track record,
                    robustness testing)       category win rate, confidence,
                          ↓                   acceptance criteria)
                   search_reports.jsonl            ↓
                          ↓                  report for human review
                   Telegram approval card
                          ↓
              User approves/rejects via Telegram → lifecycle update
                          ↓
  ┌─────────────────── Learning Cycle (weekly) ───────────────────┐
  │  ground truth snapshot → ledger delta → route pending items   │
  │  parameter suggestions → inner loop    structural → tracker   │
  └───────────────────────────────────────────────────────────────┘
                          ↓
         AutoOutcomeMeasurer → 7-day pre/post comparison → outcomes.jsonl
                          ↓
         Outcome-derived lessons → learning ledger entries
         Correction patterns (count ≥ 3) → synthesised lessons
                          ↓
         ExperimentManager auto-conclusion (when statistically decisive)
           → accept/reject linked suggestion
           → record hypothesis outcome (positive/negative)
                          ↓
  ┌─────────── Fed back into every subsequent prompt ─────────────┐
  │  ground truth trends · search reports · outcome measurements  │
  │  category scorecards · prediction accuracy · convergence      │
  │  loop health metrics · instrumentation readiness · bias data  │
  └───────────────────────────────────────────────────────────────┘
```

The system won't re-suggest rejected ideas. Categories with low success rates get their suggestions stripped before delivery. Confidence levels are capped based on historical prediction accuracy. Hypotheses that accumulate rejections and negative outcomes are auto-retired. Measured outcomes and persistent correction patterns are automatically synthesised into learning ledger entries, so the system's accumulated experience informs future analysis without manual curation.

Crucially, the learning system is self-correcting — it monitors whether its own optimisation process is working and adjusts accordingly:

- **Convergence tracking** synthesises composite score trends, prediction accuracy, outcome ratios, and scorecard evolution into an overall signal (improving/degrading/oscillating/stable). When oscillation is detected, the LLM is instructed to hold steady rather than reversing last week's suggestions.
- **Loop health metrics** quantify six operational KPIs per cycle: proposal-to-measurement latency, oscillation severity, transfer success rate, recalibration effectiveness, suggestions per cycle, and measurement coverage. These are injected into every prompt so the agent can see where the learning pipeline itself is underperforming.
- **Temporal decay on scorecards** applies 5%/week exponential decay to outcome weights, matching the learning ledger's existing decay rate. Categories recover from early failures as old negatives fade, making the inner loop adaptive rather than accumulative.
- **Directional bias correction** detects systematic optimism or pessimism per metric in the LLM's prediction track record and reduces confidence on predictions that match the bias pattern (capped at 20%).
- **Per-detector confidence calibration** gives each of the strategy engine's 19 detectors an empirical confidence multiplier derived from its outcome history. This is distinct from threshold learning (which adapts *when* a detector fires) — confidence calibration adapts *how much to trust* a detection once it fires.
- **Suggestion pre-validation** filters strategy engine suggestions against the scorecard at recording time, preventing category leakage when the scorecard was unavailable during report generation.
- **Structural proposal validation** runs six parity gates before any structural suggestion reaches a report: hypothesis track record (block if effectiveness ≤ 0 or retired), category track record (block if win rate < 0.3 with n ≥ 5), simplicity criterion (block marginal suggestions), acceptance criteria presence (require at least one well-formed metric criterion), low-confidence block (< 0.4), and empirical calibration adjustment.
- **Instrumentation readiness scoring** evaluates each bot across eight capability categories (basic analysis, process quality, exit analysis, regime analysis, slippage analysis, factor attribution, signal health, drawdown analysis) and injects per-bot readiness reports into prompts. The agent sees which bots have sufficient data coverage for which analysis types, preventing confident conclusions from incomplete instrumentation.
- **Experiment auto-conclusion** tracks parameter experiments to statistical resolution, then escalates through the full chain: conclude experiment → accept/reject linked suggestion → record hypothesis outcome (positive or negative). This closes the loop from "the system proposed a change" through "the change was tested" to "the hypothesis that motivated it was updated."
- **Discovery and convergence context** in prompt instructions tells the LLM how to use automated pattern discoveries and convergence status that were previously loaded into data but invisible to the instruction set.

## Design Philosophy: OpenClaw Governance + Hermes Memory + Autoresearch Optimisation

This system combines three reference architectures to serve a single goal: continuously improve trading bot performance while keeping a human in control of what matters.

### From OpenClaw: disposable agents, permanent knowledge

The core architectural choice from OpenClaw is that the agent is disposable and the knowledge is permanent. The CLI runtime is a stateless executor — it reads context, runs a skill, writes results, and exits. Everything that compounds value over time — outcome measurements, suggestion histories, forecast calibration, correction logs, prompt patterns — lives in local files and SQLite on your machine: owned, queryable, backed up, inspectable. The agent is the lens; the data is the asset.

This gives you three things concretely. The agent brain is swappable: when a better model arrives or a new provider launches, you swap the executor and everything continues — the skills, the memory, the learning history, the bot configurations. The system already supports four runtime providers (Claude Max, Codex Pro, ZAI, OpenRouter) with automatic fallback chains. Every interaction is auditable: because the agent reads explicit context (memory files, skill prompts, event payloads) and writes explicit outputs (run folder artifacts, JSONL records, parsed analyses), you can trace exactly what the agent saw when it made a particular recommendation. And the scheduler gives you proactive intelligence without infrastructure: APScheduler fires cron jobs for daily analysis, weekly summaries, heartbeat monitoring, and proactive scanning — just scheduled subprocesses on your local machine, not a server or cloud function.

This disposability depends on clear, enforceable boundaries between what the system can change and what only a human can change.

**Three-tier permission gates** (`memory/policies/v1/permission_gates.md`) classify every action the system can take. Routine data processing runs unattended. Strategy and parameter changes require explicit Telegram approval. Critical paths — the ground truth evaluation function, deployment scripts, kill switches, and the policy documents themselves — require double approval. The system enforces these at the file-path level during PR review.

**Separated memory tiers** keep human intent stable. `memory/policies/` contains the identity document (`soul.md`), trading rules, and permission definitions — versioned, human-edited only, never modified autonomously. `memory/findings/` is where the system writes its observations: outcome measurements, correction logs, prompt patterns, and failure modes. These are additive, time-stamped, and subject to 90-day decay. The system learns by accumulating findings; it cannot rewrite its own values.

**Deterministic routing** keeps the critical path predictable. The orchestrator brain classifies events and creates tasks without any LLM involvement. The LLM is invoked only for analysis and reasoning, never for routing or scheduling decisions.

### From Hermes: memory orchestration that improves what the next run sees

The trading assistant already had a rich learning substrate — outcome measurements, scorecards, convergence tracking, calibration, retrospective synthesis. What it lacked was the last mile: ensuring those signals actually reached the agent runtime in ranked, relevant form and that the artifacts produced by each run fed back into the next one. From Hermes, it inherits the memory orchestration layer that closes that gap.

**Prompt delivery closes the last-mile gap.** The context builder assembled corrections, skill instructions, and outcome histories — but these were sidecar files in the run directory that the CLI runtime might never read. `InvocationBuilder.build_full_prompt()` now merges all learning context directly into the prompt text: instructions, corrections, skill methodology, and ranked learning cards. The agent sees its accumulated experience in every run, so the upstream learning system actually influences the analysis it was built to improve.

**Learning cards turn past lessons into ranked retrieval.** Eight evidence types — corrections, measured outcomes, discoveries, causal outcome reasonings, confidence recalibrations, spurious outcome flags, hypothesis results, and validator blocks — are ingested into `LearningCard` objects scored by recency, impact, confidence, and context match. Cards are populated automatically after outcome reasoning and weekly memory consolidation. `ranked_for_prompt()` injects the most relevant cards first, scoped by bot for single-bot workflows, so a daily analysis for a specific bot sees the corrections and outcomes that matter to that bot rather than an unranked global dump.

**Session history flows through every assembler.** The session store is passed through all six assembler paths into `base_package()`, giving each analysis run visibility into its own recent invocation history — what was analysed, what was recommended, what the agent saw last time. This prevents redundant re-analysis and lets the agent build on prior reasoning rather than starting from a blank slate each run.

### From Autoresearch: an immutable objective function and a separable inner loop

The core insight from Autoresearch is that a self-improving system needs an evaluation function it cannot modify, and an optimisation loop that runs without the LLM in the critical path.

**The ground truth computer** (`skills/ground_truth_computer.py`) is the system's equivalent of Autoresearch's `evaluate_bpb()`. It computes a single composite performance score from daily trading data using z-score normalised metrics with fixed weights centralised in `schemas/objective_weights.py`: expected return (30%), Calmar ratio (20%), profit factor (15%), expectancy (15%), inverse drawdown (10%), and process quality (10%). These six weights are the single source of truth — the parameter searcher uses the same constants (renormalised to exclude process quality, which only applies to human-facing evaluation). This function is immutable — it lives behind the double-approval permission gate, meaning neither the system nor a single human action can change how performance is measured. Every downstream decision in the learning system — cycle verdicts, retrospective synthesis, experiment acceptance, calibration tracking — flows from this one number.

**The parameter search inner loop** (`skills/parameter_searcher.py`) runs neighbourhood exploration without any LLM involvement. When the LLM proposes a parameter change, the inner loop takes over: it builds a candidate grid around the proposed value, backtests each candidate, tests robustness across regimes and cost multipliers, and ranks results by a composite score using the same objective weights as the ground truth computer (renormalised to exclude process quality, since backtests have no process signal). The best value often differs from what the LLM suggested — the LLM identifies *what* to change, the inner loop finds the *optimal value*. Results route to APPROVE, EXPERIMENT, or DISCARD based on improvement thresholds and robustness scores.

**The weekly learning cycle** (`skills/learning_cycle.py`) ties both architectures together. Each week it computes ground truth snapshots at the start and end of the period, records the delta in the learning ledger, and routes pending suggestions by type — parameter changes to the inner loop, structural changes to the experiment tracker. The composite score trajectory is injected into every subsequent LLM prompt, so the agent always sees whether its suggestions are actually improving performance against the immutable metric.

### How they integrate into a single learning system

The three architectures are not just additive — they share a single objective function and a unified suggestion lifecycle, so that every proposal flows through the same measurement and feedback path regardless of whether it originated from the LLM or the inner loop.

**Shared objective function.** The ground truth computer and the parameter searcher draw their weights from the same source (`schemas/objective_weights.py`). The ground truth computer uses all six components with z-score normalisation against a bot's own history; the parameter searcher uses the same five non-process weights renormalised to sum to 1.0, applied as improvement ratios against a baseline simulation. The scoring *methods* differ because they serve different purposes — evaluation vs exploration — but the relative priority of metrics is identical. When weights are updated (behind double-approval), both systems move together.

**Unified suggestion lifecycle.** Whether a suggestion comes from the LLM's structural analysis or from the parameter searcher's grid exploration, it enters `SuggestionTracker` with a deterministic ID and follows the same path: proposed → approved → implemented → measured. The `AutoOutcomeMeasurer` doesn't distinguish origin — it measures pre/post performance deltas against the ground truth composite for every implemented suggestion. Outcomes feed back into per-category scorecards, which gate future suggestions of both types.

**Experiment-to-hypothesis traceability.** When the inner loop runs a parameter experiment, `ExperimentManager` tracks it to statistical conclusion. On resolution, the auto-conclusion chain updates the linked suggestion (accept or reject) and records the outcome against the hypothesis that motivated it (positive or negative). This means the LLM's next analysis prompt reflects not just "this parameter change was tested" but "the hypothesis behind it was strengthened or weakened" — connecting empirical results back to the analytical reasoning that produced them.

**Bidirectional context flow.** The weekly learning cycle is the single orchestration point. It computes ground truth deltas, routes pending parameter suggestions to the inner loop, routes structural changes to the experiment tracker, and records the period's ledger entry. Results flow back through `analysis/context_builder.py`, which loads ground truth trends, search reports, outcome measurements, category scorecards, prediction accuracy, convergence reports, loop health metrics, instrumentation readiness, and directional bias data into every prompt. The LLM sees what the inner loop explored (preventing re-proposals), what past suggestions achieved (calibrating confidence), which categories have poor track records (auto-stripping weak suggestions), whether the system is converging or oscillating (avoiding destabilising reversals), where its directional biases lie (correcting systematic over-optimism), and which bots have sufficient instrumentation (preventing conclusions from incomplete data). The inner loop, in turn, receives updated detector confidence multipliers and scorecard weights from the outcomes its suggestions produce — closing the loop in both directions.

**Governance prevents gaming.** OpenClaw's permission gates ensure neither loop can drift its own objective. Because `soul.md` and the ground truth formula sit behind human-only policy controls, the system cannot shift to something easier to optimise. Structural proposals pass through six validation gates before reaching reports, ensuring only well-supported, high-confidence suggestions with measurable acceptance criteria survive. The system learns from its own history through an evaluation function it cannot change, governed by permissions it cannot override.

### Matching the change type to the right tool

The learning system routes each type of improvement to whichever tool handles it best. 

Parameter-level changes — adjusting a stop-loss percentage, a signal threshold, a filter sensitivity — are numerical optimisation problems with a clear objective function. These belong to the autonomous inner loop, which can explore a candidate grid, backtest every value, and rank results by composite score without any LLM involvement. The inner loop is faster, cheaper, and more rigorous than asking a language model to guess the right number.

Structural changes — adding a regime-aware exit rule, splitting a strategy into variants, redesigning a filter interaction — require genuine analytical reasoning: reading performance data in context, forming hypotheses about *why* something is failing, and proposing changes that may not have an obvious metric to optimise against. These belong to the LLM, which receives the full context package (ground truth trends, outcome history, regime analysis, root cause distributions) and reasons about what to change and why. Structural proposals always surface in reports for human review — they are never auto-implemented, because the right structural change depends on judgment that neither a grid search nor a language model should make alone.

Portfolio-level changes — rebalancing family allocation weights, adjusting risk caps, modifying coordination rules between strategies, changing drawdown tier multipliers — operate above individual strategies. The strategy engine runs portfolio-level detectors (correlation crowding, family imbalance, drawdown synchronisation) and produces structured `PortfolioProposal` objects. Allocation proposals are validated through a what-if analysis (`skills/portfolio_what_if.py`) that rescales historical family PnL under the proposed weights to estimate portfolio Calmar, Sharpe, and max drawdown before any change is recorded. When per-family trade-level data is available, the what-if uses individual trades instead of daily aggregates — producing intra-day max drawdown from the per-trade equity curve, Sortino ratio, profit factor, and PnL-by-regime breakdowns that daily aggregation would mask. Portfolio outcomes are measured over a 30-day observation window (vs 7 days for strategy-level), and regime shifts during the window yield INCONCLUSIVE verdicts rather than false signals. Each strategy carries an archetype profile (`data/strategy_profiles.yaml`) that sets archetype-specific detection thresholds — trend-following strategies get looser alpha decay thresholds than intraday momentum, for example — so the engine's suggestions respect each strategy's inherent characteristics rather than applying uniform rules.

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
                                          (assemble context → provider profile → CLI runtime)
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
tests/          — pytest, asyncio_mode=auto, ~3370 tests
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

- Local runs auto-load `.env`; exported process environment variables still take precedence.
- On Windows, one-time background startup is: copy `.env`, run `powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1`, then log in. The logon task keeps the supervisor running in the background and startup catch-up replays missed cron work after downtime.

| Variable | Required | Description |
|----------|----------|-------------|
| `BOT_IDS` | Yes | Comma-separated bot identifiers |
| `RELAY_URL` | Yes | Relay VPS endpoint URL |
| `RELAY_HMAC_SECRET` | Yes | HMAC-SHA256 secret for relay auth |
| `CLAUDE_COMMAND` | No | Claude Code CLI command or absolute path |
| `CLAUDE_COMMAND_ARGS` | No | JSON array of launcher args prepended before Claude runtime args |
| `CODEX_COMMAND` | No | Codex CLI command or absolute path |
| `CODEX_COMMAND_ARGS` | No | JSON array of launcher args prepended before Codex runtime args |
| `ZAI_API_KEY` | No | Enables the Z.AI Coding Plan profile through Claude Code-compatible env overrides |
| `OPENROUTER_API_KEY` | No | Enables the OpenRouter profile through Claude Code-compatible env overrides |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token for notifications and `/settings` |
| `TELEGRAM_CHAT_ID` | No | Telegram chat for reports |
| `DISCORD_BOT_TOKEN` | No | Discord bot token |
| `DISCORD_CHANNEL_ID` | No | Discord channel for reports |
| `SMTP_HOST/PORT/USER/PASS` | No | Email delivery configuration |
| `DATA_DIR` | No | Base data directory (default: `.`) |

See `orchestrator/config.py` for all settings.

### Agent Provider Switching

Agent selection is persisted in `data/agent_preferences.json`.

- Global default applies everywhere unless a workflow override exists.
- Workflow overrides are available for `daily_analysis`, `weekly_analysis`, `wfo`, and `triage`.
- Initial values can be seeded from `AGENT_PROVIDER` and the per-workflow `*_AGENT_PROVIDER` env vars, but only when no persisted preferences file exists yet.
- `GET /agent/preferences` returns the persisted default, overrides, effective workflow selections, and provider readiness.
- `PUT /agent/preferences` updates the global default or workflow overrides. A `null` model means "use the provider default model."
- Telegram exposes the same provider switching via `/settings` and the control-panel `Settings` button. Telegram changes provider only; model overrides remain HTTP-only.

Built-in provider defaults:

- `claude_max` -> `sonnet`
- `codex_pro` -> `gpt-5.4`
- `zai_coding_plan` -> `glm-5`
- `openrouter` -> `minimax/minimax-m2.5`

Claude Max setup:

- Install Claude Code locally and log in with `claude auth login`.
- Verify the CLI reports `loggedIn=true`, `authMethod=claude.ai`, and `subscriptionType=max` via `claude auth status`.
- `GET /agent/preferences` marks `claude_max` unavailable until that Max-auth check passes.

Launcher args:

- `CLAUDE_COMMAND_ARGS` and `CODEX_COMMAND_ARGS` accept JSON arrays and are inserted before the runtime-specific CLI args.
- This supports wrapper scripts and Windows launcher patterns such as `CODEX_COMMAND=wsl.exe` with `CODEX_COMMAND_ARGS=["--","codex"]`.

Provider notes:

- `claude_max` runs through the local Claude Code CLI with `stream-json` output, fresh per-run sessions, and live runtime events on the existing SSE bus. The runner clears Anthropic API/base-url/model override env vars before Claude-backed launches so Max runs cannot silently fall back to API-credit billing.
- `codex_pro` runs through the local Codex CLI in read-only sandbox mode with streamed JSONL parsing, raw run artifacts, and the same runtime SSE events as Claude-backed providers.
- `codex_pro` requires a launchable Codex command plus local ChatGPT auth in `~/.codex/auth.json`. API-key auth is rejected for this subscription-backed profile, and OpenAI API/base-url env vars are cleared before launch so runs cannot drift into API billing.
- On Windows, `CODEX_COMMAND` may need a wrapper or launcher abstraction rather than a directly executable WindowsApp binary. Use `CODEX_COMMAND_ARGS` for that instead of editing orchestrator code.
- The app surfaces Codex diagnostics when it detects high-latency local settings such as `model_reasoning_effort = "xhigh"` or recent shared-state lock/slow-write symptoms.
- `zai_coding_plan` intentionally stays Claude Code-backed so tool use still works and usage can count against the Z.AI coding plan. It only requires the Claude CLI itself plus `ZAI_API_KEY`; it does not depend on Claude Max auth.
- `openrouter` uses the Claude-compatible Anthropic surface for parity with the existing tool-using flow. It only requires the Claude CLI itself plus `OPENROUTER_API_KEY`; it does not depend on Claude Max auth.
- Daily, weekly, and WFO Claude workflows use read-only tool allowlists (`Read`, `Grep`, `Glob`). Triage keeps `Bash` enabled alongside those read-only tools.

## Scheduled Tasks

| Task | Schedule | What It Does |
|------|----------|--------------|
| Daily Analysis | 06:00 UTC (configurable per bot timezone) | Per-bot daily report with quality gate |
| Weekly Summary | Sunday | Cross-bot review, strategy suggestions, structural proposals |
| WFO | Weekly/monthly | Walk-forward parameter optimisation |
| Proactive Scanner | Morning + evening | Anomaly detection, heartbeat monitoring |
| Learning Cycle | Weekly | Ground truth snapshots, suggestion routing, ledger deltas |
| Outcome Measurement | Sunday 10:00 UTC | Measures implemented suggestions against pre/post performance |
| Memory Consolidation | Sunday 09:00 UTC | Aggregates findings, generates hypothesis candidates |
| Transfer Outcome Tracking | Sunday 10:30 UTC | Measures cross-bot pattern transfer results |
| Threshold Learning | Periodic | Adapts detector firing thresholds from outcome history |
| Experiment Check | Periodic | Auto-concludes experiments that reach statistical significance |
| Discovery Analysis | Periodic | Raw JSONL pattern discovery outside detector coverage |
| Reliability Verification | Periodic | Verifies system health and data pipeline integrity |

Bug triage runs on-demand when HIGH+ severity errors arrive. The configured agent runtime is invoked per-task (not always-running), which keeps subscription-backed profiles at zero extra cost while idle.

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
- **Claude Code CLI** and **Codex CLI**, with Claude-compatible provider overlays for Z.AI and OpenRouter
- **Pydantic v2** for all data contracts
- **Telegram Bot API**, discord.py, SMTP for notifications
- **HMAC-SHA256** for relay authentication
