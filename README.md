# phd — Claude Deep Research Workflow for Trading Strategies

A 6-stage multi-agent pipeline that researches automated retail trading
strategies, independently reviews the findings, synthesizes them into an
executable strategy spec, runs a vectorbt backtest, performs walk-forward +
Monte Carlo robustness checks, and gates deployment to an Alpaca paper-trading
account behind a manual approval step.

## Why this exists

Single-agent research pipelines drift, dilute findings, and exhaust context
windows on long runs. This pipeline solves all three:

1. **Context ≤ 50% per stage** — each stage runs in its own
   `ClaudeSDKClient` session (fresh 0% context). A `PostToolUse` hook reads the
   SDK's real `get_context_usage()` after every tool call and, at the soft
   ceiling (50%), injects a directive to wrap up; a `PreToolUse` hook denies
   further context-growing tools at the hard ceiling (65%), forcing the agent
   to checkpoint. The orchestrator then respawns the stage with only that
   checkpoint loaded. Enforcement is *proactive and mid-turn*, not a post-hoc
   token estimate.
2. **Persistence + clean handoffs** — every stage writes a pydantic-validated
   JSON artifact under `runs/<run_id>/`. The next stage receives those
   artifacts *injected inline* into its prompt (deterministic — no "please read
   the file"), and a tool firewall blocks any stage from reading another
   stage's raw `transcript.jsonl`. Handoffs are integrity-checked (run-id match,
   and the backtest result's spec-hash must match the frozen spec).
3. **No skew / dilution** — citation-bound research claims, an independent
   review stage that re-fetches each cited URL, a frozen strategy spec, and
   per-stage tool allowlists keep findings honest as they move downstream.

## Pipeline

```
01 Research  →  02 Review  →  03 Strategy Synthesis  →  04 Backtest  →  05 Risk Check  →  06 Report
```

| Stage | Tools | In | Out |
|-------|-------|----|----|
| Research | WebSearch · WebFetch · Read · Write | thesis + universe | `findings.json` |
| Review | WebFetch · Read · Write | `findings.json` | `critique.json` + `decision.json` |
| Strategy Synthesis | Read · Write | findings | `strategy_spec.json` (frozen) |
| Backtest | phd-mcp tools | spec | `backtest_result.json` |
| Risk Check | phd-mcp tools | spec + result | `risk_assessment.json` |
| Report | Read · Write | all artifacts | `final_report.md` + `DEPLOY` / `HOLD` / `DISCARD` |

All stages run on Claude Opus 4.7 with `effort="max"` (extended thinking at
maximum budget) by default. Use `--cheap` or `--model-<stage>` to override.

## Setup

```bash
uv sync                              # install deps
uv pip install -e .                  # editable install
cp .env.example .env                 # add Alpaca paper credentials
```

The Agent SDK inherits your local Claude Code auth, so no `ANTHROPIC_API_KEY`
is needed if you're already signed into Claude Code.

Stages run with `permission_mode="default"` plus a `can_use_tool` callback that
enforces each stage's tool allowlist. (We avoid `bypassPermissions` because the
CLI refuses it when running as root, e.g. in containers/CI.)

## Usage

```bash
# Full pipeline
phd research --universe sp500 --strategy-type "post-earnings drift mean reversion"

# Smaller live smoke test
phd research --universe sp500_top10 --strategy-type "SMA crossover" --cheap

# Resume a stalled run (reuses valid artifacts; re-runs what's missing)
phd resume <run_id>
phd resume <run_id> --from-stage backtest   # force re-run from a stage onward

# Inspect a finished run
phd inspect <run_id>

# Manual deploy gate (writes to Alpaca paper account)
phd deploy <run_id> --dry-run
phd deploy <run_id>
```

## Layout

```
phd/                  # package
├── orchestrator.py   # 6-stage driver with revision loop
├── sdk_runner.py     # ClaudeSDKClient wrapper + 50%-ceiling hook
├── persistence.py    # run_id / artifact I/O
├── models.py         # pydantic inter-stage contracts
├── stages/           # per-stage prompts + tool allowlists + model config
├── prompts/          # markdown prompt templates
├── mcp_servers/      # in-process MCP tools (market data, backtest, risk)
├── backtest/         # vectorbt-driven backtest core
└── deploy/           # alpaca-py paper trading

runs/<run_id>/        # per-run artifacts (gitignored)
data_cache/           # OHLCV pickle cache (gitignored)
```

## Tests

```bash
uv run pytest                    # all
uv run pytest tests/test_models.py -v
uv run pytest -k backtest        # backtest smoke
```

## Status / risks

- **yfinance fragility**: rate-limits / endpoint drift; `phd.backtest.data`
  pickle-caches. Migrate to Alpaca data client when credentials are wired.
- **Opus quota**: `effort="max"` everywhere is heavy. Use `--cheap` for
  iteration.
- **Universe-scan token cost**: 50%-ceiling triggers checkpoint-respawns;
  expect more wall-clock time per stage than transcript depth would suggest.
