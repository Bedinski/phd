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
   `ClaudeSDKClient` session (fresh 0% context) with a per-turn usage hook
   that checkpoints and respawns the session before crossing the ceiling.
2. **Persistence + clean handoffs** — every stage writes a pydantic-validated
   JSON artifact under `runs/<run_id>/`; the next stage reads only that
   artifact, never the prior transcript.
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

## Usage

```bash
# Full pipeline
phd research --universe sp500 --strategy-type "post-earnings drift mean reversion"

# Smaller live smoke test
phd research --universe sp500_top10 --strategy-type "SMA crossover" --cheap

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
