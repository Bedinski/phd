"""Stage definition + prompt rendering tests."""

from __future__ import annotations

from phd.stages import STAGE_ORDER, STAGES, ModelConfig, render_prompt
from phd.stages._models import HAIKU, OPUS, SONNET


def test_stage_order_complete() -> None:
    assert set(STAGE_ORDER) == set(STAGES.keys())
    assert STAGE_ORDER[0] == "research"
    assert STAGE_ORDER[-1] == "report"


def test_each_stage_has_disjoint_or_intentional_tools() -> None:
    # Anti-dilution mechanism #5: per-stage tool allowlists.
    assert "WebSearch" not in STAGES["review"].allowed_tools
    assert "WebSearch" in STAGES["research"].allowed_tools
    assert "WebFetch" not in STAGES["strategy_synthesis"].allowed_tools
    # Backtest gets MCP tools but no web tools.
    assert "WebSearch" not in STAGES["backtest"].allowed_tools
    assert any(t.startswith("mcp__phd__") for t in STAGES["backtest"].allowed_tools)
    # Risk check gets its own MCP subset.
    risk_tools = set(STAGES["risk_check"].allowed_tools)
    assert "mcp__phd__walk_forward" in risk_tools
    assert "mcp__phd__param_sensitivity" in risk_tools
    assert "mcp__phd__monte_carlo" in risk_tools


def test_all_prompts_render() -> None:
    ctx = {
        "run_id": "r-1",
        "thesis": "t",
        "universe": "u",
        "strategy_type": "sx",
        "today": "2026-05-25",
        "now_iso": "2026-05-25T12:00:00",
        "revision_round": 0,
        "revision_block": "",
        "research_questions": "1. q",
    }
    for stage in STAGES.values():
        out = render_prompt(stage.prompt_template, **ctx)
        assert "{run_id}" not in out  # no unfilled placeholders
        assert len(out) > 100


def test_model_config_default_is_opus_max() -> None:
    cfg = ModelConfig()
    for s in STAGE_ORDER:
        sm = cfg.get(s)
        assert sm.model == OPUS
        assert sm.effort == "max"


def test_model_config_cheap_collapses_to_sonnet_no_thinking() -> None:
    cfg = ModelConfig()
    cfg.cheap()
    for s in STAGE_ORDER:
        sm = cfg.get(s)
        assert sm.model == SONNET
        assert sm.effort is None


def test_model_config_alias_expansion() -> None:
    cfg = ModelConfig()
    cfg.override_stage("backtest", model="haiku")
    assert cfg.get("backtest").model == HAIKU
    cfg.override_stage("research", model="sonnet-4-6")
    assert cfg.get("research").model == SONNET


def test_model_config_effort_override() -> None:
    cfg = ModelConfig()
    cfg.override_stage("backtest", effort="low")
    assert cfg.get("backtest").effort == "low"
    cfg.override_stage("backtest", effort=None)
    assert cfg.get("backtest").effort is None
