"""Tests for `sdk_runner`: the tool firewall + the run loop with a faked client.

We don't call Claude. We fake ClaudeSDKClient with a canned ResultMessage and a
get_context_usage() return so we can assert the runner records the real peak
percentage and exit reason, and we test the transcript-read firewall directly.
"""

from __future__ import annotations

from claude_agent_sdk import ResultMessage

from phd.persistence import ensure_run_layout, mint_run_id
from phd.sdk_runner import (
    CONTEXT_GROWING_TOOLS,
    StageInvocation,
    _is_transcript_read,
    run_stage,
)


def test_transcript_read_is_blocked() -> None:
    assert _is_transcript_read("Read", {"file_path": "runs/x/01_research/transcript.jsonl"})
    assert _is_transcript_read("Grep", {"pattern": "runs/x/02_review/transcript.jsonl"})
    # Reading the actual artifact is fine.
    assert not _is_transcript_read("Read", {"file_path": "runs/x/01_research/findings.json"})
    # Non-read tools are unaffected.
    assert not _is_transcript_read("Write", {"file_path": "x/transcript.jsonl"})


def test_context_growing_tools_include_web_and_mcp() -> None:
    assert "WebFetch" in CONTEXT_GROWING_TOOLS
    assert "WebSearch" in CONTEXT_GROWING_TOOLS
    assert "mcp__phd__run_backtest" in CONTEXT_GROWING_TOOLS
    # Write must never be gated — the model needs it to persist a checkpoint.
    assert "Write" not in CONTEXT_GROWING_TOOLS


def _make_result(pct_first: int) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="s",
        usage={"input_tokens": 1000, "output_tokens": 50},
    )


async def test_run_stage_records_peak_pct_and_stop(monkeypatch, tmp_path) -> None:
    """A clean turn under the ceiling returns exit_reason='stop' and the real
    peak percentage from get_context_usage()."""

    class _FakeClient:
        def __init__(self, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, prompt):
            pass

        async def receive_response(self):
            yield _make_result(20)

        async def get_context_usage(self):
            return {"percentage": 20, "totalTokens": 40000, "maxTokens": 200000}

    monkeypatch.setattr("phd.sdk_runner.ClaudeSDKClient", _FakeClient)
    monkeypatch.setattr("phd.persistence.RUNS_ROOT", tmp_path)

    rid = mint_run_id()
    ensure_run_layout(rid, root=tmp_path)

    inv = StageInvocation(
        run_id=rid,
        stage_name="research",
        system_prompt="sys",
        user_prompt="go",
        allowed_tools=["Read", "Write"],
        mcp_servers={},
        model="claude-opus-4-7",
        soft_ceiling_pct=50.0,
        hard_ceiling_pct=65.0,
    )
    result = await run_stage(inv)
    assert result.exit_reason == "stop"
    assert result.peak_context_pct == 20.0
    assert result.total_output_tokens == 50


async def test_run_stage_surfaces_errors(monkeypatch, tmp_path) -> None:
    class _BoomClient:
        def __init__(self, options):
            pass

        async def __aenter__(self):
            raise RuntimeError("connect failed")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr("phd.sdk_runner.ClaudeSDKClient", _BoomClient)
    monkeypatch.setattr("phd.persistence.RUNS_ROOT", tmp_path)

    rid = mint_run_id()
    ensure_run_layout(rid, root=tmp_path)
    inv = StageInvocation(
        run_id=rid, stage_name="research", system_prompt="s", user_prompt="g",
        allowed_tools=["Read"], mcp_servers={}, model="claude-opus-4-7",
    )
    result = await run_stage(inv)
    assert result.exit_reason == "error"
    assert "connect failed" in (result.error or "")
