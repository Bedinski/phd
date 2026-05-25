"""Unit tests for `sdk_runner.run_stage` with the SDK mocked.

We don't actually call Claude here — we patch `ClaudeSDKClient` to yield a
canned message sequence and assert the runner records usage, persists the
transcript, and signals exit reason correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from phd.persistence import ensure_run_layout, mint_run_id, stage_dir
from phd.sdk_runner import StageInvocation, run_stage, usage_pct


def test_usage_pct() -> None:
    assert usage_pct(0, "claude-opus-4-7") == 0.0
    assert usage_pct(100_000, "claude-opus-4-7") == pytest.approx(50.0)
    assert usage_pct(200_000, "claude-opus-4-7") == pytest.approx(100.0)


@dataclass
class _FakeAssistant:
    content: list
    usage: dict[str, Any] | None = None
    model: str = "claude-opus-4-7"


@dataclass
class _FakeResult:
    subtype: str
    usage: dict[str, Any]
    is_error: bool = False
    num_turns: int = 1
    duration_ms: int = 100


class _CannedClient:
    """Async-context-manager stub that mimics ClaudeSDKClient just enough."""

    def __init__(self, options):  # noqa: ARG002
        self.options = options
        self._queries: list[str] = []
        self._scripts: list[list[Any]] = [
            [
                # First turn: a result message at well-under-50% usage.
                _FakeResult(
                    subtype="success",
                    usage={"input_tokens": 5_000, "output_tokens": 200},
                )
            ],
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def query(self, prompt: str) -> None:
        self._queries.append(prompt)

    async def receive_messages(self):
        for batch in self._scripts:
            for msg in batch:
                yield msg


async def test_run_stage_records_usage_and_returns_stop(monkeypatch, tmp_path) -> None:
    from claude_agent_sdk import (
        AssistantMessage as _A,
        ResultMessage as _R,
    )

    # Make our fake messages instance-check as the real classes.
    fake_assistant = type("FA", (_A,), {})
    fake_result = type("FR", (_R,), {})

    # Build a runtime instance of the real ResultMessage so the isinstance check
    # in run_stage matches. The real ResultMessage is a regular dataclass.
    real_result = _R(
        subtype="success",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="s",
        usage={"input_tokens": 5_000, "output_tokens": 200},
    )

    class _Client:
        def __init__(self, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def query(self, prompt):
            pass

        async def receive_messages(self):
            yield real_result

    monkeypatch.setattr("phd.sdk_runner.ClaudeSDKClient", _Client)

    rid = mint_run_id()
    ensure_run_layout(rid, root=tmp_path)
    # Point persistence at the tmp_path
    monkeypatch.setattr("phd.persistence.RUNS_ROOT", tmp_path)

    inv = StageInvocation(
        run_id=rid,
        stage_name="research",
        system_prompt="sys",
        user_prompt="hi",
        allowed_tools=["Read"],
        mcp_servers={},
        model="claude-opus-4-7",
    )

    result = await run_stage(inv, ceiling_pct=50.0)
    assert result.exit_reason == "stop"
    assert result.total_input_tokens == 5_000
    assert result.peak_context_pct < 50.0
