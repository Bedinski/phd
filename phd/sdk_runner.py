"""ClaudeSDKClient wrapper: per-stage session, usage monitoring, checkpoint loop.

The orchestrator never talks to `claude_agent_sdk` directly; it calls
`run_stage(...)` here. That keeps the structural rules (fresh session per
stage, 50%-ceiling enforcement, transcript-on-disk-only) in one place.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ContextUsageResponse,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from .persistence import append_transcript

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class StageInvocation:
    """Everything `run_stage` needs to drive one ClaudeSDKClient session."""

    run_id: str
    stage_name: str
    system_prompt: str
    user_prompt: str
    allowed_tools: list[str]
    mcp_servers: dict[str, Any]
    model: str
    effort: str | None = "max"
    permission_mode: str = "bypassPermissions"
    cwd: str | None = None
    setting_sources: list[str] | None = None


@dataclass
class StageRunResult:
    """What `run_stage` returns to the orchestrator."""

    stage_name: str
    exit_reason: str  # "stop" | "ceiling_hit" | "error"
    final_text: str
    total_input_tokens: int
    total_output_tokens: int
    peak_context_pct: float
    error: str | None = None


# ---------------------------------------------------------------------------
# 50%-ceiling helpers
# ---------------------------------------------------------------------------


# Context-window sizes by model (input tokens). Used to compute the 50% gate.
# Conservative defaults; can be overridden via PHD_CONTEXT_WINDOW_<model> env var.
_DEFAULT_WINDOWS = {
    "claude-opus-4-7": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}


def context_window_for(model: str) -> int:
    env = os.environ.get(f"PHD_CONTEXT_WINDOW_{model.upper().replace('-', '_')}")
    if env:
        return int(env)
    return _DEFAULT_WINDOWS.get(model, 200_000)


def usage_pct(used_input_tokens: int, model: str) -> float:
    window = context_window_for(model)
    if window <= 0:
        return 0.0
    return (used_input_tokens / window) * 100.0


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


async def run_stage(inv: StageInvocation, *, ceiling_pct: float = 50.0) -> StageRunResult:
    """Open a fresh ClaudeSDKClient for one stage. Stream messages, persist them,
    track token usage, inject a stop-and-checkpoint instruction on ceiling.

    The 50% ceiling is informational here: this function will signal `exit_reason
    == "ceiling_hit"` so the orchestrator can decide whether to respawn the
    stage with a checkpoint loaded in. Per-stage code is responsible for asking
    the model to write the checkpoint when nudged.
    """

    options = ClaudeAgentOptions(
        allowed_tools=inv.allowed_tools,
        mcp_servers=inv.mcp_servers,
        system_prompt=inv.system_prompt,
        permission_mode=inv.permission_mode,
        setting_sources=inv.setting_sources or ["user"],
        cwd=inv.cwd,
        model=inv.model,
        effort=inv.effort,
        include_partial_messages=False,
    )

    total_in = 0
    total_out = 0
    peak_pct = 0.0
    final_text = ""
    ceiling_hit = False
    nudged_for_checkpoint = False

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(inv.user_prompt)

            async for msg in client.receive_messages():
                # Persist every message for later debugging.
                _persist(inv.run_id, inv.stage_name, msg)

                # Track usage on assistant turns.
                if isinstance(msg, AssistantMessage):
                    captured = _capture_text(msg)
                    if captured:
                        final_text = captured

                if isinstance(msg, ResultMessage):
                    usage = msg.usage or {}
                    in_tok = int(usage.get("input_tokens", 0) or 0)
                    in_tok += int(usage.get("cache_read_input_tokens", 0) or 0)
                    out_tok = int(usage.get("output_tokens", 0) or 0)
                    total_in = max(total_in, in_tok)
                    total_out += out_tok
                    pct = usage_pct(in_tok, inv.model)
                    peak_pct = max(peak_pct, pct)

                    if pct >= ceiling_pct and not nudged_for_checkpoint:
                        nudged_for_checkpoint = True
                        ceiling_hit = True
                        log.warning(
                            "stage %s: context at %.1f%% (>=%.0f%% ceiling) — "
                            "nudging for checkpoint",
                            inv.stage_name,
                            pct,
                            ceiling_pct,
                        )
                        await client.query(
                            "STOP. Context budget is approaching the 50% ceiling. "
                            "Write a checkpoint file at "
                            f"runs/{inv.run_id}/{_stage_dirname(inv.stage_name)}/"
                            "checkpoint.json capturing: (a) what you have already "
                            "completed, (b) what remains, (c) the next concrete "
                            "actions. Then end your turn without doing more work."
                        )
                        continue

                    # Any ResultMessage means the turn loop is finished — exit
                    # regardless of subtype so we never spin on unknown values.
                    return StageRunResult(
                        stage_name=inv.stage_name,
                        exit_reason="ceiling_hit" if ceiling_hit else "stop",
                        final_text=final_text,
                        total_input_tokens=total_in,
                        total_output_tokens=total_out,
                        peak_context_pct=peak_pct,
                    )

    except Exception as exc:  # noqa: BLE001 — surface any SDK error to the orchestrator
        log.exception("stage %s failed", inv.stage_name)
        return StageRunResult(
            stage_name=inv.stage_name,
            exit_reason="error",
            final_text=final_text,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            peak_context_pct=peak_pct,
            error=str(exc),
        )

    return StageRunResult(
        stage_name=inv.stage_name,
        exit_reason="ceiling_hit" if ceiling_hit else "stop",
        final_text=final_text,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        peak_context_pct=peak_pct,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _stage_dirname(stage_name: str) -> str:
    from .persistence import STAGE_DIRS

    return STAGE_DIRS[stage_name]


def _capture_text(msg: AssistantMessage) -> str:
    """Concatenate TextBlocks from an assistant message. ThinkingBlocks are
    deliberately discarded — they're large and the orchestrator never needs them."""
    out: list[str] = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            out.append(block.text)
    return "\n".join(out).strip()


def _persist(run_id: str, stage_name: str, msg: Message) -> None:
    event = _message_to_event(msg)
    if event is not None:
        append_transcript(run_id, stage_name, event)


def _message_to_event(msg: Message) -> dict | None:
    """Compact one Message into a JSON-serializable transcript event."""
    if isinstance(msg, AssistantMessage):
        return {
            "type": "assistant",
            "blocks": [_block_summary(b) for b in msg.content],
        }
    if isinstance(msg, UserMessage):
        return {"type": "user", "content_kind": type(msg.content).__name__}
    if isinstance(msg, SystemMessage):
        return {"type": "system", "subtype": msg.subtype}
    if isinstance(msg, ResultMessage):
        return {
            "type": "result",
            "subtype": msg.subtype,
            "is_error": msg.is_error,
            "num_turns": msg.num_turns,
            "duration_ms": msg.duration_ms,
            "usage": msg.usage,
        }
    return None


def _block_summary(block: Any) -> dict:
    if isinstance(block, TextBlock):
        return {"kind": "text", "len": len(block.text)}
    if isinstance(block, ThinkingBlock):
        return {"kind": "thinking", "len": len(block.thinking)}
    if isinstance(block, ToolUseBlock):
        return {"kind": "tool_use", "name": block.name, "id": block.id}
    if isinstance(block, ToolResultBlock):
        return {"kind": "tool_result", "tool_use_id": block.tool_use_id, "is_error": block.is_error}
    return {"kind": type(block).__name__}
