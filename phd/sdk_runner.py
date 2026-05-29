"""ClaudeSDKClient wrapper: per-stage session, live context-ceiling enforcement.

The orchestrator never talks to `claude_agent_sdk` directly; it calls
`run_stage(...)` here. This module owns the structural rules:

  * fresh session per stage (context isolation)
  * a *proactive* context ceiling enforced mid-turn via hooks + the SDK's real
    `get_context_usage()` (not an end-of-turn token estimate)
  * a tool firewall: each stage may only call its allowlisted tools, and no
    stage may Read another stage's raw transcript (anti-dilution)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookMatcher,
    Message,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from .persistence import append_transcript

log = logging.getLogger(__name__)


# Tools whose results grow the context window. When the hard ceiling is hit we
# deny these so the model cannot keep inflating context; Write is always allowed
# so it can still persist a checkpoint or the final artifact.
CONTEXT_GROWING_TOOLS = frozenset(
    {
        "WebSearch",
        "WebFetch",
        "Read",
        "Grep",
        "Glob",
        "mcp__phd__fetch_ohlcv",
        "mcp__phd__run_backtest",
        "mcp__phd__walk_forward",
        "mcp__phd__param_sensitivity",
        "mcp__phd__monte_carlo",
    }
)

ALWAYS_ALLOWED_TOOLS = frozenset({"Write", "Edit", "TodoWrite"})


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
    cwd: str | None = None
    setting_sources: list[str] | None = None
    soft_ceiling_pct: float = 50.0
    hard_ceiling_pct: float = 65.0
    # Required artifacts whose existence on disk is the stage's success condition.
    # The Stop hook will refuse to let the turn end until they exist.
    required_artifacts: tuple[Path, ...] = ()
    # Claude Code skills the stage may invoke (e.g. /deep-research).
    skills: tuple[str, ...] = ()


@dataclass
class StageRunResult:
    """What `run_stage` returns to the orchestrator."""

    stage_name: str
    exit_reason: str  # "stop" | "ceiling_hit" | "error"
    final_text: str
    peak_context_pct: float
    total_output_tokens: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Ceiling state shared between hooks and the run loop
# ---------------------------------------------------------------------------


@dataclass
class _CeilingState:
    """Mutable state shared by the Pre/PostToolUse hooks for one stage run."""

    inv: StageInvocation
    client: ClaudeSDKClient | None = None
    peak_pct: float = 0.0
    soft_nudged: bool = False
    hard_blocked: bool = False
    allow_set: frozenset[str] = field(default_factory=frozenset)

    async def current_pct(self) -> float | None:
        """Authoritative context occupancy via the SDK. Defensive: never raise
        into a hook (that would abort the turn)."""
        if self.client is None:
            return None
        try:
            cu = await asyncio.wait_for(self.client.get_context_usage(), timeout=15)
        except Exception as exc:  # noqa: BLE001
            log.debug("get_context_usage failed in hook: %s", exc)
            return None
        pct = float(cu.get("percentage", 0.0) or 0.0)
        self.peak_pct = max(self.peak_pct, pct)
        return pct


def _is_transcript_read(tool_name: str, tool_input: dict) -> bool:
    """Block a downstream stage from slurping an upstream stage's raw transcript
    (or any other run's artifacts) — that's the dilution vector we guard against."""
    if tool_name not in ("Read", "Grep", "Glob"):
        return False
    target = str(tool_input.get("file_path") or tool_input.get("path") or tool_input.get("pattern") or "")
    return "transcript.jsonl" in target


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


async def run_stage(inv: StageInvocation) -> StageRunResult:
    """Open a fresh ClaudeSDKClient for one stage with proactive ceiling hooks."""
    state = _CeilingState(inv=inv, allow_set=frozenset(inv.allowed_tools))

    async def can_use_tool(
        tool_name: str, tool_input: dict, ctx: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        # Firewall 1: enforce the stage allowlist ourselves (works under root,
        # where bypassPermissions is forbidden, and is stricter than allowed_tools).
        base = tool_name.split("__")[0] if not tool_name.startswith("mcp__") else tool_name
        if tool_name not in state.allow_set and base not in state.allow_set:
            return PermissionResultDeny(message=f"tool {tool_name} not in stage allowlist")
        # Firewall 2: never let a stage read another stage's raw transcript.
        if _is_transcript_read(tool_name, tool_input):
            return PermissionResultDeny(
                message="reading transcript.jsonl is forbidden — use the validated artifact"
            )
        return PermissionResultAllow()

    async def pre_tool_hook(inp: dict, tool_use_id: str | None, ctx: HookContext) -> dict:
        tool_name = inp.get("tool_name", "")
        if tool_name in CONTEXT_GROWING_TOOLS:
            pct = await state.current_pct()
            if pct is not None and pct >= inv.hard_ceiling_pct:
                state.hard_blocked = True
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": (
                            f"HARD CONTEXT CEILING reached ({pct:.0f}% >= "
                            f"{inv.hard_ceiling_pct:.0f}%). Do NOT gather more data. "
                            "Immediately write your final artifact (or checkpoint.json "
                            "if incomplete) using Write, then end your turn."
                        ),
                    }
                }
        return {}

    async def post_tool_hook(inp: dict, tool_use_id: str | None, ctx: HookContext) -> dict:
        pct = await state.current_pct()
        if pct is not None and pct >= inv.soft_ceiling_pct and not state.soft_nudged:
            state.soft_nudged = True
            log.warning("stage %s: soft ceiling %.0f%% reached at %.0f%%",
                        inv.stage_name, inv.soft_ceiling_pct, pct)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        f"[CONTEXT BUDGET WARNING] You are at {pct:.0f}% of the context "
                        f"window (soft ceiling {inv.soft_ceiling_pct:.0f}%). Wrap up now: "
                        "stop gathering new information and write your final artifact. If "
                        "you cannot finish, write a concise checkpoint.json describing "
                        "what's done, what remains, and the next concrete steps, then stop."
                    ),
                }
            }
        return {}

    async def stop_hook(inp: dict, tool_use_id: str | None, ctx: HookContext) -> dict:
        # Safety net: many models will narrate findings instead of writing the
        # artifact (observed live with Haiku 4.5). Refuse the stop until the
        # required files exist on disk. `stop_hook_active` guards against
        # infinite blocks: if we've already blocked once, let the agent finish.
        if inp.get("stop_hook_active"):
            return {}
        missing = [p for p in inv.required_artifacts if not p.exists()]
        if not missing:
            return {}
        missing_list = "\n".join(f"  - {p}" for p in missing)
        log.warning(
            "stage %s: Stop hook blocked — required artifacts missing: %s",
            inv.stage_name,
            [str(p) for p in missing],
        )
        return {
            "decision": "block",
            "reason": (
                "REQUIRED ARTIFACT MISSING. You cannot end your turn until "
                "every required output file exists on disk. Use the Write tool "
                "to create each of these paths with content that validates "
                "against the schema in your system prompt. Do not narrate; "
                "WRITE THE FILE(S):\n"
                f"{missing_list}\n"
                "After writing, re-confirm each path exists, then end your turn."
            ),
        }

    options = ClaudeAgentOptions(
        allowed_tools=inv.allowed_tools,
        mcp_servers=inv.mcp_servers,
        system_prompt=inv.system_prompt,
        permission_mode="default",  # bypassPermissions is forbidden under root
        can_use_tool=can_use_tool,
        setting_sources=inv.setting_sources or ["user"],
        cwd=inv.cwd,
        model=inv.model,
        effort=inv.effort,
        skills=list(inv.skills) if inv.skills else None,
        hooks={
            "PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_hook])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[post_tool_hook])],
            "Stop": [HookMatcher(matcher=None, hooks=[stop_hook])],
        },
    )

    total_out = 0
    final_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            state.client = client
            await client.query(inv.user_prompt)

            async for msg in client.receive_response():
                _persist(inv.run_id, inv.stage_name, msg)
                if isinstance(msg, AssistantMessage):
                    captured = _capture_text(msg)
                    if captured:
                        final_text = captured
                if isinstance(msg, ResultMessage):
                    total_out += int((msg.usage or {}).get("output_tokens", 0) or 0)
                    break

            # Authoritative final reading (between-turns call is always safe).
            final_pct = await state.current_pct()
            if final_pct is not None:
                state.peak_pct = max(state.peak_pct, final_pct)

    except Exception as exc:  # noqa: BLE001 — surface any SDK error to the orchestrator
        log.exception("stage %s failed", inv.stage_name)
        return StageRunResult(
            stage_name=inv.stage_name,
            exit_reason="error",
            final_text=final_text,
            peak_context_pct=state.peak_pct,
            total_output_tokens=total_out,
            error=str(exc),
        )

    ceiling_hit = state.soft_nudged or state.hard_blocked
    return StageRunResult(
        stage_name=inv.stage_name,
        exit_reason="ceiling_hit" if ceiling_hit else "stop",
        final_text=final_text,
        peak_context_pct=state.peak_pct,
        total_output_tokens=total_out,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _capture_text(msg: AssistantMessage) -> str:
    """Concatenate TextBlocks. ThinkingBlocks are deliberately discarded."""
    return "\n".join(b.text for b in msg.content if isinstance(b, TextBlock)).strip()


def _persist(run_id: str, stage_name: str, msg: Message) -> None:
    event = _message_to_event(msg)
    if event is not None:
        append_transcript(run_id, stage_name, event)


def _message_to_event(msg: Message) -> dict | None:
    if isinstance(msg, AssistantMessage):
        return {"type": "assistant", "blocks": [_block_summary(b) for b in msg.content]}
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
