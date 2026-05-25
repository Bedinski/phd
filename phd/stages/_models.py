"""Centralized model + effort table for all 6 stages.

This is the only place model identifiers appear. CLI overrides patch this at
runtime via `ModelConfig.override(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

OPUS = "claude-opus-4-7"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class StageModel:
    model: str
    effort: str | None  # "low" | "medium" | "high" | "xhigh" | "max" | None


DEFAULT_TABLE: dict[str, StageModel] = {
    "research":            StageModel(OPUS,   "max"),
    "review":              StageModel(OPUS,   "max"),
    "strategy_synthesis":  StageModel(OPUS,   "max"),
    "backtest":            StageModel(OPUS,   "max"),
    "risk_check":          StageModel(OPUS,   "max"),
    "report":              StageModel(OPUS,   "max"),
}


@dataclass
class ModelConfig:
    """A mutable view over the default table that supports CLI-style overrides."""

    table: dict[str, StageModel] = field(default_factory=lambda: dict(DEFAULT_TABLE))

    def get(self, stage_name: str) -> StageModel:
        if stage_name not in self.table:
            raise KeyError(f"no model configured for stage {stage_name!r}")
        return self.table[stage_name]

    # --- overrides --------------------------------------------------------

    def override_stage(self, stage_name: str, *, model: str | None = None, effort: str | None = ...) -> None:
        cur = self.table[stage_name]
        kwargs = {}
        if model is not None:
            kwargs["model"] = _expand_alias(model)
        if effort is not ...:
            kwargs["effort"] = effort
        self.table[stage_name] = replace(cur, **kwargs)

    def cheap(self) -> None:
        """`--cheap` flag: Sonnet everywhere, no extended thinking."""
        for k in self.table:
            self.table[k] = StageModel(SONNET, None)

    def all_sonnet(self) -> None:
        """`--all-sonnet` flag: Sonnet everywhere, keep thinking on."""
        for k in self.table:
            self.table[k] = StageModel(SONNET, "high")

    def thinking_budget(self, budget: int) -> None:  # noqa: ARG002 — future hook
        """Reserved for an explicit `budget_tokens` override; for now we use
        the `effort` knob."""
        # The SDK's `effort` already maps to a thinking budget; preserving this
        # method so the CLI flag exists even if it currently no-ops.
        return None


_ALIASES = {
    "opus": OPUS,
    "opus-4-7": OPUS,
    "claude-opus-4-7": OPUS,
    "sonnet": SONNET,
    "sonnet-4-6": SONNET,
    "claude-sonnet-4-6": SONNET,
    "haiku": HAIKU,
    "haiku-4-5": HAIKU,
}


def _expand_alias(name: str) -> str:
    return _ALIASES.get(name.lower(), name)
