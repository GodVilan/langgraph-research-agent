"""Per-request budget ceilings.

MIGRATION_MAP §5.9: step count is a bad proxy for spend, because v2.1's prompt grew with
the scratchpad so step 8 cost several times step 1. The ceilings here are denominated in
the units that actually matter — tokens, USD, wall-clock, and call counts.

Exceeding a ceiling never stops the run silently. The routing functions in
``src/agent/graph.py`` consult ``check_budget`` and divert to ``finalize`` with
``truncated=True`` and a reason, so the caller always receives a partial answer with an
explicit flag.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.state import Usage
from src.config import BudgetLimits


@dataclass(frozen=True)
class BudgetVerdict:
    ok: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


OK = BudgetVerdict(ok=True)


def check_budget(usage: Usage, limits: BudgetLimits) -> BudgetVerdict:
    """Return the first ceiling that has been exceeded, if any.

    Checked in a fixed order so the reported reason is deterministic and testable.
    """
    if usage.input_tokens > limits.max_input_tokens:
        return BudgetVerdict(
            False, f"input token cap exceeded ({usage.input_tokens} > {limits.max_input_tokens})"
        )
    if usage.output_tokens > limits.max_output_tokens:
        return BudgetVerdict(
            False, f"output token cap exceeded ({usage.output_tokens} > {limits.max_output_tokens})"
        )
    if usage.notional_cost_usd > limits.max_notional_cost_usd:
        return BudgetVerdict(
            False,
            f"cost cap exceeded (${usage.notional_cost_usd:.4f} notional > "
            f"${limits.max_notional_cost_usd:.4f})",
        )
    if usage.llm_calls > limits.max_llm_calls:
        return BudgetVerdict(
            False, f"LLM call cap exceeded ({usage.llm_calls} > {limits.max_llm_calls})"
        )
    if usage.tool_calls > limits.max_tool_calls:
        return BudgetVerdict(
            False, f"tool call cap exceeded ({usage.tool_calls} > {limits.max_tool_calls})"
        )
    remaining = usage.remaining_s()
    if remaining <= 0.0:
        return BudgetVerdict(False, f"wall-clock deadline exceeded ({limits.max_wall_clock_s}s)")
    return OK
