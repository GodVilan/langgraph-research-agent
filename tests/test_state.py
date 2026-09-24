"""Reducer behaviour.

These test the choices justified in docs/MIGRATION_MAP.md §2.2 — specifically the two
places where the obvious default reducer is the wrong one.
"""

from __future__ import annotations

import time

from src.agent.state import (
    AgentState,
    RetrievedChunk,
    Usage,
    initial_state,
    merge_retrieved,
    merge_usage,
)


def chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=chunk_id.split("_")[0],
        title="T",
        text="body",
        score=score,
    )


class TestMergeRetrieved:
    def test_appends_new_chunks(self) -> None:
        merged = merge_retrieved([chunk("a_0001", 0.5)], [chunk("b_0001", 0.4)])
        assert [c.chunk_id for c in merged] == ["a_0001", "b_0001"]

    def test_dedupes_on_chunk_id(self) -> None:
        """The refinement loop re-retrieves; operator.add would double the context."""
        merged = merge_retrieved([chunk("a_0001", 0.5)], [chunk("a_0001", 0.5)])
        assert len(merged) == 1

    def test_keeps_the_higher_score(self) -> None:
        merged = merge_retrieved([chunk("a_0001", 0.5)], [chunk("a_0001", 0.9)])
        assert merged[0].score == 0.9

        merged = merge_retrieved([chunk("a_0001", 0.9)], [chunk("a_0001", 0.5)])
        assert merged[0].score == 0.9

    def test_preserves_first_seen_order_when_score_improves(self) -> None:
        merged = merge_retrieved(
            [chunk("a_0001", 0.5), chunk("b_0001", 0.4)],
            [chunk("a_0001", 0.99)],
        )
        assert [c.chunk_id for c in merged] == ["a_0001", "b_0001"]

    def test_handles_none_on_either_side(self) -> None:
        assert merge_retrieved(None, None) == []
        assert len(merge_retrieved(None, [chunk("a_0001", 0.1)])) == 1
        assert len(merge_retrieved([chunk("a_0001", 0.1)], None)) == 1


class TestMergeUsage:
    def test_sums_every_counter(self) -> None:
        left = Usage(input_tokens=10, output_tokens=5, llm_calls=1, tool_calls=2, cost_usd=0.1)
        right = Usage(input_tokens=3, output_tokens=7, llm_calls=1, tool_calls=1, cost_usd=0.2)
        merged = merge_usage(left, right)
        assert merged.input_tokens == 13
        assert merged.output_tokens == 12
        assert merged.llm_calls == 2
        assert merged.tool_calls == 3
        assert merged.cost_usd == 0.30000000000000004  # float addition, asserted honestly

    def test_last_write_wins_would_lose_spend(self) -> None:
        """The regression this reducer exists to prevent."""
        left = Usage(input_tokens=1000, llm_calls=5)
        right = Usage(input_tokens=10, llm_calls=1)
        assert merge_usage(left, right).input_tokens == 1010

    def test_keeps_the_earliest_deadline(self) -> None:
        now = time.monotonic()
        merged = merge_usage(Usage(deadline_at=now + 100), Usage(deadline_at=now + 5))
        assert merged.deadline_at == now + 5

    def test_ignores_zero_deadline_from_node_deltas(self) -> None:
        now = time.monotonic()
        merged = merge_usage(Usage(deadline_at=now + 100), Usage(input_tokens=5))
        assert merged.deadline_at == now + 100

    def test_accumulates_missing_metadata_counter(self) -> None:
        merged = merge_usage(Usage(missing_usage_metadata=1), Usage(missing_usage_metadata=2))
        assert merged.missing_usage_metadata == 3

    def test_handles_none(self) -> None:
        assert merge_usage(None, None).llm_calls == 0
        assert merge_usage(None, Usage(llm_calls=2)).llm_calls == 2
        assert merge_usage(Usage(llm_calls=2), None).llm_calls == 2


class TestUsageDeadline:
    def test_no_deadline_is_infinite(self) -> None:
        assert Usage().remaining_s() == float("inf")

    def test_expired_deadline_is_negative(self) -> None:
        assert Usage(deadline_at=time.monotonic() - 1).remaining_s() < 0


class TestInitialState:
    def test_populates_every_declared_key(self) -> None:
        """total=False is for partial updates, not partial state — nodes never guard.

        Derived from ``AgentState``'s annotations rather than a hand-listed set. The list
        version had to be edited every time a field was added, which means it asserted "the
        keys are the ones I remembered" — the same drift as any hand-maintained enumeration in
        this project (see `REMEDIES`, `by_reason`). Adding a field to the state now fails this
        test until `initial_state` populates it, which is the property worth having.
        """
        state = initial_state("What is LoRA?", "thread-1")

        declared = set(AgentState.__annotations__)
        missing = declared - set(state.keys())
        extra = set(state.keys()) - declared

        assert not missing, f"initial_state does not populate {sorted(missing)}"
        assert not extra, f"initial_state sets keys AgentState does not declare: {sorted(extra)}"

    def test_deadline_is_set_when_requested(self) -> None:
        state = initial_state("q", "t", deadline_s=30.0)
        assert 29.0 < state["usage"].remaining_s() <= 30.0

    def test_request_options_are_frozen(self) -> None:
        """Request scope must not be mutable — this is the AUDIT §4.14 fix."""
        import pydantic
        import pytest

        state = initial_state("q", "t")
        with pytest.raises(pydantic.ValidationError):
            state["request"].top_k = 99  # type: ignore[misc]


class TestUsageReset:
    """`usage` is checkpointed, so without a reset the documented per-request ceilings
    silently become per-thread ceilings and a long thread truncates every later answer.

    This was found by a live two-turn run, not by reasoning about the code: turn two on the
    same thread inherited turn one's spend and reported llm_calls 3 -> 6.
    """

    def test_reset_replaces_rather_than_accumulates(self) -> None:
        accumulated = Usage(input_tokens=9000, llm_calls=15, tool_calls=11)
        merged = merge_usage(accumulated, Usage(reset=True))
        assert merged.llm_calls == 0
        assert merged.tool_calls == 0
        assert merged.input_tokens == 0

    def test_reset_flag_is_cleared_on_the_way_out(self) -> None:
        """Otherwise every later node delta would also wipe the running total."""
        merged = merge_usage(Usage(llm_calls=5), Usage(reset=True))
        assert merged.reset is False
        assert merge_usage(merged, Usage(llm_calls=1)).llm_calls == 1

    def test_node_deltas_accumulate_within_one_request(self) -> None:
        state = merge_usage(Usage(llm_calls=9), Usage(reset=True))
        for _ in range(3):
            state = merge_usage(state, Usage(llm_calls=1, input_tokens=100))
        assert state.llm_calls == 3
        assert state.input_tokens == 300

    def test_reset_preserves_the_new_requests_deadline(self) -> None:
        stale = time.monotonic() - 500
        fresh = time.monotonic() + 120
        merged = merge_usage(Usage(deadline_at=stale), Usage(reset=True, deadline_at=fresh))
        assert merged.deadline_at == fresh
        assert merged.remaining_s() > 0

    def test_initial_state_requests_a_reset(self) -> None:
        assert initial_state("q", "t")["usage"].reset is True
