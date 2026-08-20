"""Node transitions and routing.

Covers the Phase 1 deliverable "tests cover node transitions". Routers are pure functions,
so they are tested directly on hand-built states rather than by inferring behaviour from
end-to-end runs.
"""

from __future__ import annotations

import pytest

from src.agent.graph import (
    make_route_after_critique,
    make_route_after_retrieve,
    route_after_validate,
)
from src.agent.state import Critique, SubQuestion, Usage, initial_state
from src.config import Settings


def state_with(**overrides: object) -> dict[str, object]:
    base = dict(initial_state("What is LoRA?", "t1"))
    base.update(overrides)
    return base


class TestRouteAfterValidate:
    def test_refusal_goes_straight_to_finalize(self) -> None:
        assert route_after_validate(state_with(refused=True)) == "finalize"  # type: ignore[arg-type]

    def test_accepted_input_goes_to_plan(self) -> None:
        assert route_after_validate(state_with(refused=False)) == "plan"  # type: ignore[arg-type]


class TestRouteAfterRetrieve:
    def test_loops_while_sub_questions_remain(self, settings: Settings) -> None:
        route = make_route_after_retrieve(settings)
        s = state_with(
            plan=[SubQuestion(text="a"), SubQuestion(text="b")],
            plan_cursor=1,
        )
        assert route(s) == "retrieve"  # type: ignore[arg-type]

    def test_moves_on_when_the_plan_is_exhausted(self, settings: Settings) -> None:
        route = make_route_after_retrieve(settings)
        s = state_with(plan=[SubQuestion(text="a")], plan_cursor=1)
        assert route(s) == "generate"  # type: ignore[arg-type]

    def test_budget_breach_short_circuits_the_loop(self, settings: Settings) -> None:
        """A tripped ceiling must divert, not keep spending."""
        settings.budget.max_llm_calls = 2
        route = make_route_after_retrieve(settings)
        s = state_with(
            plan=[SubQuestion(text="a"), SubQuestion(text="b"), SubQuestion(text="c")],
            plan_cursor=1,
            usage=Usage(llm_calls=99),
        )
        assert route(s) == "generate"  # type: ignore[arg-type]


class TestRouteAfterCritique:
    def test_pass_finalizes(self, settings: Settings) -> None:
        route = make_route_after_critique(settings)
        s = state_with(critique=Critique(verdict="pass"))
        assert route(s) == "finalize"  # type: ignore[arg-type]

    def test_error_finalizes_and_never_refines(self, settings: Settings) -> None:
        """A crashed critic must not be able to drive a refinement round."""
        route = make_route_after_critique(settings)
        s = state_with(
            critique=Critique(verdict="error"),
            plan=[SubQuestion(text="a"), SubQuestion(text="hint")],
            plan_cursor=1,
        )
        assert route(s) == "finalize"  # type: ignore[arg-type]

    def test_retry_with_pending_hints_goes_back_to_retrieve(self, settings: Settings) -> None:
        route = make_route_after_critique(settings)
        s = state_with(
            critique=Critique(verdict="retry", search_hints=["h"]),
            plan=[SubQuestion(text="a"), SubQuestion(text="h", origin="refinement")],
            plan_cursor=1,
            refinement_count=1,
        )
        assert route(s) == "retrieve"  # type: ignore[arg-type]

    def test_refinement_ceiling_stops_the_loop(self, settings: Settings) -> None:
        settings.graph.max_refinements = 2
        route = make_route_after_critique(settings)
        s = state_with(
            critique=Critique(verdict="retry", search_hints=["h"]),
            plan=[SubQuestion(text="a"), SubQuestion(text="h", origin="refinement")],
            plan_cursor=1,
            refinement_count=3,
        )
        assert route(s) == "finalize"  # type: ignore[arg-type]

    def test_retry_without_pending_plan_entries_finalizes(self, settings: Settings) -> None:
        route = make_route_after_critique(settings)
        s = state_with(
            critique=Critique(verdict="retry", search_hints=["h"]),
            plan=[SubQuestion(text="a")],
            plan_cursor=1,
            refinement_count=1,
        )
        assert route(s) == "finalize"  # type: ignore[arg-type]

    def test_budget_breach_overrides_a_retry(self, settings: Settings) -> None:
        settings.budget.max_notional_cost_usd = 0.001
        route = make_route_after_critique(settings)
        s = state_with(
            critique=Critique(verdict="retry", search_hints=["h"]),
            plan=[SubQuestion(text="a"), SubQuestion(text="h", origin="refinement")],
            plan_cursor=1,
            refinement_count=1,
            usage=Usage(notional_cost_usd=5.0),
        )
        assert route(s) == "finalize"  # type: ignore[arg-type]

    def test_missing_critique_finalizes(self, settings: Settings) -> None:
        route = make_route_after_critique(settings)
        assert route(state_with(critique=None)) == "finalize"  # type: ignore[arg-type]


class TestGraphTopology:
    def test_compiles_and_exposes_every_node(self) -> None:
        from src.agent.graph import build_graph
        from tests.fakes import FakeRetrievalService

        graph = build_graph(FakeRetrievalService())  # type: ignore[arg-type]
        nodes = set(graph.get_graph().nodes)
        assert {"validate_input", "plan", "retrieve", "generate", "critique", "finalize"} <= nodes

    def test_draw_mermaid_needs_no_index_or_model(self) -> None:
        """`make graph` must work on a clean checkout with no build artifacts."""
        from src.agent.graph import draw_mermaid

        mermaid = draw_mermaid()
        assert "validate_input" in mermaid
        assert "finalize" in mermaid

    @pytest.mark.parametrize(
        "node", ["validate_input", "plan", "retrieve", "generate", "critique", "finalize"]
    )
    def test_every_node_is_reachable_from_start(self, node: str) -> None:
        from src.agent.graph import build_graph
        from tests.fakes import FakeRetrievalService

        graph = build_graph(FakeRetrievalService()).get_graph()  # type: ignore[arg-type]
        adjacency: dict[str, set[str]] = {}
        for edge in graph.edges:
            adjacency.setdefault(edge.source, set()).add(edge.target)

        seen: set[str] = set()
        stack = ["__start__"]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency.get(current, set()))
        assert node in seen
