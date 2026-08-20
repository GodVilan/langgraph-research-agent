"""Termination conditions — the Phase 1 deliverable that matters most.

Both bounds are tested independently, because they guard different failures:

* the explicit counters are the *intended* bound, and must stop a healthy run cleanly;
* ``recursion_limit`` is the *backstop*, and must fail loudly rather than silently burning
  tokens if a counter is ever wrong.
"""

from __future__ import annotations

import pytest
from langgraph.errors import GraphRecursionError

from src.agent import llm as llm_module
from src.agent.graph import build_graph
from src.agent.nodes import critique as critique_node
from src.agent.nodes import generate as generate_node
from src.agent.nodes import plan as plan_node
from src.agent.runner import run_query
from src.agent.state import Critique, RequestOptions, Usage, initial_state
from src.config import Settings
from tests.fakes import (
    FakeRetrievalService,
    ScriptedLLM,
    complex_plan,
    passing_critique,
    retry_critique,
    simple_plan,
)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Install a ScriptedLLM across every module that calls the LLM helpers."""

    def install(script: list[object]) -> ScriptedLLM:
        fake = ScriptedLLM(script)
        for module in (plan_node, critique_node, llm_module):
            monkeypatch.setattr(module, "call_structured", fake.call_structured, raising=False)
        monkeypatch.setattr(generate_node, "call_text", fake.call_text, raising=False)
        monkeypatch.setattr("src.agent.nodes.validate_input.call_structured", fake.call_structured)
        return fake

    return install


class TestHappyPath:
    async def test_simple_query_runs_end_to_end(self, scripted, settings: Settings) -> None:
        fake = scripted(
            [simple_plan(), "LoRA is a low-rank adapter. [2605.30350_0001]", passing_critique()]
        )
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "What is LoRA?", "t1", settings=settings)

        assert state["answer"]
        assert state["refused"] is False
        assert state["truncated"] is False
        assert len(service.calls) == 1
        assert state["critique"].verdict == "pass"
        # Exactly one pass through each LLM-calling node. "LoRA" hits the scope keyword
        # fast path, so validate_input makes no model call at all.
        assert fake.count("Plan") == 1
        assert fake.count("text") == 1
        assert fake.count("Critique") == 1
        assert fake.count("ScopeVerdict") == 0

    async def test_sources_are_projected_from_state_not_parsed_from_prose(
        self, scripted, settings: Settings
    ) -> None:
        scripted([simple_plan(), "Answer citing [2605.30350_0001].", passing_critique()])
        graph = build_graph(FakeRetrievalService(), settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "What is LoRA?", "t1", settings=settings)

        assert len(state["sources"]) == 1
        assert state["sources"][0].paper_id == "2605.30350"
        assert state["sources"][0].chunk_ids == ["2605.30350_0001"]


class TestPlanCursorTermination:
    async def test_multi_hop_visits_retrieve_once_per_sub_question(
        self, scripted, settings: Settings
    ) -> None:
        scripted([complex_plan("q1", "q2", "q3"), "Synthesised answer.", passing_critique()])
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        await run_query(graph, "Compare A and B and C", "t1", settings=settings)

        assert service.calls == ["q1", "q2", "q3"]

    async def test_sub_question_count_is_capped(self, scripted, settings: Settings) -> None:
        settings.graph.max_sub_questions = 2
        scripted([complex_plan("q1", "q2", "q3", "q4"), "Answer.", passing_critique()])
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        await run_query(graph, "many things", "t1", settings=settings)

        assert service.calls == ["q1", "q2"]


class TestRefinementTermination:
    async def test_refinement_loop_is_bounded_by_max_refinements(
        self, scripted, settings: Settings
    ) -> None:
        settings.graph.max_refinements = 2
        # A critic that always asks for another round. Without the bound this never ends.
        scripted([simple_plan(), "draft answer", retry_critique("hint")])
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "What is LoRA?", "t1", settings=settings)

        # Initial pass + max_refinements refinement passes, then it stops.
        assert state["refinement_count"] == settings.graph.max_refinements + 1
        assert len(service.calls) == 1 + settings.graph.max_refinements
        assert state["answer"]

    async def test_retry_without_hints_is_treated_as_pass(
        self, scripted, settings: Settings
    ) -> None:
        """A retry the loop cannot act on must not consume a refinement round."""
        scripted([simple_plan(), "draft", Critique(verdict="retry", search_hints=[])])
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "What is LoRA?", "t1", settings=settings)

        assert state["critique"].verdict == "pass"
        assert state["refinement_count"] == 0
        assert len(service.calls) == 1


class TestBudgetTermination:
    async def test_llm_call_ceiling_truncates_with_an_explicit_flag(
        self, scripted, settings: Settings
    ) -> None:
        """A breached ceiling yields a flagged partial answer, never a silent stop."""
        settings.budget.max_llm_calls = 2
        scripted([complex_plan("q1", "q2", "q3", "q4"), "draft", passing_critique()])
        graph = build_graph(FakeRetrievalService(), settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "big question about transformers", "t1", settings=settings)

        assert state["truncated"] is True
        assert "LLM call cap" in (state["truncation_reason"] or "")
        assert "partial" in (state["answer"] or "")

    async def test_tool_call_ceiling_stops_the_retrieve_loop_early(
        self, scripted, settings: Settings
    ) -> None:
        """The retrieve loop makes no LLM calls, so the *tool* ceiling is what bounds it."""
        settings.budget.max_tool_calls = 2
        scripted([complex_plan("q1", "q2", "q3", "q4"), "draft", passing_critique()])
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "big question about transformers", "t1", settings=settings)

        assert len(service.calls) < 4, "the tool ceiling should have cut the plan short"
        assert state["truncated"] is True
        assert "tool call cap" in (state["truncation_reason"] or "")

    async def test_cost_ceiling_is_checked_against_notional_cost(self, settings: Settings) -> None:
        """The ceiling must be live on the free tier, where billed cost is always 0."""
        from src.guardrails.budget import check_budget

        settings.budget.max_notional_cost_usd = 0.01
        usage = Usage(cost_usd=0.0, notional_cost_usd=0.5)
        verdict = check_budget(usage, settings.budget)
        assert not verdict.ok
        assert "cost cap" in (verdict.reason or "")

    async def test_wall_clock_deadline_trips(self, settings: Settings) -> None:
        import time

        from src.guardrails.budget import check_budget

        usage = Usage(deadline_at=time.monotonic() - 0.01)
        verdict = check_budget(usage, settings.budget)
        assert not verdict.ok
        assert "wall-clock" in (verdict.reason or "")


class TestRecursionBackstop:
    async def test_recursion_limit_raises_rather_than_looping_silently(
        self, scripted, settings: Settings
    ) -> None:
        """Deliberately break the counter bound and assert the backstop fires.

        max_refinements is set absurdly high so the intended bound cannot stop the loop;
        recursion_limit must.
        """
        settings.graph.max_refinements = 10_000
        settings.graph.recursion_limit = 12
        scripted([simple_plan(), "draft", retry_critique("hint")])
        graph = build_graph(FakeRetrievalService(), settings=settings)  # type: ignore[arg-type]

        with pytest.raises(GraphRecursionError):
            state = initial_state("what is attention", "t1", RequestOptions())
            await graph.ainvoke(
                state,
                config={
                    "configurable": {"thread_id": "t1"},
                    "recursion_limit": settings.graph.recursion_limit,
                },
            )

    async def test_runner_converts_the_backstop_into_a_flagged_answer(
        self, scripted, settings: Settings
    ) -> None:
        settings.graph.max_refinements = 10_000
        settings.graph.recursion_limit = 12
        scripted([simple_plan(), "draft", retry_critique("hint")])
        graph = build_graph(FakeRetrievalService(), settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "what is attention", "t1", settings=settings)

        assert state["truncated"] is True
        assert "recursion_limit" in (state["truncation_reason"] or "")


class TestRefusalTermination:
    async def test_refusal_reaches_finalize_without_retrieving(
        self, scripted, settings: Settings
    ) -> None:
        from src.agent.nodes.validate_input import ScopeVerdict

        scripted([ScopeVerdict(in_scope=False, reason="not ML")])
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "who is the president", "t1", settings=settings)

        assert state["refused"] is True
        assert service.calls == []
        assert state["sources"] == []
        assert "machine-learning" in (state["answer"] or "")

    async def test_oversized_input_is_refused_before_any_model_call(
        self, settings: Settings
    ) -> None:
        service = FakeRetrievalService()
        graph = build_graph(service, settings=settings)  # type: ignore[arg-type]

        state = await run_query(graph, "x" * 5000, "t1", settings=settings)

        assert state["refused"] is True
        assert service.calls == []
        assert any(e.kind == "input_too_long" for e in state["guardrail_events"])
