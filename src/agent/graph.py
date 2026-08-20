"""StateGraph assembly.

Topology and the reducer rationale are in docs/MIGRATION_MAP.md §2 and §4.

Two bounds operate together and are tested independently:

* the explicit counters (``plan_cursor`` against ``len(plan)``, ``refinement_count``
  against ``max_refinements``) plus the budget ceilings are the *intended* bound;
* ``recursion_limit`` is the backstop that raises ``GraphRecursionError`` loudly if a
  counter is ever wrong. A runaway loop must fail loudly, not silently burn tokens.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes.critique import critique
from src.agent.nodes.finalize import finalize
from src.agent.nodes.generate import generate
from src.agent.nodes.plan import plan
from src.agent.nodes.retrieve import make_retrieve
from src.agent.nodes.validate_input import validate_input
from src.agent.state import AgentState
from src.config import Settings, get_settings
from src.guardrails.budget import check_budget
from src.retrieval.service import RetrievalService

log = logging.getLogger(__name__)

# LangGraph's compiled-graph and checkpointer generics are parameterised over state,
# context, input, and output types that this project does not vary. Aliasing them once
# keeps every signature below readable and mypy-strict clean.
type Graph = CompiledStateGraph[AgentState, Any, Any, Any]
type Checkpointer = BaseCheckpointSaver[Any]
type RetrieveRoute = Callable[[AgentState], Literal["retrieve", "generate"]]
type CritiqueRoute = Callable[[AgentState], Literal["retrieve", "finalize"]]


# ── Routers ───────────────────────────────────────────────────────────────────


def route_after_validate(state: AgentState) -> Literal["plan", "finalize"]:
    return "finalize" if state.get("refused") else "plan"


def make_route_after_retrieve(settings: Settings) -> RetrieveRoute:
    def route_after_retrieve(state: AgentState) -> Literal["retrieve", "generate"]:
        verdict = check_budget(state["usage"], settings.budget)
        if not verdict.ok:
            log.warning("Budget tripped in retrieve loop: %s", verdict.reason)
            return "generate"
        remaining = len(state.get("plan") or []) - state.get("plan_cursor", 0)
        return "retrieve" if remaining > 0 else "generate"

    return route_after_retrieve


def make_route_after_critique(settings: Settings) -> CritiqueRoute:
    def route_after_critique(state: AgentState) -> Literal["retrieve", "finalize"]:
        verdict = check_budget(state["usage"], settings.budget)
        if not verdict.ok:
            log.warning("Budget tripped after critique: %s", verdict.reason)
            return "finalize"

        crit = state.get("critique")
        if crit is None or crit.verdict != "retry":
            return "finalize"
        if state.get("refinement_count", 0) > settings.graph.max_refinements:
            log.info("Refinement budget exhausted; finalizing")
            return "finalize"
        # critique appended hints to the plan and rewound the cursor.
        if len(state.get("plan") or []) <= state.get("plan_cursor", 0):
            return "finalize"
        return "retrieve"

    return route_after_critique


# ── Assembly ──────────────────────────────────────────────────────────────────


def build_graph(
    retrieval: RetrievalService,
    checkpointer: Checkpointer | None = None,
    settings: Settings | None = None,
) -> Graph:
    """Build and compile the graph.

    ``retrieval`` is injected rather than imported so tests can drive the whole graph
    against a fake service without loading a 1.3 GB embedding model.
    """
    s = settings or get_settings()

    builder: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)
    builder.add_node("validate_input", validate_input)
    builder.add_node("plan", plan)
    # `retrieve` is the one node built by a factory rather than declared as `async def`,
    # so mypy sees a Callable alias where add_node's overloads want a coroutine function.
    # Scoped to this call; the node's own signature is checked normally.
    builder.add_node("retrieve", make_retrieve(retrieval))  # type: ignore[arg-type]
    builder.add_node("generate", generate)
    builder.add_node("critique", critique)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "validate_input")
    builder.add_conditional_edges(
        "validate_input", route_after_validate, {"plan": "plan", "finalize": "finalize"}
    )
    builder.add_edge("plan", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        make_route_after_retrieve(s),
        {"retrieve": "retrieve", "generate": "generate"},
    )
    builder.add_edge("generate", "critique")
    builder.add_conditional_edges(
        "critique",
        make_route_after_critique(s),
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def state_model_allowlist() -> list[tuple[str, str]]:
    """Every Pydantic model that can appear in ``AgentState``, as (module, class) pairs.

    Derived by reflection rather than hand-listed so a new state model cannot silently
    fall off the allowlist.
    """
    import inspect

    from pydantic import BaseModel

    from src.agent import state as state_module

    return [
        (state_module.__name__, name)
        for name, obj in inspect.getmembers(state_module, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == state_module.__name__
    ]


def checkpoint_serde() -> object:
    """Serializer that recognises this project's state models.

    Without the allowlist LangGraph silently deserialises every ``RequestOptions``,
    ``Usage``, ``Critique``, and ``RetrievedChunk`` back as a **plain dict**. Nothing
    fails at write time; the break surfaces on resume, when a node does
    ``state["request"].prompt_version`` and gets ``AttributeError`` on a dict. Enumerating
    the models is what makes a checkpoint actually resumable.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=state_model_allowlist())


@asynccontextmanager
async def sqlite_checkpointer(
    db_path: Path | None = None,
) -> AsyncIterator[Checkpointer]:
    """Open an ``AsyncSqliteSaver`` over the thread database.

    This is the replacement for v2.1's 911-line hand-rolled thread persistence, and the
    honest justification for the rewrite: v2.1 stored *completed turns*, so a run that died
    mid-flight lost everything. This stores graph state after every node, which is what
    makes resume and interrupt possible at all.

    The async variant is required, not preferred: every node is a coroutine, and the
    synchronous ``SqliteSaver`` raises ``NotImplementedError`` on the async checkpoint API.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = db_path or get_settings().checkpoint_db
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path))
    try:
        yield AsyncSqliteSaver(conn, serde=checkpoint_serde())  # type: ignore[arg-type]
    finally:
        await conn.close()


class _TopologyOnlyService:
    """Placeholder for diagram rendering.

    The topology does not depend on the retrieval service — ``make_retrieve`` only closes
    over it — so ``make graph`` can render the diagram without loading a 1.3 GB embedding
    model or a 22 MB index.
    """

    async def retrieve(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("topology-only placeholder; not callable")


def draw_mermaid(retrieval: RetrievalService | None = None) -> str:
    """Render the compiled topology as Mermaid. Backs the ``make graph`` target."""
    service = retrieval or _TopologyOnlyService()
    return build_graph(service).get_graph().draw_mermaid()  # type: ignore[arg-type]
