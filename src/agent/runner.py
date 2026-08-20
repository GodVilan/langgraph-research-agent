"""Thin invocation layer over the compiled graph.

Exists so the CLI (now) and the FastAPI service (Phase 5) share one entry point, and so
``GraphRecursionError`` is translated into a flagged partial answer in exactly one place.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langgraph.errors import GraphRecursionError

from src.agent.graph import Graph
from src.agent.state import AgentState, RequestOptions, initial_state
from src.config import Settings, get_settings

log = logging.getLogger(__name__)


def new_thread_id() -> str:
    return uuid.uuid4().hex


def run_config(thread_id: str, settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": s.graph.recursion_limit,
    }


async def run_query(
    graph: Graph,
    question: str,
    thread_id: str | None = None,
    request: RequestOptions | None = None,
    settings: Settings | None = None,
) -> AgentState:
    """Run one query to completion.

    A ``GraphRecursionError`` means a counter was wrong — the structural backstop fired.
    That is reported as a truncated answer with an explicit reason, never swallowed.
    """
    s = settings or get_settings()
    tid = thread_id or new_thread_id()
    state = initial_state(
        question=question,
        thread_id=tid,
        request=request or RequestOptions(),
        deadline_s=s.budget.max_wall_clock_s,
    )

    try:
        # ainvoke's overloads are keyed on stream_mode literals and do not admit a
        # TypedDict input; the call is correct, the overload set cannot express it.
        final: AgentState = await graph.ainvoke(  # type: ignore[call-overload]
            state, config=run_config(tid, s)
        )
    except GraphRecursionError as exc:
        log.error("Recursion limit hit at %d steps: %s", s.graph.recursion_limit, exc)
        return AgentState(
            **{
                **state,
                "answer": (
                    "The agent exceeded its step limit before producing an answer. "
                    "This is a bug in the graph's termination conditions, not a "
                    "limitation of the question."
                ),
                "truncated": True,
                "truncation_reason": f"recursion_limit={s.graph.recursion_limit} exceeded",
            }
        )
    return final


async def stream_events(
    graph: Graph,
    question: str,
    thread_id: str | None = None,
    request: RequestOptions | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Node-level progress events.

    Replaces v2.1's ``step_callback``. Phase 5 maps these onto SSE frames.
    """
    s = settings or get_settings()
    tid = thread_id or new_thread_id()
    state = initial_state(
        question=question,
        thread_id=tid,
        request=request or RequestOptions(),
        deadline_s=s.budget.max_wall_clock_s,
    )

    async for event in graph.astream_events(  # type: ignore[call-overload]
        state, config=run_config(tid, s), version="v2"
    ):
        kind = event.get("event", "")
        if kind in {"on_chain_start", "on_chain_end"} and event.get("name") in {
            "validate_input",
            "plan",
            "retrieve",
            "generate",
            "critique",
            "finalize",
        }:
            yield {"event": kind, "node": event["name"]}
        elif kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            text = getattr(chunk, "content", "") if chunk is not None else ""
            if text:
                yield {"event": "token", "text": text}
