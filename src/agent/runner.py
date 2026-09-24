"""Thin invocation layer over the compiled graph.

Exists so the CLI (now) and the FastAPI service (Phase 5) share one entry point, and so
``GraphRecursionError`` is translated into a flagged partial answer in exactly one place.
"""

from __future__ import annotations

import logging
import time
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
    """Graph config, with the Langfuse callback attached when it is configured.

    The handler is what makes the whole run arrive as one nested trace instead of a scatter
    of log lines. It is a no-op list when Langfuse is off.
    """
    s = settings or get_settings()
    from src.observability.langfuse import callback_handler

    handler = callback_handler()
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": s.graph.recursion_limit,
        "callbacks": [handler] if handler is not None else [],
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
    options = request or RequestOptions()
    state = initial_state(
        question=question,
        thread_id=tid,
        request=options,
        deadline_s=s.budget.max_wall_clock_s,
    )

    from src.observability import langfuse as lf
    from src.observability import metrics

    started = time.monotonic()
    with lf.trace_run(
        name="query",
        thread_id=tid,
        question=question,
        model=s.model_name(),
        prompt_version=options.prompt_version,
    ) as root:
        state = AgentState(**{**state, "trace_id": lf.current_trace_id()})
        try:
            # ainvoke's overloads are keyed on stream_mode literals and do not admit a
            # TypedDict input; the call is correct, the overload set cannot express it.
            final: AgentState = await graph.ainvoke(  # type: ignore[call-overload]
                state, config=run_config(tid, s)
            )
        except GraphRecursionError as exc:
            log.error("Recursion limit hit at %d steps: %s", s.graph.recursion_limit, exc)
            final = AgentState(
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
            metrics.record_error("graph_recursion_limit", time.monotonic() - started)
        except Exception as exc:
            metrics.record_error(type(exc).__name__, time.monotonic() - started)
            lf.flush()
            raise

        # The graph's reducers do not carry `trace_id` through, so restore it onto the final
        # state: it was set before the invoke and is a property of the run, not of any node.
        final = AgentState(**{**final, "trace_id": state["trace_id"]})
        lf.record_state(root, dict(final))

    metrics.record_run(dict(final), time.monotonic() - started)
    lf.flush()
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
