"""Thin invocation layer over the compiled graph.

Exists so the CLI, the eval harness and the FastAPI service share one entry point, and so
``GraphRecursionError`` is translated into a flagged partial answer in exactly one place.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, cast

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


type EventSink = Callable[[dict[str, Any]], Awaitable[None]]

# Nodes whose completion is reported as progress. Token events are forwarded from
# `generate` only: the scope classifier, planner and critic also call the model, but their
# output is structured JSON, not answer text.
PROGRESS_NODES = frozenset(
    {"validate_input", "plan", "retrieve", "generate", "critique", "finalize"}
)


async def run_query(
    graph: Graph,
    question: str,
    thread_id: str | None = None,
    request: RequestOptions | None = None,
    settings: Settings | None = None,
    on_event: EventSink | None = None,
    flush: bool = True,
) -> AgentState:
    """Run one query to completion.

    With ``on_event`` the same run streams: node completions and answer tokens are handed
    to the sink as they happen, and the terminal state is still returned. There is one run
    either way. The CLI's ``--stream`` used to call a separate streaming function and then
    this one, running the whole graph — and paying for it — twice per question.

    ``flush`` pushes pending spans out before returning. Short-lived callers — the CLI, the
    eval harness — need it or they exit before the batch exporter fires. The server must
    not: a flush waits on the exporter, so a slow, down or rate-limiting Langfuse would sit
    on the request path. Measured before this parameter existed: 3.2 s added to one query
    against a backend that never answered (tests/test_api_observability_faults.py). The
    server flushes once, at shutdown.

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
            if on_event is None:
                # ainvoke's overloads are keyed on stream_mode literals and do not admit a
                # TypedDict input; the call is correct, the overload set cannot express it.
                final: AgentState = await graph.ainvoke(  # type: ignore[call-overload]
                    state, config=run_config(tid, s)
                )
            else:
                final = await _stream(graph, state, run_config(tid, s), on_event)
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
            if flush:
                lf.flush()
            raise

        # The graph's reducers do not carry `trace_id` through, so restore it onto the final
        # state: it was set before the invoke and is a property of the run, not of any node.
        final = AgentState(**{**final, "trace_id": state["trace_id"]})
        lf.record_state(root, dict(final))

    metrics.record_run(dict(final), time.monotonic() - started)
    if flush:
        lf.flush()
    return final


async def _stream(
    graph: Graph, state: AgentState, config: dict[str, Any], on_event: EventSink
) -> AgentState:
    """Drive the graph with ``astream`` and return the last full state it emitted."""
    final: dict[str, Any] | None = None
    async for mode, chunk in graph.astream(  # type: ignore[call-overload]
        state, config=config, stream_mode=["updates", "messages", "values"]
    ):
        if mode == "values":
            final = chunk
        elif mode == "updates":
            for node in chunk:
                if node in PROGRESS_NODES:
                    await on_event({"event": "node", "node": node})
        elif mode == "messages":
            message, meta = chunk
            if meta.get("langgraph_node") != "generate":
                continue
            # `.text` is a str subclass in langchain-core 1.x (callable only for backward
            # compatibility, and deprecated as a call).
            text = str(getattr(message, "text", "") or "")
            if text:
                await on_event({"event": "token", "text": text})
    if final is None:
        raise RuntimeError("the graph emitted no state")
    return cast(AgentState, final)
