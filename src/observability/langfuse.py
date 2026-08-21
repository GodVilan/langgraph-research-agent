"""Langfuse integration.

The LangChain callback handler is attached to the graph's config, so the whole run — every
node, every model call, every token count — arrives as one nested trace rather than a
scatter of log lines. That is the capability AUDIT §1.1 found missing: v2.1's only
instrumentation was ``log.info``, so a slow or expensive query could not be attributed to a
stage without re-running it by hand.

What the handler does not know about is added here: guardrail events, retrieval hit ids,
notional cost, and the prompt version each node resolved to. Those live in ``AgentState``
and would otherwise never reach the trace.

Everything degrades to a no-op when Langfuse is unconfigured. An observability backend
being unreachable is not a reason a query fails.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from src.observability.config import get_observability_settings


def _set_trace_attributes(**values: Any) -> None:
    """Set trace-level fields on the active span.

    This Langfuse version exposes no ``update_trace`` on a span object — trace-level fields
    are OTel span attributes under reserved keys. Getting this wrong is not a silent no-op:
    an ``AttributeError`` raised inside the root context manager ends the span before the
    graph runs, and the LangChain handler then starts a *second, separate* trace. The
    symptom is two traces per query, one of which has all the cost and none of the session
    id. Hence the direct attribute route.
    """
    import json

    from langfuse._client.attributes import LangfuseOtelSpanAttributes as Attr
    from opentelemetry import trace as otel_trace

    current = otel_trace.get_current_span()
    if not current.is_recording():
        return

    mapping = {
        "session_id": Attr.TRACE_SESSION_ID,
        "name": Attr.TRACE_NAME,
        "tags": Attr.TRACE_TAGS,
        "metadata": Attr.TRACE_METADATA,
        "input": Attr.TRACE_INPUT,
        "output": Attr.TRACE_OUTPUT,
    }
    for field, value in values.items():
        key = mapping.get(field)
        if key is None or value is None:
            continue
        if field == "tags":
            # JSON, not an OTel array: an array attribute round-trips through the exporter
            # as `{"arrayValue":{}}` and the tags arrive unusable.
            current.set_attribute(key, json.dumps([str(t) for t in value]))
        elif isinstance(value, str):
            current.set_attribute(key, value)
        else:
            current.set_attribute(key, json.dumps(value, default=str))


if TYPE_CHECKING:
    from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger(__name__)

_CLIENT: Any = None
_CLIENT_TRIED = False


def get_client() -> Any:
    """Return the Langfuse client, or None when it is not configured or cannot start."""
    global _CLIENT, _CLIENT_TRIED
    if _CLIENT_TRIED:
        return _CLIENT
    _CLIENT_TRIED = True

    settings = get_observability_settings()
    if not settings.langfuse_enabled:
        log.debug("Langfuse disabled: no key pair configured")
        return None

    try:
        from langfuse import Langfuse

        _CLIENT = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            environment=settings.langfuse_environment,
            release=settings.release,
        )
        log.info("Langfuse enabled: %s (%s)", settings.langfuse_host, settings.langfuse_environment)
    except Exception as exc:  # tracing must never break the agent
        log.warning("Could not start Langfuse: %s", exc)
        _CLIENT = None
    return _CLIENT


def callback_handler() -> BaseCallbackHandler | None:
    """The LangChain handler to attach to the graph config, or None when disabled."""
    if get_client() is None:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:  # tracing must never break the agent
        log.warning("Could not build the Langfuse callback handler: %s", exc)
        return None


@contextlib.contextmanager
def trace_run(
    *,
    name: str,
    thread_id: str,
    question: str,
    model: str,
    prompt_version: str,
) -> Iterator[Any]:
    """Wrap one query in a root span carrying the trace-level identifiers.

    ``session_id`` is the thread id, so multi-turn conversations group in the Langfuse UI
    the way they group in the checkpointer.
    """
    client = get_client()
    if client is None:
        yield None
        return

    settings = get_observability_settings()
    try:
        with client.start_as_current_observation(
            name=name,
            as_type="agent",
            input={"question": question},
            metadata={
                "thread_id": thread_id,
                "model": model,
                "prompt_version": prompt_version,
                "release": settings.release,
            },
        ) as root:
            _set_trace_attributes(
                name=name,
                session_id=thread_id,
                input={"question": question},
                tags=[f"model:{model}", f"prompts:{prompt_version}"],
            )
            yield root
    except Exception as exc:  # tracing must never break the agent
        log.warning("Langfuse root span failed: %s", exc)
        yield None


def record_state(root: Any, state: dict[str, Any]) -> None:
    """Attach everything the callback handler cannot see.

    The LangChain handler captures model calls and node boundaries. Guardrail events,
    retrieval hit ids, notional cost, and the truncation flag live in ``AgentState``, and
    without this they would never reach the trace — which would leave the trace unable to
    answer the two questions it exists for: what did this cost, and what did the guardrails
    do.
    """
    if root is None:
        return
    from src.agent.state import GuardrailEvent, RetrievalEvent, ToolCallRecord, Usage

    try:
        usage = state.get("usage")
        guardrails = [
            e for e in (state.get("guardrail_events") or []) if isinstance(e, GuardrailEvent)
        ]
        retrievals = [
            e for e in (state.get("retrieval_events") or []) if isinstance(e, RetrievalEvent)
        ]
        tools = [t for t in (state.get("tool_calls") or []) if isinstance(t, ToolCallRecord)]

        metadata: dict[str, Any] = {
            "refused": bool(state.get("refused")),
            "truncated": bool(state.get("truncated")),
            "truncation_reason": state.get("truncation_reason"),
            "refinement_count": state.get("refinement_count", 0),
            "critique_verdict": getattr(state.get("critique"), "verdict", None),
            "sub_questions": [s.text for s in (state.get("plan") or [])],
            "tool_calls": [
                {"name": t.name, "args": t.args, "ok": t.ok, "latency_ms": t.latency_ms}
                for t in tools
            ],
            "retrieval": [
                {
                    "query": e.query,
                    "retriever": e.retriever,
                    "k": e.k,
                    "n_hits": e.n_hits,
                    "latency_ms": e.latency_ms,
                    "hit_chunk_ids": e.hit_chunk_ids,
                }
                for e in retrievals
            ],
            "guardrail_events": [
                {
                    "kind": e.kind,
                    "severity": e.severity,
                    "node": e.node,
                    "chunk_id": e.chunk_id,
                    "detail": e.detail,
                }
                for e in guardrails
            ],
            "guardrail_summary": {
                "block": sum(1 for e in guardrails if e.severity == "block"),
                "warn": sum(1 for e in guardrails if e.severity == "warn"),
            },
        }

        if isinstance(usage, Usage):
            metadata["usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "thinking_tokens": usage.reasoning_tokens,
                "llm_calls": usage.llm_calls,
                "tool_calls": usage.tool_calls,
                "cost_usd_billed": usage.cost_usd,
                "cost_usd_notional": usage.notional_cost_usd,
                "missing_usage_metadata": usage.missing_usage_metadata,
            }
            root.update(
                usage_details={
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "total": usage.input_tokens + usage.output_tokens,
                },
                # Billed and notional are kept apart: on the free tier the first is always
                # zero, and collapsing them would either read zero forever or imply spend
                # that is not happening.
                cost_details={"total": usage.cost_usd},
            )

        root.update(output={"answer": state.get("answer")}, metadata=metadata)
        _set_trace_attributes(
            output={"answer": state.get("answer")},
            metadata=metadata,
            tags=_trace_tags(state, guardrails),
        )

        for event in guardrails:
            if event.severity == "info":
                continue
            with contextlib.suppress(Exception):
                root.create_event(
                    name=f"guardrail:{event.kind}",
                    level="WARNING" if event.severity == "warn" else "ERROR",
                    metadata={
                        "node": event.node,
                        "chunk_id": event.chunk_id,
                        "detail": event.detail,
                    },
                )
    except Exception as exc:  # tracing must never break the agent
        log.warning("Could not attach state to the Langfuse trace: %s", exc)


def _trace_tags(state: dict[str, Any], guardrails: list[Any]) -> list[str]:
    """Outcome tags, plus the identity tags set at trace start.

    Tags are written whole, not merged, so the final write has to re-state the model and
    prompt-version tags or a clean run ends up with no tags at all.
    """
    tags: list[str] = []
    request = state.get("request")
    if request is not None:
        from src.config import get_settings

        tags.append(f"model:{get_settings().model_name()}")
        tags.append(f"prompts:{getattr(request, 'prompt_version', 'v1')}")
    if state.get("refused"):
        tags.append("refused")
    if state.get("truncated"):
        tags.append("truncated")
    if any(e.severity == "block" for e in guardrails):
        tags.append("guardrail:block")
    elif any(e.severity == "warn" for e in guardrails):
        tags.append("guardrail:warn")
    if state.get("refinement_count", 0):
        tags.append("refined")
    return tags


def flush() -> None:
    """Flush pending spans. Short-lived processes exit before the batch worker fires."""
    client = get_client()
    if client is not None:
        with contextlib.suppress(Exception):
            client.flush()


def trace_url() -> str | None:
    client = get_client()
    if client is None:
        return None
    with contextlib.suppress(Exception):
        return str(client.get_trace_url())
    return None
