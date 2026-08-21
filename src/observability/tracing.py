"""OpenTelemetry spans for the non-LLM work.

v2.1 could tell you a query took 4 seconds. It could not tell you whether that was the
model, the FAISS search, or loading the embedding model, because the only instrumentation
was ``log.info`` (AUDIT §1.1). These spans make the non-LLM half of the latency visible.

Langfuse v4 installs itself as the **global OTel tracer provider**, so when Langfuse is
enabled these spans nest inside the same trace tree as the LLM calls — one picture of the
whole run rather than two systems to correlate by hand. When Langfuse is off they go to an
OTLP endpoint if configured, and to a no-op tracer otherwise.

Nothing here may raise into the agent. A tracer that cannot reach its collector is not a
reason a query fails.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from src.observability.config import get_observability_settings

log = logging.getLogger(__name__)

INSTRUMENTATION_NAME = "arxiv-agent-v3"

T = TypeVar("T")


@functools.lru_cache(maxsize=1)
def _configure_provider() -> None:
    """Install a tracer provider when Langfuse has not already installed one.

    Langfuse sets the global provider in its own constructor. Calling
    ``set_tracer_provider`` twice logs a warning and the second call is ignored, so this
    only acts when Langfuse is disabled.
    """
    settings = get_observability_settings()
    if settings.langfuse_enabled:
        return  # Langfuse owns the provider; our spans join its trace tree.
    if not settings.otlp_endpoint and not settings.otel_console_export:
        return  # No sink configured: the default no-op provider is the right answer.

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": INSTRUMENTATION_NAME, "service.version": settings.release}
            )
        )
        if settings.otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
            )
        if settings.otel_console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
    except Exception as exc:  # tracing must never break the agent
        log.warning("Could not configure OpenTelemetry: %s", exc)


def get_tracer() -> trace.Tracer:
    _configure_provider()
    return trace.get_tracer(INSTRUMENTATION_NAME)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a span, recording exceptions and never raising from the tracer itself."""
    try:
        tracer = get_tracer()
    except Exception:  # tracing must never break the agent
        yield trace.INVALID_SPAN
        return

    with tracer.start_as_current_span(name) as current:
        with contextlib.suppress(Exception):
            set_attributes(current, **attributes)
        try:
            yield current
        except Exception as exc:
            with contextlib.suppress(Exception):
                current.record_exception(exc)
                current.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def set_attributes(current: Span, **attributes: Any) -> None:
    """Set span attributes, coercing values OTel will not accept."""
    if not current.is_recording():
        return
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            current.set_attribute(key, value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            current.set_attribute(key, [str(v) for v in value])
        else:
            current.set_attribute(key, str(value))


def traced(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form, for synchronous functions."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with span(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
