"""Model access and usage accounting.

Everything routes through ``init_chat_model`` so a provider swap is config, not code.

Every call returns a ``Usage`` delta alongside its result. Nodes merge that delta into
state, which is what makes per-node cost attribution fall out of the reducer instead of
requiring bespoke bookkeeping (docs/MIGRATION_MAP.md §2.2).
"""

from __future__ import annotations

import contextvars
import datetime as dt
import functools
import json
import logging
import os
import sys
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from src.agent.state import Usage
from src.config import Settings, get_settings

log = logging.getLogger(__name__)


class StructuredOutputError(RuntimeError):
    """Raised when structured output could not be parsed within the attempt budget.

    Callers must translate this into a visible state — a ``GuardrailEvent`` plus a
    degraded-but-flagged result — never into a silent permissive default. v2.1 returned a
    clean ``pass`` on exactly this failure (AUDIT §4.3).
    """


# ── The usage log ──────────────────────────────────────────────────────────────
#
# Every model response that passes through this module appends one row to a local JSONL log
# (`Settings.usage_log`) — inside `usage_from_message`, which `call_text` and `call_structured`
# both call, so a caller that discards the `Usage` it is handed cannot discard the row. The
# reconciliation against Google's bill found 76% of billed prompt tokens recorded nowhere,
# because eval construction, the necessity checks and the probes all threw their usage away
# (DECISIONS D-046). This is the mechanism fix; `tests/test_usage_log.py` asserts a call leaves
# a row, and fails if any other module builds a model or calls Gemini's HTTP API around it.

USAGE_ACTIVITY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_activity", default=None
)


def current_activity() -> str:
    """What the row is attributed to: an explicit `USAGE_ACTIVITY`, else the running
    program — `evals.build_set`, `scripts/guardrail_variance.py`, `uvicorn` …"""
    explicit = USAGE_ACTIVITY.get()
    if explicit:
        return explicit
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and getattr(spec, "name", None):
        return str(spec.name)
    return os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown"


def record_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    notional_cost_usd: float | None = None,
    missing_usage_metadata: bool = False,
    provider: str = "google",
    activity: str | None = None,
) -> None:
    """Append one row to the usage log. Never raises into a query: a log that cannot be written
    is reported loudly (error log) rather than taking the request down with it."""
    path = get_settings().usage_log
    row = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
        "activity": activity or current_activity(),
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "notional_cost_usd": notional_cost_usd,
        "missing_usage_metadata": missing_usage_metadata,
        "pid": os.getpid(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError as exc:
        log.error("usage log %s could not be written — this call is unrecorded: %s", path, exc)


@functools.lru_cache(maxsize=16)
def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    pinned: bool = False,
    overrides: tuple[tuple[str, Any], ...] = (),
) -> BaseChatModel:
    """The configured chat model. ``pinned`` adds ``seed=0, top_k=1``.

    Pinned is used by the input scope classifier only (D-035). Measured with
    ``make guardrail-probe``: over 28 classifier-judged questions x 5 draws, the unpinned
    model's output varied on all 28 and its verdict flipped on 3; pinned, the output was
    byte-identical on all 140 calls. D-014's "determinism is unrecoverable" held for the
    parameters it tried (temperature, reasoning effort, thinking budget) and did not try
    these. The generator, planner and critic stay unpinned: changing them moves the Phase 4
    baseline, and whether it should is a measured Phase 6 question (BACKLOG).
    """
    from langchain.chat_models import init_chat_model

    settings = get_settings()
    key = settings.google_api_key.get_secret_value()
    kwargs: dict[str, Any] = {
        "temperature": settings.agent_temperature if temperature is None else temperature,
        # Pinned rather than defaulted: thinking bills at the output rate, so the model
        # default moving would silently change the bill. See config.agent_reasoning_effort.
        "reasoning_effort": settings.agent_reasoning_effort,
    }
    if key:
        kwargs["api_key"] = key
    if pinned:
        kwargs["seed"] = 0
        kwargs["top_k"] = 1
    # Probe-only settings (e.g. D-014's thinking budgets), so no script needs to build a
    # model of its own — the one construction site is here, next to the usage log.
    kwargs.update(dict(overrides))
    limiter = model_rate_limiter(settings.agent_requests_per_minute)
    if limiter is not None:
        kwargs["rate_limiter"] = limiter
    return cast(BaseChatModel, init_chat_model(model or settings.agent_model, **kwargs))


@functools.lru_cache(maxsize=4)
def model_rate_limiter(requests_per_minute: float) -> Any:
    """One process-wide pacer per rate, shared by every cached model, or None when off.

    Cached so that the scope classifier, planner, generator and critic draw from the same
    bucket: the provider's quota is per key, not per call site.

    The bucket holds ``MODEL_CALL_BURST`` tokens and starts full. The first version held one
    and started empty (the library default), which made every call after the first wait a
    full refill interval even with a single user — a 3-call query took 16.6 s in the
    container, most of it the pacer waiting on nobody. With a burst of 3 an ordinary query
    runs unpaced, and any 60 s window still admits at most ``burst + rpm`` calls: 3 + 12 =
    15, the free-tier quota.
    """
    if requests_per_minute <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=MODEL_CALL_BURST,
    )
    limiter.available_tokens = float(MODEL_CALL_BURST)
    return limiter


MODEL_CALL_BURST = 3


def usage_from_message(message: AIMessage, settings: Settings | None = None) -> Usage:
    """Build a ``Usage`` delta from one model response.

    A response with no ``usage_metadata`` increments ``missing_usage_metadata`` rather than
    contributing a silent zero, so an under-reported token count is visible in the trace
    instead of looking like a cheap call.
    """
    s = settings or get_settings()
    pricing = s.pricing()
    meta = getattr(message, "usage_metadata", None)
    model_name = str((message.response_metadata or {}).get("model_name") or s.model_name())

    if not meta:
        record_usage(model_name, 0, 0, missing_usage_metadata=True)
        return Usage(llm_calls=1, missing_usage_metadata=1)

    input_tokens = int(meta.get("input_tokens", 0))
    output_tokens = int(meta.get("output_tokens", 0))
    details = meta.get("output_token_details") or {}
    reasoning_tokens = int(details.get("reasoning", 0))
    # Gemini's prompt count includes the cached part; LangChain reports that part separately.
    cached = int((meta.get("input_token_details") or {}).get("cache_read", 0) or 0)
    cached = min(cached, input_tokens)
    cached_rate = (
        pricing.notional_cached_input_usd
        if pricing.notional_cached_input_usd is not None
        else pricing.notional_input_usd  # unverified cached rate: overstate, never understate
    )

    # Reasoning tokens bill at the output rate and are already counted inside output_tokens
    # by the provider; they are tracked separately for reporting only. No billed figure is
    # computed: billing comes from the provider's record (D-046), so `cost_usd` stays None.
    usage = Usage(
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        notional_cost_usd=(
            (input_tokens - cached) * pricing.notional_input_usd
            + cached * cached_rate
            + output_tokens * pricing.notional_output_usd
        )
        / 1_000_000,
        llm_calls=1,
    )
    record_usage(
        model_name,
        input_tokens,
        output_tokens,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning_tokens,
        notional_cost_usd=usage.notional_cost_usd,
    )
    return usage


async def call_text(
    system: str, user: str, model: BaseChatModel | None = None
) -> tuple[str, Usage]:
    """One free-text call. Returns the text and the usage delta."""
    llm = model or get_chat_model()
    response = await llm.ainvoke([("system", system), ("user", user)])
    assert isinstance(response, AIMessage)
    return str(response.text).strip(), usage_from_message(response)


async def call_structured[T: BaseModel](
    schema: type[T],
    system: str,
    user: str,
    max_attempts: int = 2,
    model: BaseChatModel | None = None,
) -> tuple[T, Usage]:
    """Structured output with bounded repair.

    Raises ``StructuredOutputError`` when every attempt fails. Usage from the failed
    attempts is *not* lost — it is attached to the exception so the caller still charges
    the budget for tokens that were genuinely spent.
    """
    llm = model or get_chat_model()
    structured = llm.with_structured_output(schema, include_raw=True)

    total = Usage()
    last_error: str = "unknown"
    messages: list[tuple[str, str]] = [("system", system), ("user", user)]

    for attempt in range(1, max_attempts + 1):
        result: dict[str, Any] = await structured.ainvoke(messages)  # type: ignore[assignment]

        raw = result.get("raw")
        if isinstance(raw, AIMessage):
            total = _merge(total, usage_from_message(raw))

        parsed = result.get("parsed")
        if parsed is not None:
            return cast(T, parsed), total

        last_error = str(result.get("parsing_error") or "no parsed output")
        log.warning("Structured output attempt %d/%d failed: %s", attempt, max_attempts, last_error)
        if attempt < max_attempts:
            messages = [
                ("system", system),
                ("user", user),
                (
                    "user",
                    f"Your previous response could not be parsed into the required schema "
                    f"({last_error}). Respond again, matching the schema exactly.",
                ),
            ]

    error = StructuredOutputError(
        f"{schema.__name__} could not be parsed in {max_attempts} attempts: {last_error}"
    )
    error.usage = total  # type: ignore[attr-defined]
    raise error


def _merge(left: Usage, right: Usage) -> Usage:
    from src.agent.state import merge_usage

    return merge_usage(left, right)
