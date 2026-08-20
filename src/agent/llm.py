"""Model access and usage accounting.

Everything routes through ``init_chat_model`` so a provider swap is config, not code.

Every call returns a ``Usage`` delta alongside its result. Nodes merge that delta into
state, which is what makes per-node cost attribution fall out of the reducer instead of
requiring bespoke bookkeeping (docs/MIGRATION_MAP.md §2.2).
"""

from __future__ import annotations

import functools
import logging
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


@functools.lru_cache(maxsize=4)
def get_chat_model(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
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
    return cast(BaseChatModel, init_chat_model(model or settings.agent_model, **kwargs))


def usage_from_message(message: AIMessage, settings: Settings | None = None) -> Usage:
    """Build a ``Usage`` delta from one model response.

    A response with no ``usage_metadata`` increments ``missing_usage_metadata`` rather than
    contributing a silent zero, so an under-reported token count is visible in the trace
    instead of looking like a cheap call.
    """
    s = settings or get_settings()
    pricing = s.pricing()
    meta = getattr(message, "usage_metadata", None)

    if not meta:
        return Usage(llm_calls=1, missing_usage_metadata=1)

    input_tokens = int(meta.get("input_tokens", 0))
    output_tokens = int(meta.get("output_tokens", 0))
    details = meta.get("output_token_details") or {}
    reasoning_tokens = int(details.get("reasoning", 0))

    # Reasoning tokens bill at the output rate and are already counted inside
    # output_tokens by the provider; they are tracked separately for reporting only.
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=(input_tokens * pricing.input_usd + output_tokens * pricing.output_usd)
        / 1_000_000,
        notional_cost_usd=(
            input_tokens * pricing.notional_input_usd + output_tokens * pricing.notional_output_usd
        )
        / 1_000_000,
        llm_calls=1,
    )


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
