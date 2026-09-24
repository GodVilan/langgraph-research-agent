"""Request and response bodies. Pydantic v2 throughout, like every other I/O schema here."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.agent.nodes.validate_input import MAX_QUESTION_CHARS
from src.agent.state import AgentState, GuardrailEvent, Source, Usage
from src.guardrails.decision import GuardrailDecision, input_decision

# Thread ids are minted by the service (uuid4 hex) and act as bearer capabilities: anyone
# holding one can read and continue that thread. Accepting only this shape keeps arbitrary
# strings out of the checkpointer's keyspace.
THREAD_ID_PATTERN = r"^[0-9a-f]{32}$"
PAPER_ID_PATTERN = r"^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # Server-sent events by default; `false` returns one JSON body when the run completes.
    stream: bool = True
    top_k: int = Field(default=5, ge=1, le=10)
    # Restrict retrieval to these corpus papers. The per-request scoping v2.1 kept as
    # mutable attributes on a shared agent (AUDIT §4.14) — here it is request state.
    paper_ids: list[str] | None = Field(default=None, max_length=20)

    model_config = {"extra": "forbid"}

    def validated_paper_ids(self) -> frozenset[str] | None:
        import re

        if not self.paper_ids:
            return None
        bad = [p for p in self.paper_ids if not re.match(PAPER_ID_PATTERN, p)]
        if bad:
            raise ValueError(f"not arXiv ids: {bad[:3]}")
        return frozenset(self.paper_ids)


class SourceOut(BaseModel):
    paper_id: str
    title: str
    score: float

    @classmethod
    def of(cls, source: Source) -> SourceOut:
        return cls(paper_id=source.paper_id, title=source.title, score=source.score)


BILLED_UNVERIFIED = (
    "unverified: billing is known only from the provider's own record, after the fact, and is "
    "never assumed to be $0 (DECISIONS D-046)"
)


class UsageOut(BaseModel):
    llm_calls: int
    tool_calls: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    # Two figures, never merged (D-004 / D-020). Notional prices the tokens at paid standard
    # rates and is what every ceiling checks. Billed is null unless a provider record supplied
    # it — never 0 by default: the $0 this used to report was a tier assumption (D-046).
    billed_cost_usd: float | None
    billed_cost_basis: str
    notional_cost_usd: float

    @classmethod
    def of(cls, usage: Usage) -> UsageOut:
        return cls(
            llm_calls=usage.llm_calls,
            tool_calls=usage.tool_calls,
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            output_tokens=usage.output_tokens,
            billed_cost_usd=None if usage.cost_usd is None else round(usage.cost_usd, 6),
            billed_cost_basis="provider record"
            if usage.cost_usd is not None
            else BILLED_UNVERIFIED,
            notional_cost_usd=round(usage.notional_cost_usd, 6),
        )


class EventOut(BaseModel):
    kind: str
    severity: Literal["info", "warn", "block"]
    node: str


class QueryResponse(BaseModel):
    request_id: str
    thread_id: str
    # The Langfuse trace this request wrote to; "" when tracing is off for this deployment.
    trace_id: str
    answer: str
    sources: list[SourceOut]
    # True only when the *input* guardrail refused the question. A generated answer that
    # says the corpus does not cover the question is not a block (Phase 4 rename).
    guardrail_blocked: bool
    guardrail: GuardrailDecision
    # Non-info guardrail events from any stage — e.g. a retrieved passage quarantined.
    guardrail_events: list[EventOut]
    # A budget ceiling stopped the run; `answer` is partial and says so (never silent).
    truncated: bool
    truncation_reason: str | None
    usage: UsageOut
    latency_ms: float

    @classmethod
    def from_state(cls, state: AgentState, request_id: str, latency_ms: float) -> QueryResponse:
        events = [e for e in state.get("guardrail_events") or [] if isinstance(e, GuardrailEvent)]
        usage = state.get("usage")
        refused = bool(state.get("refused"))
        return cls(
            request_id=request_id,
            thread_id=state["thread_id"],
            trace_id=state.get("trace_id") or "",
            answer=state.get("answer") or "",
            sources=[SourceOut.of(s) for s in state.get("sources") or []],
            guardrail_blocked=refused,
            guardrail=input_decision(refused, events, _classifier_pinned()),
            guardrail_events=[
                EventOut(kind=e.kind, severity=e.severity, node=e.node)
                for e in events
                if e.severity != "info"
            ],
            truncated=bool(state.get("truncated")),
            truncation_reason=state.get("truncation_reason"),
            usage=UsageOut.of(usage if isinstance(usage, Usage) else Usage()),
            latency_ms=round(latency_ms, 1),
        )


def _classifier_pinned() -> bool:
    from src.config import get_settings

    return get_settings().pin_scope_classifier


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ThreadView(BaseModel):
    thread_id: str
    turns: list[Turn]
    last_answer: str | None
    last_guardrail_blocked: bool
    last_truncated: bool


class ErrorBody(BaseModel):
    error: str
    detail: str
    retry_after_s: int | None = None

    def json_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)
