"""AgentState schema and its reducers.

Container is a ``TypedDict`` so nodes can return partial updates without full-model
revalidation on every merge; every non-scalar value is a Pydantic v2 model. Rationale and
the per-field reducer justifications are in docs/MIGRATION_MAP.md §2.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ── Value models ──────────────────────────────────────────────────────────────


class RequestOptions(BaseModel):
    """Per-request scope.

    This model is the fix for AUDIT §4.14: the four attributes v2.1 mutated on a shared
    ``ReActAgent`` instance become checkpointed request state, so two concurrent requests
    cannot see each other's scoping.
    """

    allowed_paper_ids: frozenset[str] | None = None
    use_arxiv: bool = False
    top_k: int = 5
    section_filter: str | None = None
    # Bumped to v2 in Phase 2: the generate prompt gained the data-not-instruction framing
    # that the injection defence depends on. Prompt versions are recorded per request so a
    # change in answer quality is attributable rather than guessed at.
    prompt_version: str = "v2"

    model_config = {"frozen": True}


class SubQuestion(BaseModel):
    text: str
    origin: Literal["plan", "refinement"] = "plan"


class RetrievedChunk(BaseModel):
    """A retrieval hit carried as structured data.

    v2.1 reconstructed provenance by regex over formatted display strings, which silently
    returned nothing for three of its six tools (AUDIT §4.19). Carrying the object means
    ``sources`` is a projection of state rather than a parse of prose.
    """

    chunk_id: str
    paper_id: str
    title: str
    text: str
    score: float
    source: str = "corpus"
    section_type: str = "general"
    retriever: Literal["dense", "sparse", "arxiv"] = "dense"


class RetrievalEvent(BaseModel):
    """One retriever invocation. Append-only audit log; duplicates are meaningful."""

    query: str
    retriever: Literal["dense", "sparse", "arxiv"]
    k: int
    n_hits: int
    latency_ms: float
    hit_chunk_ids: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Replaces v2.1's ``scratchpad: list[Step]`` with structured, countable data."""

    name: str
    args: dict[str, object] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    latency_ms: float = 0.0


class GuardrailEvent(BaseModel):
    """Every detection must reach the trace; nothing here may be collapsed or overwritten."""

    kind: str
    severity: Literal["info", "warn", "block"] = "info"
    node: str = ""
    detail: str = ""
    chunk_id: str | None = None


class Critique(BaseModel):
    """Structured critique verdict.

    ``error`` exists so a crashed critic is distinguishable from a passing one — v2.1
    returned a clean ``pass`` on any exception, making the verdict unfalsifiable
    (AUDIT §4.3, MIGRATION_MAP §5.5).
    """

    verdict: Literal["pass", "retry", "error"] = "pass"
    grounded: bool = True
    complete: bool = True
    specific: bool = True
    gaps: list[str] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    """Structured planner output."""

    kind: Literal["simple", "complex"] = "simple"
    sub_questions: list[str] = Field(default_factory=list)


class Source(BaseModel):
    paper_id: str
    title: str
    score: float
    chunk_ids: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    """Accumulated spend.

    ``cost_usd`` is what we are actually billed (0.0 on the Gemini free tier).
    ``notional_cost_usd`` prices the same tokens at paid rates so the budget ceiling is a
    live guard rather than dead code. See docs/DECISIONS.md D-004.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    notional_cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    # Set once at entry; nodes leave it at 0.0 in their deltas and merge_usage keeps the
    # earliest non-zero value.
    deadline_at: float = 0.0
    # Counts LLM responses that arrived with no usage_metadata. Non-zero means the token
    # figures below understate reality, so it is surfaced rather than assumed to be zero.
    missing_usage_metadata: int = 0
    # Set only by `initial_state`. Because `usage` is checkpointed and `merge_usage` sums,
    # a second turn on the same thread would otherwise inherit the first turn's spend, and
    # the "per-request" ceilings would quietly become per-thread ceilings — a thread would
    # hit the cap after a handful of turns and truncate every answer after that. Observed
    # live before this was added: two turns on one thread went llm_calls 3 -> 6.
    # This flag makes a new request replace the running total rather than add to it.
    reset: bool = False

    def remaining_s(self) -> float:
        if self.deadline_at <= 0.0:
            return float("inf")
        return self.deadline_at - time.monotonic()


# ── Reducers ──────────────────────────────────────────────────────────────────


def merge_retrieved(
    left: list[RetrievedChunk] | None, right: list[RetrievedChunk] | None
) -> list[RetrievedChunk]:
    """Append, dedupe on ``chunk_id`` keeping the max score, preserve first-seen order.

    Not ``operator.add``: the refinement loop re-enters ``retrieve`` and re-surfaces
    overlapping chunks. Plain concatenation would inflate the generation prompt, the token
    bill, and the Recall@k denominator in the eval.
    """
    merged: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunk in [*(left or []), *(right or [])]:
        existing = merged.get(chunk.chunk_id)
        if existing is None:
            merged[chunk.chunk_id] = chunk
            order.append(chunk.chunk_id)
        elif chunk.score > existing.score:
            merged[chunk.chunk_id] = chunk
    return [merged[cid] for cid in order]


def merge_usage(left: Usage | None, right: Usage | None) -> Usage:
    """Sum the counters; keep the earliest deadline; honour a reset.

    Neither default reducer works: last-write-wins drops every node's spend except the
    last, and ``operator.add`` is undefined for a model. Nodes return only their own delta,
    which keeps them independent and makes per-node cost attribution fall out for free.

    A ``right`` carrying ``reset=True`` replaces rather than adds, and the flag is cleared
    on the way out so subsequent node deltas accumulate normally. Only ``initial_state``
    sets it, which is what keeps the ceilings per-request even though ``usage`` is
    checkpointed and survives into the next turn on the same thread.
    """
    if left is None:
        return right or Usage()
    if right is None:
        return left
    if right.reset:
        return right.model_copy(update={"reset": False})

    deadlines = [d for d in (left.deadline_at, right.deadline_at) if d > 0.0]
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        cost_usd=left.cost_usd + right.cost_usd,
        notional_cost_usd=left.notional_cost_usd + right.notional_cost_usd,
        llm_calls=left.llm_calls + right.llm_calls,
        tool_calls=left.tool_calls + right.tool_calls,
        deadline_at=min(deadlines) if deadlines else 0.0,
        missing_usage_metadata=left.missing_usage_metadata + right.missing_usage_metadata,
    )


# ── State ─────────────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """Graph state. See docs/MIGRATION_MAP.md §2.2 for the reducer justification per field."""

    # Identity — last-write-wins; set once at entry.
    thread_id: str
    question: str
    request: RequestOptions
    # The Langfuse trace this run wrote to, "" when tracing is off. Carried in state rather
    # than in a module global so that scores can be attached to the right trace afterwards,
    # and so Phase 5 can return it per request without cross-request leakage.
    trace_id: str

    # add_messages, not operator.add: it dedupes and updates by message id, which is what
    # makes checkpoint resume idempotent. operator.add would duplicate every message on
    # replay.
    messages: Annotated[list[AnyMessage], add_messages]

    # LWW: written whole by `plan`, and by `critique` when it appends refinement hints.
    # Both do a full read-modify-write and never run concurrently.
    plan: list[SubQuestion]
    # LWW: a position, not a tally. Under operator.add a replayed node would advance the
    # cursor twice and skip a sub-question.
    plan_cursor: int

    retrieved: Annotated[list[RetrievedChunk], merge_retrieved]
    retrieval_events: Annotated[list[RetrievalEvent], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    draft_answer: str | None
    critique: Critique | None
    # LWW, incremented node-side. Explicitly not operator.add: this is a safety bound, and
    # a double-write from a node retry or replayed checkpoint would inflate it and
    # terminate a healthy run early. A guard that fires spuriously is as bad as one that
    # never fires.
    refinement_count: int

    guardrail_events: Annotated[list[GuardrailEvent], operator.add]
    refused: bool
    refusal_reason: str | None

    usage: Annotated[Usage, merge_usage]
    truncated: bool
    truncation_reason: str | None

    answer: str | None
    sources: list[Source]


def initial_state(
    question: str,
    thread_id: str,
    request: RequestOptions | None = None,
    deadline_s: float | None = None,
) -> AgentState:
    """Build a fully populated starting state.

    Every key is set explicitly so nodes never have to guard against a missing key, and so
    ``total=False`` stays a convenience for partial *updates* rather than a licence for
    partial *state*.
    """
    opts = request or RequestOptions()
    usage = Usage(reset=True)
    if deadline_s is not None:
        usage.deadline_at = time.monotonic() + deadline_s
    return AgentState(
        thread_id=thread_id,
        question=question,
        trace_id="",
        request=opts,
        messages=[],
        plan=[],
        plan_cursor=0,
        retrieved=[],
        retrieval_events=[],
        tool_calls=[],
        draft_answer=None,
        critique=None,
        refinement_count=0,
        guardrail_events=[],
        refused=False,
        refusal_reason=None,
        usage=usage,
        truncated=False,
        truncation_reason=None,
        answer=None,
        sources=[],
    )
