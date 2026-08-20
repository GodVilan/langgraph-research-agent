"""Retrieval for the sub-question at ``plan_cursor``.

Advances the cursor by exactly one. The graph's self-edge re-enters this node until the
plan is exhausted, which reproduces v2.1's sequential per-sub-question behaviour without
its per-sub-question ReAct loop.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from src.agent.state import (
    AgentState,
    GuardrailEvent,
    RetrievalEvent,
    ToolCallRecord,
)
from src.agent.tools.retrieval_tools import to_retrieved
from src.guardrails.screening import screen_chunks
from src.retrieval.service import RetrievalService

log = logging.getLogger(__name__)

NODE = "retrieve"

RetrieveNode = Callable[[AgentState], Awaitable[dict[str, object]]]


def make_retrieve(service: RetrievalService) -> RetrieveNode:
    async def retrieve(state: AgentState) -> dict[str, object]:
        steps = state.get("plan") or []
        cursor = state.get("plan_cursor", 0)

        if cursor >= len(steps):
            # Defensive: the router should never route here with an exhausted plan.
            log.warning("retrieve called with cursor %d beyond plan of %d", cursor, len(steps))
            return {"plan_cursor": cursor}

        sub_question = steps[cursor].text
        options = state["request"]
        t0 = time.monotonic()

        try:
            result = await service.retrieve(
                sub_question,
                top_k=options.top_k,
                allowed_paper_ids=options.allowed_paper_ids,
                use_arxiv=options.use_arxiv,
                section_filter=options.section_filter,
            )
        except Exception as exc:
            log.exception("Retrieval failed for %r", sub_question[:80])
            return {
                "plan_cursor": cursor + 1,
                "tool_calls": [
                    ToolCallRecord(
                        name="search_corpus",
                        args={"query": sub_question, "top_k": options.top_k},
                        ok=False,
                        error=str(exc),
                        latency_ms=round((time.monotonic() - t0) * 1000, 2),
                    )
                ],
                "guardrail_events": [
                    GuardrailEvent(
                        kind="retrieval_failed", severity="warn", node=NODE, detail=str(exc)
                    )
                ],
            }

        retrieved = [to_retrieved(c, s, r) for c, s, r in result.hits]

        # Screen at the point untrusted text enters state, not at render time: a detection
        # must be recorded even if a budget ceiling later stops the run before `generate`
        # executes. Neutralisation is unconditional; quarantine drops BLOCK-level chunks
        # from the context entirely (src/guardrails/injection.py).
        screening = screen_chunks(retrieved, node=NODE)
        chunks = screening.kept

        # Counted over `retrieved`, not `chunks`: this log records what retrieval returned.
        # What screening withheld is a separate fact with its own guardrail events.
        events: list[RetrievalEvent] = [
            RetrievalEvent(
                query=sub_question,
                retriever="dense",
                k=options.top_k,
                n_hits=sum(1 for c in retrieved if c.retriever == "dense"),
                latency_ms=result.latency_ms,
                hit_chunk_ids=[c.chunk_id for c in retrieved if c.retriever == "dense"],
            )
        ]
        if result.used_sparse:
            events.append(
                RetrievalEvent(
                    query=sub_question,
                    retriever="sparse",
                    k=options.top_k,
                    n_hits=sum(1 for c in retrieved if c.retriever == "sparse"),
                    latency_ms=result.latency_ms,
                    hit_chunk_ids=[c.chunk_id for c in retrieved if c.retriever == "sparse"],
                )
            )
        if result.used_arxiv:
            events.append(
                RetrievalEvent(
                    query=sub_question,
                    retriever="arxiv",
                    k=options.top_k,
                    n_hits=sum(1 for c in retrieved if c.retriever == "arxiv"),
                    latency_ms=result.latency_ms,
                    hit_chunk_ids=[c.chunk_id for c in retrieved if c.retriever == "arxiv"],
                )
            )

        tool_names = ["search_corpus"]
        if result.used_sparse:
            tool_names.append("keyword_search")
        if result.used_arxiv:
            tool_names.append("fetch_arxiv")

        return {
            "plan_cursor": cursor + 1,
            "retrieved": chunks,
            "retrieval_events": events,
            "guardrail_events": screening.events,
            "tool_calls": [
                ToolCallRecord(
                    name=name,
                    args={"query": sub_question, "top_k": options.top_k},
                    ok=True,
                    latency_ms=result.latency_ms,
                )
                for name in tool_names
            ],
            "usage": _tool_usage(len(tool_names)),
        }

    return retrieve


def _tool_usage(n_tools: int) -> object:
    from src.agent.state import Usage

    return Usage(tool_calls=n_tools)
