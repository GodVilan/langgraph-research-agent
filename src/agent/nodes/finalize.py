"""Terminal node. Every path out of the graph goes through here.

Refusals, truncations, and successes all leave by the same exit so the caller gets one
response shape. ``sources`` is *projected* from ``retrieved`` — v2.1 reconstructed it by
regex over formatted tool output, which silently returned nothing for three of its six
tools (AUDIT §4.19).
"""

from __future__ import annotations

import logging
import re

from src.agent.nodes.validate_input import REFUSAL_MESSAGE
from src.agent.state import AgentState, RetrievedChunk, Source
from src.config import get_settings
from src.guardrails.budget import check_budget

log = logging.getLogger(__name__)

NODE = "finalize"

CHUNK_CITE_RE = re.compile(r"\[([0-9]{4}\.[0-9]{4,5}(?:v\d+)?_\d{4})\]")

TRUNCATION_NOTE = (
    "\n\n---\n*This answer is partial: the request hit a budget ceiling "
    "({reason}) before the agent finished.*"
)
CRITIQUE_ERROR_NOTE = (
    "\n\n---\n*The self-critique step did not run for this answer, so it has not been "
    "checked for groundedness.*"
)


def project_sources(chunks: list[RetrievedChunk], cited_ids: set[str]) -> list[Source]:
    """Group chunks by paper, keeping the best score and every contributing chunk id.

    Papers whose chunks were actually cited sort first; within each group, by score.
    """
    by_paper: dict[str, Source] = {}
    for chunk in chunks:
        existing = by_paper.get(chunk.paper_id)
        if existing is None:
            by_paper[chunk.paper_id] = Source(
                paper_id=chunk.paper_id,
                title=chunk.title,
                score=round(chunk.score, 4),
                chunk_ids=[chunk.chunk_id],
            )
        else:
            existing.chunk_ids.append(chunk.chunk_id)
            existing.score = round(max(existing.score, chunk.score), 4)

    def sort_key(src: Source) -> tuple[int, float]:
        was_cited = any(cid in cited_ids for cid in src.chunk_ids)
        return (0 if was_cited else 1, -src.score)

    return sorted(by_paper.values(), key=sort_key)


async def finalize(state: AgentState) -> dict[str, object]:
    if state.get("refused"):
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "truncated": False,
        }

    # Truncation is decided here rather than in a router, because routers must stay pure —
    # LangGraph may evaluate one more than once.
    verdict = check_budget(state["usage"], get_settings().budget)
    truncated = not verdict.ok

    answer = state.get("draft_answer") or ""
    chunks = state.get("retrieved") or []
    cited_ids = set(CHUNK_CITE_RE.findall(answer))

    # Citations the model invented. Reported rather than silently dropped, because a
    # fabricated id is a groundedness signal Phase 4 will want to measure.
    known_ids = {c.chunk_id for c in chunks}
    hallucinated = cited_ids - known_ids
    if hallucinated:
        log.warning("Answer cites %d unknown chunk id(s)", len(hallucinated))

    critique = state.get("critique")
    if critique is not None and critique.verdict == "error":
        answer += CRITIQUE_ERROR_NOTE

    if truncated:
        answer += TRUNCATION_NOTE.format(reason=verdict.reason or "unspecified")

    updates: dict[str, object] = {
        "answer": answer,
        "sources": project_sources(chunks, cited_ids & known_ids),
        "truncated": truncated,
        "truncation_reason": verdict.reason,
    }
    if hallucinated:
        from src.agent.state import GuardrailEvent

        updates["guardrail_events"] = [
            GuardrailEvent(
                kind="unknown_citation",
                severity="warn",
                node=NODE,
                detail=f"{len(hallucinated)} cited id(s) not in retrieved set: "
                + ", ".join(sorted(hallucinated)[:5]),
            )
        ]
    return updates
