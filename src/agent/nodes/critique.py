"""Self-critique with a falsifiable verdict.

The one rule that matters here: **a critique that failed to run must never look like a
critique that passed.** v2.1 returned ``CritiqueResult("pass", …)`` from a bare
``except Exception`` (AUDIT §4.3), so a permanently broken critic and a clean approval
were indistinguishable — and the CLI printed "passed" for both. A parse failure here sets
``verdict="error"``, which routes to ``finalize`` (never to a refinement round) and lands
in the response as a flag.

On ``retry`` the node appends the critic's search hints to ``plan`` and rewinds
``plan_cursor`` to the first of them, so the refinement loop reuses the ``retrieve`` node
instead of v2.1's separate corrective loop.
"""

from __future__ import annotations

import logging

from src.agent.llm import StructuredOutputError, call_structured
from src.agent.nodes.generate import render_context
from src.agent.prompts import load_prompt
from src.agent.state import AgentState, Critique, GuardrailEvent, SubQuestion, Usage
from src.config import get_settings

log = logging.getLogger(__name__)

NODE = "critique"

MAX_HINTS = 2


async def critique(state: AgentState) -> dict[str, object]:
    settings = get_settings()
    question = state["question"]
    draft = state.get("draft_answer") or ""
    chunks = state.get("retrieved") or []
    already = state.get("refinement_count", 0)

    try:
        verdict, usage = await call_structured(
            Critique,
            system=load_prompt("critique", state["request"].prompt_version),
            user=(
                f"Question: {question}\n\n"
                f"Proposed answer:\n{draft}\n\n"
                f"Retrieved passages:\n{render_context(chunks)}"
            ),
            max_attempts=settings.graph.max_structured_output_attempts,
        )
    except StructuredOutputError as exc:
        log.warning("Critique failed to parse; marking verdict=error: %s", exc)
        return {
            "critique": Critique(verdict="error", gaps=[f"critique unavailable: {exc}"]),
            "refinement_count": already,
            "usage": getattr(exc, "usage", Usage()),
            "guardrail_events": [
                GuardrailEvent(
                    kind="critique_parse_failed", severity="warn", node=NODE, detail=str(exc)
                )
            ],
        }

    # A `retry` with no actionable hint cannot drive a useful refinement pass. Treat it as a
    # pass rather than spending another round on the same query.
    hints = [h.strip() for h in verdict.search_hints if h.strip()][:MAX_HINTS]
    if verdict.verdict == "retry" and not hints:
        log.info("Critique returned retry with no hints; treating as pass")
        return {
            "critique": Critique(**{**verdict.model_dump(), "verdict": "pass"}),
            "refinement_count": already,
            "usage": usage,
            "guardrail_events": [
                GuardrailEvent(kind="retry_without_hints", severity="info", node=NODE)
            ],
        }

    if verdict.verdict != "retry":
        return {"critique": verdict, "refinement_count": already, "usage": usage}

    existing = list(state.get("plan") or [])
    cursor = len(existing)
    existing.extend(SubQuestion(text=h, origin="refinement") for h in hints)

    log.info("Critique: retry with %d hint(s), refinement %d", len(hints), already + 1)
    return {
        "critique": verdict,
        "plan": existing,
        "plan_cursor": cursor,
        # Read-modify-write in the single node that owns this counter. See state.py.
        "refinement_count": already + 1,
        "usage": usage,
    }
