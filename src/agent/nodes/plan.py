"""Multi-hop query decomposition.

v2.1 regex-stripped fences off a JSON response and fell back to a single-hop plan on any
exception, with the only trace being a WARNING log (AUDIT §4.4). Here a parse failure still
degrades to single-hop — that is the right behaviour — but it emits a ``GuardrailEvent``
so the degradation is visible rather than invisible.
"""

from __future__ import annotations

import logging

from src.agent.llm import StructuredOutputError, call_structured
from src.agent.prompts import load_prompt
from src.agent.state import AgentState, GuardrailEvent, Plan, SubQuestion, Usage
from src.config import get_settings

log = logging.getLogger(__name__)

NODE = "plan"


async def plan(state: AgentState) -> dict[str, object]:
    settings = get_settings()
    question = state["question"]
    version = state["request"].prompt_version
    max_subs = settings.graph.max_sub_questions

    system = load_prompt("plan", version).format(max_sub_questions=max_subs)

    try:
        parsed, usage = await call_structured(
            Plan,
            system=system,
            user=f"Question: {question}",
            max_attempts=settings.graph.max_structured_output_attempts,
        )
    except StructuredOutputError as exc:
        log.warning("Planner failed; degrading to single-hop: %s", exc)
        return {
            "plan": [SubQuestion(text=question, origin="plan")],
            "plan_cursor": 0,
            "usage": getattr(exc, "usage", Usage()),
            "guardrail_events": [
                GuardrailEvent(
                    kind="plan_parse_failed",
                    severity="warn",
                    node=NODE,
                    detail=f"degraded to single-hop: {exc}",
                )
            ],
        }

    sub_texts = [s.strip() for s in parsed.sub_questions if s.strip()][:max_subs]
    if parsed.kind != "complex" or not sub_texts:
        steps = [SubQuestion(text=question, origin="plan")]
    else:
        steps = [SubQuestion(text=t, origin="plan") for t in sub_texts]

    log.info("Plan: %s, %d step(s)", parsed.kind, len(steps))
    return {"plan": steps, "plan_cursor": 0, "usage": usage}
