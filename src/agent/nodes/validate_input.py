"""Input validation and scope classification.

Fails **closed**. v2.1's scope guard returned "in scope" on any exception (AUDIT §4.2), so
an expired key or a network blip routed every query — including the ones the guard existed
to reject — into the full pipeline. On a public endpoint with a real API key behind it, the
failure mode of a cost guard must not be "spend money".

Phase 2 expands this module (length caps are here already; encoding validation and the
injection detector land there).
"""

from __future__ import annotations

import logging
import unicodedata

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agent.llm import StructuredOutputError, call_structured, get_chat_model
from src.agent.prompts import load_prompt
from src.agent.state import AgentState, GuardrailEvent, Usage
from src.config import get_settings
from src.guardrails.injection import INVISIBLE_RE, Severity, scan, worst_severity

log = logging.getLogger(__name__)

NODE = "validate_input"

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 2_000

REFUSAL_MESSAGE = (
    "I answer questions about machine-learning and AI research papers. "
    "I can help with ML methods, models, benchmarks, and the literature — for example "
    "'What is LoRA?', 'Compare RLHF approaches', or 'What does the corpus say about "
    "continual learning?'"
)

# Presence of any of these skips the classifier call entirely. Ported from v2.1; it removes
# one model round-trip from the hot path on the common case.
IN_SCOPE_TERMS = frozenset(
    {
        "machine learning",
        "deep learning",
        "neural network",
        "transformer",
        "llm",
        "language model",
        "embedding",
        "fine-tun",
        "pretrain",
        "rlhf",
        "lora",
        "diffusion",
        "gradient",
        "attention",
        "arxiv",
        "paper",
        "dataset",
        "benchmark",
        "reinforcement learning",
        "supervised",
        "unsupervised",
        "convolution",
        "gan",
        "bert",
        "gpt",
        "backprop",
        "overfit",
        "regulariz",
        "hyperparameter",
        "quantization",
        "distillation",
        "retrieval",
        "rag",
        "catastrophic forgetting",
        "continual learning",
        "few-shot",
        "zero-shot",
    }
)


class ScopeVerdict(BaseModel):
    in_scope: bool = Field(description="True if the question is about ML/AI research.")
    reason: str = Field(default="", description="One short sentence.")


def _refuse(reason: str, events: list[GuardrailEvent]) -> dict[str, object]:
    return {
        "refused": True,
        "refusal_reason": reason,
        "guardrail_events": events,
        "messages": [AIMessage(content=REFUSAL_MESSAGE)],
    }


async def validate_input(state: AgentState) -> dict[str, object]:
    question = (state.get("question") or "").strip()
    events: list[GuardrailEvent] = []

    if len(question) < MIN_QUESTION_CHARS:
        events.append(
            GuardrailEvent(
                kind="input_too_short", severity="block", node=NODE, detail=f"{len(question)} chars"
            )
        )
        return _refuse("Question is empty or too short.", events)

    if len(question) > MAX_QUESTION_CHARS:
        events.append(
            GuardrailEvent(
                kind="input_too_long",
                severity="block",
                node=NODE,
                detail=f"{len(question)} chars > {MAX_QUESTION_CHARS}",
            )
        )
        return _refuse(f"Question exceeds {MAX_QUESTION_CHARS} characters.", events)

    # Control characters other than tab/newline indicate a malformed or hostile payload.
    if any(unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in question):
        events.append(GuardrailEvent(kind="control_characters", severity="block", node=NODE))
        return _refuse("Question contains control characters.", events)

    # The same encoding tricks used to hide instructions inside retrieved documents work on
    # the way in: bidi overrides and zero-width joiners make the question a human reviewer
    # sees differ from the one the model receives. Refused rather than stripped, because a
    # legitimate research question has no use for them.
    if INVISIBLE_RE.search(question):
        events.append(
            GuardrailEvent(
                kind="invisible_characters",
                severity="block",
                node=NODE,
                detail=f"{len(INVISIBLE_RE.findall(question))} zero-width or bidi character(s)",
            )
        )
        return _refuse("Question contains hidden formatting characters.", events)

    # Scope matching, and everything downstream, sees the normalised form. Without this a
    # fullwidth or compatibility spelling walks past the keyword list and the classifier.
    question = unicodedata.normalize("NFKC", question)

    # An injection aimed at the *input* rather than at a retrieved document. Detected with
    # the same rules, and refused rather than quarantined: unlike a paper, a question that
    # is entirely an injection has no legitimate remainder to answer from.
    injections = scan(question)
    if worst_severity(injections) is Severity.BLOCK:
        events.append(
            GuardrailEvent(
                kind="injection_in_question",
                severity="block",
                node=NODE,
                detail="; ".join(d.describe()[:100] for d in injections[:3]),
            )
        )
        return _refuse("Question contains an embedded instruction directed at the model.", events)

    lowered = question.lower()
    if any(term in lowered for term in IN_SCOPE_TERMS):
        events.append(GuardrailEvent(kind="scope_keyword_fastpath", severity="info", node=NODE))
        return {
            "refused": False,
            "guardrail_events": events,
            "messages": [HumanMessage(content=question)],
        }

    version = state["request"].prompt_version
    settings = get_settings()
    try:
        verdict, usage = await call_structured(
            ScopeVerdict,
            system=load_prompt("scope", version),
            user=f"Question: {question}",
            max_attempts=settings.graph.max_structured_output_attempts,
            # Pinned sampling (D-035): the same question gets the same decision.
            model=get_chat_model(pinned=True) if settings.pin_scope_classifier else None,
        )
    except StructuredOutputError as exc:
        # Fail closed. A guard that cannot run must not wave the request through.
        events.append(
            GuardrailEvent(kind="scope_check_failed", severity="block", node=NODE, detail=str(exc))
        )
        result = _refuse("Scope check could not run; refusing rather than proceeding.", events)
        result["usage"] = getattr(exc, "usage", Usage())
        return result

    if not verdict.in_scope:
        events.append(
            GuardrailEvent(kind="out_of_scope", severity="block", node=NODE, detail=verdict.reason)
        )
        result = _refuse("Question is outside the ML/AI research scope.", events)
        result["usage"] = usage
        return result

    # Recorded on a pass as well as a refusal. The classifier is nondeterministic — the same
    # question is refused on some draws and not others (`make guardrail-variance`) — so a
    # pass it decided is a different fact from a pass the keyword list decided, and the API
    # reports which one happened (src/guardrails/decision.py).
    events.append(
        GuardrailEvent(
            kind="scope_classifier_pass", severity="info", node=NODE, detail=verdict.reason
        )
    )
    return {
        "refused": False,
        "guardrail_events": events,
        "usage": usage,
        "messages": [HumanMessage(content=question)],
    }
