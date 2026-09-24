"""The input guardrail's decision, stated so a caller can read it.

The input guardrail is a chain of checks, and they are not alike. Length, encoding,
injection and the keyword fast path are deterministic: the same question gets the same
answer every time. The model classifier is not. Across seven v3 runs on identical input it
blocked between 3 and 6 of the same 43 verified factual questions, and three questions
flipped between answered and refused with nothing changed (`make guardrail-variance`).

On a public endpoint that means a caller can ask the same question twice and get an answer
once and a refusal once. This module does not make the decision stable. It makes it
*legible*: which stage decided, why, and whether asking again could change the result
(DECISIONS D-035). The classifier now runs with pinned sampling, measured stable; the
decision is still reported as not deterministic, because stable-in-a-probe is not a
guarantee the provider makes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel

from src.agent.state import GuardrailEvent

Stage = Literal[
    "input_validation",
    "injection_check",
    "keyword_fastpath",
    "scope_classifier",
    "scope_check_failed",
]

# Event kind -> the stage that emitted it, and whether that stage is deterministic.
_STAGES: dict[str, tuple[Stage, bool]] = {
    "input_too_short": ("input_validation", True),
    "input_too_long": ("input_validation", True),
    "control_characters": ("input_validation", True),
    "invisible_characters": ("input_validation", True),
    "injection_in_question": ("injection_check", True),
    "scope_keyword_fastpath": ("keyword_fastpath", True),
    "out_of_scope": ("scope_classifier", False),
    "scope_classifier_pass": ("scope_classifier", False),
    # A model call that failed to parse — depends on the provider, not the question.
    "scope_check_failed": ("scope_check_failed", False),
}

NONDETERMINISTIC_NOTE = (
    "This decision was made by a model classifier that is not deterministic: the same "
    "question can be refused on one request and answered on another. See the README's "
    "'Known limitations'."
)
PINNED_NOTE = (
    "This decision was made by a model classifier with pinned sampling (seed=0, top_k=1). "
    "Pinned, it gave byte-identical output on 140 of 140 probe calls, so the same question "
    "should get the same decision — but the provider does not guarantee determinism, and a "
    "model update can change the decision. See the README's 'Known limitations'."
)


class GuardrailDecision(BaseModel):
    """What the input guardrail decided, returned in every API response."""

    decision: Literal["passed", "refused"]
    stage: Stage
    reason: str
    deterministic: bool
    note: str | None = None


def input_decision(
    refused: bool, events: Iterable[GuardrailEvent], classifier_pinned: bool = True
) -> GuardrailDecision:
    """Project the input-stage decision out of a run's guardrail events.

    The *last* input-stage event decides, because the chain stops at the first refusal and
    a pass is recorded only by the stage that let the question through.
    """
    decided: tuple[GuardrailEvent, Stage, bool] | None = None
    for event in events:
        stage = _STAGES.get(event.kind)
        if stage is not None:
            decided = (event, stage[0], stage[1])

    if decided is None:
        # No input-stage event at all: a run that never passed through validate_input, or
        # an event kind this table does not know. Stated, not guessed.
        return GuardrailDecision(
            decision="refused" if refused else "passed",
            stage="input_validation",
            reason="no input guardrail event was recorded for this run",
            deterministic=False,
            note="The guardrail decision could not be attributed to a stage.",
        )

    event, stage_name, deterministic = decided
    reason = event.detail or event.kind.replace("_", " ")
    return GuardrailDecision(
        decision="refused" if refused else "passed",
        stage=stage_name,
        reason=reason,
        deterministic=deterministic,
        note=None
        if deterministic
        else (PINNED_NOTE if classifier_pinned else NONDETERMINISTIC_NOTE),
    )
