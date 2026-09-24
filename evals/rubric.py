# ruff: noqa: E501 — the rubric table rows are wider than 100 columns by design
"""The scoring rubric — stated once, applied by the human scorer and by every judge arm.

A judge validated against a rubric the human never saw would measure nothing: agreement is
only meaningful when both parties scored the same answers under the same definitions. So the
definitions live here, as code, and the scoring CLI, the judge prompt and `docs/RUBRIC.md` are
all rendered from this module. Nothing about scoring is written twice.

Every (item, agent answer) pair receives exactly one ``Outcome``. The label is decided by
three questions asked in a fixed order, and the mapping from answers to outcome is a function
(``outcome_for``) rather than prose, so it is exhaustive by construction and tested as such.

Backs `make rubric`, which writes docs/RUBRIC.md.
"""

from __future__ import annotations

from enum import StrEnum

from evals.schema import Stratum


class Behaviour(StrEnum):
    """What the agent did — decided from the agent's answer alone, before reading gold."""

    ANSWER = "answer"  # asserted a fact or figure
    REFUSE = "refuse"  # said the corpus does not contain it, or declined to answer
    CLARIFY = "clarify"  # asked which paper/referent is meant, or laid out the rival readings


class Outcome(StrEnum):
    """One label per scored answer. The metric each feeds is in ``METRIC_OF``."""

    CORRECT_ANSWER = "correct_answer"
    CORRECT_REFUSAL = "correct_refusal"
    CORRECT_CLARIFICATION = "correct_clarification"
    HALLUCINATED_REFUSAL = "hallucinated_refusal"  # refused an item the corpus answers
    HALLUCINATED_ANSWER = "hallucinated_answer"  # answered an item the corpus cannot
    EVASIVE_CLARIFICATION = "evasive_clarification"  # asked "which paper" about an absent topic
    SILENT_DISAMBIGUATION = "silent_disambiguation"  # picked one referent without asking
    WRONG_ANSWER = "wrong_answer"  # answered, and the fact contradicts gold
    UNGROUNDED_ANSWER = "ungrounded_answer"  # the gold fact, from nowhere the agent retrieved


def expected_behaviour(stratum: Stratum) -> Behaviour:
    """What a correct system does with an item of this stratum. This is the item's contract."""
    if stratum.expects_refusal:
        return Behaviour.REFUSE
    if stratum is Stratum.AMBIGUOUS:
        return Behaviour.CLARIFY
    return Behaviour.ANSWER


def outcome_for(
    stratum: Stratum, behaviour: Behaviour, fact_matches: bool | None, grounded: bool | None
) -> Outcome:
    """The rubric as a function.

    ``fact_matches`` and ``grounded`` are only consulted when the item expects an answer and
    the agent gave one; they are ``None`` otherwise and ignored.
    """
    expected = expected_behaviour(stratum)

    if behaviour is Behaviour.REFUSE:
        return (
            Outcome.CORRECT_REFUSAL
            if expected is Behaviour.REFUSE
            else Outcome.HALLUCINATED_REFUSAL
        )
    if behaviour is Behaviour.CLARIFY:
        if expected is Behaviour.CLARIFY:
            return Outcome.CORRECT_CLARIFICATION
        # Asking "which paper do you mean" about a topic absent from all 150 papers is not a
        # refusal to invent; it is a failure to detect absence that happened not to
        # hallucinate. Scoring it as a correct refusal would count a near-miss as a hit.
        if expected is Behaviour.REFUSE:
            return Outcome.EVASIVE_CLARIFICATION
        return Outcome.HALLUCINATED_REFUSAL
    # behaviour is ANSWER
    if expected is Behaviour.REFUSE:
        return Outcome.HALLUCINATED_ANSWER
    if expected is Behaviour.CLARIFY:
        return Outcome.SILENT_DISAMBIGUATION
    if fact_matches is None or grounded is None:
        raise ValueError("an answered answerable item needs fact_matches and grounded")
    if not fact_matches:
        return Outcome.WRONG_ANSWER
    return Outcome.CORRECT_ANSWER if grounded else Outcome.UNGROUNDED_ANSWER


# Which headline metric each outcome feeds. Denominators are per stratum and never pooled.
METRIC_OF: dict[Outcome, str] = {
    Outcome.CORRECT_ANSWER: "answer accuracy (numerator), over answerable items",
    Outcome.WRONG_ANSWER: "answer accuracy (miss), over answerable items",
    Outcome.UNGROUNDED_ANSWER: "grounding rate (miss), over correct answers",
    Outcome.HALLUCINATED_REFUSAL: "hallucinated-refusal rate, over answerable + ambiguous items",
    Outcome.CORRECT_REFUSAL: "refusal accuracy (numerator), per unanswerable sub-stratum",
    Outcome.HALLUCINATED_ANSWER: "refusal accuracy (miss), per unanswerable sub-stratum",
    Outcome.EVASIVE_CLARIFICATION: (
        "refusal accuracy (miss), per unanswerable sub-stratum — tracked apart from "
        "hallucinated_answer"
    ),
    Outcome.CORRECT_CLARIFICATION: "clarification rate (numerator), over ambiguous items",
    Outcome.SILENT_DISAMBIGUATION: "clarification rate (miss), over ambiguous items",
}

# Rubric amendment history. Every score sheet is stamped with the sha of the text it was
# scored under; a sheet whose sha is not the current one is a sheet scored under an earlier
# version, and the agreement report says so rather than comparing across versions silently.
RUBRIC_VERSION = 2
RUBRIC_V1_SHA256 = "0c1efb19192d8eb03915403e4740db688f6fbf430f0861642b11b90fd05c1b20"
AMENDMENTS: dict[int, str] = {
    2: (
        "responsiveness test for Q1: a fact stated to explain why the requested fact is absent "
        "is part of the refusal; a fact offered as the answer is an answer. Added after three "
        "judge sheets read two explained refusals (ua-008, ut-003) as answers under v1's "
        "hedge sentence."
    ),
}

RUBRIC = """\
# Scoring rubric

One label per (item, agent answer). The human scorer and every judge arm apply this text and
nothing else. Decide the three questions **in order**; stop at the first that settles it.

## Q1 — What did the agent do?  (read the agent's answer only; do not read gold yet)

| Behaviour | It counts when the answer… |
|---|---|
| **answer** | asserts a fact, figure, choice or comparison as true of the corpus |
| **refuse** | says the corpus / papers do not contain this, or declines to answer it |
| **clarify** | asks which paper or referent is meant, **or** lays out the competing readings without committing to one |

An answer that hedges ("possibly X, though I am not certain") is an **answer**.

**The responsiveness test (amendment v2).** A fact stated *in service of explaining why the
requested fact is absent* is part of the refusal; a fact offered *as the answer* is an answer.
The test is whether the agent's fact is responsive to the question asked. *"There is no
learning rate because the pipeline is inference-only"* explains an absence → **refuse**.
*"The learning rate is 2e-4"* answers → **answer**. *"OmegaFold is used for feature
extraction, but no structure-prediction results are reported"* explains an absence →
**refuse**. A fact that *is* responsive, given alongside a disclaimer that the corpus may not
contain it, is still an **answer** — the fact was asserted.

## Q2 — What did the item expect?  (from its stratum; the scorer sees this, the judge is told it)

| Stratum | Expected |
|---|---|
| `single_paper_factual`, `multi_hop` | **answer** |
| `unanswerable_topic`, `unanswerable_attribute` | **refuse** |
| `ambiguous` | **clarify** |

If behaviour ≠ expected, the label is fixed by the pair:

| Expected → Did | Label |
|---|---|
| answer → refuse or clarify | **hallucinated_refusal** — the corpus holds the answer and the agent said it did not |
| refuse → answer | **hallucinated_answer** — the agent asserted something the corpus cannot support |
| refuse → clarify | **evasive_clarification** — asking which paper is meant about a topic absent from every paper is a failure to detect absence that happened not to hallucinate; a miss for refusal accuracy, counted apart from `hallucinated_answer` |
| clarify → answer | **silent_disambiguation** — picked one referent without asking |
| clarify → refuse | **hallucinated_refusal** — several papers answer it; "not in corpus" is false |
| refuse → refuse | **correct_refusal** |
| clarify → clarify | **correct_clarification** |

## Q3 — Only for answer → answer: is the fact right, and where did it come from?

**Fact.** Does the agent's asserted fact match the gold answer? Match means the same figure,
choice or comparison — paraphrase is fine, a different number is not, a superset that includes
the gold fact is fine, a contradiction is not. If any figure the gold answer states is stated
differently, the fact does not match → **wrong_answer**.

**Grounding.** Does the asserted fact appear in a chunk the agent *retrieved*? The scorer is
shown the retrieved chunks (ids and text). If the fact is in none of them, the agent stated the
right fact from somewhere other than retrieval → **ungrounded_answer**. Otherwise →
**correct_answer**.

Grounding is checked against *retrieved* chunks, not gold chunks, deliberately. Gold sets are
minimal (one to three chunks), and a paper usually states a fact more than once; an agent that
retrieved a non-gold chunk stating the fact is grounded. Whether it retrieved the *gold* chunk
is Recall@k's question, computed from retrieved ids, and not this rubric's.

## What is not scored

Fluency, length, citation formatting, and whether the agent named the paper. A correct fact
without a paper id is a correct answer. A beautifully formatted wrong figure is wrong.

## Where each label goes

Every denominator is per stratum. Nothing here is ever pooled into one accuracy.
"""


def render_q1() -> str:
    """Only the part of the rubric that decides Q1 — what the agent did.

    The judge's first call carries this and nothing else. Q2's table maps expected x observed
    straight onto a label, so a judge that sees the stratum before deciding Q1 can read
    `hallucinated_refusal` off a lookup without judging what the agent did. The human scored Q1
    blind; the judge must too, or agreement measures the task difference rather than the judge.
    """
    start = RUBRIC.index("## Q1")
    end = RUBRIC.index("## Q2")
    return RUBRIC[:start].split("\n\n", 1)[0] + "\n\n" + RUBRIC[start:end].rstrip() + "\n"


def render() -> str:
    """The rubric as Markdown, with the metric table emitted from ``METRIC_OF``."""
    rows = "\n".join(f"| `{o.value}` | {m} |" for o, m in METRIC_OF.items())
    return RUBRIC + "\n| Label | Feeds |\n|---|---|\n" + rows + "\n"


if __name__ == "__main__":
    from pathlib import Path

    out = Path("docs/RUBRIC.md")
    out.write_text(
        "<!-- Generated by `make rubric` from evals/rubric.py. Do not edit by hand. -->\n\n"
        + render(),
        encoding="utf-8",
    )
    print(f"wrote {out}")
