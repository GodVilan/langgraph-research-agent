"""The scorer's view of one agent answer, and the record a score becomes.

One renderer, two consumers: `evals/score_cli.py` shows the human exactly ``view()``, and the
judge prompt (`evals/judge.py`) embeds exactly ``view()``. That is what makes "the judge prompt
carries exactly the fields the CLI shows" a property of the code rather than a promise — a
field added to one is added to both, and a disagreement between human and judge is a
difference of judgement, never of information.

The view is in two parts because the rubric reads them in order: ``before_gold`` (question and
agent answer — enough to decide Q1, what the agent did) and ``with_gold`` (stratum contract,
gold answer, gold chunk text, retrieved chunk text — what Q2 and Q3 need). Full chunk text on
both sides, never ids alone: a scorer with ids cannot check grounding, and every grounding
disagreement would then be an artifact of who could see the text.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from evals.absence import load_corpus
from evals.rubric import Behaviour, Outcome, expected_behaviour, render
from evals.run_set import ItemRun
from evals.schema import Stratum

CHUNK_LIMIT = 1600  # characters shown per chunk; a scorer needs the fact, not the appendix


def rubric_sha256() -> str:
    """Stamped onto every score so a rubric edited after scoring is detectable."""
    return hashlib.sha256(render().encode("utf-8")).hexdigest()


def _chunk_text(chunk_id: str) -> str:
    for chunk in load_corpus().chunks:
        if str(chunk["chunk_id"]) == chunk_id:
            return " ".join(str(chunk["text"]).split())
    return "(not in corpus)"


def _clip(text: str, limit: int = CHUNK_LIMIT) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def before_gold(run: ItemRun) -> str:
    """What Q1 is decided from. Nothing here reveals the expected behaviour."""
    return (
        f"ITEM {run.item_id}\n\n"
        f"QUESTION\n{run.question}\n\n"
        f"AGENT ANSWER\n{run.answer or '(no answer produced)'}\n"
    )


def with_gold(run: ItemRun) -> str:
    """What Q2 and Q3 are decided from."""
    stratum = Stratum(run.stratum)
    expected = expected_behaviour(stratum).value
    lines = [
        f"STRATUM  {stratum.value}   (expected behaviour: {expected})",
        "",
        "GOLD ANSWER",
        run.gold_answer or "(none — this item expects a refusal)",
        "",
        f"GOLD CHUNKS ({len(run.gold_chunk_ids)})",
    ]
    for cid in run.gold_chunk_ids:
        lines += [f"  [{cid}]", f"  {_clip(_chunk_text(cid))}", ""]
    if not run.gold_chunk_ids:
        lines += ["  (none)", ""]
    lines.append(f"RETRIEVED CHUNKS ({len(run.retrieved)}) — grounding is judged against these")
    for c in run.retrieved:
        lines += [f"  [{c.chunk_id}]  score {c.score:.3f}", f"  {_clip(c.text)}", ""]
    if not run.retrieved:
        lines += ["  (the agent retrieved nothing)", ""]
    return "\n".join(lines)


def view(run: ItemRun) -> str:
    """The complete scorer's view — the judge prompt embeds this string verbatim."""
    return before_gold(run) + "\n" + with_gold(run)


class Score(BaseModel):
    """One scorer's label for one item, with every input to the rubric function recorded."""

    item_id: str
    scorer: str  # "human" or a judge arm id
    behaviour: Behaviour
    fact_matches: bool | None = None
    grounded: bool | None = None
    outcome: Outcome
    reason: str = ""
    scored_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ScoreSheet(BaseModel):
    """Scores against one run, stamped with the run, set and rubric they were made against."""

    scorer: str
    run_path: str
    set_sha256: str
    rubric_sha256: str
    sample_path: str
    # The one input the artifact cannot pin. Every other input above is a hash or a seed;
    # no provider exposes a weights revision for either judge model (D-030b), so the sheet
    # says so in the same place it records everything it *can* pin.
    judge_weights_revision: str = "n/a — human scorer"
    # Set when a sheet was rescored under an amended rubric: which items were re-judged and
    # the sha they were originally scored under. A post-amendment agreement figure is never
    # printed without these beside it.
    previous_rubric_sha256: str | None = None
    rescored_items: list[str] = Field(default_factory=list)
    rescore_note: str = ""
    scores: dict[str, Score] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> ScoreSheet:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
