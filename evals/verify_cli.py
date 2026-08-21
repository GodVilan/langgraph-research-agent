"""Human verification of eval items, one at a time.

Backs `make verify-evals`. Srikanth verifies 25 items by hand; this is what he sees.

Design constraints that are not negotiable:

* **Every automated check runs and is shown before the question.** A verifier who reads the
  question first is anchored by it. The absence checks and the multi-hop sufficiency check
  are what the automated pipeline believes; showing them first makes disagreement visible
  rather than polite.
* **Rejecting is as cheap as accepting.** A verification tool that makes rejection
  effortful produces a verified set that means nothing.
* **Progress is saved after every decision.** A crash at item 20 of 25 must not cost the
  first 19.
* **The generator's intent is recorded separately from the human label**, so disagreement
  between them can be reported instead of silently resolved in the generator's favour.

Usage:
    python -m evals.verify_cli evals/datasets/draft.json
    python -m evals.verify_cli evals/datasets/draft.json --stratum multi_hop
    python -m evals.verify_cli evals/datasets/draft.json --report
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from evals.absence import load_corpus, verify_attribute_absent, verify_topic_absent
from evals.schema import EvalItem, EvalSet, Stratum

app = typer.Typer(add_completion=False, help="Verify eval items by hand.")

ACCEPT, EDIT, REJECT, SKIP, QUIT = "a", "e", "r", "s", "q"


def _c(text: str, colour: str) -> str:
    return typer.style(text, fg=colour)


def show_chunk(chunk_id: str, limit: int = 400) -> None:
    corpus = load_corpus()
    for chunk in corpus.chunks:
        if str(chunk["chunk_id"]) == chunk_id:
            text = " ".join(str(chunk["text"]).split())
            typer.echo(f"    [{chunk_id}] {text[:limit]}{'...' if len(text) > limit else ''}")
            return
    typer.echo(_c(f"    [{chunk_id}] NOT FOUND IN CORPUS", typer.colors.RED))


def run_automated_checks(item: EvalItem) -> list[tuple[bool, str]]:
    """Everything the machine can decide, run before the human sees the question."""
    results: list[tuple[bool, str]] = []

    if item.stratum is Stratum.UNANSWERABLE_TOPIC:
        outcome = verify_topic_absent(item.absent_term)
        results.append((outcome.absent, f"topic absence (all chunks): {outcome.reason}"))
    elif item.stratum is Stratum.UNANSWERABLE_ATTRIBUTE:
        outcome = verify_attribute_absent(item.anchor_paper_id, item.absent_term)
        results.append((outcome.absent, f"attribute absence (all chunks): {outcome.reason}"))
    else:
        corpus_ids = {str(c["chunk_id"]) for c in load_corpus().chunks}
        missing = [g for g in item.gold_chunk_ids if g not in corpus_ids]
        detail = f"missing from the corpus: {missing}" if missing else "all present"
        results.append((not missing, f"gold chunks exist: {detail}"))

    if item.stratum is Stratum.MULTI_HOP:
        checked = item.verification.single_paper_sufficiency_checked
        alone = item.verification.answerable_by_one_paper
        results.append(
            (
                checked and not alone,
                "single-paper sufficiency: "
                + (
                    "NOT CHECKED — run the drafter's check first"
                    if not checked
                    else "one paper answers it alone; this is not multi-hop"
                    if alone
                    else "no single paper answers it"
                ),
            )
        )
    return results


def render(item: EvalItem, index: int, total: int) -> None:
    typer.echo("\n" + "=" * 78)
    typer.echo(
        f"Item {index}/{total}   {_c(item.stratum.value, typer.colors.CYAN)}   {item.item_id}"
    )
    typer.echo("=" * 78)

    typer.echo(_c("\nAutomated checks", typer.colors.BRIGHT_BLACK))
    for ok, message in run_automated_checks(item):
        mark = _c("  PASS", typer.colors.GREEN) if ok else _c("  FAIL", typer.colors.RED)
        typer.echo(f"{mark}  {message}")

    typer.echo(_c("\nQuestion", typer.colors.BRIGHT_BLACK))
    typer.echo(f"  {item.question}")

    if item.stratum.expects_refusal:
        typer.echo(_c("\nExpected", typer.colors.BRIGHT_BLACK))
        typer.echo(f"  a refusal. absent term: {item.absent_term!r}")
        if item.anchor_paper_id:
            typer.echo(f"  anchored on {item.anchor_paper_id}, shape: {item.absence_shape}")
    else:
        typer.echo(_c("\nGold answer", typer.colors.BRIGHT_BLACK))
        typer.echo(f"  {item.gold_answer or '(none recorded)'}")
        typer.echo(_c("\nGold chunks", typer.colors.BRIGHT_BLACK))
        for chunk_id in item.gold_chunk_ids:
            show_chunk(chunk_id)

    typer.echo(_c("\nProvenance", typer.colors.BRIGHT_BLACK))
    typer.echo(
        f"  {item.provenance.generator_model} "
        f"{item.provenance.generator_snapshot} prompt={item.provenance.prompt_version}"
    )
    typer.echo(f"  papers: {', '.join(item.provenance.source_paper_ids) or '-'}")


def prompt_decision() -> str:
    typer.echo("")
    answer: str = typer.prompt(
        typer.style("[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit", fg=typer.colors.YELLOW),
        default=ACCEPT,
    )
    return answer.strip().lower()[:1]


def edit(item: EvalItem) -> EvalItem:
    """Edit the question or gold answer in place. The stratum is never editable.

    Changing an item's stratum would silently move it between reporting groups — and the two
    unanswerable sub-strata must stay separate (schema docstring). Reject it and draft a new
    one instead.
    """
    question = typer.prompt("question", default=item.question)
    updated = item.model_copy(update={"question": question})
    if not item.stratum.expects_refusal:
        updated = updated.model_copy(
            update={"gold_answer": typer.prompt("gold answer", default=item.gold_answer)}
        )
    return updated


@app.command()
def main(
    path: Annotated[Path, typer.Argument(help="Draft eval set to verify.")],
    stratum: Annotated[str | None, typer.Option(help="Only verify one stratum.")] = None,
    limit: Annotated[int, typer.Option(help="Stop after this many decisions.")] = 25,
    report: Annotated[bool, typer.Option("--report", help="Print status and exit.")] = False,
    out: Annotated[Path | None, typer.Option(help="Write here instead of in place.")] = None,
) -> None:
    evalset = EvalSet.read(path)
    target = out or path

    if report:
        verified = [i for i in evalset.items if i.verification.human_verified]
        typer.echo(f"{path}: {len(evalset.items)} items, {len(verified)} human-verified")
        for name, count in evalset.counts().items():
            done = sum(
                1
                for i in evalset.items
                if i.stratum.value == name and i.verification.human_verified
            )
            typer.echo(f"  {name:28} {count:3d} drafted  {done:3d} verified")
        shapes = evalset.absence_shape_counts()
        if shapes:
            typer.echo(f"  absence shapes: {shapes}")
        raise typer.Exit(0)

    queue = [
        i
        for i in evalset.items
        if not i.verification.human_verified and (stratum is None or i.stratum.value == stratum)
    ]
    if not queue:
        typer.echo("Nothing left to verify.")
        raise typer.Exit(0)

    by_id = {i.item_id: n for n, i in enumerate(evalset.items)}
    decisions = {"accepted": 0, "rejected": 0, "edited": 0}

    for n, item in enumerate(queue[:limit], start=1):
        render(item, n, min(len(queue), limit))
        choice = prompt_decision()

        if choice == QUIT:
            break
        if choice == SKIP:
            continue

        if choice == EDIT:
            item = edit(item)
            decisions["edited"] += 1
            choice = ACCEPT

        if choice == REJECT:
            notes = typer.prompt("why rejected", default="")
            item = item.model_copy(
                update={
                    "verification": item.verification.model_copy(
                        update={"human_verified": False, "human_notes": f"REJECTED: {notes}"}
                    )
                }
            )
            decisions["rejected"] += 1
        else:
            notes = typer.prompt("notes (optional)", default="")
            item = item.model_copy(
                update={
                    "verification": item.verification.model_copy(
                        update={"human_verified": True, "human_notes": notes}
                    )
                }
            )
            decisions["accepted"] += 1

        evalset.items[by_id[item.item_id]] = item
        # Written after every decision: a crash at item 20 must not cost the first 19.
        evalset.write(target)

    typer.echo(
        f"\n{decisions['accepted']} accepted ({decisions['edited']} after editing), "
        f"{decisions['rejected']} rejected. Written to {target}."
    )
    rejected = [i for i in evalset.items if i.verification.human_notes.startswith("REJECTED")]
    if rejected:
        typer.echo(
            _c(
                f"{len(rejected)} rejected items remain in the file. They are the "
                f"generator-vs-human disagreement rate and should be reported, not deleted.",
                typer.colors.YELLOW,
            )
        )


if __name__ == "__main__":
    sys.exit(app())
