"""Human verification of eval items, one at a time.

Backs `make verify-evals`. Srikanth verifies 25 items by hand; this is what he sees.

Design constraints that are not negotiable:

* **The human decides first; automated verdicts are revealed afterwards.** This is the
  important one. Showing "necessity: PASS" before the decision makes the label dependent on
  the check, and the resulting agreement rate then measures agreement-with-the-machine
  rather than generator-vs-human disagreement. Multi-hop and unanswerable-attribute have no
  validation *other* than that number, so it has to be arrived at independently.
* **Rejecting is as cheap as accepting.** A verification tool that makes rejection
  effortful produces a verified set that means nothing.
* **Progress is saved after every decision.** A crash at item 20 of 25 must not cost the
  first 19.
* **The generator's intent is recorded separately from the human label**, so disagreement
  between them can be reported instead of silently resolved in the generator's favour.

Usage:
    python -m evals.verify_cli evals/datasets/phase4.json
    python -m evals.verify_cli evals/datasets/phase4.json --stratum multi_hop
    python -m evals.verify_cli evals/datasets/phase4.json --report
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Annotated

import typer

from evals.absence import load_corpus, verify_attribute_absent, verify_topic_absent
from evals.schema import EvalItem, EvalSet, MachineCheck, Stratum

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


def run_automated_checks(item: EvalItem) -> list[tuple[str, bool, str]]:
    """Every automated verdict, as (name, passed, detail).

    Named rather than positional so a disagreement is attributable to a specific check
    instead of to "the pipeline". These are computed only after the human has ruled.
    """
    results: list[tuple[str, bool, str]] = []

    if item.stratum is Stratum.UNANSWERABLE_TOPIC:
        outcome = verify_topic_absent(item.absent_term)
        results.append(("topic_absence", outcome.absent, outcome.reason))
    elif item.stratum is Stratum.UNANSWERABLE_ATTRIBUTE:
        outcome = verify_attribute_absent(item.anchor_paper_id, item.absent_term)
        results.append(("attribute_absence", outcome.absent, outcome.reason))
    else:
        corpus_ids = {str(c["chunk_id"]) for c in load_corpus().chunks}
        missing = [g for g in item.gold_chunk_ids if g not in corpus_ids]
        detail = f"missing from the corpus: {missing}" if missing else "all present"
        results.append(("gold_chunks_exist", not missing, detail))

    if item.stratum is Stratum.MULTI_HOP:
        checked = item.verification.single_paper_sufficiency_checked
        alone = item.verification.answerable_by_one_paper
        results.append(
            (
                "multi_hop_necessity",
                checked and not alone,
                "NOT CHECKED — run the drafter's check first"
                if not checked
                else "one paper answers it alone; this is not multi-hop"
                if alone
                else "no single paper answers it, and the papers together do",
            )
        )
    return results


def render(item: EvalItem, index: int, total: int) -> None:
    """Everything the human needs to rule, and **nothing the machine concluded**.

    The automated verdicts are deliberately withheld until after the decision is taken. A
    verifier shown "necessity: PASS" before the question is anchored by it, and the
    resulting agreement rate then measures agreement-with-the-checker rather than the
    generator-vs-human disagreement the multi-hop and attribute strata rest on. Those two
    strata have no other validation, so that number has to be independent to be worth
    anything.
    """
    typer.echo("\n" + "=" * 78)
    typer.echo(
        f"Item {index}/{total}   {_c(item.stratum.value, typer.colors.CYAN)}   {item.item_id}"
    )
    typer.echo("=" * 78)

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


def reveal(item: EvalItem) -> None:
    """Show the automated verdicts *after* the human has ruled, and name any disagreement."""
    checks = item.verification
    typer.echo(
        _c(
            "\n  ---- automated verdicts (revealed after your decision) ----",
            typer.colors.BRIGHT_BLACK,
        )
    )
    for check in checks.machine_checks:
        mark = _c("PASS", typer.colors.GREEN) if check.passed else _c("FAIL", typer.colors.RED)
        typer.echo(f"  {mark}  {check.name}: {check.detail}")

    if checks.agrees is True:
        typer.echo(_c("  agreement: human and pipeline concur", typer.colors.GREEN))
    elif checks.agrees is False:
        named = ", ".join(c.name for c in checks.disagreeing_checks()) or "-"
        verdict = (
            "you accepted, the pipeline rejected"
            if checks.human_accepted
            else "you rejected, the pipeline accepted"
        )
        typer.echo(_c(f"  DISAGREEMENT: {verdict} (checks: {named})", typer.colors.YELLOW))
        typer.echo(
            _c(
                "  recorded, not resolved. This is the number the stratum's validity rests on.",
                typer.colors.BRIGHT_BLACK,
            )
        )


VALID_DECISIONS = {ACCEPT: "ACCEPT", EDIT: "EDIT", REJECT: "REJECT", SKIP: "SKIP", QUIT: "QUIT"}
NOTES_TERMINATOR = "."


def _drain_pending_input() -> None:
    """Discard anything already sitting in the terminal buffer.

    This is the fix for the defect that corrupted a whole verification session. Notes pasted
    as several lines were consumed one line per subsequent prompt, so every answer after the
    paste landed one slot early: decisions were stored as notes and notes were read as
    decisions. A session recorded 22 accepts and 2 rejects against an intent of roughly 17
    and 7 — turning a 28% disagreement rate into 8%, on the single number the exercise
    exists to produce. Anything left in the buffer when a decision is requested was not
    typed in answer to that question, so it is dropped rather than interpreted.
    """
    if not sys.stdin.isatty():
        return
    with contextlib.suppress(Exception):
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def prompt_decision() -> str:
    """Read one decision. No default, strict validation, echoed back before proceeding.

    A default meant an empty line — or a stray line from a paste — silently became "accept".
    Nothing here advances until the verifier has typed one of the five keys.
    """
    _drain_pending_input()
    while True:
        typer.echo("")
        raw: str = typer.prompt(
            typer.style("[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit", fg=typer.colors.YELLOW),
            default="",
            show_default=False,
        )
        answer = str(raw)
        choice = answer.strip().lower()
        if choice in VALID_DECISIONS:
            typer.echo(_c(f"  -> recorded as {VALID_DECISIONS[choice]}", typer.colors.CYAN))
            return choice
        typer.echo(
            _c(
                f"  {answer.strip()!r} is not a decision. Type exactly one of a, e, r, s, q.",
                typer.colors.RED,
            )
        )


def prompt_notes(label: str) -> str:
    """Read notes that may span several lines, terminated explicitly.

    Multi-line input is the thing that desynchronised the session, so it is handled rather
    than forbidden: lines are collected until a lone "." so that no line can escape into the
    next prompt, and pasted text ending without the terminator cannot silently consume the
    following decision.
    """
    typer.echo(_c(f"  {label} (end with a single '.' on its own line):", typer.colors.YELLOW))
    lines: list[str] = []
    while True:
        try:
            line = input("  | ")
        except EOFError:
            break
        if line.strip() == NOTES_TERMINATOR:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def edit(item: EvalItem) -> EvalItem:
    """Edit the question, gold answer, and — for refusal items — the absent term.

    The stratum is never editable: changing it would move the item between reporting groups,
    and the two unanswerable sub-strata must stay separate.

    ``absent_term`` *is* editable, and must be. A question rewritten past the term it was
    built around leaves the absence check certifying something the question no longer asks,
    which is how an item ended up carrying `absent_term="warmup"` after being rewritten into
    a question about confidence thresholds.
    """
    typer.echo(_c("  editing — the stratum cannot change; reject and redraft instead", "yellow"))
    question = typer.prompt("  question", default=item.question)
    updated = item.model_copy(update={"question": question})

    if item.stratum.expects_refusal:
        if question.strip() != item.question.strip():
            typer.echo(
                _c(
                    f"  the question changed; confirm the term whose absence it tests "
                    f"(was {item.absent_term!r})",
                    typer.colors.YELLOW,
                )
            )
        updated = updated.model_copy(
            update={"absent_term": typer.prompt("  absent term", default=item.absent_term)}
        )
    else:
        updated = updated.model_copy(
            update={"gold_answer": typer.prompt("  gold answer", default=item.gold_answer)}
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
        anchors = evalset.anchor_class_counts()
        if anchors:
            typer.echo(f"  anchor classes: {anchors}")

        # Per stratum, never pooled: multi-hop and unanswerable-attribute rest entirely on
        # automated checks, so their agreement rate is evidence about those checks.
        # Averaging in single-paper factual, which is close to free, would dilute it.
        agreement = evalset.agreement_by_stratum()
        if any(row["ruled"] for row in agreement.values()):
            typer.echo("\n  human-vs-pipeline agreement (independent: human ruled first)")
            for name, row in sorted(agreement.items()):
                if not row["ruled"]:
                    continue
                rate = row["agree"] / row["ruled"]
                typer.echo(
                    f"    {name:28} {row['agree']:2d}/{row['ruled']:2d} agree "
                    f"({rate:.0%}), {row['disagree']} disagree"
                )
        for item_id, stratum, names in evalset.disagreements():
            typer.echo(f"    disagreement {item_id} ({stratum}) -> {', '.join(names) or '-'}")
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
            choice = prompt_decision()
            if choice in {QUIT, SKIP}:
                continue

        accepted = choice != REJECT
        notes = prompt_notes("notes (optional)" if accepted else "why rejected")
        decisions["accepted" if accepted else "rejected"] += 1

        # Re-run against the *edited* item, not the drafted one. An item rewritten from a
        # warmup question into a confidence-threshold question kept `absent_term="warmup"`
        # and was certified against a term the new question never asks about — the checks
        # had validated a question that no longer existed.
        checks = [
            MachineCheck(name=name, passed=passed, detail=detail)
            for name, passed, detail in run_automated_checks(item)
        ]
        item = item.model_copy(
            update={
                "verification": item.verification.model_copy(
                    update={
                        "human_accepted": accepted,
                        "human_verified": accepted,
                        "human_notes": notes if accepted else f"REJECTED: {notes}",
                        "machine_checks": checks,
                    }
                )
            }
        )
        reveal(item)

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
