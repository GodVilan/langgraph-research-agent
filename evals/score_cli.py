"""Human scoring of agent answers, one at a time, under the rubric — and nothing else.

Backs `make score`. The same disclosure discipline as `evals/verify_cli.py`: the human decides
before anything automated is shown. Here that discipline has a second layer, because the
rubric itself orders its questions — Q1 (what did the agent do) is decided from the question
and the agent's answer *alone*, and only then are the stratum, the gold answer, and the chunk
texts revealed for Q2 and Q3. The outcome label is computed by ``rubric.outcome_for`` from the
scorer's three answers; the scorer never picks a label from a menu.

What is revealed after the score: the run's own flags (`refused`, guardrail events,
truncation). They are metadata about the run, not judgements, and they are shown last so they
cannot anchor one.

Usage:
    python -m evals.score_cli                     # the 25-item judge sample, human scorer
    python -m evals.score_cli --all               # every item in the run
    python -m evals.score_cli --report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from evals.rubric import Behaviour, Outcome, expected_behaviour, outcome_for
from evals.run_set import RunRecord, run_path
from evals.schema import EvalSet, Stratum
from evals.scoring import Score, ScoreSheet, before_gold, rubric_sha256, with_gold
from evals.verify_cli import _c, _drain_pending_input, prompt_notes

app = typer.Typer(add_completion=False, help=__doc__)

BEHAVIOURS = {"a": Behaviour.ANSWER, "r": Behaviour.REFUSE, "c": Behaviour.CLARIFY}


def prompt_choice(label: str, choices: dict[str, str]) -> str:
    """One key from ``choices``; no default; echoed back before proceeding."""
    _drain_pending_input()
    while True:
        typer.echo("")
        raw = str(
            typer.prompt(typer.style(label, fg=typer.colors.YELLOW), default="", show_default=False)
        )
        key = raw.strip().lower()
        if key in choices:
            typer.echo(_c(f"  -> recorded as {choices[key]}", typer.colors.CYAN))
            return key
        typer.echo(_c(f"  {raw.strip()!r} is not a choice. Type one of {sorted(choices)}.", "red"))


def yes_no(label: str) -> bool:
    return prompt_choice(label, {"y": "YES", "n": "NO"}) == "y"


def scores_path(set_sha: str, scorer: str) -> Path:
    return Path("evals/runs") / f"scores_{scorer}_{set_sha[:8]}.json"


@app.command()
def main(
    all_items: Annotated[bool, typer.Option("--all")] = False,
    report: Annotated[bool, typer.Option("--report")] = False,
    scorer: str = "human",
    set_path: Path = Path("evals/datasets/phase4.json"),
    sample_path: Path = Path("evals/datasets/judge_sample.json"),
) -> None:
    evalset = EvalSet.read(set_path)
    run_file = run_path(evalset.sha256)
    record = RunRecord.model_validate_json(run_file.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    if sample["source_sha256"] != evalset.sha256:
        raise typer.BadParameter("the judge sample was drawn from a different set sha")

    out = scores_path(evalset.sha256, scorer)
    sheet = (
        ScoreSheet.load(out)
        if out.exists()
        else ScoreSheet(
            scorer=scorer,
            run_path=str(run_file),
            set_sha256=evalset.sha256,
            rubric_sha256=rubric_sha256(),
            sample_path=str(sample_path),
        )
    )
    if sheet.rubric_sha256 != rubric_sha256():
        typer.echo(_c("The rubric has changed since these scores were started. Stop.", "red"))
        raise typer.Exit(2)

    ids = list(record.items) if all_items else list(sample["item_ids"])
    if report:
        _report(sheet, ids)
        return

    todo = [i for i in ids if i not in sheet.scores]
    typer.echo(f"{len(todo)} to score, {len(sheet.scores)} done — scores go to {out}")
    for n, item_id in enumerate(todo, start=1):
        run = record.items[item_id]
        typer.echo("\n" + "=" * 78)
        typer.echo(_c(f"[{n}/{len(todo)}]", typer.colors.BRIGHT_BLACK))
        typer.echo(before_gold(run))
        key = prompt_choice(
            "Q1 — the agent [a]nswered / [r]efused / [c]larified? (q quits)",
            {
                **{k: v.value for k, v in BEHAVIOURS.items()},
                "q": "QUIT",
            },
        )
        if key == "q":
            break
        behaviour = BEHAVIOURS[key]

        typer.echo("\n" + _c("— gold and retrieved chunks —", typer.colors.BRIGHT_BLACK))
        typer.echo(with_gold(run))

        fact = grounded = None
        stratum = Stratum(run.stratum)
        if behaviour is Behaviour.ANSWER and expected_behaviour(stratum) is Behaviour.ANSWER:
            fact = yes_no("Q3a — does the asserted fact match the gold answer? [y/n]")
            grounded = (
                yes_no("Q3b — is that fact stated in a RETRIEVED chunk above? [y/n]")
                if fact
                else False
            )
        outcome = outcome_for(stratum, behaviour, fact, grounded)
        typer.echo(_c(f"  outcome: {outcome.value}", typer.colors.GREEN))
        reason = prompt_notes("reason")

        sheet.scores[item_id] = Score(
            item_id=item_id,
            scorer=scorer,
            behaviour=behaviour,
            fact_matches=fact,
            grounded=grounded,
            outcome=outcome,
            reason=reason,
        )
        sheet.save(out)

        # Revealed last: the run's own metadata. Not a judgement, and never shown before one.
        flags = []
        if run.guardrail_blocked:
            flags.append(f"guardrail block: {run.guardrail_reason}")
        if run.truncated:
            flags.append(f"truncated: {run.truncation_reason}")
        blocks = [e for e in run.guardrail_events if e.get("severity") == "block"]
        if blocks:
            flags.append(f"{len(blocks)} guardrail block(s)")
        typer.echo(_c("  run flags: " + ("; ".join(flags) or "none"), typer.colors.BRIGHT_BLACK))

    _report(sheet, ids)


def _report(sheet: ScoreSheet, ids: list[str]) -> None:
    done = [sheet.scores[i] for i in ids if i in sheet.scores]
    typer.echo(f"\n{len(done)} of {len(ids)} scored by {sheet.scorer}")
    counts: dict[str, int] = {}
    for s in done:
        counts[s.outcome.value] = counts.get(s.outcome.value, 0) + 1
    for outcome in Outcome:
        if outcome.value in counts:
            typer.echo(f"  {outcome.value:24} {counts[outcome.value]:3d}")


if __name__ == "__main__":
    app()
