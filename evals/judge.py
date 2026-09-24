"""The judge arms: one prompt, three models, agreement against the human sheet.

Backs `make judge-*`. The judge receives **exactly what the human scorer saw, in the
order the human saw it**, under **exactly the rubric the human applied**. Two calls per item:

    stage q1   system = the rubric's Q1 section only;  user = question + agent answer.
               No stratum, no gold, no chunks — asserted structurally by test. Q2's table
               maps expected x observed straight onto a label, so a judge that sees the
               stratum first can read `hallucinated_refusal` off a lookup without judging
               what the agent did. The human decided Q1 blind; so does the judge.
    stage q3   only when the judge said `answer` and the item expects one: system = the
               full rubric; user = the complete scorer's view (``scoring.view()``, full
               gold and retrieved chunk text) with the judge's own Q1 verdict restated.

Every other case is settled by Q1 and the stratum, exactly as the scoring CLI settled it,
and costs one call. The judge never picks a label; ``rubric.outcome_for`` computes it from
its answers, the same function that computed the human's. A disagreement is therefore a
difference of judgement, never of information, ordering, or label vocabulary.

Three arms, decided as an experiment rather than assumed:

    oss120b       openai/gpt-oss-120b via the Hugging Face inference router — $0
    luna-low      gpt-5.6-luna, reasoning_effort=low,    OpenAI Batch API
    luna-medium   gpt-5.6-luna, reasoning_effort=medium, OpenAI Batch API

Batch is mandatory for the OpenAI arms (standard is exactly 2x). Nothing is submitted without
``--i-confirmed-cap-and-alert``, which stands for a human having confirmed the $5 hard cap and
$2 alert on the dashboard; the flag is not a substitute for the confirmation, it records it.

Agreement is reported per arm, per stratum, and — the discriminating question — separately on
the items the human labelled `correct_refusal` and `hallucinated_refusal`. Pooled, a judge that
calls everything a refusal scores 17/17 on this sample. The only thing being measured is
whether it can tell a refusal that should have happened from one that should not, which is
the distinction the agent itself fails.

    python -m evals.judge print-one --arm luna-low --item sp-036 [--stage q3]
    python -m evals.judge estimate
    python -m evals.judge submit --arm luna-low --stage q1 --i-confirmed-cap-and-alert
    python -m evals.judge collect --arm luna-low --stage q1     # then --stage q3 for both
    python -m evals.judge run --arm oss120b                     # synchronous, free router
    python -m evals.judge agreement
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from pydantic import BaseModel, Field, ValidationError

from evals.rubric import (
    AMENDMENTS,
    RUBRIC_VERSION,
    Behaviour,
    Outcome,
    expected_behaviour,
    outcome_for,
    render,
    render_q1,
)
from evals.run_set import ItemRun, RunRecord, load_record, run_path
from evals.schema import EvalSet, Stratum
from evals.scoring import Score, ScoreSheet, before_gold, rubric_sha256, view

app = typer.Typer(add_completion=False, help=__doc__)


@app.callback()
def scope(
    all_items: Annotated[
        bool, typer.Option("--all", help="judge a whole run, not the sample")
    ] = False,
    tag: Annotated[str, typer.Option(help="repeat-run tag (r2, r3 …)")] = "",
) -> None:
    SCOPE["all"] = all_items
    SCOPE["tag"] = tag


RUNS = Path("evals/runs")


@dataclass(frozen=True)
class Arm:
    id: str
    model: str
    provider: str  # "openai_batch" | "openai_compatible"
    base_url: str
    reasoning_effort: str | None
    price_in_per_m: float  # what this arm is actually billed at
    price_out_per_m: float
    assumed_output_tokens: int  # visible + reasoning, for the estimate only
    key_env: str = "OPENAI_API_KEY"
    billed_to: str = "the $5 OpenAI ceiling"
    # What agreement with this arm would establish, and what it would not. Stated on the arm
    # because a judge experiment is only as meaningful as the question each arm answers.
    establishes: str = ""
    proposed: bool = False  # costed and listed, not yet chosen


ARMS: dict[str, Arm] = {
    "oss120b": Arm(
        id="oss120b",
        model="openai/gpt-oss-120b",
        provider="openai_compatible",
        base_url="https://router.huggingface.co/v1",
        reasoning_effort=None,
        price_in_per_m=0.35,  # cerebras, the provider the router chose by default (2026-09-18)
        price_out_per_m=0.75,
        assumed_output_tokens=600,
        key_env="HF_TOKEN",
        billed_to="Hugging Face inference credit — outside the $5 OpenAI ceiling",
        establishes=(
            "independence from the generator's vendor: open weights from a third lab. "
            "Weights revision unpinnable (D-030b)."
        ),
    ),
    # A third arm on a key already held: open weights from the generator's own vendor. Tests
    # independence from the generator *model* but not from the vendor — a different, weaker
    # question than oss120b's. Free tier; the Gemini OpenAI-compatible endpoint.
    "gemma-4-31b": Arm(
        id="gemma-4-31b",
        model="gemma-4-31b-it",
        provider="openai_compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        reasoning_effort=None,
        price_in_per_m=0.0,
        price_out_per_m=0.0,
        assumed_output_tokens=300,
        key_env="GOOGLE_API_KEY",
        billed_to="Google AI free tier — $0, rate-limited",
        establishes=(
            "independence from the generator model (Gemma is not Gemini) but NOT from its "
            "vendor; open weights, revision not exposed by the API either."
        ),
        proposed=True,
    ),
    # The self-judging control: the generator's own model as judge. Establishes the size of
    # same-model preference, which is a different thing from independence and worth knowing.
    "flash-lite-self": Arm(
        id="flash-lite-self",
        model="gemini-3.5-flash-lite",
        provider="openai_compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        reasoning_effort=None,
        price_in_per_m=0.0,
        price_out_per_m=0.0,
        assumed_output_tokens=400,
        key_env="GOOGLE_API_KEY",
        billed_to="Google AI free tier — $0, rate-limited (15 RPM)",
        establishes=(
            "the self-preference control: the generator judging its own answers. Measures "
            "same-model bias; establishes nothing about independence."
        ),
        proposed=True,
    ),
    # Batch rates: exactly half of the $0.20 / $1.20 standard card in docs/BUDGET.md.
    "luna-low": Arm(
        id="luna-low",
        model="gpt-5.6-luna",
        provider="openai_batch",
        base_url="https://api.openai.com/v1",
        reasoning_effort="low",
        price_in_per_m=0.10,
        price_out_per_m=0.60,
        assumed_output_tokens=300 + 300,
        establishes=(
            "a closed reasoning judge from a third vendor at low effort; with luna-medium it "
            "tests reasoning effort, not independence. No dated snapshot alias."
        ),
    ),
    "luna-medium": Arm(
        id="luna-medium",
        model="gpt-5.6-luna",
        provider="openai_batch",
        base_url="https://api.openai.com/v1",
        reasoning_effort="medium",
        price_in_per_m=0.10,
        price_out_per_m=0.60,
        assumed_output_tokens=300 + 800,
        establishes="the same judge at medium effort; the reasoning-effort comparison only.",
    ),
}


class Q1Verdict(BaseModel):
    """Stage one: what the agent did. Decided from the question and the answer alone."""

    behaviour: Behaviour = Field(description="Q1: what the agent did")
    reason: str = Field(default="", description="One sentence")


class Q3Verdict(BaseModel):
    """Stage two, only for answer -> answer: fact and grounding."""

    fact_matches: bool = Field(description="Q3a: does the asserted fact match the gold answer")
    grounded: bool = Field(description="Q3b: is that fact stated in a RETRIEVED chunk")
    reason: str = Field(default="", description="One or two sentences")


class JudgeVerdict(BaseModel):
    """The rubric's three answers, assembled across the two stages. Never a label."""

    behaviour: Behaviour
    fact_matches: bool | None = None
    grounded: bool | None = None
    reason: str = ""


Q1_INSTRUCTIONS = """\
Apply the rubric above. You are shown a question and an agent's answer, nothing else. Decide
ONLY what the agent did. Answer as JSON:

{"behaviour": "answer" | "refuse" | "clarify", "reason": "<one sentence>"}

Output the JSON object only.
"""

Q3_INSTRUCTIONS = """\
Apply the rubric above to the item below. In an earlier step you decided the agent's
behaviour was: **{behaviour}**. Do not revisit that. Answer ONLY Q3, as JSON:

{{"fact_matches": true | false, "grounded": true | false, "reason": "<one or two sentences>"}}

`grounded` is judged against the RETRIEVED chunks shown, not the gold chunks. Do not name an
outcome label; it is computed from your answers. Output the JSON object only.
"""


def q1_messages(run: ItemRun) -> list[dict[str, str]]:
    """Stage one: the Q1 rubric section and the pre-gold view. Nothing that reveals Q2."""
    return [
        {"role": "system", "content": render_q1()},
        {"role": "user", "content": Q1_INSTRUCTIONS + "\n" + before_gold(run)},
    ]


def q3_messages(run: ItemRun, behaviour: Behaviour) -> list[dict[str, str]]:
    """Stage two: the full rubric and the complete scorer's view, verbatim."""
    return [
        {"role": "system", "content": render()},
        {
            "role": "user",
            "content": Q3_INSTRUCTIONS.format(behaviour=behaviour.value) + "\n" + view(run),
        },
    ]


def needs_q3(run: ItemRun, behaviour: Behaviour) -> bool:
    """Q3 is asked only where the scoring CLI asked it: answer -> answer."""
    return (
        behaviour is Behaviour.ANSWER
        and expected_behaviour(Stratum(run.stratum)) is Behaviour.ANSWER
    )


def request_body(
    arm: Arm, run: ItemRun, stage: str = "q1", behaviour: Behaviour | None = None
) -> dict[str, Any]:
    if stage == "q1":
        messages = q1_messages(run)
    elif behaviour is not None:
        messages = q3_messages(run, behaviour)
    else:
        raise ValueError("stage q3 needs the stage-q1 behaviour")
    body: dict[str, Any] = {
        "model": arm.model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if arm.reasoning_effort:
        body["reasoning_effort"] = arm.reasoning_effort
    return body


def approx_tokens(text: str) -> int:
    """Characters / 4. Stated as approximate wherever it is printed."""
    return len(text) // 4


SCOPE = {"tag": "", "all": False}  # set by the CLI; the sample is the default scope


def sample_runs() -> tuple[EvalSet, RunRecord, list[ItemRun], dict[str, Any]]:
    """The items to judge: the 25-item validation sample, or a whole run (`--scope all`)."""
    evalset = EvalSet.read(Path("evals/datasets/phase4.json"))
    record = load_record(run_path(evalset.sha256, str(SCOPE["tag"])))
    if record is None:
        raise typer.BadParameter("no run recorded; run `make run-set` first")
    sample = json.loads(Path("evals/datasets/judge_sample.json").read_text(encoding="utf-8"))
    if sample["source_sha256"] != evalset.sha256:
        raise typer.BadParameter("judge sample was drawn from a different set sha")
    if SCOPE["all"]:
        runs = list(record.items.values())
        sample = {**sample, "item_ids": list(record.items), "scope": "all"}
    else:
        runs = [record.items[i] for i in sample["item_ids"]]
    return evalset, record, runs, sample


def _suffix() -> str:
    parts = []
    if SCOPE["all"]:
        parts.append("all")
    if SCOPE["tag"]:
        parts.append(str(SCOPE["tag"]))
    return ("_" + "_".join(parts)) if parts else ""


def estimate_arm(arm: Arm, runs: list[ItemRun]) -> dict[str, float]:
    """Stage q1 for every item; stage q3 assumed for every *answerable* item.

    The q3 count is an upper bound — q3 runs only where the judge says `answer` on an item
    that expects one — so the estimate is conservative in the direction of over-stating.
    """
    q1_in = sum(approx_tokens(json.dumps(q1_messages(r))) for r in runs)
    answerable = [r for r in runs if expected_behaviour(Stratum(r.stratum)) is Behaviour.ANSWER]
    q3_in = sum(approx_tokens(json.dumps(q3_messages(r, Behaviour.ANSWER))) for r in answerable)
    tokens_out = arm.assumed_output_tokens * (len(runs) + len(answerable))
    tokens_in = q1_in + q3_in
    cost = tokens_in / 1e6 * arm.price_in_per_m + tokens_out / 1e6 * arm.price_out_per_m
    return {
        "requests": len(runs) + len(answerable),
        "q1_requests": len(runs),
        "q3_requests_max": len(answerable),
        "approx_input_tokens": tokens_in,
        "assumed_output_tokens": tokens_out,
        "estimated_usd": round(cost, 4),
    }


def sheet_path(arm: Arm, set_sha: str) -> Path:
    return RUNS / f"scores_{arm.id}_{set_sha[:8]}{_suffix()}.json"


def stage_path(arm: Arm, stage: str) -> Path:
    return RUNS / f"judge_{arm.id}_{stage}{_suffix()}.json"


def load_stage(arm: Arm, stage: str) -> dict[str, Any]:
    path = stage_path(arm, stage)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_stage(arm: Arm, stage: str, data: dict[str, Any]) -> None:
    stage_path(arm, stage).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def assert_served_model(arm: Arm, response: dict[str, Any]) -> None:
    """The response must name the model that was asked for. A silent substitution is the
    exact failure the availability assertion exists to prevent, and a router that falls back
    to another model on quota exhaustion would otherwise pass unnoticed."""
    served = str(response.get("model", ""))
    # The HF router drops the organisation prefix: a request for `openai/gpt-oss-120b` is
    # answered as `gpt-oss-120b`. Compare the model name itself; any other name is rejected.
    if served.rsplit("/", 1)[-1].lower() != arm.model.rsplit("/", 1)[-1].lower():
        raise typer.BadParameter(f"{arm.id}: asked for {arm.model}, response says {served!r}")


def to_score(arm: Arm, run: ItemRun, verdict: JudgeVerdict) -> Score:
    stratum = Stratum(run.stratum)
    fact, grounded = verdict.fact_matches, verdict.grounded
    if (
        verdict.behaviour is Behaviour.ANSWER
        and not stratum.expects_refusal
        and stratum is not Stratum.AMBIGUOUS
    ):
        # The judge may omit Q3 fields; treat an omitted answer as "no", never as "yes".
        fact = bool(fact)
        grounded = bool(grounded) if fact else False
    else:
        fact = grounded = None
    return Score(
        item_id=run.item_id,
        scorer=arm.id,
        behaviour=verdict.behaviour,
        fact_matches=fact,
        grounded=grounded,
        outcome=outcome_for(stratum, verdict.behaviour, fact, grounded),
        reason=verdict.reason,
    )


def _env(name: str) -> str:
    """The environment, then `.env` — the same file `src.config` reads, without importing it."""
    if os.environ.get(name):
        return os.environ[name]
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def _headers(arm: Arm) -> dict[str, str]:
    key = _env(arm.key_env)
    if not key:
        raise typer.BadParameter(f"no API key in the environment for arm {arm.id}")
    return {"Authorization": f"Bearer {key}"}


def assert_model_available(arm: Arm) -> str:
    """Fail loudly; never silently substitute a judge. Returns the provider's model id."""
    with httpx.Client(timeout=30, headers=_headers(arm)) as client:
        r = client.get(f"{arm.base_url}/models/{arm.model}")
        if r.status_code == 200:
            return str(r.json().get("id", arm.model))
        # Some OpenAI-compatible routers serve only the list endpoint.
        listing = client.get(f"{arm.base_url}/models")
    ids = (
        {str(m.get("id")) for m in listing.json().get("data", [])}
        if listing.status_code == 200
        else set()
    )
    if arm.model in ids:
        return arm.model
    raise typer.BadParameter(
        f"{arm.model} is not available at {arm.base_url} ({r.status_code}): {r.text[:200]}"
    )


# ── commands ──────────────────────────────────────────────────────────────────────────────


@app.command("print-one")
def print_one(arm: str = "luna-low", item: str = "sp-036") -> None:
    """The exact request body for one item, verbatim."""
    _, record, _, _ = sample_runs()
    typer.echo(
        json.dumps(request_body(ARMS[arm], record.items[item]), indent=2, ensure_ascii=False)
    )


@app.command()
def estimate() -> None:
    """Request count and estimated cost per arm. Tokens are chars/4, so approximate."""
    _, _, runs, _ = sample_runs()
    total = 0.0
    for arm in ARMS.values():
        e = estimate_arm(arm, runs)
        if not arm.proposed:
            total += e["estimated_usd"]
        typer.echo(
            ("PROPOSED  " if arm.proposed else "          ") + f"{arm.id}: {arm.establishes}"
        )
        typer.echo(
            f"{arm.id:12} {arm.model:22} {arm.provider:18} requests {int(e['requests'])} "
            f"({int(e['q1_requests'])} q1 + up to {int(e['q3_requests_max'])} q3)  "
            f"~{int(e['approx_input_tokens']):,} in + {int(e['assumed_output_tokens']):,} out "
            f"(assumed)  -> ${e['estimated_usd']:.4f} at "
            f"${arm.price_in_per_m}/${arm.price_out_per_m} per 1M"
        )
    typer.echo(
        f"\nestimated total for the chosen arms ${total:.4f}; OpenAI-billed arms count against "
        f"the $5 lifetime ceiling ($2 alert), others are billed as named per arm"
    )
    if SCOPE["all"]:
        for line in recent_actuals():
            typer.echo(line)


def recent_actuals(arm_id: str = "luna-low", n: int = 3) -> list[str]:
    """The last ``n`` comparable full runs' actual cost, beside the conservative estimate.

    The estimate stays deliberately conservative (Phase 5 review: do not recalibrate it); this
    puts the measured figures next to it so the gap is visible at the moment of deciding.
    Comparable = a full-run (``--all``) judging of the shipping configuration by the same arm,
    q1 and q3 together; source: ``evals/runs/judge_spend.json`` (`make judge-spend`).
    """
    spend_file = RUNS / "judge_spend.json"
    if not spend_file.exists():
        return ["(no evals/runs/judge_spend.json: run `make judge-spend` for measured actuals)"]
    detail = json.loads(spend_file.read_text(encoding="utf-8")).get("batches_detail", [])
    arms = ("dense_only", "section_filter", "v21")
    runs: dict[str, float] = {}
    for row in detail:
        name = str(row["receipt"])
        prefix = f"batch_{arm_id}_"
        if not name.startswith(prefix) or "_all" not in name or row.get("usd") is None:
            continue
        if any(a in name for a in arms) or name.count(".") > 1:  # arms, archived receipts
            continue
        tag = name.removeprefix(prefix).split("_all", 1)[1].removesuffix(".json") or "_r1"
        runs[tag] = runs.get(tag, 0.0) + float(row["usd"])
    if not runs:
        return ["(no comparable full runs recorded)"]

    def submitted(tag: str) -> str:
        # Chronological, from the q1 receipt: sorting tags by name put "traced" after "r3"
        # and dropped the newest run from "the last three".
        suffix = "" if tag == "_r1" else tag
        receipt = RUNS / f"batch_{arm_id}_q1_all{suffix}.json"
        if receipt.exists():
            return str(json.loads(receipt.read_text(encoding="utf-8")).get("submitted_at") or "")
        return ""

    order = sorted(runs, key=lambda t: (submitted(t), t))[-n:]
    return [
        f"measured, last {len(order)} comparable full runs ({arm_id}, q1+q3): "
        + ", ".join(f"{t.strip('_')} ${runs[t]:.4f}" for t in order)
    ]


def _q3_targets(
    arm: Arm, record: RunRecord, runs: list[ItemRun]
) -> list[tuple[ItemRun, Behaviour]]:
    """Items whose stage-q1 verdict was `answer` on an answerable item."""
    q1 = load_stage(arm, "q1")
    if not q1:
        raise typer.BadParameter(f"{arm.id}: stage q1 has not been collected yet")
    out: list[tuple[ItemRun, Behaviour]] = []
    for run in runs:
        v = q1.get(run.item_id)
        if not v:
            continue
        behaviour = Behaviour(v["behaviour"])
        if needs_q3(run, behaviour):
            out.append((run, behaviour))
    return out


def _requests_for(
    arm: Arm, stage: str, record: RunRecord, runs: list[ItemRun], only: set[str] | None = None
) -> list[dict[str, Any]]:
    keep = [r for r in runs if not only or r.item_id in only]
    if stage == "q1":
        return [{"custom_id": r.item_id, "body": request_body(arm, r, "q1")} for r in keep]
    return [
        {"custom_id": r.item_id, "body": request_body(arm, r, "q3", b)}
        for r, b in _q3_targets(arm, record, keep)
    ]


def _new_sheet(arm: Arm, evalset: EvalSet) -> ScoreSheet:
    return ScoreSheet(
        scorer=arm.id,
        # The *scoped* run, not the default one. Every tagged sheet recorded
        # `v3_<sha>.json` regardless of which run it judged, so six of seven sheets carried a
        # provenance claim that was false — caught by `push_scores`' refusal to attach a
        # sheet's labels to a run it did not score, which is the check doing its job against a
        # defect it was not written for.
        run_path=str(run_path(evalset.sha256, str(SCOPE["tag"]))),
        set_sha256=evalset.sha256,
        rubric_sha256=rubric_sha256(),
        sample_path="evals/datasets/judge_sample.json",
        judge_weights_revision=(
            f"NOT PINNED: {arm.model} — no provider exposes a weights revision; served model "
            f"id asserted per response (D-030b)"
        ),
    )


def _assemble(arm: Arm, evalset: EvalSet, record: RunRecord, runs: list[ItemRun]) -> ScoreSheet:
    """Both stages into one score sheet; the label is computed here and nowhere else."""
    q1, q3 = load_stage(arm, "q1"), load_stage(arm, "q3")
    sheet = _new_sheet(arm, evalset)
    rescore = RUNS / f"rescore_{arm.id}.json"
    if rescore.exists():
        meta = json.loads(rescore.read_text(encoding="utf-8"))
        sheet.previous_rubric_sha256 = meta["previous_rubric_sha256"]
        sheet.rescored_items = list(meta["items"])
        sheet.rescore_note = meta.get("note", "")
    for run in runs:
        v1 = q1.get(run.item_id)
        if not v1:
            continue
        behaviour = Behaviour(v1["behaviour"])
        verdict = JudgeVerdict(behaviour=behaviour, reason=v1.get("reason", ""))
        if needs_q3(run, behaviour):
            v3 = q3.get(run.item_id)
            if not v3:
                continue  # q3 not collected yet for this item
            verdict = JudgeVerdict(
                behaviour=behaviour,
                fact_matches=bool(v3["fact_matches"]),
                grounded=bool(v3["grounded"]),
                reason=f"{v1.get('reason', '')} | {v3.get('reason', '')}",
            )
        sheet.scores[run.item_id] = to_score(arm, run, verdict)
    sheet.save(sheet_path(arm, evalset.sha256))
    return sheet


def _begin_rescore(arm: Arm, evalset: EvalSet, items: str) -> set[str]:
    """Archive the arm's current sheet and stage files before re-judging under an amended rubric.

    The "before" numbers come from the archived sheet; the live sheet carries the previous
    sha and the rescored ids so the "after" number can never be printed alone.
    """
    only = {i.strip() for i in items.split(",") if i.strip()}
    live = sheet_path(arm, evalset.sha256)
    if live.exists():
        prior = ScoreSheet.load(live)
        if prior.rubric_sha256 != rubric_sha256():
            tag = f"rubric-sha-{prior.rubric_sha256[:8]}"
            live.with_suffix(f".{tag}.json").write_text(
                live.read_text(encoding="utf-8"), encoding="utf-8"
            )
            for stage in ("q1", "q3"):
                sp = stage_path(arm, stage)
                if sp.exists():
                    sp.with_suffix(f".{tag}.json").write_text(
                        sp.read_text(encoding="utf-8"), encoding="utf-8"
                    )
            (RUNS / f"rescore_{arm.id}.json").write_text(
                json.dumps(
                    {
                        "previous_rubric_sha256": prior.rubric_sha256,
                        "items": sorted(only),
                        "note": f"rubric v{RUBRIC_VERSION}: {AMENDMENTS[RUBRIC_VERSION]}",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return only


def write_receipt(path: Path, info: dict[str, Any]) -> None:
    """Persist a submission receipt and prove it is readable, or raise.

    A missing or unreadable receipt means the batch cannot be collected later, so "submitted"
    would be a claim with nothing behind it. The `batch_id` is required: an info dict without
    one is a failed submit that got this far.
    """
    if not info.get("batch_id"):
        raise typer.BadParameter(f"submit produced no batch_id; nothing was queued: {info}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a receipt for a different batch: archive it beside the new one. An
    # overwrite hid three paid batches (117 requests) from `make judge-spend` until the
    # account's own batch list was checked against the receipts (Phase 5 review).
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        old_id = str(previous.get("batch_id", ""))
        if old_id and old_id != info["batch_id"]:
            archived = path.with_name(f"{path.stem}.{old_id[-12:]}.json")
            archived.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    written = json.loads(path.read_text(encoding="utf-8"))
    if written.get("batch_id") != info["batch_id"]:
        raise typer.BadParameter(
            f"submission receipt {path} did not persist; refusing to report success"
        )


def _parse_stage(stage: str, text: str) -> dict[str, Any]:
    model = Q1Verdict if stage == "q1" else Q3Verdict
    return model.model_validate_json(text).model_dump(mode="json")


@app.command()
def submit(
    arm: str = "luna-low",
    stage: str = "q1",
    items: str = "",
    i_confirmed_cap_and_alert: Annotated[bool, typer.Option("--i-confirmed-cap-and-alert")] = False,
) -> None:
    """Upload a Batch JSONL for one stage and create the batch. Refuses without the flag."""
    a = ARMS[arm]
    if a.provider != "openai_batch":
        raise typer.BadParameter(f"{arm} is not a batch arm; use `run`")
    if not i_confirmed_cap_and_alert:
        raise typer.BadParameter(
            "refusing to submit: pass --i-confirmed-cap-and-alert only after the $5 hard cap "
            "and $2 alert have been confirmed on the OpenAI dashboard"
        )
    evalset, record, runs, _ = sample_runs()
    served = assert_model_available(a)
    only = _begin_rescore(a, evalset, items) if items else None
    reqs = _requests_for(a, stage, record, runs, only)
    if not reqs:
        typer.echo(f"{arm} {stage}: nothing to submit")
        return
    jsonl = (
        "\n".join(
            json.dumps(
                {
                    "custom_id": r["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": r["body"],
                }
            )
            for r in reqs
        )
        + "\n"
    )
    with httpx.Client(timeout=120, headers=_headers(a)) as client:
        up = client.post(
            f"{a.base_url}/files",
            files={
                "file": (f"judge_{arm}_{stage}.jsonl", jsonl.encode("utf-8"), "application/jsonl")
            },
            data={"purpose": "batch"},
        )
        up.raise_for_status()
        file_id = up.json()["id"]
        batch = client.post(
            f"{a.base_url}/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {
                    "arm": arm,
                    "stage": stage,
                    "set_sha": evalset.sha256[:16],
                    "rubric": rubric_sha256()[:16],
                },
            },
        )
        batch.raise_for_status()
    info = {
        "arm": arm,
        "stage": stage,
        "model_served": served,
        "file_id": file_id,
        "batch_id": batch.json()["id"],
        "requests": len(reqs),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # The receipt is the proof the submit happened, so it is written and then *verified* before
    # anything reports success. Two submissions once 401'd against a deactivated account, wrote
    # no file, and left the driver log reading "SUBMITTED" for work that did not exist — a step
    # reporting success without having happened (D-026's class, not an outage's). A submit that
    # cannot produce a readable receipt raises here instead.
    receipt = RUNS / f"batch_{arm}_{stage}{_suffix()}.json"
    write_receipt(receipt, info)
    typer.echo(json.dumps(info, indent=2))


@app.command()
def collect(arm: str = "luna-low", stage: str = "q1") -> None:
    """Fetch a finished batch for one stage; assemble the score sheet when q3 is in."""
    a = ARMS[arm]
    evalset, record, runs, _ = sample_runs()
    info = json.loads((RUNS / f"batch_{arm}_{stage}{_suffix()}.json").read_text(encoding="utf-8"))
    with httpx.Client(timeout=120, headers=_headers(a)) as client:
        b = client.get(f"{a.base_url}/batches/{info['batch_id']}")
        b.raise_for_status()
        status = b.json()
        # An `expired` batch is not an empty one: the 24h completion window closed while some
        # requests were still queued, and whatever finished is in the output file. The
        # `dense_only` batch expired at 67 of 69 while the account was deactivated, and
        # refusing to read it would have thrown away 67 paid verdicts. Collect what exists,
        # say how much is missing, and let a resubmission cover the remainder.
        if status["status"] not in {"completed", "expired"} or not status.get("output_file_id"):
            typer.echo(
                f"batch {info['batch_id']} is {status['status']}: {status.get('request_counts')}"
            )
            raise typer.Exit(1)
        if status["status"] == "expired":
            counts = status.get("request_counts", {})
            typer.echo(
                f"batch {info['batch_id']} EXPIRED with "
                f"{counts.get('completed')} of {counts.get('total')} completed — collecting the "
                f"partial output; resubmit the remainder with `submit --items …`"
            )
        out = client.get(f"{a.base_url}/files/{status['output_file_id']}/content")
        out.raise_for_status()
    stage_data = load_stage(a, stage)
    usage_in = usage_out = 0
    failures: list[str] = []
    for line in out.text.splitlines():
        row = json.loads(line)
        item_id = row["custom_id"]
        resp = row.get("response", {}).get("body", {})
        try:
            assert_served_model(a, resp)
            stage_data[item_id] = _parse_stage(stage, resp["choices"][0]["message"]["content"])
        except (KeyError, IndexError, ValidationError, typer.BadParameter) as exc:
            failures.append(f"{item_id}: {exc}")
            continue
        usage_in += int(resp.get("usage", {}).get("prompt_tokens", 0))
        usage_out += int(resp.get("usage", {}).get("completion_tokens", 0))
    save_stage(a, stage, stage_data)
    billed = usage_in / 1e6 * a.price_in_per_m + usage_out / 1e6 * a.price_out_per_m
    typer.echo(
        f"{arm} {stage}: {len(stage_data)} verdicts, {len(failures)} unparseable; "
        f"tokens {usage_in:,} in / {usage_out:,} out; billed ~${billed:.4f} at batch rates"
    )
    for f in failures:
        typer.echo(f"  unparseable {f}")
    sheet = _assemble(a, evalset, record, runs)
    pending = [r.item_id for r in runs if r.item_id not in sheet.scores]
    typer.echo(f"assembled {len(sheet.scores)} scores; awaiting q3 for {pending or 'nothing'}")


@app.command()
def run(arm: str = "oss120b", items: str = "") -> None:
    """Synchronous two-stage calls for the free, non-batch arm."""
    a = ARMS[arm]
    if a.provider != "openai_compatible":
        raise typer.BadParameter(f"{arm} is a batch arm; use `submit`")
    evalset, record, runs, _ = sample_runs()
    served = assert_model_available(a)
    only = _begin_rescore(a, evalset, items) if items else None
    all_runs = runs
    if only:
        runs = [r for r in runs if r.item_id in only]
    failures: list[str] = []
    served_ids: set[str] = set()
    spend: dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "providers": set()}
    with httpx.Client(timeout=180, headers=_headers(a)) as client:

        def call(stage: str, body: dict[str, Any]) -> dict[str, Any] | None:
            resp = client.post(f"{a.base_url}/chat/completions", json=body)
            if resp.status_code != 200:
                failures.append(f"{stage} HTTP {resp.status_code} {resp.text[:160]}")
                return None
            data = resp.json()
            assert_served_model(a, data)
            # The router names the provider it routed to only in headers; keep both, since
            # price varies ~10x across the eleven providers and the weights revision is not
            # exposed anywhere (D-030b).
            provider = resp.headers.get("x-inference-provider", "")
            # `system_fingerprint` is the serving stack's fingerprint, not a weights revision;
            # recorded because it is the closest thing to a pin the response offers.
            served_ids.add(
                f"{data.get('model')}@{provider or 'provider-not-reported'}"
                f"#{data.get('system_fingerprint') or 'no-fingerprint'}"
            )
            usage = data.get("usage", {})
            spend["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            spend["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            # Every synchronous judge call — including the Gemini arms, which call Gemini's
            # OpenAI-compatible endpoint around the LangChain wrapper — lands in the usage log.
            from src.agent.llm import record_usage

            record_usage(
                str(data.get("model") or a.model),
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
                provider=a.base_url,
                activity=f"judge:{a.id}",
            )
            spend["providers"].add(provider or "provider-not-reported")
            return _parse_stage(stage, data["choices"][0]["message"]["content"])

        q1 = load_stage(a, "q1")
        for r in runs:
            if r.item_id in q1 and not only:
                continue
            v = call("q1", request_body(a, r, "q1"))
            if v is not None:
                q1[r.item_id] = v
                save_stage(a, "q1", q1)
        if not q1:
            raise typer.BadParameter(
                f"{arm}: no stage-q1 verdict was obtained; first failure: "
                f"{failures[0] if failures else 'none recorded'}"
            )
        q3 = load_stage(a, "q3")
        for r, behaviour in _q3_targets(a, record, runs):
            if r.item_id in q3 and not only:
                continue
            v = call("q3", request_body(a, r, "q3", behaviour))
            if v is not None:
                q3[r.item_id] = v
                save_stage(a, "q3", q3)
    sheet = _assemble(a, evalset, record, all_runs)
    prior_spend = RUNS / f"spend_{arm}.json"
    if prior_spend.exists():
        # Accumulate across invocations (a rescore is a second, smaller run of the same arm).
        earlier = json.loads(prior_spend.read_text(encoding="utf-8"))
        spend["prompt_tokens"] += int(earlier.get("prompt_tokens", 0) or 0)
        spend["completion_tokens"] += int(earlier.get("completion_tokens", 0) or 0)
        spend["providers"] |= set(earlier.get("providers", []))
    spend_record = {
        "arm": arm,
        "model_served": served,
        "responses_named": sorted(served_ids),
        "providers": sorted(spend["providers"]),
        "prompt_tokens": spend["prompt_tokens"],
        "completion_tokens": spend["completion_tokens"],
        "billed_to": a.billed_to,
        "establishes": a.establishes,
    }
    (RUNS / f"spend_{arm}.json").write_text(
        json.dumps(spend_record, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(
        f"{arm} ({served}; responses named {sorted(served_ids)}): {len(sheet.scores)} scored, "
        f"{len(failures)} failed; tokens {spend['prompt_tokens']:,} in / "
        f"{spend['completion_tokens']:,} out, billed to {a.billed_to} "
        f"(evals/runs/spend_{arm}.json)"
    )
    for f in failures:
        typer.echo(f"  {f}")


@app.command()
def agreement() -> None:
    """Per-arm agreement with the human sheet: overall, per stratum, and per refusal label."""
    evalset, record, _runs, _ = sample_runs()
    human = ScoreSheet.load(RUNS / f"scores_human_{evalset.sha256[:8]}.json")
    if human.rubric_sha256 != rubric_sha256():
        typer.echo("the human sheet was scored under a different rubric; agreement is meaningless")
        raise typer.Exit(2)
    for a in ARMS.values():
        path = sheet_path(a, evalset.sha256)
        if not path.exists():
            typer.echo(f"{a.id:12} no score sheet")
            continue
        judge = ScoreSheet.load(path)
        if judge.rubric_sha256 != rubric_sha256():
            typer.echo(
                f"\n{a.id}: scored under rubric sha {judge.rubric_sha256[:8]}, current is "
                f"{rubric_sha256()[:8]} — not compared; rescore first"
            )
            continue
        rows = [(i, human.scores[i], judge.scores[i]) for i in human.scores if i in judge.scores]
        amended = (
            f" — AFTER rubric v{RUBRIC_VERSION} amendment, rescored {judge.rescored_items}"
            if judge.rescored_items
            else ""
        )
        typer.echo(
            f"\n{a.id} ({a.model}, {a.reasoning_effort or 'default'}): "
            f"{len(rows)} items scored{amended}"
        )
        if judge.previous_rubric_sha256:
            before_path = path.with_suffix(f".rubric-sha-{judge.previous_rubric_sha256[:8]}.json")
            if before_path.exists():
                before = ScoreSheet.load(before_path)
                b_rows = [
                    (i, human.scores[i], before.scores[i])
                    for i in human.scores
                    if i in before.scores
                ]
                b_agree = sum(h.outcome is s.outcome for _, h, s in b_rows)
                typer.echo(
                    f"  BEFORE amendment (sha {judge.previous_rubric_sha256[:8]}): agreed on "
                    f"{b_agree} of {len(b_rows)}"
                )
        if len(rows) < len(human.scores):
            # A partial arm has not tested what it exists to test unless the discriminating
            # cells are covered; no rate is computed, only what is and is not covered.
            pair = (Outcome.CORRECT_REFUSAL, Outcome.HALLUCINATED_REFUSAL)
            covered = {lb.value: sum(1 for _, h, _ in rows if h.outcome is lb) for lb in pair}
            totals = {
                lb.value: sum(1 for h in human.scores.values() if h.outcome is lb) for lb in pair
            }
            typer.echo(
                f"  PARTIAL — {len(rows)} of {len(human.scores)}; no agreement computed. "
                f"Discriminating cells covered: correct_refusal "
                f"{covered['correct_refusal']} of {totals['correct_refusal']}, "
                f"hallucinated_refusal {covered['hallucinated_refusal']} of "
                f"{totals['hallucinated_refusal']}."
            )
            continue

        def line(
            label: str,
            subset: list[tuple[str, Score, Score]],
            rescored: list[str] = list(judge.rescored_items),  # noqa: B006 — bound per arm
        ) -> None:
            if not subset:
                return
            agree = sum(h.outcome is j.outcome for _, h, j in subset)
            # Counts, never percentages: at n=7 one item is 14 points and arms differ by noise.
            # A perfect cell after a rescore carries the amendment in the same line — a
            # post-amendment perfect agreement presented alone is the near-perfect-metric trap.
            touched = [i for i, _, _ in subset if i in rescored]
            tag = (
                f"  (after rubric v{RUBRIC_VERSION} amendment; rescored {touched})"
                if agree == len(subset) and touched
                else ""
            )
            typer.echo(f"  {label:44} agreed on {agree} of {len(subset)}{tag}")

        line("overall", rows)
        for s in Stratum:
            line(f"stratum {s.value}", [r for r in rows if record.items[r[0]].stratum == s.value])
        typer.echo("  — the discriminating pair, never pooled —")
        for label in (Outcome.CORRECT_REFUSAL, Outcome.HALLUCINATED_REFUSAL):
            line(f"human said {label.value}", [r for r in rows if r[1].outcome is label])
        for label in Outcome:
            confused = [
                (i, j.outcome.value)
                for i, h, j in rows
                if h.outcome is label and j.outcome is not label
            ]

    # Cross-arm: do the arms agree with each other more than with the human? Shared
    # model-family bias is not what the experiment set out to measure, but it would change
    # which arm ships, so it is printed as counts per cell like everything else.
    complete = {
        a.id: ScoreSheet.load(sheet_path(a, evalset.sha256))
        for a in ARMS.values()
        if sheet_path(a, evalset.sha256).exists()
        and len(ScoreSheet.load(sheet_path(a, evalset.sha256)).scores) == len(human.scores)
    }
    if len(complete) >= 2:
        typer.echo("\ncross-arm agreement (complete arms only) — with each other vs with the human")
        ids = list(human.scores)
        names = list(complete)
        for x in range(len(names)):
            for y in range(x + 1, len(names)):
                sx, sy = complete[names[x]], complete[names[y]]
                both = sum(sx.scores[i].outcome is sy.scores[i].outcome for i in ids)
                hx = sum(sx.scores[i].outcome is human.scores[i].outcome for i in ids)
                hy = sum(sy.scores[i].outcome is human.scores[i].outcome for i in ids)
                typer.echo(
                    f"  {names[x]} vs {names[y]}: agree with each other on {both} of {len(ids)}; "
                    f"with the human on {hx} and {hy} of {len(ids)}"
                )
                # Items where the arms agree with each other and both disagree with the human.
                shared = [
                    i
                    for i in ids
                    if sx.scores[i].outcome is sy.scores[i].outcome
                    and sx.scores[i].outcome is not human.scores[i].outcome
                ]
                if shared:
                    typer.echo(f"    both disagree with the human, identically, on {shared}")
            if confused:
                said = sorted({v for _, v in confused})
                items = [i for i, _ in confused]
                typer.echo(f"    {label.value} -> judge said {said} on {items}")


if __name__ == "__main__":
    app()
