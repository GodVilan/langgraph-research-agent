"""Attach judge scores to the Langfuse traces the judged run wrote.

Backs `make push-scores`. The dashboard's Scores panel reads scores, not traces: with 69
traces and no scores it shows "No data", which is what the pre-Phase-4 screenshot shows.

**The scores must belong to the traced run's own answers.** The generator is nondeterministic
(D-014), so a different run produced different answers and its sheet describes different text;
pushing r1's scores onto a later run's traces would attach labels to answers nobody scored.
This refuses to run unless the score sheet's `run_path` names the run being pushed.

One score per item per axis, so the panel has something to aggregate:
  `outcome_correct`   1.0 when the rubric label is the stratum's correct outcome, else 0.0
  `retrieval_recall5` gold chunks retrieved in the top 5, as a fraction (answerable only)
plus the rubric label itself as the score comment, so a trace in the UI shows why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import has_answer_gold, recall_at
from evals.rubric import Outcome
from evals.run_set import RunRecord
from evals.schema import EvalSet
from evals.scoring import ScoreSheet
from src.observability import langfuse as lf

CORRECT = {
    Outcome.CORRECT_ANSWER,
    Outcome.CORRECT_REFUSAL,
    Outcome.CORRECT_CLARIFICATION,
}


def _client() -> tuple[Any, str]:
    """An authenticated client for the Langfuse public API, and its host."""
    import base64

    import httpx

    from src.observability.config import get_observability_settings

    s = get_observability_settings()
    auth = base64.b64encode(
        f"{s.langfuse_public_key.get_secret_value()}:"
        f"{s.langfuse_secret_key.get_secret_value()}".encode()
    ).decode()
    return (
        httpx.Client(
            base_url=s.langfuse_host, headers={"Authorization": f"Basic {auth}"}, timeout=60
        ),
        s.langfuse_host,
    )


def _score_ids(client: Any, trace_ids: set[str]) -> list[str]:
    ids: list[str] = []
    page = 1
    while True:
        r = client.get("/api/public/scores", params={"limit": 100, "page": page})
        r.raise_for_status()
        body = r.json()
        ids += [str(x["id"]) for x in body.get("data", []) if x.get("traceId") in trace_ids]
        if page >= int(body.get("meta", {}).get("totalPages", 1)):
            break
        page += 1
    return ids


def _count_scores(client: Any, trace_ids: set[str]) -> int:
    return len(_score_ids(client, trace_ids))


def delete_scores_for(trace_ids: set[str]) -> int:
    """Remove every score attached to these traces, and verify they are gone.

    `create_score` is not idempotent: pushing twice attaches two scores to one trace and the
    dashboard averages both.

    **Deletion is asynchronous.** Langfuse answers `202 Score deletion queued successfully`,
    and a first version counted 202 as done, pushed immediately, and produced 239 scores where
    115 were intended. A queued operation is not a completed one (D-026, fourth instance), so
    this polls until the scores are actually gone and raises if they are not.
    """
    import time

    client, _ = _client()
    with client:
        doomed = _score_ids(client, trace_ids)
        for score_id in doomed:
            client.delete(f"/api/public/scores/{score_id}")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            left = _count_scores(client, trace_ids)
            if not left:
                return len(doomed)
            time.sleep(3)
        raise SystemExit(
            f"{_count_scores(client, trace_ids)} score(s) still present after 120s; "
            f"Langfuse's worker drains this queue slowly. Refusing to push on top of them."
        )


def _await_scores(trace_ids: set[str], expected: int, timeout_s: float = 120.0) -> int:
    """Poll until the expected number of scores is visible, and return what actually is."""
    import time

    client, _ = _client()
    seen = 0
    with client:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            seen = _count_scores(client, trace_ids)
            if seen >= expected:
                return seen
            time.sleep(3)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete this run's existing scores first; `create_score` is not idempotent",
    )
    args = parser.parse_args()

    evalset = EvalSet.read(args.set)
    record = RunRecord.model_validate_json(args.run.read_text(encoding="utf-8"))
    sheet = ScoreSheet.load(args.sheet)

    # The sheet has to be about *this* run, or the scores describe other answers entirely.
    if Path(sheet.run_path).name != args.run.name:
        raise SystemExit(
            f"refusing: the sheet was scored against {sheet.run_path}, not {args.run}. "
            f"The generator is nondeterministic; another run's answers are not these answers."
        )
    if lf.get_client() is None:
        raise SystemExit("Langfuse is not configured; nothing to push (make langfuse-up)")

    gold = {i.item_id: i for i in evalset.items}
    if args.replace:
        removed = delete_scores_for({r.trace_id for r in record.items.values() if r.trace_id})
        print(f"deleted {removed} existing score(s) on this run's traces")
    pushed = skipped = 0
    for item_id, score in sorted(sheet.scores.items()):
        run = record.items.get(item_id)
        if run is None or not run.trace_id:
            skipped += 1
            continue
        ok = lf.record_score(
            run.trace_id,
            "outcome_correct",
            1.0 if score.outcome in CORRECT else 0.0,
            comment=f"{score.outcome.value} — {score.reason[:200]}",
        )
        item = gold[item_id]
        if has_answer_gold(item.stratum):
            lf.record_score(
                run.trace_id,
                "retrieval_recall5",
                recall_at([c.chunk_id for c in run.retrieved], list(item.gold_chunk_ids), 5),
                comment=f"{item.stratum.value}; gold {len(item.gold_chunk_ids)} chunk(s)",
            )
        pushed += 1 if ok else 0
    lf.flush()
    # `create_score` returning without raising means the SDK accepted it, not that Langfuse
    # ingested it — ingestion is asynchronous, exactly like the deletion above. Inferring the
    # effect from the absence of an exception is the same defect as inferring it from a 202
    # (D-026), so the count is read back from the server and reported as the actual.
    expected = pushed + sum(
        1
        for item_id in sheet.scores
        if item_id in record.items
        and record.items[item_id].trace_id
        and has_answer_gold(gold[item_id].stratum)
    )
    landed = _await_scores({r.trace_id for r in record.items.values() if r.trace_id}, expected)
    print(
        f"pushed scores for {pushed} of {len(sheet.scores)} scored items "
        f"({skipped} had no trace id); judge {sheet.scorer}, rubric {sheet.rubric_sha256[:8]}"
    )
    print(f"verified {landed} of {expected} score(s) present at the server")
    return 0 if landed == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
