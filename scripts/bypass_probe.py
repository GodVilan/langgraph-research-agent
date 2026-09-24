"""Settle whether the five guardrail-refused items are answerable: retrieve, bypassing the guard.

Backs `make bypass-probe`. The input scope classifier refused five verified factual items whose
questions the paraphrase rule had stripped of every ML term. Two readings were possible —
the guardrail is wrong, or the items are under-anchored — and one of them makes the headline
number smaller, so it is not decided by argument after seeing the result. It is measured:

    retrieval returns the gold chunk  ->  the corpus answers the question; the guardrail failed
    retrieval does not                ->  the item is genuinely under-anchored

Retrieval is called directly on the verbatim question, exactly as the agent's `retrieve` node
would call it after the input layer, at the agent's top_k and at 10. No LLM call, no OpenAI.

    python scripts/bypass_probe.py               # the items the last run blocked
    python scripts/bypass_probe.py --items sp-003,sp-014
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.run_set import RunRecord, load_record, run_path
from evals.schema import EvalSet
from evals.verify_items import bare_figures, claims_in, claims_supported_by


def fact_in_run_context(record: RunRecord, item_id: str) -> tuple[bool, str]:
    """Whether the gold fact was in the chunks the *run* retrieved — gold chunk or not.

    "Gold chunk retrieved" is the wrong split between retrieval and generation failure, and
    `sp-036` is why: its gold chunk was not retrieved, but a neighbouring chunk of the same
    paper stating the same fact was — "For Llama 3.2 models, we use context length 8192" —
    the agent quoted it, and refused anyway. That is a generation failure the gold-chunk
    criterion files under retrieval. So the split is made the way the rubric grounds an
    answer: is the fact stated in what the agent retrieved.
    """
    run = record.items[item_id]
    joined = " ".join(c.text for c in run.retrieved)
    claims = claims_in(run.gold_answer)
    supported = claims_supported_by(joined, claims)
    if any(supported.values()):
        return True, "claim in retrieved text: " + str(
            {k: sorted(v) for k, v in supported.items() if v}
        )
    loose = bare_figures(run.gold_answer)
    if loose and loose <= bare_figures(joined):
        # A common figure present somewhere in context; name the papers it came from so a
        # coincidental "1000" from another paper is visible as such.
        papers = sorted({c.paper_id for c in run.retrieved if loose & bare_figures(c.text)})
        gold_papers = {g.split("_")[0] for g in run.gold_chunk_ids}
        if set(papers) & gold_papers:
            return True, f"figure {sorted(loose)} in a retrieved chunk of the gold paper"
        return False, f"figure {sorted(loose)} appears only in other papers {papers}"
    return False, "no claim of the gold answer in any retrieved chunk"


async def probe(item_ids: list[str], evalset: EvalSet, top_k: int) -> list[dict[str, object]]:
    from src.retrieval.service import RetrievalService

    service = RetrievalService.load()
    by_id = {i.item_id: i for i in evalset.items}
    rows: list[dict[str, object]] = []
    for item_id in item_ids:
        item = by_id[item_id]
        result = await service.retrieve(item.question, top_k=top_k)
        chunks = [hit[0] for hit in result.hits]
        hit_ids = [str(c.chunk_id) for c in chunks]
        ranks = {g: (hit_ids.index(g) + 1 if g in hit_ids else None) for g in item.gold_chunk_ids}
        gold_papers = {g.split("_")[0] for g in item.gold_chunk_ids}
        paper_hit = any(str(c.paper_id) in gold_papers for c in chunks)
        rows.append(
            {
                "item_id": item_id,
                "question": item.question,
                "gold_chunk_ids": list(item.gold_chunk_ids),
                "top_k": top_k,
                "hits": hit_ids,
                "gold_rank": ranks,
                "gold_chunk_retrieved": any(r is not None for r in ranks.values()),
                "gold_paper_retrieved": paper_hit,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--items", default="")
    parser.add_argument(
        "--from-scores",
        type=Path,
        default=None,
        help="probe every item a score sheet labelled hallucinated_refusal",
    )
    parser.add_argument("--out", type=Path, default=Path("evals/runs/bypass_probe.json"))
    args = parser.parse_args()

    evalset = EvalSet.read(args.set)
    record = load_record(run_path(evalset.sha256))
    if record is None:
        raise SystemExit("no run recorded for this set")
    if args.items:
        ids = [i.strip() for i in args.items.split(",") if i.strip()]
    elif args.from_scores:
        sheet = json.loads(args.from_scores.read_text(encoding="utf-8"))
        ids = [i for i, sc in sheet["scores"].items() if sc["outcome"] == "hallucinated_refusal"]
    else:
        ids = [r.item_id for r in record.items.values() if r.status == "guardrail_blocked"]

    rows = asyncio.run(probe(ids, evalset, 5)) + asyncio.run(probe(ids, evalset, 10))
    for row in rows:
        item_id = str(row["item_id"])
        run = record.items[item_id]
        in_context, why = fact_in_run_context(record, item_id)
        row["run_status"] = run.status
        row["run_retrieved"] = [c.chunk_id for c in run.retrieved]
        row["fact_in_run_context"] = in_context
        row["fact_in_run_context_why"] = why
        row["bucket"] = (
            "guardrail"
            if run.status == "guardrail_blocked"
            else "generation"
            if in_context
            else "retrieval"
        )
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['item_id']}  k={row['top_k']:2d}  gold chunk retrieved: "
            f"{row['gold_chunk_retrieved']!s:5}  rank {row['gold_rank']}  "
            f"gold paper in hits: {row['gold_paper_retrieved']}"
        )
    at5 = [r for r in rows if r["top_k"] == 5]
    print(
        f"\nat the agent's top_k=5: gold chunk retrieved for "
        f"{sum(bool(r['gold_chunk_retrieved']) for r in at5)} of {len(at5)}"
    )
    print("\nbucket (from the run's own retrieved context, the rubric's grounding criterion):")
    for row in at5:
        print(f"  {row['item_id']}  {row['bucket']:10}  {row['fact_in_run_context_why']}")
    counts: dict[str, int] = {}
    for row in at5:
        counts[str(row["bucket"])] = counts.get(str(row["bucket"]), 0) + 1
    print(f"  -> {counts} (n={len(at5)}); wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
