"""Phase 4 metrics: every number per stratum, with n, from artifacts a reader can regenerate.

Backs `make metrics-report`. Two families, kept apart because they come from different
instruments:

**Retrieval** (no judge, no model): Recall@k and MRR over the chunk ids the agent retrieved
against the gold chunk ids in the frozen set. Reported per stratum, and for factual items
split by `gold_support` — 14 of 43 factual gold sets pass containment only on the token or
common-figure fallback, and a material strong-vs-weak gap is a finding about the gold set,
not the retriever. Multi-hop is n=3: per item by name, never a rate.

**Outcomes** (from a score sheet, human or judge): answer accuracy, grounding, the refusal
pair, clarification. **Refusal accuracy and hallucinated-refusal rate are printed on one line,
always** — 10 of 10 on the unanswerable strata next to 7 of 10 refused on the answerable ones
describes an agent that refuses readily, and the first number alone is the near-perfect-metric
trap. `unanswerable_topic` is per item by name (n=2, effectively 1 clean — D-031).

**Variance**: with several runs of the agent each scored by the same judge, every metric is
reported as its spread across runs (min, max, std). D-014 established the generator is
nondeterministic; the necessity fixture measured the *judge-side* instrument at 1-of-3 and
2-of-3 on a real multi-hop pair. One run is a draw, not a measurement.

Every figure is labelled "local batch run, single-user" — nothing here is live traffic.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.rubric import Outcome
from evals.run_set import RunRecord, load_record, run_path
from evals.schema import EvalSet, Stratum
from evals.scoring import ScoreSheet

RUNS = Path("evals/runs")


# ── retrieval ─────────────────────────────────────────────────────────────────────────────


def recall_at(run_hits: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(run_hits[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def reciprocal_rank(run_hits: list[str], gold: list[str]) -> float:
    for rank, cid in enumerate(run_hits, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def has_answer_gold(stratum: Stratum) -> bool:
    """Whether this stratum's gold chunks are *answer support*, so Recall@k means something.

    The one predicate, because it was briefly two. `scripts/push_scores.py` had its own
    version that excluded only the refusal strata, so it pushed a `retrieval_recall5` score
    over 56 items while `make metrics-report` computed Recall@5 over 46 — the same name on
    two populations, which is the merge-two-measurements rule (CLAUDE.md §8.5) with a
    dashboard as the second reader. Ambiguous gold chunks are *competing referents*, not
    answer support: recall over them asks a different question and does not belong under this
    name.
    """
    return not stratum.expects_refusal and stratum is not Stratum.AMBIGUOUS


def retrieval_rows(evalset: EvalSet, record: RunRecord) -> list[dict[str, Any]]:
    rows = []
    for item in evalset.items:
        if not has_answer_gold(item.stratum):
            continue
        run = record.items.get(item.item_id)
        if run is None:
            continue
        hits = [c.chunk_id for c in run.retrieved]
        rows.append(
            {
                "item_id": item.item_id,
                "stratum": item.stratum.value,
                "gold_support": item.gold_support.value if item.gold_support else "unknown",
                "n_retrieved": len(hits),
                "recall@5": recall_at(hits, item.gold_chunk_ids, 5),
                "recall@10": recall_at(hits, item.gold_chunk_ids, 10),
                "rr": reciprocal_rank(hits, item.gold_chunk_ids),
                "gold_paper_hit": any(
                    c.paper_id in {g.split("_")[0] for g in item.gold_chunk_ids}
                    for c in run.retrieved
                ),
                "guardrail_blocked": run.status == "guardrail_blocked",
            }
        )
    return rows


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per stratum, then factual by gold support. Multi-hop is listed per item."""
    out: dict[str, dict[str, Any]] = {}
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r["stratum"])].append(r)
        if r["stratum"] == Stratum.SINGLE_PAPER.value:
            by[f"{r['stratum']}/{r['gold_support']}"].append(r)
    for key, group in sorted(by.items()):
        reached = [r for r in group if not r["guardrail_blocked"]]
        k_max = max((int(r["n_retrieved"]) for r in group), default=0)
        entry: dict[str, Any] = {
            "n": len(group),
            # Blocked items retrieved nothing and count as zero: the metric is the system's,
            # and the input guardrail is part of the system. The same figure over the items
            # that reached retrieval is beside it, so the two failures are separable.
            "recall@5": round(_mean([float(r["recall@5"]) for r in group]), 3),
            "recall@5_reached_retrieval": round(_mean([float(r["recall@5"]) for r in reached]), 3),
            "n_reached_retrieval": len(reached),
            "mrr": round(_mean([float(r["rr"]) for r in group]), 3),
            "gold_paper_in_hits": sum(1 for r in group if r["gold_paper_hit"]),
            "guardrail_blocked": sum(1 for r in group if r["guardrail_blocked"]),
            "k_retrieved": k_max,
        }
        if k_max < 10:
            entry["recall@10"] = f"not computable: the run retrieved k={k_max}"
        else:
            entry["recall@10"] = round(_mean([float(r["recall@10"]) for r in group]), 3)
        if key == Stratum.MULTI_HOP.value:
            entry["per_item"] = {
                str(r["item_id"]): {"recall@5": r["recall@5"], "rr": r["rr"]} for r in group
            }
            entry["note"] = "n=3 — a case study, never a rate"
        out[key] = entry
    return out


# ── outcomes ──────────────────────────────────────────────────────────────────────────────


def outcome_summary(evalset: EvalSet, sheet: ScoreSheet) -> dict[str, Any]:
    """The rubric's metrics, per stratum, refusal pair on one line, n everywhere."""
    stratum_of = {i.item_id: i.stratum for i in evalset.items}
    scored = {i: s for i, s in sheet.scores.items() if i in stratum_of}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item_id, s in scored.items():
        counts[stratum_of[item_id].value][s.outcome.value] += 1

    def n(stratum: str) -> int:
        return sum(counts[stratum].values())

    answerable = [Stratum.SINGLE_PAPER.value, Stratum.MULTI_HOP.value]
    n_ans = sum(n(s) for s in answerable)
    correct = sum(counts[s][Outcome.CORRECT_ANSWER.value] for s in answerable)
    ungrounded = sum(counts[s][Outcome.UNGROUNDED_ANSWER.value] for s in answerable)
    hall_ref = (
        sum(counts[s][Outcome.HALLUCINATED_REFUSAL.value] for s in answerable)
        + counts[Stratum.AMBIGUOUS.value][Outcome.HALLUCINATED_REFUSAL.value]
    )
    amb_hall = counts[Stratum.AMBIGUOUS.value][Outcome.HALLUCINATED_REFUSAL.value]
    ua = counts[Stratum.UNANSWERABLE_ATTRIBUTE.value]
    ut_items = {
        i: s.outcome.value for i, s in scored.items() if stratum_of[i] is Stratum.UNANSWERABLE_TOPIC
    }
    amb = counts[Stratum.AMBIGUOUS.value]
    return {
        "scorer": sheet.scorer,
        "rubric_sha256": sheet.rubric_sha256[:8],
        "n_scored": len(scored),
        "by_stratum_outcomes": {k: dict(v) for k, v in counts.items()},
        # The pair, on one line, always.
        "refusal_pair": (
            f"refusal accuracy {ua[Outcome.CORRECT_REFUSAL.value]} of "
            f"{n(Stratum.UNANSWERABLE_ATTRIBUTE.value)} (attribute) "
            f"+ topic per item {ut_items or '{}'}  |  hallucinated refusals "
            f"{hall_ref - amb_hall} of {n_ans} answerable "
            f"(+{amb_hall} of {n(Stratum.AMBIGUOUS.value)} ambiguous) — read together: an "
            f"agent that refuses readily scores the first perfectly by refusing everything"
        ),
        "answer_accuracy": f"{correct} of {n_ans} answerable items correct",
        "grounding": f"{ungrounded} of {correct + ungrounded} correct facts ungrounded",
        "clarification": (
            f"{amb[Outcome.CORRECT_CLARIFICATION.value]} of {n(Stratum.AMBIGUOUS.value)} "
            f"ambiguous items clarified; {amb[Outcome.SILENT_DISAMBIGUATION.value]} silently "
            f"disambiguated; {amb[Outcome.HALLUCINATED_REFUSAL.value]} refused"
        ),
        "multi_hop_per_item": {
            i: s.outcome.value for i, s in scored.items() if stratum_of[i] is Stratum.MULTI_HOP
        },
    }


# ── run-level ─────────────────────────────────────────────────────────────────────────────


def run_summary(record: RunRecord) -> dict[str, Any]:
    lat = sorted(r.latency_s for r in record.items.values())
    notional = [float(r.usage.get("notional_cost_usd", 0.0) or 0.0) for r in record.items.values()]
    return {
        "label": "local batch run, single-user — not live traffic",
        "n": len(record.items),
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p95_s": round(lat[max(0, int(0.95 * len(lat)) - 1)], 2) if lat else None,
        "notional_usd_per_query_median": round(statistics.median(notional), 4)
        if notional
        else None,
        "billed_usd": 0.0,
        "statuses": {
            s: sum(1 for r in record.items.values() if r.status == s)
            for s in sorted({r.status for r in record.items.values()})
        },
    }


def compare(evalset: EvalSet, tags: list[str]) -> str:
    """Retrieval metrics side by side across runs, with the max-min spread over repeats.

    Judge-free, so it is available while judging is not. Repeat runs of the shipping config
    (`r1`, `r2`, `r3`) give the spread; each arm is one draw against it. An arm that differs
    from the shipping config by less than the spread has not been shown to differ.
    """
    rows: list[str] = []
    values: dict[str, dict[str, float]] = {}
    for tag in tags:
        path = run_path(evalset.sha256, "" if tag == "r1" else tag)
        record = load_record(path)
        if record is None:
            rows.append(f"{tag:16} (no run)")
            continue
        summary = retrieval_summary(retrieval_rows(evalset, record))
        if "single_paper_factual" not in summary:
            rows.append(
                f"{tag:16} arm={record.arm:16} partial: {len(record.items)} items, "
                f"no factual rows yet"
            )
            continue
        s = summary["single_paper_factual"]
        values[tag] = {"recall@5": float(s["recall@5"]), "mrr": float(s["mrr"])}
        rows.append(
            f"{tag:16} arm={record.arm:16} factual recall@5 {s['recall@5']:.3f} "
            f"(reached-retrieval {s['recall@5_reached_retrieval']:.3f}, "
            f"n={s['n_reached_retrieval']})  "
            f"mrr {s['mrr']:.3f}  gold-paper-in-top5 {s['gold_paper_in_hits']}/43  "
            f"guardrail-blocked {s['guardrail_blocked']}"
        )
    repeats = [v for t, v in values.items() if t in {"r1", "r2", "r3"}]
    if len(repeats) >= 2:
        for metric in ("recall@5", "mrr"):
            xs = [r[metric] for r in repeats]
            rows.append(
                f"{'spread ' + metric:16} over {len(xs)} repeat runs: min {min(xs):.3f} "
                f"max {max(xs):.3f} max-min {max(xs) - min(xs):.3f} — three draws, no interval"
            )
    return "\n".join(rows)


OUTCOME_KEYS = (
    "correct",
    "wrong",
    "hall_ref_answerable",
    "ungrounded",
    "attr_correct_refusal",
    "topic_correct_refusal",
    "amb_clarified",
    "amb_silent",
    "amb_refused",
)


def outcome_counts(evalset: EvalSet, sheet: ScoreSheet) -> dict[str, int]:
    """The gated outcome counts for one (run, sheet) pair. Counts, never rates."""
    by = outcome_summary(evalset, sheet)["by_stratum_outcomes"]
    answerable = (Stratum.SINGLE_PAPER.value, Stratum.MULTI_HOP.value)

    def g(stratum: str, outcome: Outcome) -> int:
        return int(by.get(stratum, {}).get(outcome.value, 0))

    return {
        "correct": sum(g(s, Outcome.CORRECT_ANSWER) for s in answerable),
        "wrong": sum(g(s, Outcome.WRONG_ANSWER) for s in answerable),
        "hall_ref_answerable": sum(g(s, Outcome.HALLUCINATED_REFUSAL) for s in answerable),
        "ungrounded": sum(g(s, Outcome.UNGROUNDED_ANSWER) for s in answerable),
        "attr_correct_refusal": g(Stratum.UNANSWERABLE_ATTRIBUTE.value, Outcome.CORRECT_REFUSAL),
        "topic_correct_refusal": g(Stratum.UNANSWERABLE_TOPIC.value, Outcome.CORRECT_REFUSAL),
        "amb_clarified": g(Stratum.AMBIGUOUS.value, Outcome.CORRECT_CLARIFICATION),
        "amb_silent": g(Stratum.AMBIGUOUS.value, Outcome.SILENT_DISAMBIGUATION),
        "amb_refused": g(Stratum.AMBIGUOUS.value, Outcome.HALLUCINATED_REFUSAL),
    }


def sheet_for(set_sha: str, tag: str) -> Path:
    suffix = "" if tag == "r1" else f"_{tag}"
    return RUNS / f"scores_luna-low_{set_sha[:8]}_all{suffix}.json"


def spread(evalset: EvalSet, repeats: list[str], arms: list[str]) -> str:
    """Three draws of every outcome metric, their max-min, and each arm against that spread.

    No confidence interval: three draws cannot support one (docs/EVALS.md). An arm differing
    from the shipping configuration by no more than the spread has not been shown to differ.
    """
    counts = {
        t: outcome_counts(evalset, ScoreSheet.load(sheet_for(evalset.sha256, t)))
        for t in repeats + arms
        if sheet_for(evalset.sha256, t).exists()
    }
    present = [t for t in repeats if t in counts]
    rows = [
        "outcome counts over the frozen 69 — local batch, single-user, judge gpt-5.6-luna @ low",
        "",
        f"{'metric':24} " + " ".join(f"{t:>6}" for t in present) + "   min  max  spread",
    ]
    spreads: dict[str, int] = {}
    for key in OUTCOME_KEYS:
        xs = [counts[t][key] for t in present]
        spreads[key] = max(xs) - min(xs)
        rows.append(
            f"{key:24} "
            + " ".join(f"{x:6}" for x in xs)
            + f"   {min(xs):3}  {max(xs):3}  {spreads[key]:3}"
        )
    rows += ["", "arms, each one draw, against that spread:", ""]
    rows.append(f"{'metric':24} " + " ".join(f"{a:>15}" for a in arms if a in counts))
    for key in OUTCOME_KEYS:
        cells = []
        for a in arms:
            if a not in counts:
                continue
            delta = counts[a][key] - counts[present[0]][key]
            mark = "" if abs(delta) <= spreads[key] else "  *"
            cells.append(f"{counts[a][key]:>6} ({delta:+d}){mark:<3}")
        rows.append(f"{key:24} " + " ".join(f"{c:>15}" for c in cells))
    rows += [
        "",
        "* marks a difference from r1 larger than the three-run spread. Everything unmarked",
        "  sits inside it and has not been shown to differ. Three draws, no interval.",
    ]
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--run", type=Path, default=None, help="a run record; default: the set's")
    parser.add_argument("--sheet", type=Path, default=None, help="a score sheet for outcomes")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--spread",
        default="",
        help="comma-separated arm tags to compare against the r1/r2/r3 outcome spread",
    )
    parser.add_argument(
        "--compare",
        default="",
        help="comma-separated run tags (r1,r2,r3,dense_only,…): judge-free retrieval side by side",
    )
    args = parser.parse_args()

    evalset = EvalSet.read(args.set)
    if args.spread:
        print(
            spread(
                evalset,
                ["r1", "r2", "r3"],
                [a.strip() for a in args.spread.split(",") if a.strip()],
            )
        )
        return 0
    if args.compare:
        print(compare(evalset, [t.strip() for t in args.compare.split(",") if t.strip()]))
        return 0
    record = (
        RunRecord.model_validate_json(args.run.read_text(encoding="utf-8"))
        if args.run
        else load_record(run_path(evalset.sha256))
    )
    if record is None:
        raise SystemExit("no run recorded")
    report: dict[str, Any] = {
        "set_sha256": evalset.sha256[:8],
        "corpus_sha256": record.corpus_sha256[:8],
        "generator": record.generator_model,
        "run": run_summary(record),
        "retrieval": retrieval_summary(retrieval_rows(evalset, record)),
    }
    if args.sheet:
        sheet = ScoreSheet.load(args.sheet)
        report["outcomes"] = outcome_summary(evalset, sheet)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(
        f"set {report['set_sha256']}  corpus {report['corpus_sha256']}  "
        f"generator {report['generator']}"
    )
    print(f"run: {report['run']}")
    print("\nretrieval (per stratum; factual split by gold support):")
    for key, entry in report["retrieval"].items():
        print(f"  {key:40} {entry}")
    if "outcomes" in report:
        o = report["outcomes"]
        print(f"\noutcomes — scorer {o['scorer']}, rubric {o['rubric_sha256']}, n={o['n_scored']}")
        for k in ("refusal_pair", "answer_accuracy", "grounding", "clarification"):
            print(f"  {k:18} {o[k]}")
        print(f"  multi_hop per item {o['multi_hop_per_item']}")
        print(f"  by stratum {o['by_stratum_outcomes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
