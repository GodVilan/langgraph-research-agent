"""The regression gate: compare a run's metrics to the committed baseline and fail on a drop.

Backs `make gate`, and CI runs it on every push over the committed artifacts. A gate that
has never fired has never been tested, so `tests/test_gate.py` injects a regression into a
copy of the baseline run and asserts this exits non-zero — and the workflow has once been run
against a deliberately regressed artifact, with the failing run recorded in docs/EVALS.md.

What "regression" means here is decided by the variance estimate, not by a round number. A
metric moving by less than the spread measured across three runs of the same system is not a
move; the tolerance for each gated metric is the max-min spread observed in
`evals/baseline_metrics.json`, and a drop larger than that fails. Until the spread is measured
the gate refuses to run, rather than gating on a tolerance nobody derived.

Gated metrics (per stratum, never pooled):
  * Recall@5 on `single_paper_factual` (retrieval; no judge needed)
  * hallucinated-refusal count on the answerable strata, from the judge sheet if present
  * refusal accuracy on `unanswerable_attribute`, only ever printed beside the line above
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.metrics import outcome_summary, retrieval_rows, retrieval_summary
from evals.run_set import RunRecord
from evals.schema import EvalSet
from evals.scoring import ScoreSheet

BASELINE = Path("evals/baseline_metrics.json")


def gated_values(evalset: EvalSet, record: RunRecord, sheet: ScoreSheet | None) -> dict[str, float]:
    retrieval = retrieval_summary(retrieval_rows(evalset, record))
    factual = retrieval.get("single_paper_factual", {})
    values: dict[str, float] = {
        "factual_recall@5": float(factual.get("recall@5", 0.0)),
        "factual_mrr": float(factual.get("mrr", 0.0)),
    }
    if sheet is not None:
        o = outcome_summary(evalset, sheet)
        by = o["by_stratum_outcomes"]
        answerable = ["single_paper_factual", "multi_hop"]
        values["hallucinated_refusals"] = float(
            sum(by.get(s, {}).get("hallucinated_refusal", 0) for s in answerable)
        )
        values["correct_answers"] = float(
            sum(by.get(s, {}).get("correct_answer", 0) for s in answerable)
        )
        values["attribute_correct_refusals"] = float(
            by.get("unanswerable_attribute", {}).get("correct_refusal", 0)
        )
    return values


# Deterministic by construction (the frozen index; `make index-verify` enforces identical
# rankings): their tolerance is their own observed spread. Everything else is an outcome.
RETRIEVAL_METRICS = frozenset({"factual_recall@5", "factual_mrr"})

# Higher is better for these; lower is better for the rest.
HIGHER_IS_BETTER = {
    "factual_recall@5",
    "factual_mrr",
    "correct_answers",
    "attribute_correct_refusals",
}


def evaluate(values: dict[str, float], baseline: dict[str, Any]) -> list[str]:
    """Every gated metric that regressed past its measured tolerance."""
    failures: list[str] = []
    for name, current in values.items():
        if name not in baseline["metrics"]:
            continue
        ref = baseline["metrics"][name]
        centre, tolerance = float(ref["value"]), float(ref["tolerance"])
        drop = (centre - current) if name in HIGHER_IS_BETTER else (current - centre)
        if drop > tolerance:
            failures.append(
                f"{name}: {current:.3f} vs baseline {centre:.3f} "
                f"(tolerance ±{tolerance:.3f} from the measured spread) — regression"
            )
    return failures


def derive_baseline(
    per_run: list[dict[str, float]],
    records: list[RunRecord],
    reference: dict[str, Any],
    run_paths: list[Path],
) -> dict[str, Any]:
    """The one derivation rule, for any number of runs of one configuration.

    * **n ≥ 2:** value = the mean of the runs. Tolerance: for **retrieval** metrics, the runs'
      max-min spread — retrieval is deterministic by construction and `make index-verify`
      enforces it. For **outcome** metrics (anything downstream of generation or judging),
      ``max(own spread, reference spread)``: generation and judging stay unpinned (D-042
      measured pinning, did not ship it), so three equal draws from a process known to vary
      are a small-sample artifact, not evidence the variance is gone — a zero tolerance there
      is a flaky gate (Phase 5 review, D-044 note). The reference is the unpinned Phase 4
      baseline; deriving Phase 4 against itself leaves it unchanged, and
      `tests/test_gate.py` asserts both committed baselines regenerate exactly.
    * **n = 1:** value = the run; tolerance **carried** from ``reference`` — one run cannot
      measure its own spread. This was the first pinned baseline (D-044), and it loosened the
      gate: the centre was the one draw already worse than all three unpinned runs.

    Only metrics every run reports and the reference gates are included. The written file is
    the function's output; baselines are never edited by hand.
    """
    names = [n for n in reference["metrics"] if all(n in v for v in per_run)]
    metrics: dict[str, Any] = {}
    for name in names:
        runs = [v[name] for v in per_run]
        own = max(runs) - min(runs)
        floor = float(reference["metrics"][name]["tolerance"])
        if len(runs) < 2:
            tolerance, basis = floor, "reference spread, carried (one run)"
        elif name in RETRIEVAL_METRICS:
            tolerance, basis = own, "own spread (retrieval is deterministic by construction)"
        else:
            tolerance = max(own, floor)
            basis = "own spread" if own >= floor else "floor: reference (unpinned) spread"
        metrics[name] = {
            "value": sum(runs) / len(runs),
            "tolerance": tolerance,
            "tolerance_basis": basis,
            "runs": [int(r) if float(r).is_integer() and r > 1 else r for r in runs],
        }
    pinned = {r.scope_classifier_pinned for r in records}
    if len(pinned) != 1 or len({(r.set_sha256, r.corpus_sha256) for r in records}) != 1:
        raise ValueError("a baseline is derived from runs of one configuration on one set")
    n = len(per_run)
    return {
        "set_sha256": records[0].set_sha256,
        "corpus_sha256": records[0].corpus_sha256,
        "judge": reference.get("judge"),
        "rubric_version": reference.get("rubric_version"),
        "configuration": {"scope_classifier_pinned": pinned.pop()},
        "derivation": (
            f"value = mean of {n} complete runs. Tolerance: retrieval metrics = their max-min "
            f"spread (deterministic by construction); outcome metrics = max(their max-min "
            f"spread, the reference baseline's spread for the same metric), because generation "
            f"and judging stay unpinned and a few equal draws do not show the variance is gone "
            f"(D-044 note). A drop larger than the tolerance is a regression. {n} draws, no "
            f"confidence interval."
            if n >= 2
            else "value = this single run; tolerance = the reference baseline's spread, carried "
            "(one run cannot measure its own spread)."
        ),
        "tolerance_reference": reference.get("_path", "the first --baseline"),
        "runs": [str(p) for p in run_paths],
        "metrics": metrics,
    }


def strictest(baselines: list[dict[str, Any]]) -> dict[str, Any]:
    """One baseline whose every bound is the stricter of the given ones.

    For a higher-is-better metric the bound is ``value - tolerance`` and the stricter is the
    higher; for the rest, ``value + tolerance`` and the lower. Expressed back as a centre with
    zero tolerance, so `evaluate` applies it unchanged. Used while a baseline is being
    re-measured, so the gate is never looser than either (Phase 5 review).
    """
    out: dict[str, Any] = {
        "metrics": {},
        "derivation": "stricter bound of each metric across baselines",
    }
    names = set().union(*(b["metrics"] for b in baselines))
    for name in sorted(names):
        bounds = []
        for b in baselines:
            ref = b["metrics"].get(name)
            if ref is None:
                continue
            centre, tol = float(ref["value"]), float(ref["tolerance"])
            bounds.append(centre - tol if name in HIGHER_IS_BETTER else centre + tol)
        bound = max(bounds) if name in HIGHER_IS_BETTER else min(bounds)
        out["metrics"][name] = {"value": bound, "tolerance": 0.0}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--run", type=Path, default=None)
    parser.add_argument("--sheet", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        type=Path,
        action="append",
        default=None,
        help="repeatable: with more than one, each metric is gated on the stricter bound",
    )
    parser.add_argument(
        "--derive-baseline",
        type=Path,
        default=None,
        help="write a baseline derived from the --derive-from runs to this path",
    )
    parser.add_argument(
        "--derive-from",
        action="append",
        default=[],
        metavar="RUN.json:SHEET.json",
        help="repeatable: the runs of one configuration a baseline is derived from",
    )
    args = parser.parse_args()
    evalset = EvalSet.read(args.set)
    baseline_paths: list[Path] = args.baseline or [BASELINE]

    for path in baseline_paths:
        if not path.exists():
            print(
                f"no baseline {path}: the gate refuses to run on an underived tolerance",
                file=sys.stderr,
            )
            return 2
    loaded = [
        {**json.loads(p.read_text(encoding="utf-8")), "_path": str(p)} for p in baseline_paths
    ]

    if args.derive_baseline is not None:
        if not args.derive_from:
            print("--derive-baseline needs at least one --derive-from RUN:SHEET", file=sys.stderr)
            return 2
        per_run, records, paths = [], [], []
        for pair in args.derive_from:
            run_path, sheet_path = (Path(x) for x in pair.split(":", 1))
            record = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
            per_run.append(gated_values(evalset, record, ScoreSheet.load(sheet_path)))
            records.append(record)
            paths.append(run_path)
        derived = derive_baseline(per_run, records, loaded[0], paths)
        args.derive_baseline.write_text(json.dumps(derived, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.derive_baseline} from {len(paths)} run(s)")
        for name, ref in derived["metrics"].items():
            print(
                f"  {name:28} {ref['value']:.3f} ±{ref['tolerance']:.3f}  runs {ref['runs']}  "
                f"({ref['tolerance_basis']})"
            )
        return 0

    if args.run is None:
        print("--run is required to gate", file=sys.stderr)
        return 2
    baseline = loaded[0] if len(loaded) == 1 else strictest(loaded)
    record = RunRecord.model_validate_json(args.run.read_text(encoding="utf-8"))
    sheet = ScoreSheet.load(args.sheet) if args.sheet else None
    values = gated_values(evalset, record, sheet)
    failures = evaluate(values, baseline)
    if len(loaded) > 1:
        print(
            f"gating on the stricter bound of {len(loaded)} baselines: "
            + ", ".join(str(p) for p in baseline_paths)
        )
    for name, value in values.items():
        ref = baseline["metrics"].get(name)
        print(
            f"  {name:28} {value:.3f}"
            + (
                f"   baseline {ref['value']:.3f} ±{ref['tolerance']:.3f}"
                if ref
                else "   (not gated)"
            )
        )
    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\ngate passed (within the measured spread of the baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
