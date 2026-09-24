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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()

    if not args.baseline.exists():
        print("no baseline: the gate refuses to run on an underived tolerance", file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    evalset = EvalSet.read(args.set)
    record = RunRecord.model_validate_json(args.run.read_text(encoding="utf-8"))
    sheet = ScoreSheet.load(args.sheet) if args.sheet else None
    values = gated_values(evalset, record, sheet)
    failures = evaluate(values, baseline)
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
