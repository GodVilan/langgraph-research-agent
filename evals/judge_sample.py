"""The 25-item judge-validation sample: a seeded, documented draw from the frozen set.

Backs `make judge-sample`. The human scores these 25 agent answers under `evals/rubric.py`;
each judge arm scores the same 25; agreement is reported per arm, per stratum.

Drawn from the *frozen* artifact and recorded with its sha, never from a reviewer's earlier
session — item ids are positional and an earlier session's ids no longer resolve. The draw is
seeded so `make judge-sample` regenerates it byte-for-byte.

Weighted toward refusal, because judges disagree most on whether a refusal was correct and
refusal accuracy is a headline metric:

    unanswerable_attribute   8 of 11
    unanswerable_topic       2 of  2   (all of them — this is not a sample)
    single_paper_factual    10 of 43   stratified 7 strong-gold / 3 weak-gold, in proportion
    ambiguous                5 of 10
    multi_hop                0 of  3   (n=3 is a case study; every item is scored, none sampled)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evals.schema import EvalItem, EvalSet, GoldSupport, Stratum

SEED = 20260910
QUOTA: dict[Stratum, int] = {
    Stratum.UNANSWERABLE_ATTRIBUTE: 8,
    Stratum.UNANSWERABLE_TOPIC: 2,
    Stratum.SINGLE_PAPER: 10,
    Stratum.AMBIGUOUS: 5,
}
FACTUAL_STRONG = 7  # of the 10; the rest are weak-gold, matching the 29:14 split


def draw(evalset: EvalSet, seed: int = SEED) -> list[EvalItem]:
    rng = random.Random(seed)
    chosen: list[EvalItem] = []
    for stratum, n in QUOTA.items():
        pool = sorted((i for i in evalset.items if i.stratum is stratum), key=lambda i: i.item_id)
        if stratum is Stratum.SINGLE_PAPER:
            strong = [i for i in pool if i.gold_support is GoldSupport.STRONG]
            weak = [i for i in pool if i.gold_support is not GoldSupport.STRONG]
            chosen += rng.sample(strong, FACTUAL_STRONG) + rng.sample(weak, n - FACTUAL_STRONG)
        else:
            chosen += rng.sample(pool, min(n, len(pool)))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--out", type=Path, default=Path("evals/datasets/judge_sample.json"))
    args = parser.parse_args()

    evalset = EvalSet.read(args.set)
    items = draw(evalset)
    record = {
        "source_set": str(args.set),
        "source_sha256": evalset.sha256,
        "seed": SEED,
        "quota": {s.value: n for s, n in QUOTA.items()},
        "factual_strong_weak": [FACTUAL_STRONG, QUOTA[Stratum.SINGLE_PAPER] - FACTUAL_STRONG],
        "item_ids": [i.item_id for i in items],
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    by: dict[str, int] = {}
    for i in items:
        by[i.stratum.value] = by.get(i.stratum.value, 0) + 1
    print(f"{len(items)} items from {args.set} (sha {evalset.sha256[:16]}), seed {SEED}: {by}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
