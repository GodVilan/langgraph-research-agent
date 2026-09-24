"""Every entity the premise check would assert, across the whole set.

Backs `make audit-entities`. The stop list was grown one entry at a time — GPU, CPU, FLOPS,
AUC, RAM each added after causing a false positive — which encodes the terms that happened
to be hit and leaves the rest untested until one of them rejects an item. This prints the
whole surface at once so borderline cases are reviewed as a list rather than discovered one
rejection at a time.

Read it against the criterion in ``evals/verify_items``: a named entity is a proper noun for
a *specific* dataset, model, benchmark or system. Anything here naming a category rather
than a thing belongs in ``GENERIC_VOCABULARY``.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.multihop import paper_text
from evals.schema import EvalSet, Stratum
from evals.verify_items import GENERIC_VOCABULARY, entities_in, false_premises


def main() -> int:
    evalset = EvalSet.read(Path("evals/datasets/phase4.json"))
    asserted: collections.Counter[str] = collections.Counter()
    flagged: list[tuple[str, list[str]]] = []
    by_item: dict[str, set[str]] = {}

    for item in evalset.items:
        if item.stratum.expects_refusal or item.stratum is Stratum.AMBIGUOUS:
            continue
        found = entities_in(item.question)
        by_item[item.item_id] = found
        asserted.update(found)
        papers = [paper_text(p, 40_000) for p in item.provenance.source_paper_ids]
        if papers:
            missing = false_premises(item.question, papers)
            if missing:
                flagged.append((item.item_id, missing))

    print(f"{len(by_item)} answerable items; {len(asserted)} distinct entities asserted\n")
    print("every entity the premise check treats as a name (count, term):")
    for entity, count in asserted.most_common():
        print(f"  {count:3d}  {entity}")

    print(f"\nexcluded as generic vocabulary ({len(GENERIC_VOCABULARY)} terms):")
    print("  " + ", ".join(sorted(GENERIC_VOCABULARY)))

    print(f"\nitems whose premise fails ({len(flagged)}):")
    for item_id, missing in flagged:
        print(f"  {item_id}: {missing}")
    if not flagged:
        print("  none")

    print(
        "\nReview: anything above naming a *category* rather than a specific dataset, model, "
        "benchmark or system belongs in GENERIC_VOCABULARY."
    )
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
