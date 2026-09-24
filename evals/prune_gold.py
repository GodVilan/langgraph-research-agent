"""Prune gold sets to the minimal jointly-supporting set, and drop rejected items.

Backs `make prune-gold`. Two operations, deliberately asymmetric:

**Pruning removes, never adds.** Selecting the top three chunks per paper grew multi-hop gold
from 3-4 chunks to 5-6, and the surplus is *interchangeable* rather than dead — three chunks of
one paper each supporting the same four benchmark names. `unsupported_chunks` cannot see that,
because each one supports something. Recall@k divides by the size of the gold set, so
interchangeable evidence turns one retrieval into a fractional credit and dilutes MRR@k in the
retriever's favour.

**Re-selection is reported, not applied.** A human verified specific gold chunks; silently
swapping them would invalidate that verification while the item still reads as accepted. So
where the corrected ineligibility rule newly admits a chunk carrying a claim the pruned gold
cannot support, this *names* the item and stops. Re-selection is the reviewer's call, and a
rejection citing several defects is not cleared by fixing one of them.

Usage:
    python -m evals.prune_gold --drop mh-001,mh-004,mh-014
    python -m evals.prune_gold --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.absence import load_corpus
from evals.build_set import chunk_index, gold_chunks_for
from evals.schema import EvalItem, EvalSet, GoldSupport, Stratum
from evals.verify_items import (
    claims_in,
    claims_supported_by,
    gold_chunks_support_jointly,
    minimal_gold,
)


def support_class(item: EvalItem, texts: dict[str, str]) -> GoldSupport:
    """Whether the gold set grounds the answer strongly or only on the token fallback."""
    ok, why = gold_chunks_support_jointly(
        item.gold_answer, [texts[g] for g in item.gold_chunk_ids if g in texts]
    )
    return GoldSupport.WEAK if (not ok or "WEAK" in why) else GoldSupport.STRONG


def better_chunk_available(item: EvalItem, texts: dict[str, str], index: object) -> list[str]:
    """Chunks selection would now choose that carry a claim the current gold cannot.

    The corrected reference-list rule un-bars 274 chunks, among them the table headed "Language
    models used as starting checkpoints for maze training" and the passage reading "reduce the
    dimensionality from 1024 to 100" — both of which a human named as the correct gold for items
    whose selector had been forbidden to see them.
    """
    claims = claims_in(item.gold_answer)
    covered = claims_supported_by(
        " ".join(texts[g] for g in item.gold_chunk_ids if g in texts), claims
    )
    outstanding = {
        (kind, value) for kind, values in claims.items() for value in values - covered[kind]
    }
    better: list[str] = []
    for paper_id in item.provenance.source_paper_ids:
        for chunk_id in gold_chunks_for(item.gold_answer, paper_id, index):  # type: ignore[arg-type]
            if chunk_id in item.gold_chunk_ids or chunk_id not in texts:
                continue
            support = claims_supported_by(texts[chunk_id], claims)
            gains = {(k, v) for k, values in support.items() for v in values} & outstanding
            if gains:
                better.append(chunk_id)
    return sorted(set(better))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--drop", default="", help="comma-separated item ids to remove")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    evalset = EvalSet.read(args.path)
    texts = {str(c["chunk_id"]): str(c["text"]) for c in load_corpus().chunks}
    index = chunk_index()
    dropped = {i.strip() for i in args.drop.split(",") if i.strip()}

    kept: list[EvalItem] = []
    removed_chunks = 0
    needs_review: dict[str, list[str]] = {}
    for item in evalset.items:
        if item.item_id in dropped:
            print(f"dropped {item.item_id} ({item.stratum.value})")
            continue
        # Ambiguous gold chunks are competing referents, not answer support: there is no claim
        # to minimise against, and pruning them would delete a rival reading.
        if item.stratum is Stratum.AMBIGUOUS or not item.gold_answer:
            kept.append(item)
            continue
        present = {g: texts[g] for g in item.gold_chunk_ids if g in texts}
        minimal = minimal_gold(item.gold_answer, present, list(item.provenance.source_paper_ids))
        surplus = [g for g in item.gold_chunk_ids if g not in minimal]
        if surplus:
            removed_chunks += len(surplus)
            was = len(item.gold_chunk_ids)
            print(f"{item.item_id}: {was} -> {len(minimal)} gold; dropped {surplus}")
        pruned = item.model_copy(update={"gold_chunk_ids": minimal})
        pruned = pruned.model_copy(update={"gold_support": support_class(pruned, texts)})
        better = better_chunk_available(pruned, texts, index)
        if better:
            needs_review[item.item_id] = better
        kept.append(pruned)

    print(f"\nitems: {len(evalset.items)} -> {len(kept)}; gold chunks removed: {removed_chunks}")
    if needs_review:
        print("\nRE-SELECTION AVAILABLE — a reviewer decides, this tool does not:")
        for item_id, chunks in sorted(needs_review.items()):
            print(f"  {item_id}: {chunks} carry claims the pruned gold cannot support")

    rebuilt = EvalSet(name=evalset.name, corpus_sha256=evalset.corpus_sha256, items=kept)
    print(f"\ncounts: {rebuilt.counts()}")
    print(f"gold support: {rebuilt.gold_support_counts()}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    rebuilt.write(args.path)
    print(f"\nwrote {args.path} ({len(rebuilt.items)} items, sha {rebuilt.sha256[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
