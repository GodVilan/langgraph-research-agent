"""Deterministic, stratified selection of the 11 absent-attribute anchors.

Backs `make select-attributes`. The selection must be regenerable: an eval set whose items
were picked by an unrecorded process is a set nobody can re-derive, which is the failure
`AUDIT.md` §5 catalogues in v2.1.

The procedure, in order, is:

1. **Enumerate.** For each of the six absence shapes, probe every paper for every term in
   that shape's family and keep the (paper, term, shape) triples where the term is verified
   absent from the whole corpus, not just the anchor paper (`evals/absence.py`).
2. **Sort canonically** by (shape, anchor class, paper id, term). No dependence on dict or
   filesystem ordering.
3. **Prefer strong anchors.** Papers with a distinctive method name are drawn first, because
   the cross-citation check can actually recognise them; only 45 of 150 qualify. A weak
   anchor is used only when a shape cannot be filled from strong ones, and the item records
   ``anchor_class`` either way so the difference stays visible per item.
4. **Round-robin the shapes** so no shape dominates, then fill the remainder by seeded
   shuffle. Eleven items across six shapes cannot be balanced exactly: the split is 5 shapes
   with 2 and 1 with 1, and which shape gets the short straw is decided by the seed rather
   than by whichever happened to be enumerated first.
5. **One item per anchor paper.** Two absences from the same paper share its retrieval
   behaviour and would not be independent measurements.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.absence import _mentions, load_corpus, paper_handles
from evals.schema import AbsenceShape, AnchorClass

SEED = 20260521
TARGET = 11

# Terms whose absence from a paper is a checkable fact, grouped by what kind of absence it
# is. Every term must be one that *some* papers in the corpus do report, or its absence is
# uninformative — an absent term nobody uses is an absent topic, not an absent attribute.
SHAPE_TERMS: dict[AbsenceShape, list[str]] = {
    AbsenceShape.BENCHMARK: ["imagenet", "cifar", "mnist", "gsm8k", "mmlu", "humaneval", "coco"],
    AbsenceShape.DATASET: ["wikitext", "librispeech", "openwebtext", "laion", "pile"],
    AbsenceShape.ABLATION: ["ablation", "ablation study", "leave-one-out"],
    AbsenceShape.BASELINE: ["random baseline", "zero-shot baseline", "majority class"],
    AbsenceShape.HYPERPARAMETER: ["learning rate", "batch size", "weight decay", "warmup"],
    AbsenceShape.COMPUTE: ["gpu hours", "flops", "wall-clock", "a100", "throughput"],
}


@dataclass(frozen=True)
class Candidate:
    paper_id: str
    term: str
    shape: AbsenceShape
    anchor_class: AnchorClass

    @property
    def sort_key(self) -> tuple[str, int, str, str]:
        # Strong anchors sort first within a shape.
        return (
            self.shape.value,
            0 if self.anchor_class is AnchorClass.STRONG else 1,
            self.paper_id,
            self.term,
        )


def anchor_class_of(paper_id: str) -> AnchorClass:
    """Strong when the cross-citation check has a distinctive name to match on."""
    return (
        AnchorClass.STRONG
        if len(paper_handles().get(paper_id, frozenset())) > 1
        else AnchorClass.WEAK
    )


def enumerate_candidates() -> list[Candidate]:
    """Every (paper, term, shape) whose absence survives full-corpus verification.

    Indexed by term rather than looping ``verify_attribute_absent`` over every pair: that
    would be ~4,000 full scans of 5,401 chunks. Each term is scanned once, and the
    cross-citation rule is then applied only to the papers that could be affected — the
    same logic, arranged so it finishes.
    """
    corpus = load_corpus()
    handles = paper_handles()
    candidates: list[Candidate] = []

    for shape, terms in SHAPE_TERMS.items():
        for term in terms:
            needle = term.lower()
            chunks_with_term = [c for c in corpus.chunks if needle in str(c["text"]).lower()]
            # A term no paper reports is an absent *topic*; its absence from a specific
            # paper says nothing about that paper.
            if not chunks_with_term:
                continue

            # Papers another paper attributes this term to. Excluded even though the term is
            # absent from their own text: the corpus would still answer the question.
            attributed: set[str] = set()
            for chunk in chunks_with_term:
                text = str(chunk["text"])
                for paper_id, hs in handles.items():
                    if paper_id == str(chunk["paper_id"]) or paper_id in attributed:
                        continue
                    if any(_mentions(text, h) for h in hs):
                        attributed.add(paper_id)

            for paper_id in sorted(corpus.text_by_paper):
                if needle in corpus.text_by_paper[paper_id] or paper_id in attributed:
                    continue
                candidates.append(Candidate(paper_id, term, shape, anchor_class_of(paper_id)))
    return sorted(candidates, key=lambda c: c.sort_key)


def select(candidates: list[Candidate], target: int = TARGET, seed: int = SEED) -> list[Candidate]:
    """Round-robin across shapes, strong anchors first, one item per paper."""
    by_shape: dict[AbsenceShape, list[Candidate]] = {}
    for candidate in candidates:
        by_shape.setdefault(candidate.shape, []).append(candidate)

    rng = random.Random(seed)
    # Shuffle *within* a shape so the choice among equally-eligible strong anchors is seeded
    # rather than alphabetical, while the strong-before-weak ordering is preserved.
    for shape in by_shape:
        strong = [c for c in by_shape[shape] if c.anchor_class is AnchorClass.STRONG]
        weak = [c for c in by_shape[shape] if c.anchor_class is AnchorClass.WEAK]
        rng.shuffle(strong)
        rng.shuffle(weak)
        by_shape[shape] = strong + weak

    shapes = sorted(by_shape, key=lambda s: s.value)
    rng.shuffle(shapes)  # decides which shape gets the short straw at 11/6

    chosen: list[Candidate] = []
    used_papers: set[str] = set()
    while len(chosen) < target:
        progressed = False
        for shape in shapes:
            if len(chosen) >= target:
                break
            for candidate in by_shape[shape]:
                if candidate.paper_id in used_papers:
                    continue
                chosen.append(candidate)
                used_papers.add(candidate.paper_id)
                by_shape[shape].remove(candidate)
                progressed = True
                break
        if not progressed:
            break  # exhausted; the caller reports the shortfall rather than padding
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    candidates = enumerate_candidates()
    chosen = select(candidates, args.target, args.seed)

    if args.json:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "candidates": len(candidates),
                    "selected": [
                        {
                            "paper_id": c.paper_id,
                            "term": c.term,
                            "shape": c.shape.value,
                            "anchor_class": c.anchor_class.value,
                        }
                        for c in chosen
                    ],
                },
                indent=2,
            )
        )
        return 0

    by_shape: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for c in candidates:
        by_shape[c.shape.value] = by_shape.get(c.shape.value, 0) + 1
    print(f"verified candidates: {len(candidates):,}  (seed {args.seed})")
    for shape, count in sorted(by_shape.items()):
        print(f"  {shape:16} {count:5d}")

    print(f"\nselected {len(chosen)}/{args.target}:")
    for c in chosen:
        by_class[c.anchor_class.value] = by_class.get(c.anchor_class.value, 0) + 1
        print(f"  {c.shape.value:16} {c.paper_id}  {c.anchor_class.value:6} {c.term!r}")
    print(f"\nshape spread: {shape_counts(chosen)}")
    print(f"anchor classes: {by_class}")
    if len(chosen) < args.target:
        print(f"\nSHORTFALL: {args.target - len(chosen)} short, candidates exhausted")
        return 1
    return 0


def shape_counts(chosen: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chosen:
        counts[c.shape.value] = counts.get(c.shape.value, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
