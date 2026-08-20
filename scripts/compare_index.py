"""Compare two FAISS indexes built from the same chunks.

Answers the question `make index-verify` raises: a rebuild is not bit-identical across
torch/transformers versions or across devices, so is it *equivalent*?

Reports element-wise divergence, cosine similarity, and — the part that actually matters —
whether retrieval rankings change. Reads only the `.faiss` files, so it needs neither the
pickled sidecar nor the embedding model.

Usage:
    python scripts/compare_index.py data/indices/BGE_cs512.faiss other/BGE_cs512.faiss
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--probes", type=int, default=500, help="Probe queries to compare")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--k", type=int, nargs="+", default=[5, 10], help="Ranking depths to compare"
    )
    args = parser.parse_args()

    left = faiss.read_index(str(args.left))
    right = faiss.read_index(str(args.right))

    if (left.ntotal, left.d) != (right.ntotal, right.d):
        print(
            f"Shape mismatch: {left.ntotal}x{left.d} vs {right.ntotal}x{right.d} — "
            f"these indexes were not built from the same chunks."
        )
        return 1

    a = left.reconstruct_n(0, left.ntotal)
    b = right.reconstruct_n(0, right.ntotal)
    diff = np.abs(a - b)
    cos = (a * b).sum(axis=1)

    print(f"vectors            : {left.ntotal} x {left.d}")
    print(f"bit-identical rows : {int((diff.max(axis=1) == 0).sum())} / {len(a)}")
    print(f"max |elem diff|    : {diff.max():.3e}")
    print(f"mean |elem diff|   : {diff.mean():.3e}")
    print(f"cosine similarity  : min={cos.min():.8f}  mean={cos.mean():.8f}")

    rng = np.random.default_rng(args.seed)
    probes = a[rng.choice(left.ntotal, min(args.probes, left.ntotal), replace=False)]

    print(f"\nranking agreement over {len(probes)} probe queries (seed={args.seed}):")
    divergent = 0
    for k in args.k:
        _, ia = left.search(probes, k)
        _, ib = right.search(probes, k)
        ordering = int((ia == ib).all(axis=1).sum())
        same_set = sum(1 for r in range(len(probes)) if set(ia[r]) == set(ib[r]))
        divergent += len(probes) - ordering
        print(
            f"  k={k:<3} identical ordering {ordering}/{len(probes)}  "
            f"identical set {same_set}/{len(probes)}"
        )

    if divergent:
        print(f"\n{divergent} ranking difference(s) — the indexes are NOT interchangeable.")
        return 1
    print("\nRankings agree at every depth checked; the indexes are interchangeable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
