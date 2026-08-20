"""Run the injection detector over the whole committed corpus.

Backs `make screen-corpus`. Pure regex over 5,401 chunks — no LLM calls, no API spend,
seconds of local compute.

This is the only false-positive number that means anything. A hand-written benign set says
what the author thought to write down; the corpus says what the rules actually do to real
PDF-extracted text. Any rule whose hits here are predominantly legitimate academic prose is
a broken rule.

Usage:
    python scripts/screen_corpus.py                    # summary
    python scripts/screen_corpus.py --rule letter_spacing --limit 20
    python scripts/screen_corpus.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.guardrails.injection import Severity, scan, worst_severity
from src.retrieval.chunker import load_chunks

EXCERPTS_PER_RULE = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=str, default=None, help="Show every hit for one rule")
    parser.add_argument("--limit", type=int, default=EXCERPTS_PER_RULE)
    parser.add_argument("--json", type=Path, default=None, help="Write the full result as JSON")
    args = parser.parse_args()

    settings = get_settings()
    chunks = load_chunks(settings.chunks_path, classify_on_load=False)

    by_rule: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    excerpts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    flagged_chunks: set[str] = set()
    quarantined_corpus: set[str] = set()
    quarantined_strict: set[str] = set()

    for chunk in chunks:
        detections = scan(chunk.text, strict=False)
        strict_detections = scan(chunk.text, strict=True)

        if detections:
            flagged_chunks.add(chunk.chunk_id)
        if worst_severity(detections) is Severity.BLOCK:
            quarantined_corpus.add(chunk.chunk_id)
        if worst_severity(strict_detections) is Severity.BLOCK:
            quarantined_strict.add(chunk.chunk_id)

        for d in detections:
            by_rule[d.pattern] += 1
            by_severity[d.severity.value] += 1
            by_category[d.category.value] += 1
            excerpts[d.pattern].append((chunk.chunk_id, d.excerpt))

    total = len(chunks)

    if args.rule:
        hits = excerpts.get(args.rule, [])
        print(f"rule {args.rule!r}: {len(hits)} hit(s) over {total} chunks\n")
        for chunk_id, excerpt in hits[: args.limit]:
            print(f"  {chunk_id}  {excerpt!r}")
        return 0

    print("CORPUS SCREEN — injection detector over the committed corpus")
    print("=" * 78)
    print(f"chunks screened                  : {total:,}")
    print(
        f"chunks with >=1 detection        : {len(flagged_chunks):,}"
        f"  ({len(flagged_chunks) / total:.2%})"
    )
    print(
        f"would be quarantined (corpus)    : {len(quarantined_corpus):,}"
        f"  ({len(quarantined_corpus) / total:.2%})"
    )
    print(
        f"would be quarantined (strict)    : {len(quarantined_strict):,}"
        f"  ({len(quarantined_strict) / total:.2%})"
    )
    print()

    if not by_rule:
        print("No rule fired on any chunk.")
    else:
        print(f"{'rule':<28} {'hits':>7}  {'severity':<8} category")
        print("-" * 78)
        from src.guardrails.injection import RULES

        meta = {r.name: (r.severity.value, r.category.value) for r in RULES}
        meta["invisible_characters"] = ("block", "encoding_evasion")
        meta["letter_spacing"] = ("block", "encoding_evasion")
        for rule, count in by_rule.most_common():
            severity, category = meta.get(rule, ("?", "?"))
            print(f"{rule:<28} {count:>7,}  {severity:<8} {category}")
        print()
        print(f"by severity: {dict(by_severity)}")
        print(f"by category: {dict(by_category)}")
        print()

        print(f"TOP EXCERPTS (up to {args.limit} most frequent shapes per rule)")
        print("=" * 78)
        for rule, _ in by_rule.most_common():
            print(f"\n{rule}:")
            shapes = Counter(e for _, e in excerpts[rule])
            for excerpt, n in shapes.most_common(args.limit):
                marker = f" (x{n})" if n > 1 else ""
                print(f"  {excerpt[:110]!r}{marker}")

    if args.json:
        payload: dict[str, Any] = {
            "chunks_screened": total,
            "chunks_flagged": len(flagged_chunks),
            "quarantined_corpus_policy": len(quarantined_corpus),
            "quarantined_strict_policy": len(quarantined_strict),
            "by_rule": dict(by_rule),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "excerpts": {r: v[: args.limit] for r, v in excerpts.items()},
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
