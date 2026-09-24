"""Print gold chunks in full, and diff a draft's gold against an earlier revision's.

Backs `make gold-report`. Two jobs, both of which human verification asked for by hand:

**Confirmation.** Human review of multi-hop items reported three cases where "no gold chunk
visibly contains the claim". In all three the claim was present and the excerpt shown by
`evals/verify_cli.py` had truncated it — 1,634 sits 900 characters into its chunk, "Llama 3.2
1B" is a table row, "Qwen2.5 7B" is the last sentence of a hyperparameter appendix. A reviewer
cannot confirm containment against a window, so this prints the whole chunk and locates every
claim string inside it with its surrounding context.

**Regression.** Gold selection has been rewritten twice, and the second rewrite *lost* chunks
the first had chosen correctly. Item ids are positional and unstable across rebuilds, so the
diff keys on the source paper pair, which is stable, and reports for every dropped chunk
whether it supports a claim the retained set does not — and whether the structural
ineligibility rule is what made it unselectable.

    python scripts/gold_report.py --stratum multi_hop
    python scripts/gold_report.py --against ee95155
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.absence import load_corpus
from evals.schema import EvalItem, EvalSet
from evals.verify_items import (
    claims_in,
    claims_supported_by,
    is_ineligible_gold,
    minimal_gold,
)

CONTEXT = 110


def corpus_texts() -> dict[str, str]:
    return {str(c["chunk_id"]): str(c["text"]) for c in load_corpus().chunks}


def flat(text: str) -> str:
    return " ".join(text.split())


def locate(text: str, needle: str) -> list[str]:
    """Every occurrence of a claim string with its surrounding context."""
    body = flat(text)
    windows = []
    for match in re.finditer(re.escape(needle), body, re.IGNORECASE):
        start = max(0, match.start() - CONTEXT)
        windows.append(f"…{body[start : match.end() + CONTEXT]}…")
    return windows[:2]


def confirm(item: EvalItem, texts: dict[str, str]) -> None:
    """Print every gold chunk of one item in full, with its claims located."""
    claims = claims_in(item.gold_answer)
    wanted = [(kind, value) for kind, values in claims.items() for value in sorted(values)]
    print("=" * 100)
    print(f"{item.item_id}  {item.stratum.value}  papers {item.provenance.source_paper_ids}")
    print(f"\nQ: {item.question}")
    print(f"A: {item.gold_answer}")
    print(f"\nclaims to confirm ({len(wanted)}): {wanted or 'none extractable'}")

    covered: set[tuple[str, str]] = set()
    for chunk_id in item.gold_chunk_ids:
        text = texts.get(chunk_id, "")
        if not text:
            print(f"\n--- {chunk_id}: NOT IN CORPUS ---")
            continue
        support = claims_supported_by(text, claims)
        here = [(kind, v) for kind, values in support.items() for v in sorted(values)]
        covered |= set(here)
        print(
            f"\n--- {chunk_id}  ({len(text)} chars, eligible: {is_ineligible_gold(text) or 'yes'})"
        )
        print(f"    supports: {here or 'NOTHING'}")
        for kind, value in here:
            for window in locate(text, value):
                print(f"      {kind} {value!r}: {window}")
        print(f"\n    FULL TEXT:\n{flat(text)}\n")

    unconfirmed = [c for c in wanted if c not in covered]
    print(f"UNCONFIRMED CLAIMS: {unconfirmed or 'none — every claim string located above'}")


def revision(rev: str, path: Path) -> list[dict[str, Any]]:
    """An older draft's items as raw dictionaries.

    Deliberately *not* validated against today's ``EvalItem``. Every multi-hop item in the
    first draft has an empty ``gold_answer``, which the schema now forbids — so validating
    would make the tool that diagnoses a regression unable to read the revision it regressed
    from. A diff against history has to accept history's shape.
    """
    blob = subprocess.run(
        ["git", "cat-file", "-p", f"{rev}:{path}"], capture_output=True, text=True, check=True
    ).stdout
    items: list[dict[str, Any]] = json.loads(blob)["items"]
    return items


def key(item: EvalItem) -> frozenset[str]:
    """Source papers: the only identity stable across rebuilds. Ids are positional."""
    return frozenset(item.provenance.source_paper_ids)


def old_key(item: dict[str, Any]) -> frozenset[str]:
    return frozenset(item["provenance"]["source_paper_ids"])


def diff_gold(new: EvalSet, old: list[dict[str, Any]], stratum: str, texts: dict[str, str]) -> None:
    """Report, per item, which chunks the rewrite dropped and what they carried."""
    old_by_key = {old_key(i): i for i in old if i["stratum"] == stratum}
    new_items = [i for i in new.items if i.stratum.value == stratum]
    unmatched = [i for i in new_items if key(i) not in old_by_key]

    print("=" * 100)
    print(f"GOLD DIFF, stratum {stratum}: {len(new_items)} current items, {len(old_by_key)} prior")
    print(f"matched on source-paper pair: {len(new_items) - len(unmatched)}")
    culled = len(old_by_key) - len(new_items) + len(unmatched)
    print(f"prior items with no current counterpart (culled): {culled}\n")

    for item in new_items:
        prior = old_by_key.get(key(item))
        if prior is None:
            print(f"{item.item_id}: no prior item on this pair")
            continue
        prior_gold: list[str] = prior["gold_chunk_ids"]
        dropped = [c for c in prior_gold if c not in item.gold_chunk_ids]
        retained = [c for c in item.gold_chunk_ids if c in prior_gold]
        claims = claims_in(item.gold_answer)
        kept_support = claims_supported_by(
            " ".join(texts.get(c, "") for c in item.gold_chunk_ids), claims
        )
        print(f"{item.item_id} (was {prior['item_id']}) {sorted(key(item))}")
        print(f"   prior gold {prior_gold}")
        print(f"   current    {item.gold_chunk_ids}   retained {retained or 'none'}")
        for chunk_id in dropped:
            text = texts.get(chunk_id, "")
            if not text:
                print(f"   dropped {chunk_id}: not in corpus")
                continue
            support = claims_supported_by(text, claims)
            unique = {
                kind: sorted(values - kept_support[kind])
                for kind, values in support.items()
                if values - kept_support[kind]
            }
            barred = is_ineligible_gold(text)
            print(
                f"   dropped {chunk_id}: carries-uniquely={unique or 'nothing'} "
                f"structurally-barred={barred or 'no'}"
            )
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--stratum", default="multi_hop")
    parser.add_argument("--against", default="", help="git revision to diff gold against")
    parser.add_argument("--minimal", action="store_true", help="confirm against the pruned gold")
    args = parser.parse_args()

    texts = corpus_texts()
    evalset = EvalSet.read(args.path)
    items = [i for i in evalset.items if i.stratum.value == args.stratum]

    if args.against:
        diff_gold(evalset, revision(args.against, args.path), args.stratum, texts)
        return 0

    for item in items:
        if args.minimal and item.gold_answer:
            present = {c: texts[c] for c in item.gold_chunk_ids if c in texts}
            item = item.model_copy(
                update={
                    "gold_chunk_ids": minimal_gold(
                        item.gold_answer, present, list(item.provenance.source_paper_ids)
                    )
                }
            )
        confirm(item, texts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
