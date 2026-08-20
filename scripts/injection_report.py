"""Report the injection guardrail's detection and false-positive rates.

Backs `make injection-report`. Prints the numbers that go in the README, so the reported
rate cannot drift from the corpus the way a hand-typed figure would.

Reports detection rate *and* false-positive rate together, because either one alone is
trivially gamed: a detector that flags everything scores 100% detection, and one that flags
nothing scores 0% false positives.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guardrails.injection import Severity, scan, worst_severity
from tests.adversarial_corpus import ADVERSARIAL, BENIGN


def main() -> int:
    expected = [c for c in ADVERSARIAL if c.expected_caught]
    gaps = [c for c in ADVERSARIAL if not c.expected_caught]

    detected = [c for c in expected if scan(c.text)]
    blocked = [c for c in expected if worst_severity(scan(c.text)) is Severity.BLOCK]
    leaked = [c for c in gaps if scan(c.text)]
    strict_blocked = [
        c for c in expected if worst_severity(scan(c.text, strict=True)) is Severity.BLOCK
    ]
    # A WARN on benign text changes nothing — the chunk still reaches the model. Only a
    # BLOCK is a real false positive, because it silently withholds evidence.
    benign_blocked = [c for c in BENIGN if worst_severity(scan(c.text)) is Severity.BLOCK]
    benign_warned = [c for c in BENIGN if scan(c.text) and c not in benign_blocked]
    undeclared = [c for c in benign_warned if not c.expected_warn]

    total = len(ADVERSARIAL)
    caught_overall = len(detected) + len(leaked)

    print("INJECTION GUARDRAIL — DETECTION REPORT")
    print("=" * 72)
    print(
        f"adversarial corpus        : n={total} ({len(expected)} targeted, {len(gaps)} known gaps)"
    )
    print(f"benign corpus             : n={len(BENIGN)}")
    print()
    print(
        f"detected (targeted)       : {len(detected)}/{len(expected)}"
        f"  ({len(detected) / len(expected):.0%})"
    )
    print(f"  quarantined, corpus policy: {len(blocked)}/{len(expected)}")
    print(f"  quarantined, strict policy: {len(strict_blocked)}/{len(expected)}")
    print(f"detected (whole corpus)   : {caught_overall}/{total}  ({caught_overall / total:.0%})")
    print()
    print(f"benign quarantined (BLOCK): {len(benign_blocked)}/{len(BENIGN)}   <- the real FP rate")
    print(f"benign warned (declared)  : {len(benign_warned)}/{len(BENIGN)}")
    print(f"benign warned (undeclared): {len(undeclared)}/{len(BENIGN)}")
    print()
    print("Real-corpus false positives are the number that matters; run `make screen-corpus`.")
    print("Last measured over all 5,401 committed chunks: 0 quarantined (0.00%),")
    print("60 warned (1.11%). See docs/SCREEN.md.")
    print()

    if benign_blocked or undeclared:
        print("BENIGN FAILURES:")
        for case in benign_blocked:
            print(f"  BLOCK {case.name}: {[d.pattern for d in scan(case.text)]}")
        for case in undeclared:
            print(f"  UNDECLARED WARN {case.name}: {[d.pattern for d in scan(case.text)]}")
        print()

    print(f"KNOWN GAPS ({len(gaps)}) — undetected by design, kept in the corpus:")
    for case in gaps:
        status = "NOW DETECTED (update the corpus)" if scan(case.text) else "undetected"
        print(f"  {case.name:<26} {status}")
        if case.note:
            print(f"      {case.note.strip()[:150]}")
    print()
    print("Caveat: the targeted cases and the rules were written by the same author, so the")
    print("targeted rate measures internal consistency more than robustness. The known-gap")
    print("set exists because that number would otherwise be meaningless. Structural")
    print("neutralisation, not detection, is the layer that holds against unseen attacks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
