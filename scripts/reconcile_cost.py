"""Reconcile our notional cost against Langfuse's own figure, trace by trace.

Backs `make reconcile-cost`. This exists because two cost figures disagreed and the gap was
logged as "different rate cards" — which is only a valid explanation if both figures are
internally consistent. One of them was not, and the reason was not a rate card at all.

What it checks, in order of how much each one is worth:

1. **Three-way agreement on priced traces.** For every trace carrying a non-zero notional
   cost, it recomputes the cost from that trace's own token counts at the rates in
   ``src/config.PRICING`` and compares against both the stored figure and Langfuse's
   independently-computed ``total_cost``. Three numbers derived three ways must agree.

2. **The blended-rate invariant.** Total notional divided by total tokens must land between
   the input and output per-1M rates. No token mix can put it outside that interval, so a
   blend below the input floor is proof that the cost and the tokens describe *different
   populations of traces* — which is exactly what had happened.

3. **Population census.** Traces are classified rather than summed blindly:
   priced / unpriced-with-tokens / no-usage-metadata. Synthetic traces from the test suite
   land in the second bucket; pre-instrumentation and duplicate-root traces land in the
   third. Reporting the buckets separately is the whole point — merging them is the bug.

4. **Duplicate-root detection.** The pre-fix Langfuse integration emitted two roots per
   query, one holding the cost and one holding the metadata (see
   ``src/observability/langfuse._set_trace_attributes``). Those pairs still sit in any
   window wide enough to include them and still inflate the dashboard's total.

Exits non-zero when a check fails, so it can gate a release the way `make check` does.

The live store was reset at the start of Phase 4 to give the eval runs a clean window, so
the window this finding was derived from is committed as a checksummed fixture at
``data/traces_d021.json``. `--from-fixture` reconciles against it, and the test suite asserts
its numbers. A documented finding whose evidence has been deleted is exactly what AUDIT §5
catalogues in v2.1; this is the same mistake declined.

Usage:
    python scripts/reconcile_cost.py                 # live store, last 30 days
    python scripts/reconcile_cost.py --days 7
    python scripts/reconcile_cost.py --from-fixture   # reproduce D-021 (make reconcile-d021)
    python scripts/reconcile_cost.py --dump           # capture the live window to a fixture
    python scripts/reconcile_cost.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelPricing, get_settings
from src.observability.config import get_observability_settings
from src.observability.langfuse import get_client

# Two roots for one query land within a few hundred milliseconds of each other; the pre-fix
# pairs observed here were 6-20ms apart. One second is generous without being loose enough
# to pair two genuinely separate queries.
DUPLICATE_WINDOW_S = 1.0
TOLERANCE_USD = 1e-9

# The committed evidence for D-021. Reconciling against this rather than the live store is
# what lets the finding outlive `make langfuse-reset`.
FIXTURE = Path(__file__).resolve().parent.parent / "data" / "traces_d021.json"


def fetch_traces(days: int) -> list[Any]:
    """Page through every trace in the window, or exit if Langfuse is unreachable."""
    client = get_client()
    if client is None:
        settings = get_observability_settings()
        print(
            "Langfuse is not configured, so there is nothing to reconcile.\n"
            f"  host: {settings.langfuse_host}\n"
            "  Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env and start the stack "
            "with `make langfuse-up`.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    since = datetime.now(UTC) - timedelta(days=days)
    traces: list[Any] = []
    page = 1
    while True:
        response = client.api.trace.list(from_timestamp=since, page=page, limit=100)
        batch = getattr(response, "data", []) or []
        traces.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return traces


def to_fixture_row(trace: Any) -> dict[str, Any]:
    """Reduce a Langfuse trace to the fields this reconciliation reads.

    Deliberately lossy. Storing whole traces would commit answer text and prompt bodies for
    no analytical gain; these eight fields are the entire evidential basis of D-021.
    """
    metadata = getattr(trace, "metadata", None) or {}
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    trace_input = getattr(trace, "input", None)
    question = trace_input.get("question") if isinstance(trace_input, dict) else None
    if isinstance(question, str):
        # Only used to pair duplicate roots, and the input-length guardrail test contributes
        # 5,000-character strings that would otherwise be most of the fixture. Two questions
        # would have to share a 120-character prefix *and* land within a second of each
        # other to pair wrongly.
        question = question[:120]
    return {
        "id": trace.id,
        "name": trace.name,
        "timestamp": trace.timestamp.isoformat(),
        "session_id": getattr(trace, "session_id", None),
        "environment": getattr(trace, "environment", None),
        "total_cost": float(getattr(trace, "total_cost", 0) or 0),
        "usage": usage if isinstance(usage, dict) else None,
        "question": question,
    }


def _payload_checksum(rows: list[dict[str, Any]]) -> str:
    """Checksum the trace rows alone, so the header can carry it without self-reference."""
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dump_fixture(traces: list[Any], path: Path, days: int) -> dict[str, Any]:
    """Freeze the trace window to a committed JSON fixture.

    D-021 is a finding about 214 specific traces. Resetting the store to give Phase 4 a
    clean window would destroy the only evidence for it — leaving a documented result whose
    backing data no longer exists, which is precisely the failure AUDIT §5 catalogues in
    v2.1 (a README metric whose dataset was deleted). So the evidence is committed before
    the store is reset, and the finding stays reproducible from the fixture forever.
    """
    rows = sorted((to_fixture_row(t) for t in traces), key=lambda r: (r["timestamp"], r["id"]))
    fixture = {
        "schema_version": 1,
        "generated_by": "scripts/reconcile_cost.py --dump",
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": days,
        "n_traces": len(rows),
        "sha256": _payload_checksum(rows),
        "description": (
            "Langfuse trace window backing DECISIONS D-021. Captured before "
            "`make langfuse-reset` cleared the store for Phase 4. Reduced to the fields the "
            "reconciliation reads; no answer text or prompt bodies."
        ),
        "traces": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    return fixture


def load_fixture(path: Path) -> list[Any]:
    """Rehydrate the fixture into objects with the attribute surface the checks expect.

    The checksum is verified on load: a fixture that has been edited by hand is not
    evidence, and silently reconciling against a tampered file would be worse than failing.
    """
    fixture = json.loads(path.read_text(encoding="utf-8"))
    rows = fixture["traces"]

    actual = _payload_checksum(rows)
    if actual != fixture["sha256"]:
        raise SystemExit(
            f"{path} failed its checksum: expected {fixture['sha256'][:16]}…, got "
            f"{actual[:16]}…. The fixture has been modified since it was captured, so it is "
            f"no longer evidence of anything. Restore it from git."
        )

    return [
        SimpleNamespace(
            id=row["id"],
            name=row["name"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            session_id=row["session_id"],
            environment=row["environment"],
            total_cost=row["total_cost"],
            metadata={"usage": row["usage"]} if row["usage"] is not None else {},
            input={"question": row["question"]} if row["question"] else None,
        )
        for row in rows
    ]


def _usage(trace: Any) -> dict[str, Any] | None:
    metadata = getattr(trace, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return None
    usage = metadata.get("usage")
    return usage if isinstance(usage, dict) else None


def notional_for(
    input_tokens: int, output_tokens: int, pricing: ModelPricing, cached_input_tokens: int = 0
) -> float:
    """Price tokens the same way ``src/agent/llm.usage_from_message`` does: the cached part of
    the prompt at the cached rate (the full input rate when that is unverified)."""
    cached = min(cached_input_tokens, input_tokens)
    cached_rate = (
        pricing.notional_cached_input_usd
        if pricing.notional_cached_input_usd is not None
        else pricing.notional_input_usd
    )
    return (
        (input_tokens - cached) * pricing.notional_input_usd
        + cached * cached_rate
        + output_tokens * pricing.notional_output_usd
    ) / 1_000_000


def rate_floor(pricing: ModelPricing, input_tokens: float = 0.0, cached: float = 0.0) -> float:
    """The lowest blended rate these tokens could carry: their input at the input rate, with
    the *recorded* cached share at the cached rate.

    Not simply the cached rate. Using $0.03 as the floor for every token would make the D-021
    blend ($0.10/1M, impossible for uncached input) look possible and silently disable the
    check that caught it. A population with no recorded cached tokens keeps the $0.30 floor.
    """
    if not input_tokens or not cached or pricing.notional_cached_input_usd is None:
        return pricing.notional_input_usd
    share = min(cached, input_tokens) / input_tokens
    return share * pricing.notional_cached_input_usd + (1 - share) * pricing.notional_input_usd


def classify(traces: list[Any], pricing: ModelPricing) -> dict[str, Any]:
    """Split traces into populations that can honestly be summed together."""
    priced: list[dict[str, Any]] = []
    unpriced: list[dict[str, Any]] = []
    no_usage: list[dict[str, Any]] = []

    for trace in traces:
        row = {
            "id": trace.id,
            "name": trace.name,
            "timestamp": trace.timestamp,
            "session_id": getattr(trace, "session_id", None),
            "langfuse_cost": float(getattr(trace, "total_cost", 0) or 0),
            "question": (trace.input or {}).get("question")
            if isinstance(getattr(trace, "input", None), dict)
            else None,
        }
        usage = _usage(trace)
        if usage is None:
            no_usage.append(row)
            continue

        row["input_tokens"] = int(usage.get("input_tokens") or 0)
        row["output_tokens"] = int(usage.get("output_tokens") or 0)
        row["cached_input_tokens"] = int(usage.get("cached_input_tokens") or 0)
        row["stored_notional"] = float(usage.get("cost_usd_notional") or 0)
        row["recomputed_notional"] = notional_for(
            row["input_tokens"],
            row["output_tokens"],
            pricing,
            int(usage.get("cached_input_tokens") or 0),  # absent on pre-D-046 traces: 0
        )
        (priced if row["stored_notional"] > 0 else unpriced).append(row)

    return {"priced": priced, "unpriced": unpriced, "no_usage": no_usage}


def find_duplicate_roots(traces: list[Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair traces that are two roots of one query — the pre-fix double-trace symptom.

    The tell is a ``query`` root and a ``LangGraph`` root, seconds apart, carrying the same
    question, where the cost sits on one and the metadata on the other.
    """
    rows = [
        {
            "id": t.id,
            "name": t.name,
            "timestamp": t.timestamp,
            "cost": float(getattr(t, "total_cost", 0) or 0),
            "question": (t.input or {}).get("question")
            if isinstance(getattr(t, "input", None), dict)
            else None,
        }
        for t in traces
    ]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[str] = set()
    for a in sorted(rows, key=lambda r: r["timestamp"]):
        if a["name"] != "query" or a["id"] in used or not a["question"]:
            continue
        for b in rows:
            if b["name"] != "LangGraph" or b["id"] in used or b["question"] != a["question"]:
                continue
            if abs((b["timestamp"] - a["timestamp"]).total_seconds()) <= DUPLICATE_WINDOW_S:
                pairs.append((a, b))
                used.update({a["id"], b["id"]})
                break
    return pairs


def report(groups: dict[str, Any], duplicates: list[Any], pricing: ModelPricing) -> list[str]:
    """Run every check. Returns the list of failures; empty means reconciled."""
    failures: list[str] = []
    priced = groups["priced"]

    # ── 1. three-way agreement ────────────────────────────────────────────────
    stored = sum(r["stored_notional"] for r in priced)
    recomputed = sum(r["recomputed_notional"] for r in priced)
    langfuse_priced = sum(r["langfuse_cost"] for r in priced)
    disagreeing = [
        r for r in priced if abs(r["stored_notional"] - r["recomputed_notional"]) > TOLERANCE_USD
    ]

    print(
        f"Rates in use: ${pricing.notional_input_usd}/1M input, "
        f"${pricing.notional_output_usd}/1M output (verified={pricing.verified})"
    )
    print(f"  source: {pricing.source}\n")

    print(f"Priced traces: {len(priced)}")
    print(f"  stored notional      ${stored:.5f}")
    print(f"  recomputed at rates  ${recomputed:.5f}")
    print(f"  Langfuse total_cost  ${langfuse_priced:.5f}")
    if disagreeing:
        failures.append(
            f"{len(disagreeing)} priced traces disagree with a recompute at the configured "
            f"rates — the calculator and the rate table are out of step"
        )

    # ── 2. blended-rate invariant ─────────────────────────────────────────────
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in priced)
    if tokens:
        blended = stored / tokens * 1_000_000
        low = rate_floor(
            pricing,
            sum(r["input_tokens"] for r in priced),
            sum(r.get("cached_input_tokens", 0) for r in priced),
        )
        high = pricing.notional_output_usd
        print(f"  tokens {tokens:,} -> blended ${blended:.4f}/1M (must sit in ${low}-${high})")
        if not low - 1e-6 <= blended <= high + 1e-6:
            failures.append(
                f"blended rate ${blended:.4f}/1M falls outside ${low}-${high}/1M. No token "
                f"mix can do that, so the cost and the tokens describe different traces"
            )

    # ── 3. population census ──────────────────────────────────────────────────
    unpriced, no_usage = groups["unpriced"], groups["no_usage"]
    unpriced_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in unpriced)
    print(f"\nUnpriced traces (usage metadata, zero cost): {len(unpriced)}")
    print(f"  carrying {unpriced_tokens:,} tokens that must never be summed with the cost above")
    print(f"No-usage traces: {len(no_usage)}")
    print(f"  holding ${sum(r['langfuse_cost'] for r in no_usage):.5f} of Langfuse-computed cost")

    # ── 4. duplicate roots ────────────────────────────────────────────────────
    dup_cost = sum(b["cost"] for _, b in duplicates)
    print(f"\nDuplicate-root pairs: {len(duplicates)} (${dup_cost:.5f} on the orphan roots)")
    for a, b in duplicates:
        print(
            f"  {a['timestamp']:%Y-%m-%d %H:%M:%S}  {a['id'][:12]} (query, $0) "
            f"↔ {b['id'][:12]} (LangGraph, ${b['cost']:.5f})  {str(a['question'])[:44]}"
        )

    total_langfuse = langfuse_priced + sum(r["langfuse_cost"] for r in no_usage)
    total_langfuse += sum(r["langfuse_cost"] for r in unpriced)
    print(f"\nLangfuse dashboard total over the window: ${total_langfuse:.5f}")
    print(f"  of which reconciled against our own instrumentation: ${langfuse_priced:.5f}")
    print(f"  of which duplicate roots of already-counted runs:    ${dup_cost:.5f}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    parser.add_argument(
        "--from-fixture",
        nargs="?",
        const=str(FIXTURE),
        metavar="PATH",
        help="Reconcile against the committed trace fixture instead of the live store. "
        "This is what keeps D-021 reproducible after `make langfuse-reset`.",
    )
    parser.add_argument(
        "--dump",
        nargs="?",
        const=str(FIXTURE),
        metavar="PATH",
        help="Capture the live trace window to a fixture, then reconcile against it.",
    )
    args = parser.parse_args()

    pricing = get_settings().pricing()

    if args.from_fixture:
        path = Path(args.from_fixture)
        if not path.exists():
            print(f"No fixture at {path}. Capture one with --dump.", file=sys.stderr)
            return 2
        traces = load_fixture(path)
        source = f"fixture {path}"
    else:
        traces = fetch_traces(args.days)
        source = f"the live store, last {args.days} days"
        if args.dump:
            fixture = dump_fixture(traces, Path(args.dump), args.days)
            print(
                f"Wrote {args.dump}: {fixture['n_traces']} traces, "
                f"sha256 {fixture['sha256'][:16]}…\n"
            )
            source = f"fixture {args.dump}"

    groups = classify(traces, pricing)
    duplicates = find_duplicate_roots(traces)

    if args.json:
        print(
            json.dumps(
                {
                    "traces": len(traces),
                    "priced": len(groups["priced"]),
                    "unpriced": len(groups["unpriced"]),
                    "no_usage": len(groups["no_usage"]),
                    "duplicate_root_pairs": len(duplicates),
                    "stored_notional": sum(r["stored_notional"] for r in groups["priced"]),
                    "recomputed_notional": sum(r["recomputed_notional"] for r in groups["priced"]),
                    "langfuse_cost_priced": sum(r["langfuse_cost"] for r in groups["priced"]),
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"Reconciling {len(traces)} traces from {source}\n")
    failures = report(groups, duplicates, pricing)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    if not groups["priced"]:
        # An empty store passes every check above trivially. Saying "reconciled" here would
        # be a false green of exactly the kind this script exists to catch, so it says what
        # actually happened instead.
        print(
            "\nNothing to reconcile: no priced traces in this window. This is a vacuous "
            "pass, not a verification.\n"
            "  The D-021 finding is reproducible from the committed fixture:\n"
            "    python scripts/reconcile_cost.py --from-fixture"
        )
        return 0

    print("\nReconciled: stored, recomputed, and Langfuse agree on every priced trace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
