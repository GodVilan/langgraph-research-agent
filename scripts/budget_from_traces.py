"""Generate the spend table in docs/BUDGET.md from Langfuse trace data.

Backs `make budget`. This is the control BUDGET.md already committed to: the spend column
is generated from traces, not typed by hand, so it cannot drift from what was actually
spent — the same failure mode `make readme-stats` exists to prevent for the test count.

Reads traces via the Langfuse API and rewrites the region between the SPEND markers.
Requires a Langfuse key pair; exits non-zero with a clear message when there is none, rather
than writing a table full of zeros that looks like a measurement.

Usage:
    python scripts/budget_from_traces.py                # last 30 days
    python scripts/budget_from_traces.py --days 7
    python scripts/budget_from_traces.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.observability.config import get_observability_settings
from src.observability.langfuse import get_client

REPO = Path(__file__).resolve().parent.parent
BUDGET_MD = REPO / "docs" / "BUDGET.md"
START = "<!-- SPEND:START -->"
END = "<!-- SPEND:END -->"


def fetch_traces(days: int) -> list[Any]:
    client = get_client()
    if client is None:
        settings = get_observability_settings()
        print(
            "Langfuse is not configured, so there are no traces to read.\n"
            f"  host: {settings.langfuse_host}\n"
            "  Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env, and start the "
            "stack with:\n"
            "    docker compose -f infra/docker-compose.langfuse.yml up -d",
            file=sys.stderr,
        )
        return []

    since = datetime.now(UTC) - timedelta(days=days)
    traces: list[Any] = []
    page = 1
    while True:
        try:
            response = client.api.trace.list(from_timestamp=since, page=page, limit=100)
        except Exception as exc:  # a reachability failure is a result, not a crash
            print(f"Could not read traces from Langfuse: {exc}", file=sys.stderr)
            break
        batch = getattr(response, "data", []) or []
        traces.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return traces


def summarise(traces: list[Any]) -> dict[str, Any]:
    """Aggregate per environment, counting only traces whose tokens *and* cost are both real.

    The table used to sum tokens from every trace carrying usage metadata while summing cost
    from the subset that had been priced. Those are two different populations, and merging
    them produced a blended rate of $0.17/1M — below the $0.30 input floor, which no token
    mix can produce. 187 of 214 traces were synthetic runs from the test suite: real token
    counts in the metadata, zero cost, because ``tests/fakes.py`` never priced them.

    So a trace contributes to the token columns only if it also contributes to the cost
    columns. Everything else is counted and reported, never summed in (DECISIONS D-021).
    ``scripts/reconcile_cost.py`` is the check that this stays true.
    """
    rows: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "traces": 0.0,
            "priced": 0.0,
            "unpriced": 0.0,
            "no_usage": 0.0,
            "input": 0.0,
            "cached": 0.0,
            "output": 0.0,
            "thinking": 0.0,
            "notional": 0.0,
        }
    )
    for trace in traces:
        env = getattr(trace, "environment", None) or "unknown"
        row = rows[env]
        row["traces"] += 1

        metadata = getattr(trace, "metadata", None) or {}
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
        if not isinstance(usage, dict):
            # No fallback to Langfuse's own `total_cost`. It is that provider's own
            # estimate at its own rate card, and folding it into the billed column
            # produced a table showing $0.015 billed on a free tier — and billed above
            # notional, which is impossible. A trace we cannot attribute is counted as
            # unattributed rather than guessed at.
            row["no_usage"] += 1
            continue

        notional = float(usage.get("cost_usd_notional") or 0)
        if notional <= 0:
            # Tokens with no cost. Either a synthetic trace from the test suite or a real
            # run against a model with no PRICING entry — both of which would drag the
            # blended rate below the input floor if their tokens were counted here.
            row["unpriced"] += 1
            continue

        row["priced"] += 1
        row["input"] += float(usage.get("input_tokens") or 0)
        row["cached"] += float(usage.get("cached_input_tokens") or 0)
        row["output"] += float(usage.get("output_tokens") or 0)
        row["thinking"] += float(usage.get("thinking_tokens") or 0)
        row["notional"] += notional
    return dict(rows)


def is_test_environment(env: str) -> bool:
    """Environments that carry test traffic rather than agent spend.

    `tests/test_trace_integration.py` writes genuine traces — that is the point of it — but
    to `integration-test`, so they can be shown without being counted. Naming the boundary
    here rather than hardcoding one string keeps a future `eval-test` on the right side of
    it automatically.
    """
    return "test" in env.lower()


def check_blended_rate(rows: dict[str, Any]) -> str | None:
    """Assert the blended rate lands between the input and output per-1M rates.

    Total cost over total tokens is a weighted average of the two rates, so it cannot fall
    outside them. When it does, the cost and the tokens are describing different traces —
    which is the exact failure this script shipped before. Returns an error string, or None.
    """
    pricing = get_settings().pricing()
    billable = [r for e, r in rows.items() if not is_test_environment(e)]
    tokens = sum(r["input"] + r["output"] for r in billable)
    notional = sum(r["notional"] for r in billable)
    if not tokens or not notional:
        return None
    blended = notional / tokens * 1_000_000
    from scripts.reconcile_cost import rate_floor

    low = rate_floor(
        pricing,
        sum(r["input"] for r in billable),
        sum(r.get("cached", 0.0) for r in billable),
    )
    high = pricing.notional_output_usd
    if low - 1e-6 <= blended <= high + 1e-6:
        return None
    return (
        f"Blended rate ${blended:.4f}/1M sits outside the ${low}-${high}/1M rate card. "
        f"The cost column and the token columns are summing different traces. "
        f"Run `make reconcile-cost` before publishing this table."
    )


def render(rows: dict[str, Any], days: int, n_traces: int) -> str:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if not rows:
        body = (
            "_No traces found. Start Langfuse "
            "(`docker compose -f infra/docker-compose.langfuse.yml up -d`), set the key "
            "pair in `.env`, run a query, then `make budget`._"
        )
    else:
        lines = [
            "| Environment | Priced traces | Input | Output | Thinking | Notional USD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for env, row in sorted(rows.items()):
            label = f"`{env}`" + (" _(test traffic)_" if is_test_environment(env) else "")
            lines.append(
                f"| {label} | {int(row['priced']):,} | "
                f"{int(row['input']):,} | {int(row['output']):,} | {int(row['thinking']):,} | "
                f"${row['notional']:.5f} |"
            )
        # The total covers agent spend only. `tests/test_trace_integration.py` writes real
        # traces to its own environment, and folding those into a figure labelled "agent
        # spend" would be D-021's category error committed a second time, in miniature.
        billable = {e: r for e, r in rows.items() if not is_test_environment(e)}
        total_notional = sum(r["notional"] for r in billable.values())
        total_priced = int(sum(r["priced"] for r in billable.values()))
        total_input = sum(r["input"] for r in billable.values())
        total_output = sum(r["output"] for r in billable.values())
        lines.append(
            f"| **total** | **{total_priced:,}** | **{int(total_input):,}** | "
            f"**{int(total_output):,}** | | **${total_notional:.5f}** |"
        )

        tokens = total_input + total_output
        if tokens and total_notional:
            blended = total_notional / tokens * 1_000_000
            pricing = get_settings().pricing()
            lines.append("")
            lines.append(
                f"_Blended ${blended:.4f} per 1M tokens, between the "
                f"${pricing.notional_input_usd} input and ${pricing.notional_output_usd} "
                f"output rates as a mostly-input workload should be (cached input, at "
                f"${pricing.notional_cached_input_usd}, is the only thing allowed below it). "
                f"`make reconcile-cost` checks this._"
            )

        unpriced = int(sum(r["unpriced"] for r in rows.values()))
        no_usage = int(sum(r["no_usage"] for r in rows.values()))
        if unpriced or no_usage:
            lines.append("")
            lines.append(
                f"_Excluded from every column above, not estimated: {unpriced} traces "
                f"carrying tokens but no cost (synthetic runs from the test suite, which "
                f"priced nothing), and {no_usage} traces with no usage metadata "
                f"(pre-instrumentation runs and the duplicate roots of the double-trace "
                f"bug). {total_priced} of {n_traces} traces in the window are real, priced "
                f"agent runs. Counting the excluded traces' tokens against the priced "
                f"traces' cost is what produced an impossible blended rate before "
                f"(DECISIONS D-021)._"
            )
        body = "\n".join(lines)

    return f"""{START}
<!-- Generated by `make budget` from Langfuse traces. Do not edit by hand. -->

**Agent-side spend, last {days} days** (generated {today}):

{body}

Notional prices the tokens at paid standard rates (DECISIONS D-004). **There is no billed
column.** It used to read $0 from a free-tier assumption while the key's project was billed;
billing comes only from the provider's own record — see the Gemini line above, from
docs/billing/gemini.json (DECISIONS D-046). **This table is the agent side only.** OpenAI
judge spend is a different provider on a different ceiling and is never summed with the figures
above; regenerate it with `make judge-spend`, which reads usage from the batch objects
themselves.
{END}"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    traces = fetch_traces(args.days)
    rows = summarise(traces)

    # Refuse to write a table that fails the structural check. A wrong rate propagating into
    # the one document whose purpose is being right about money is worse than no table, and
    # machine-generated numbers get trusted more than hand-written ones, not less.
    error = check_blended_rate(rows)
    if error:
        print(f"Refusing to write {BUDGET_MD}: {error}", file=sys.stderr)
        return 1

    block = render(rows, args.days, len(traces))

    if args.dry_run:
        print(block)
        return 0

    text = BUDGET_MD.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"{BUDGET_MD} is missing the {START} / {END} markers", file=sys.stderr)
        return 1
    BUDGET_MD.write_text(
        re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.DOTALL),
        encoding="utf-8",
    )
    print(block)
    print(f"\nWrote {BUDGET_MD}")
    return 0 if traces else 1


if __name__ == "__main__":
    raise SystemExit(main())
