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
    """Aggregate per environment, which is how dev and deployed traffic stay separable."""
    rows: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "traces": 0.0,
            "attributed": 0.0,
            "input": 0.0,
            "output": 0.0,
            "thinking": 0.0,
            "billed": 0.0,
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
            continue

        row["attributed"] += 1
        row["input"] += float(usage.get("input_tokens") or 0)
        row["output"] += float(usage.get("output_tokens") or 0)
        row["thinking"] += float(usage.get("thinking_tokens") or 0)
        row["billed"] += float(usage.get("cost_usd_billed") or 0)
        row["notional"] += float(usage.get("cost_usd_notional") or 0)
    return dict(rows)


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
            "| Environment | Traces | Attributed | Input | Output | Thinking | Billed USD |"
            " Notional USD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for env, row in sorted(rows.items()):
            lines.append(
                f"| `{env}` | {int(row['traces']):,} | {int(row['attributed']):,} | "
                f"{int(row['input']):,} | {int(row['output']):,} | {int(row['thinking']):,} | "
                f"${row['billed']:.5f} | ${row['notional']:.5f} |"
            )
        total_billed = sum(r["billed"] for r in rows.values())
        total_notional = sum(r["notional"] for r in rows.values())
        total_attributed = int(sum(r["attributed"] for r in rows.values()))
        lines.append(
            f"| **total** | **{n_traces:,}** | **{total_attributed:,}** | | | | "
            f"**${total_billed:.5f}** | **${total_notional:.5f}** |"
        )
        unattributed = n_traces - total_attributed
        if unattributed:
            lines.append("")
            lines.append(
                f"_{unattributed} of {n_traces} traces carry no usage metadata and are "
                f"excluded from the cost columns rather than estimated. Those predate the "
                f"Phase 3 instrumentation._"
            )
        body = "\n".join(lines)

    return f"""{START}
<!-- Generated by `make budget` from Langfuse traces. Do not edit by hand. -->

**Agent-side spend, last {days} days** (generated {today}):

{body}

Billed is what the provider charges — $0 on the Gemini free tier. Notional prices the same
tokens at paid standard rates (DECISIONS D-004). OpenAI judge spend is tracked separately
in the allocation table above and is $0.00: no judge call has been made.
{END}"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    traces = fetch_traces(args.days)
    block = render(summarise(traces), args.days, len(traces))

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
