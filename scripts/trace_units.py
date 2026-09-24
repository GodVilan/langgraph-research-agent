"""What tracing every request costs, in Langfuse Cloud units, and the sampling policy it implies.

    make trace-units

A Langfuse unit is any data point ingested: a trace, each observation in it (span, event,
generation), and each score. The Hobby plan includes 50,000 units a month (pricing page,
checked 2026-09-23). This reads the 69 traces of the Phase 4 traced run
(``evals/runs/v3_de699d68_traced.json``) from the local Langfuse, counts units per trace, and
asks: at the most queries the daily cost ceiling can admit, does tracing every request fit?

Scores are counted separately and left out of the serving figure: the 115 scores on the
Phase 4 window were pushed by the eval harness, and the deployed service writes none.

The query rate is the ceiling's, not a forecast: ``daily_notional_ceiling_usd`` divided by
the mean notional cost per query over the same 69 runs is the most the ceiling admits in a
day. Real traffic on a demo endpoint is expected to be far lower; the policy is sized for
the worst case the service itself permits, so it cannot be exceeded by traffic alone.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RUN = REPO / "evals" / "runs" / "v3_de699d68_traced.json"
HOBBY_UNITS_PER_MONTH = 50_000  # Langfuse Cloud Hobby, pricing page checked 2026-09-23
DAYS_PER_MONTH = 30


def main() -> None:
    from src.config import get_settings
    from src.observability.langfuse import get_client

    client = get_client()
    if client is None:
        raise SystemExit("Langfuse is not configured; `make langfuse-up` and set the keys")

    items = json.loads(RUN.read_text(encoding="utf-8"))["items"]
    per_trace: list[int] = []
    scores = 0
    notional: list[float] = []
    for item in items.values():
        tid = item.get("trace_id")
        if not tid:
            continue
        trace = client.api.trace.get(tid)
        observations = len(getattr(trace, "observations", None) or [])
        per_trace.append(1 + observations)  # the trace itself, plus each observation
        scores += len(getattr(trace, "scores", None) or [])
        notional.append(float(item["usage"]["notional_cost_usd"]))

    if not per_trace:
        raise SystemExit("no traces found for the traced run")
    ceiling = get_settings().api.daily_notional_ceiling_usd
    mean_cost = statistics.mean(notional)
    max_queries_day = ceiling / mean_cost
    monthly = max_queries_day * DAYS_PER_MONTH * statistics.mean(per_trace)
    rate = min(1.0, HOBBY_UNITS_PER_MONTH / monthly)

    print(f"Phase 4 traced run: n={len(per_trace)} traces from the local Langfuse")
    print(
        f"units per trace (trace + observations): median {statistics.median(per_trace):g}, "
        f"mean {statistics.mean(per_trace):.1f}, max {max(per_trace)}"
    )
    print(f"scores on these traces: {scores} (eval harness only; the service writes none)")
    print(f"mean notional cost per query: ${mean_cost:.5f} (same {len(notional)} runs)")
    print(
        f"most queries the ${ceiling:.2f}/day ceiling admits: {max_queries_day:.0f}/day "
        f"-> {monthly:,.0f} units/month if every one is traced"
    )
    print(f"Langfuse Cloud Hobby includes {HOBBY_UNITS_PER_MONTH:,} units/month")
    print(
        f"sample rate that keeps the ceiling's worst case inside it: {rate:.2f}"
        + (" (tracing everything fits)" if rate >= 1.0 else "")
    )


if __name__ == "__main__":
    main()
