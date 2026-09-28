"""The fixed trace window behind the published spend table and its reconciliation.

`make budget` and `make reconcile-cost` used to read "the last 30 days", so the committed table
was a function of the day it was generated: regenerated on 2026-09-28 its `integration-test` row
read 6 traces against the committed 8, and once the Phase 4 run (2026-09-22) left the window —
around 2026-10-22 — every row would have moved (D-059). A published figure has to be
regenerable, so the window is two absolute instants:

* from 2026-09-01 00:00 UTC — after the August integration-test traffic, before anything the
  published figures describe;
* to 2026-09-25 00:00 UTC, exclusive — covering the Phase 4 traced run (2026-09-22) and the
  2026-09-24 runs, and ending before the four test traces a mis-aimed local run wrote on
  2026-09-26 (D-058).

Nothing here reads the clock. `--days N` remains for ad-hoc looks, and never writes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

PUBLISHED_WINDOW: tuple[datetime, datetime] = (
    datetime(2026, 9, 1, tzinfo=UTC),
    datetime(2026, 9, 25, tzinfo=UTC),
)


def label(window: tuple[datetime, datetime] = PUBLISHED_WINDOW) -> str:
    start, end = window
    last = end.fromtimestamp(end.timestamp() - 1, UTC)
    return f"{start:%Y-%m-%d} to {last:%Y-%m-%d} (UTC, inclusive)"


def _timestamp(trace: Any) -> datetime | None:
    raw = getattr(trace, "timestamp", None)
    if raw is None and isinstance(trace, dict):
        raw = trace.get("timestamp")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, str):
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return None


def within(traces: list[Any], window: tuple[datetime, datetime]) -> list[Any]:
    """Keep traces whose timestamp is in [start, end). Filtered here as well as asked of the
    API, so the boundary does not depend on how a Langfuse version treats `to_timestamp`."""
    start, end = window
    kept = []
    for trace in traces:
        ts = _timestamp(trace)
        if ts is not None and start <= ts < end:
            kept.append(trace)
    return kept


def fetch(client: Any, window: tuple[datetime, datetime]) -> list[Any]:
    """Every trace in the window, paginated. Raises on a read failure; callers decide."""
    start, end = window
    traces: list[Any] = []
    page = 1
    while True:
        response = client.api.trace.list(
            from_timestamp=start, to_timestamp=end, page=page, limit=100
        )
        batch = getattr(response, "data", []) or []
        traces.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return within(traces, window)
