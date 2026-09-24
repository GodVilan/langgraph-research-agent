"""Read a trace back from Langfuse through the v2 observations API, and judge completeness.

Langfuse Cloud answers the legacy ``GET /api/public/traces`` with **410** for organisations
created on or after 2026-09-16 (found 2026-09-24, D-046). The replacement is
``GET /api/public/v2/observations`` — Cloud now, self-hosted from **Langfuse v4**; the local
v3 stack answers it 404. Both `make smoke-live` and `scripts/span_loss_probe.py` read through
here, so there is one definition of "the trace arrived".

**Arrived is not complete** (`span_loss_probe.py`, 2026-09-24): after a SIGKILL a trace
*existed* with 15 of its 32 observations, the root span and the late nodes still buffered.
So completeness is two facts, both required — the observation count has stopped growing, and
the **root observation** (the ``query`` span, which ends last) carries the answer as its output.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import httpx


def fetch_observations(
    host: str, auth: tuple[str, str], trace_id: str, window_h: float = 24.0
) -> list[dict[str, Any]]:
    """Every observation of one trace. ``fromStartTime``/``toStartTime`` are required by the
    API; the window reaches back ``window_h`` hours and an hour ahead of now."""
    now = dt.datetime.now(dt.UTC)
    params: dict[str, Any] = {
        "traceId": trace_id,
        "fromStartTime": (now - dt.timedelta(hours=window_h)).isoformat(),
        "toStartTime": (now + dt.timedelta(hours=1)).isoformat(),
        "fields": "core,basic,io",
        "limit": 1000,
    }
    out: list[dict[str, Any]] = []
    while True:
        r = httpx.get(
            f"{host.rstrip('/')}/api/public/v2/observations", params=params, auth=auth, timeout=30
        )
        if r.status_code in {404, 410}:
            raise RuntimeError(
                f"{host} answered {r.status_code} for /api/public/v2/observations — self-hosted "
                f"Langfuse needs v4+ for this endpoint"
            )
        r.raise_for_status()
        body = r.json()
        out.extend(body.get("data") or [])
        cursor = (body.get("meta") or {}).get("cursor")
        if not cursor:
            return out
        params["cursor"] = cursor


def summarise(observations: list[dict[str, Any]]) -> dict[str, Any]:
    roots = [o for o in observations if not o.get("parentObservationId")]
    root = roots[0] if len(roots) == 1 else None
    output = (root or {}).get("output")
    if isinstance(output, str):
        has_answer = bool(output.strip())
    else:
        has_answer = bool((output or {}).get("answer")) if isinstance(output, dict) else False
    return {
        "observations": len(observations),
        "roots": len(roots),
        "root_name": (root or {}).get("name"),
        "root_has_answer": has_answer,
    }


def wait_for_complete(
    host: str, auth: tuple[str, str], trace_id: str, timeout_s: float, poll_s: float = 5.0
) -> dict[str, Any] | None:
    """Poll until the trace is complete and has stopped growing, or ``timeout_s`` passes.

    Returns the last summary seen (possibly incomplete) with ``complete`` and
    ``seconds_until_first_seen`` / ``seconds_until_complete``, or None if nothing ever arrived.
    Never fetches once: the server's batch exporter and the backend's ingestion both delay it.
    """
    started = time.monotonic()
    first_seen: float | None = None
    last: dict[str, Any] | None = None
    stable = 0
    while time.monotonic() - started < timeout_s:
        summary = summarise(fetch_observations(host, auth, trace_id))
        if summary["observations"]:
            if first_seen is None:
                first_seen = time.monotonic() - started
            grew = last is None or summary["observations"] != last["observations"]
            stable = 0 if grew else stable + 1
            last = summary
            if summary["root_has_answer"] and stable >= 2:
                return {
                    **summary,
                    "complete": True,
                    "seconds_until_first_seen": round(first_seen, 1),
                    "seconds_until_complete": round(time.monotonic() - started, 1),
                }
        time.sleep(poll_s)
    if last is None:
        return None
    return {**last, "complete": False, "seconds_until_first_seen": round(first_seen or 0, 1)}
