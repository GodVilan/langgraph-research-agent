"""The load check against a deployed instance, to the Phase 5 review's G-3 spec.

    make load-check URL=https://… SPACE=owner/name [C=10] [SERVED=30] [MAX_MIN=15]
    make load-report LABEL=deployed

**What it measures, and what it refuses to mix.**

* ``C`` concurrent users (10) post questions back to back until at least ``SERVED`` (30)
  requests have been *served*, or ``MAX_MIN`` minutes pass. The service is quota-bound — about
  3 queries a minute at the container's 10 model calls/min — so n ≥ 30 served takes roughly
  10 to 11 minutes, and most attempts in that window are turned away. Sizing by served count, not
  by requests sent, is what makes the p95 worth printing.
* p50/p95 are over **served requests only**, with n inline. At n = 30 the nearest-rank p95 is
  the 29th of 30 values — the second-highest — and the report says so.
* Every other response is counted **by status and reason** — admission ``503 busy``,
  per-IP ``429 rate_limited``, cost ``429 daily_cost_ceiling``, upstream ``503
  model_quota_exhausted`` — never dropped.
* **Cold start is measured separately and first** (with ``SPACE``): the Space is restarted, and
  the time to ``/ready`` and the first request's latency are recorded under ``cold_start`` —
  never in p50/p95.
* The questions are the frozen eval set's 43 verified factual questions, cycled in order.

**The per-IP limiter.** Every request comes from one address, so the per-IP bucket would
refuse nearly all of them. ``LOADCHECK_TOKEN`` (the value set on the instance) bypasses that
bucket *only*; the ceiling and the gate still apply (D-040). ``--no-token`` runs a short
check without it, whose 429s are the limiter firing on the deployed instance.

**The key must be exclusive.** Another process on the same Gemini key during the window
would spend the quota the check is measuring. The run refuses to start without
``--key-exclusive`` (the operator's attestation) and refuses if a known key-using script of
this repo is running locally; both are recorded in the artifact — together with their limit:
they cover **local processes only**, and the public endpoint stays **open** during the run,
so any real callers share the quota with the load.

Labelled "load check against deployed instance": one instance, one client machine — not
live traffic, and not comparable to Phase 4's local single-user batch figures.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "evals" / "runs"
DATASET = REPO / "evals" / "datasets" / "phase4.json"
LABEL = "load check against deployed instance"
# Hard stop on requests sent, whatever else happens: the served target needs ~30-40 served and
# at most a few hundred attempts at this service's throughput.
MAX_REQUESTS = 600
# This repo's scripts that call Gemini with the same key.
KEY_USERS = (
    "evals.run_set",
    "guardrail_variance",
    "generator_determinism",
    "span_loss_probe",
    "determinism_probe",
    "injection_live_probe",
    "evals.build_set",
)


def questions() -> list[str]:
    items = json.loads(DATASET.read_text(encoding="utf-8"))["items"]
    return [it["question"] for it in items if it["stratum"] == "single_paper_factual"]


def local_key_users() -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    me = str(os.getpid())
    return [
        line.strip()[:120]
        for line in out.splitlines()
        if any(k in line for k in KEY_USERS) and not line.strip().startswith(me)
    ]


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile: a value that was actually observed, not an interpolation."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


async def cold_start(url: str, space: str) -> dict[str, Any]:
    from dotenv import dotenv_values
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or str(dotenv_values(REPO / ".env").get("HF_TOKEN") or "")
    async with httpx.AsyncClient(timeout=30) as client:
        before = (await client.get(f"{url}/ready")).json().get("boot_id")
        if not before:
            return {"error": "/ready reports no boot_id; cannot tell the new container apart"}
        HfApi(token=token or None).restart_space(space)
        t0 = time.monotonic()
        # Ready means ready *from a new process*. The first version polled for any 200 and
        # timed the old container, which kept answering for seconds after the restart call.
        while True:
            try:
                r = await client.get(f"{url}/ready")
                if r.status_code == 200 and r.json().get("boot_id") not in (None, before):
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() - t0 > 900:
                return {"error": "no new container ready within 15 min of restart"}
            await asyncio.sleep(2)
        to_ready = time.monotonic() - t0
        t1 = time.monotonic()
        r = await client.post(
            f"{url}/query", json={"question": questions()[0], "stream": False}, timeout=300
        )
    return {
        "method": "HfApi.restart_space, then poll /ready every 2 s until a new boot_id answers",
        "seconds_to_ready": round(to_ready, 1),
        "first_request_status": r.status_code,
        "first_request_s": round(time.monotonic() - t1, 1),
    }


async def run(
    url: str,
    concurrency: int,
    min_served: int,
    max_minutes: float,
    token: str,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[dict[str, Any]], float]:
    qs = questions()
    records: list[dict[str, Any]] = []
    headers = {"X-Loadcheck-Token": token} if token else {}
    started = time.monotonic()
    counter = {"next": 0}

    def served() -> int:
        return sum(1 for r in records if r["status"] == 200)

    stop = {"reason": ""}

    def done() -> bool:
        if stop["reason"]:
            return True
        if len(records) >= MAX_REQUESTS:
            stop["reason"] = f"request cap {MAX_REQUESTS} reached"
            return True
        return served() >= min_served or time.monotonic() - started > max_minutes * 60

    async def user(uid: int, client: httpx.AsyncClient) -> None:
        while not done():
            i = counter["next"]
            counter["next"] += 1
            t0 = time.monotonic()
            retry_after: str | None = None
            rec: dict[str, Any] = {"i": i, "user": uid, "sent_s": round(t0 - started, 2)}
            try:
                r = await client.post(
                    f"{url}/query",
                    json={"question": qs[i % len(qs)], "stream": False},
                    headers=headers,
                )
                rec["status"] = r.status_code
                retry_after = r.headers.get("retry-after")
                ctype = r.headers.get("content-type", "")
                body = r.json() if ctype.startswith("application/json") else {}
                if not ctype.startswith("application/json") and r.status_code != 200:
                    body = {"error": "platform (non-JSON — the host, not this service)"}
                if r.status_code == 200:
                    rec |= {
                        "server_ms": body.get("latency_ms"),
                        "guardrail_blocked": body.get("guardrail_blocked"),
                        "truncated": body.get("truncated"),
                        "llm_calls": (body.get("usage") or {}).get("llm_calls"),
                    }
                else:
                    rec["reason"] = body.get("error") or r.text[:80]
            except httpx.HTTPError as exc:
                rec["status"] = 0
                rec["reason"] = type(exc).__name__
            rec["client_ms"] = round((time.monotonic() - t0) * 1000, 1)
            records.append(rec)
            # Back off instead of retrying at once. The first deployed run retried instantly
            # and sent ~74,000 requests in minutes, 70,510 of them answered by Hugging Face's
            # own platform rate limiter (D-048).
            if rec.get("reason") == "daily_cost_ceiling":
                stop["reason"] = "daily cost ceiling reached; it resets at 00:00 UTC"
            elif rec["status"] in (429, 503) or rec["status"] == 0:
                wait = float(retry_after or 0) or (30.0 if rec["status"] == 429 else 5.0)
                await asyncio.sleep(min(wait, 30.0))
            print(
                f"  #{i:>3} u{uid} {rec['status']} {rec['client_ms']:>8.0f} ms "
                f"{rec.get('reason', '')}  served {served()}",
                flush=True,
            )

    async with httpx.AsyncClient(
        timeout=timeout_s, limits=httpx.Limits(max_connections=concurrency), transport=transport
    ) as client:
        await asyncio.gather(*(user(u, client) for u in range(concurrency)))
    if stop["reason"]:
        print(f"stopped early: {stop['reason']}", flush=True)
    return records, time.monotonic() - started


def report(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = data["records"]
    ok = [r for r in recs if r["status"] == 200]
    minutes = data["wall_s"] / 60
    print(f"{LABEL.capitalize()} — {data['url']}")
    print(
        f"concurrency {data['concurrency']}, {len(recs)} requests sent over "
        f"{data['wall_s']:.0f} s ({minutes:.1f} min), started {data['started_utc']}, "
        f"per-IP bypass {'on' if data['per_ip_bypass'] else 'off'}, key exclusive: "
        f"{data['key_exclusive']['attested']} (local key users: "
        f"{len(data['key_exclusive']['local_processes'])})"
    )
    print(
        f"key exclusivity: {data['key_exclusive'].get('scope', 'local processes only')}\n"
        f"public endpoint open during the run: {data.get('public_endpoint_open_during_run', True)}"
        " — any real callers shared the quota, gate and ceiling with this load"
    )
    print("One instance, one client — not live traffic, not comparable to Phase 4's local batch.\n")
    print(f"served: {len(ok)} of {len(recs)}")
    codes = Counter(
        f"{r['status']} {r.get('reason', '')}".strip() for r in recs if r["status"] != 200
    )
    for code, count in codes.most_common():
        print(f"rejected: {count:>3}  {code}")
    if ok:
        lat = [r["client_ms"] / 1000 for r in ok]
        n = len(ok)
        rank = max(1, math.ceil(0.95 * n))
        print(
            f"\nlatency of served requests only, n={n} at concurrency {data['concurrency']}: "
            f"p50 {pct(lat, 50):.1f} s, p95 {pct(lat, 95):.1f} s (client-measured, end to end, "
            f"nearest rank — at n={n} the p95 is value {rank} of {n}"
            + (", the second-highest" if rank == n - 1 else "")
            + ")"
        )
        server = [r["server_ms"] / 1000 for r in ok if r.get("server_ms") is not None]
        if server:
            print(
                f"server-side graph time p50 {pct(server, 50):.1f} s, p95 {pct(server, 95):.1f} s"
                " — the rest is queueing for a slot, and the network"
            )
        calls = statistics.median(r.get("llm_calls") or 0 for r in ok)
        print(
            f"throughput: {n / minutes:.1f} served queries/min. Ceiling: the Gemini free-tier "
            f"quota — the container paces model calls at {data['pacer']} and a query makes a "
            f"median {calls:g} calls here"
        )
    cold = data.get("cold_start")
    if cold:
        print(f"\ncold start (separate, not in the figures above): {json.dumps(cold)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--url")
    parser.add_argument("--space", default="", help="owner/name: measure cold start first")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--min-served", type=int, default=30)
    parser.add_argument("--max-minutes", type=float, default=15.0)
    parser.add_argument("--label", default="deployed")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-token", action="store_true", help="run without the bypass")
    parser.add_argument(
        "--key-exclusive",
        action="store_true",
        help="attest that nothing else uses this Gemini key during the run",
    )
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    out = RUNS / f"loadcheck_{args.label}.json"

    if args.report:
        report(out)
        return
    if not args.url:
        raise SystemExit("--url is required")
    if not args.key_exclusive:
        raise SystemExit("refusing: attest --key-exclusive (nothing else on the Gemini key)")
    others = local_key_users()
    if others:
        raise SystemExit("refusing: local processes are using the key:\n  " + "\n  ".join(others))
    token = "" if args.no_token else os.environ.get("LOADCHECK_TOKEN", "")
    url = args.url.rstrip("/")

    result: dict[str, Any] = {
        "label": LABEL,
        "url": url,
        "concurrency": args.concurrency,
        "min_served": args.min_served,
        "per_ip_bypass": bool(token),
        "pacer": "10 calls/min, burst 3 (infra/Dockerfile)",
        "key_exclusive": {
            "attested": True,
            "local_processes": others,
            "scope": (
                "local processes only: --key-exclusive and the process check cover this "
                "machine. They cannot see other holders of the same Gemini key elsewhere."
            ),
        },
        # The endpoint is public and stays open during the check: real callers, if any, share
        # the quota, the concurrency gate and the daily ceiling with the load. Not excluded,
        # not detectable from here — recorded so no reader assumes an isolated instance.
        "public_endpoint_open_during_run": True,
        "client": "scripts/load_check.py (httpx), one machine",
    }
    if args.space:
        print("measuring cold start first…", file=sys.stderr)
        result["cold_start"] = asyncio.run(cold_start(url, args.space))
    result["started_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    records, wall = asyncio.run(
        run(url, args.concurrency, args.min_served, args.max_minutes, token, args.timeout)
    )
    result |= {"wall_s": round(wall, 1), "records": sorted(records, key=lambda r: r["i"])}
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out}\n")
    report(out)


if __name__ == "__main__":
    main()
