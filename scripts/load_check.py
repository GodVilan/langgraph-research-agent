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
sys.path.insert(0, str(REPO))

from scripts.smoke_live import cited_source_papers  # noqa: E402

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


class ClientClock:
    """Did the machine running the check sleep? Every timing here is ``time.monotonic``, which
    on macOS (and Linux's CLOCK_MONOTONIC) stops while the machine is asleep. The first G-3 rerun
    ran on a laptop that slept for most of it: the tool reported 600 s and a 2.3 s cold start for
    a run that took two hours of wall time, and its requests timed out on dead connections
    (D-052). Wall time running ahead of monotonic time is the signature; past
    ``TOLERANCE_S`` the run is invalid."""

    TOLERANCE_S = 5.0

    def __init__(self) -> None:
        self.wall0 = time.time()
        self.mono0 = time.monotonic()

    def slept_s(self) -> float:
        return (time.time() - self.wall0) - (time.monotonic() - self.mono0)

    def slept(self) -> bool:
        return self.slept_s() > self.TOLERANCE_S


def invalid_reason(slept_s: float) -> str:
    return (
        f"the client machine slept ~{slept_s:.0f} s during the run: every timing is measured "
        "on a clock that stops in sleep, and in-flight requests died with it. INVALID — not a "
        "result."
    )


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile: a value that was actually observed, not an interpolation."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


async def cold_start(url: str, space: str) -> dict[str, Any]:
    from dotenv import dotenv_values
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or str(dotenv_values(REPO / ".env").get("HF_TOKEN") or "")
    clock = ClientClock()
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
    if clock.slept():
        return {"error": invalid_reason(clock.slept_s())}
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
    sink: Any = None,
    clock: ClientClock | None = None,
) -> tuple[list[dict[str, Any]], float]:
    qs = questions()
    records: list[dict[str, Any]] = []
    headers = {"X-Loadcheck-Token": token} if token else {}
    started = time.monotonic()
    clock = clock or ClientClock()
    counter = {"next": 0}

    def served() -> int:
        return sum(1 for r in records if r["status"] == 200)

    stop = {"reason": ""}

    def done() -> bool:
        if stop["reason"]:
            return True
        if clock.slept():
            stop["reason"] = invalid_reason(clock.slept_s())
            if sink is not None:
                sink({"invalid": stop["reason"]})  # the stream itself says so, even if killed
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
                    sources = [str(x["paper_id"]) for x in body.get("sources") or []]
                    rec |= {
                        "server_ms": body.get("latency_ms"),
                        "guardrail_blocked": body.get("guardrail_blocked"),
                        "truncated": body.get("truncated"),
                        "llm_calls": (body.get("usage") or {}).get("llm_calls"),
                        # Does the served answer cite a source it returned? Same test as
                        # `make smoke-live`, over every served answer rather than one example.
                        "cited": bool(cited_source_papers(str(body.get("answer", "")), sources)),
                        "n_sources": len(sources),
                        # Kept so an uncited response can be read: a refusal cites nothing by
                        # design, and the service flags only the guardrail's refusals.
                        "answer": str(body.get("answer", "")),
                        "trace_id": body.get("trace_id"),
                    }
                else:
                    rec["reason"] = body.get("error") or r.text[:80]
            except httpx.HTTPError as exc:
                rec["status"] = 0
                rec["reason"] = type(exc).__name__
            rec["client_ms"] = round((time.monotonic() - t0) * 1000, 1)
            rec["clock_drift_s"] = round(clock.slept_s(), 1)  # > TOLERANCE_S: the client slept
            records.append(rec)
            if sink is not None:
                sink(rec)  # written and flushed now: an abort keeps every request so far
            # Back off instead of retrying at once. The first deployed run retried instantly
            # and sent ~74,000 requests in minutes, 70,510 of them answered by Hugging Face's
            # own platform rate limiter (D-048).
            if rec.get("reason") == "daily_cost_ceiling":
                stop["reason"] = "daily cost ceiling reached; it resets at 00:00 UTC"
            elif rec.get("reason") == "cost_ledger_unavailable":
                stop["reason"] = "the service's cost ledger is failing; not a load result"
            elif rec["status"] == 500:
                # Builds before 2026-09-25 answered a malformed ledger result with an unhandled
                # 500 rather than `cost_ledger_unavailable` (D-054, since fixed and redeployed).
                # An unexplained 500 is still not a load result, so it stops the run.
                stop["reason"] = "unhandled 500 from the service (possibly the ledger); stopped"
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


def load_artifact(path: Path) -> dict[str, Any]:
    """The JSONL written during the run (header line, then one line per request) — or the JSON
    summary written at the end. The JSONL survives an abort; the summary does not."""
    if path.suffix == ".jsonl":
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        header = lines[0]
        records = [line for line in lines[1:] if "i" in line]
        header |= {k: v for line in lines[1:] if "i" not in line for k, v in line.items()}
        wall = max((r["sent_s"] + r["client_ms"] / 1000 for r in records), default=0.0)
        data = {**header, "records": records, "wall_s": round(wall, 1)}
    else:
        data = dict(json.loads(path.read_text(encoding="utf-8")))
    drift = max((r.get("clock_drift_s", 0.0) for r in data["records"]), default=0.0)
    if drift > ClientClock.TOLERANCE_S and not data.get("invalid"):
        data["invalid"] = invalid_reason(drift)  # an aborted JSONL carries the evidence too
    return data


def report(path: Path) -> None:
    data = load_artifact(path)
    recs = data["records"]
    if data.get("invalid"):
        print(f"{LABEL.capitalize()} — {data['url']}\n")
        print(f"INVALID: {data['invalid']}")
        print(f"{len(recs)} requests recorded; no latency, throughput or rate is printed from it.")
        return
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
        cited = [r for r in ok if "cited" in r]
        if cited:
            uncited = [r for r in cited if not r["cited"]]
            blocked = [r for r in uncited if r.get("guardrail_blocked")]
            other = [r for r in uncited if not r.get("guardrail_blocked")]
            print(
                f"uncited: {len(uncited)} of {len(cited)} served responses cite no returned "
                f"source (n={len(cited)}, live Space) — {len(blocked)} guardrail-blocked (the "
                f"service's flag), {len(other)} not blocked"
                + (", read below:" if other and all("answer" in r for r in other) else "")
            )
            for r in other:
                if "answer" in r:
                    print(f"    #{r['i']:>3}  {' '.join(str(r['answer']).split())[:110]}")
        calls = statistics.median(r.get("llm_calls") or 0 for r in ok)
        print(
            f"throughput: {n / minutes:.1f} served queries/min. Ceiling: the Gemini free-tier "
            f"quota — the container paces model calls at {data['pacer']} and a query makes a "
            f"median {calls:g} calls here"
        )
    cold = data.get("cold_start")
    if cold:
        print(f"\ncold start (separate, not in the figures above): {json.dumps(cold)}")


SINGLE_USER = RUNS / "latency_single_user.json"


async def single_user(
    url: str,
    n: int,
    spacing_s: float,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
    clock: ClientClock | None = None,
    pause: Any = asyncio.sleep,
) -> list[dict[str, Any]]:
    """``n`` requests, one at a time, ``spacing_s`` apart start to start: one user against a
    warm instance. No bypass token — the per-IP bucket (4/min) admits one request every 30 s.
    Never merged with the concurrent figures: different load, different question."""
    clock = clock or ClientClock()
    qs = questions()
    records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
        for i in range(n):
            t0 = time.monotonic()
            rec: dict[str, Any] = {"i": i, "sent_utc": dt.datetime.now(dt.UTC).isoformat()}
            try:
                r = await client.post(f"{url}/query", json={"question": qs[i], "stream": False})
                rec["status"] = r.status_code
                body = (
                    r.json()
                    if r.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                if r.status_code == 200:
                    rec |= {
                        "server_ms": body.get("latency_ms"),
                        "llm_calls": (body.get("usage") or {}).get("llm_calls"),
                        "guardrail_blocked": body.get("guardrail_blocked"),
                        "trace_id": body.get("trace_id"),
                    }
                else:
                    rec["reason"] = body.get("error") or r.text[:80]
            except httpx.HTTPError as exc:
                rec["status"], rec["reason"] = 0, type(exc).__name__
            rec["client_ms"] = round((time.monotonic() - t0) * 1000, 1)
            rec["clock_drift_s"] = round(clock.slept_s(), 1)
            records.append(rec)
            print(f"  {i + 1}/{n} {rec['status']} {rec['client_ms']:.0f} ms", flush=True)
            if clock.slept():
                break
            if i < n - 1:
                await pause(max(0.0, spacing_s - (time.monotonic() - t0)))
    return records


def report_single(path: Path = SINGLE_USER) -> dict[str, Any]:
    data = dict(json.loads(path.read_text(encoding="utf-8")))
    recs = data["records"]
    drift = max((r.get("clock_drift_s", 0.0) for r in recs), default=0.0)
    if drift > ClientClock.TOLERANCE_S:
        data["invalid"] = invalid_reason(drift)
    if data.get("invalid"):
        print(f"INVALID: {data['invalid']}")
        return data
    ok = [r for r in recs if r["status"] == 200]
    lat = [r["client_ms"] / 1000 for r in ok]
    print(
        f"Single user, warm — {data['url']}, {data['started_utc']}: {len(ok)} of {len(recs)} "
        f"served, one at a time, {data['spacing_s']:.0f} s apart"
    )
    if ok:
        n = len(ok)
        print(
            f"latency n={n}: p50 {pct(lat, 50):.1f} s, p95 {pct(lat, 95):.1f} s (client, end to "
            f"end, nearest rank — at n={n} the p95 is value {max(1, math.ceil(0.95 * n))} of {n})"
        )
    return data


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
    parser.add_argument("--single-user", type=int, default=0, help="N sequential requests")
    parser.add_argument("--spacing", type=float, default=30.0, help="seconds, start to start")
    args = parser.parse_args()
    if args.single_user or (args.report and args.label == "single_user"):
        if not args.report:
            if not args.url:
                raise SystemExit("--url is required")
            started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            clock = ClientClock()
            recs = asyncio.run(
                single_user(
                    args.url.rstrip("/"), args.single_user, args.spacing, args.timeout, clock=clock
                )
            )
            single: dict[str, Any] = {
                "label": "single user, warm",
                "url": args.url.rstrip("/"),
                "started_utc": started,
                "spacing_s": args.spacing,
                "client": "scripts/load_check.py --single-user (httpx), one machine, no bypass",
                "records": recs,
            }
            if clock.slept():
                single["invalid"] = invalid_reason(clock.slept_s())
            SINGLE_USER.write_text(json.dumps(single, indent=1), encoding="utf-8")
            print(f"wrote {SINGLE_USER}")
        report_single()
        return
    out = RUNS / f"loadcheck_{args.label}.json"
    stream = RUNS / f"loadcheck_{args.label}.jsonl"

    if args.report:
        report(out if out.exists() else stream)
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
    clock = ClientClock()
    result["max_requests"] = MAX_REQUESTS
    fh = open(stream, "w", encoding="utf-8")  # noqa: SIM115 — held open for the whole run
    fh.write(json.dumps(result) + "\n")
    fh.flush()

    def sink(rec: dict[str, Any]) -> None:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    try:
        records, wall = asyncio.run(
            run(
                url,
                args.concurrency,
                args.min_served,
                args.max_minutes,
                token,
                args.timeout,
                sink=sink,
                clock=clock,
            )
        )
    finally:
        fh.close()
    result |= {"wall_s": round(wall, 1), "records": sorted(records, key=lambda r: r["i"])}
    if clock.slept():
        result["invalid"] = invalid_reason(clock.slept_s())
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out}\n")
    report(out)


if __name__ == "__main__":
    main()
