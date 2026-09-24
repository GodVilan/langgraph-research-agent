"""The load-check report: percentiles are observed values, and non-200s are counted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.load_check import pct, report


def test_nearest_rank_percentiles() -> None:
    values = [float(v) for v in range(1, 21)]  # 1..20
    assert pct(values, 50) == 10.0
    assert pct(values, 95) == 19.0
    assert pct([3.0], 95) == 3.0
    assert pct(values, 100) == 20.0


def test_report_counts_rejections(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ok = [
        {"i": i, "status": 200, "client_ms": 1000.0 * (i + 1), "server_ms": 900.0, "llm_calls": 4}
        for i in range(30)
    ]
    rejected = [
        {"i": 30 + i, "status": 503, "reason": "busy", "client_ms": 20000.0} for i in range(12)
    ]
    rejected.append({"i": 99, "status": 429, "reason": "daily_cost_ceiling", "client_ms": 50.0})
    artifact = tmp_path / "lc.json"
    artifact.write_text(
        json.dumps(
            {
                "url": "http://x",
                "concurrency": 10,
                "per_ip_bypass": True,
                "pacer": "10 calls/min, burst 3",
                "key_exclusive": {"attested": True, "local_processes": []},
                "started_utc": "t",
                "wall_s": 600.0,
                "records": ok + rejected,
                "cold_start": {"seconds_to_ready": 70.0, "first_request_s": 9.0},
            }
        )
    )
    report(artifact)
    out = capsys.readouterr().out
    assert "served: 30 of 43" in out
    assert "rejected:  12  503 busy" in out
    assert "429 daily_cost_ceiling" in out
    assert "value 29 of 30, the second-highest" in out  # the G-3 spec's p95 caveat
    assert "p95 29.0 s" in out  # served only: the 20 s rejections are not in it
    assert "cold start (separate" in out
    assert "not live traffic" in out
    assert "local processes only" in out
    assert "public endpoint open during the run: True" in out


async def test_the_ceiling_stops_the_run_instead_of_a_hot_loop() -> None:
    """The first deployed run retried instant 429s and sent ~74,000 requests (D-048)."""
    import httpx

    from scripts.load_check import run

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            429, json={"error": "daily_cost_ceiling"}, headers={"Retry-After": "3600"}
        )

    records, _ = await run(
        "http://svc", 10, 30, 1.0, "", 5.0, transport=httpx.MockTransport(handler)
    )
    assert calls["n"] <= 10  # each user stops after its first ceiling response
    assert all(r["reason"] == "daily_cost_ceiling" for r in records)


async def test_a_platform_429_is_recorded_as_the_host_not_the_service() -> None:
    import httpx

    from scripts.load_check import run

    served = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if served["n"] == 0:
            served["n"] += 1
            return httpx.Response(429, text="<!doctype html>", headers={"Retry-After": "0.01"})
        return httpx.Response(200, json={"latency_ms": 1.0, "usage": {"llm_calls": 3}})

    records, _ = await run("http://svc", 1, 1, 1.0, "", 5.0, transport=httpx.MockTransport(handler))
    assert records[0]["reason"].startswith("platform")
    assert records[-1]["status"] == 200
