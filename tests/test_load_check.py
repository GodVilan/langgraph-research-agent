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


async def test_every_request_is_written_as_it_happens_and_a_ledger_error_stops_the_run(
    tmp_path: Path,
) -> None:
    import httpx

    from scripts.load_check import load_artifact, run

    answers = iter(
        [
            httpx.Response(
                200,
                json={
                    "answer": "LoRA [30179_0006].",
                    "sources": [{"paper_id": "2605.30179"}],
                    "latency_ms": 1.0,
                    "usage": {"llm_calls": 3},
                },
            ),
            httpx.Response(200, json={"answer": "no cite", "sources": [{"paper_id": "2605.1"}]}),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            return next(answers)
        except StopIteration:
            return httpx.Response(503, json={"error": "cost_ledger_unavailable"})

    stream = tmp_path / "lc.jsonl"
    stream.write_text(json.dumps({"url": "http://svc", "concurrency": 1}) + "\n")
    written: list[dict[str, object]] = []

    def sink(rec: dict[str, object]) -> None:
        written.append(rec)
        with open(stream, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    records, _ = await run(
        "http://svc", 1, 30, 1.0, "", 5.0, transport=httpx.MockTransport(handler), sink=sink
    )
    assert len(written) == len(records) == 3  # every request reached the sink as it happened
    assert records[-1]["reason"] == "cost_ledger_unavailable"  # and the run stopped there
    assert [r.get("cited") for r in records[:2]] == [True, False]
    artifact = load_artifact(stream)  # what an abort leaves behind is still readable
    assert len(artifact["records"]) == 3


async def test_an_unhandled_500_stops_the_run() -> None:
    """The deployed build answers a malformed ledger result with a plain-text 500, not
    `cost_ledger_unavailable`; the run must stop on it rather than read it as the platform."""
    import httpx

    from scripts.load_check import run

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    records, _ = await run(
        "http://svc", 1, 30, 1.0, "", 5.0, transport=httpx.MockTransport(handler)
    )
    assert len(records) == 1 and records[0]["status"] == 500


class FakeClock:
    """A client clock whose wall time jumps ahead of monotonic time — a laptop lid closing."""

    def __init__(self, after: int, slept_s: float) -> None:
        self.calls, self.after, self.jump = 0, after, slept_s

    def slept_s(self) -> float:
        self.calls += 1
        return self.jump if self.calls > self.after else 0.0

    def slept(self) -> bool:
        from scripts.load_check import ClientClock

        return self.slept_s() > ClientClock.TOLERANCE_S


async def test_a_client_that_sleeps_stops_the_run_and_the_report_refuses_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D-052: the first G-3 rerun's laptop slept for two hours; the tool reported 600 s."""
    import httpx

    from scripts.load_check import load_artifact, report, run

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "x", "sources": [], "latency_ms": 1.0})

    stream = tmp_path / "lc.jsonl"
    stream.write_text(json.dumps({"url": "http://svc", "concurrency": 1}) + "\n")

    def sink(rec: dict[str, object]) -> None:
        with open(stream, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    records, _ = await run(
        "http://svc",
        1,
        30,
        1.0,
        "",
        5.0,
        transport=httpx.MockTransport(handler),
        sink=sink,
        clock=FakeClock(after=4, slept_s=7200.0),  # type: ignore[arg-type]
    )
    assert 0 < len(records) < 30  # stopped on the sleep, not on the served target
    artifact = load_artifact(stream)  # an aborted JSONL carries the evidence per record
    assert "INVALID" in artifact["invalid"]
    capsys.readouterr()
    report(stream)
    out = capsys.readouterr().out
    assert "INVALID" in out
    assert "p50" not in out and "served queries/min" not in out and "uncited" not in out


def test_the_real_clock_does_not_report_sleep_on_an_awake_machine() -> None:
    from scripts.load_check import ClientClock

    assert not ClientClock().slept()


async def test_single_user_sends_one_request_at_a_time_and_waits_between() -> None:
    import httpx

    from scripts.load_check import single_user

    in_flight, peak, pauses = 0, 0, []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        in_flight -= 1
        return httpx.Response(200, json={"answer": "x", "latency_ms": 5.0, "usage": {}})

    async def pause(seconds: float) -> None:
        pauses.append(seconds)

    records = await single_user(
        "http://svc", 4, 30.0, 5.0, transport=httpx.MockTransport(handler), pause=pause
    )
    assert [r["status"] for r in records] == [200] * 4
    assert peak == 1  # never concurrent
    assert len(pauses) == 3 and all(25.0 < p <= 30.0 for p in pauses)  # spaced start to start


async def test_single_user_stops_when_the_client_sleeps() -> None:
    import httpx

    from scripts.load_check import single_user

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "x", "latency_ms": 5.0})

    async def pause(seconds: float) -> None:
        return None

    records = await single_user(
        "http://svc",
        10,
        30.0,
        5.0,
        transport=httpx.MockTransport(handler),
        clock=FakeClock(after=2, slept_s=600.0),  # type: ignore[arg-type]
        pause=pause,
    )
    assert len(records) < 10
