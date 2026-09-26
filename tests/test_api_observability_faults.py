"""Observability is not on the critical path of a request.

At ~43 Langfuse units per query, Langfuse Cloud's Hobby allowance covers ~38 traced queries a
day, and the daily ceiling admits more (D-039) — so a deployed instance can meet a Langfuse
that rate-limits it, and any hosted backend can be slow or down. None of that may reach the
caller: `/query` must still answer, and its latency must not carry export retries.

Real Langfuse client, real OTLP exporter, pointed at a local stub that either answers every
export with 429 or accepts the connection and never answers. Nothing leaves the machine and
nothing is written to any store (CLAUDE.md §8.10).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.api.ledger import MemoryLedger
from src.config import Settings
from src.observability import langfuse as lf
from tests.api_harness import running
from tests.fakes import FakeRetrievalService
from tests.test_api import scripted  # noqa: F401  (fixture)

# Budget for one faked query through the whole app. Without a tracing fault it takes well
# under a second; an exporter blocking the response would take its 10 s timeout or more.
LATENCY_BUDGET_S = 3.0


class _TooManyRequests(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.send_response(429)
        self.send_header("Retry-After", "30")
        self.end_headers()

    def log_message(self, *args: object) -> None:
        return


class _QuickServer(ThreadingHTTPServer):
    def server_bind(self) -> None:
        # HTTPServer.server_bind reverse-resolves the bind address (socket.getfqdn), which
        # took ~35 s on macOS and made this test look like an exporter stall. It was not.
        import socketserver

        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


@pytest.fixture
def rate_limited_langfuse() -> Iterator[str]:
    server = _QuickServer(("127.0.0.1", 0), _TooManyRequests)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def blackhole_langfuse() -> Iterator[str]:
    """Accepts connections and never replies: the slow-backend case."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(64)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept() -> None:
        sock.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
                held.append(conn)
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    stop.set()
    for conn in held:
        conn.close()
    sock.close()


class Installed:
    """The client, the provider it exports through, and the threads creating it started."""

    def __init__(self, client: object, provider: object, threads: set[threading.Thread]) -> None:
        self.client, self.provider, self.threads = client, provider, threads


def install_client(monkeypatch: pytest.MonkeyPatch, host: str) -> Installed:
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider

    # Own provider, so no process-global one outlives the test; the exporter is Langfuse's
    # real OTLP exporter, aimed at the stub.
    before = set(threading.enumerate())
    provider = TracerProvider()
    client = Langfuse(
        public_key="pk-test-faults",
        secret_key="sk-test-faults",
        host=host,
        tracer_provider=provider,
        # Bounds the exporter's own wait, so the synchronous shutdown below ends in seconds
        # against a stub that never answers. It does not touch the request path under test.
        timeout=1,
    )
    monkeypatch.setattr(lf, "_CLIENT", client)
    monkeypatch.setattr(lf, "_CLIENT_TRIED", True)
    monkeypatch.setattr(lf, "callback_handler", lambda: None)
    return Installed(client, provider, set(threading.enumerate()) - before)


@pytest.fixture
def api_settings(settings: Settings, tmp_path: Path) -> Settings:
    settings.checkpoint_db = tmp_path / "threads.sqlite"
    settings.api.daily_notional_ceiling_usd = 100.0
    return settings


@pytest.mark.parametrize("backend", ["rate_limited_langfuse", "blackhole_langfuse"])
async def test_query_answers_promptly_when_langfuse_misbehaves(
    backend: str,
    request: pytest.FixtureRequest,
    api_settings: Settings,
    scripted,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.fakes import passing_critique, simple_plan

    host = request.getfixturevalue(backend)
    installed = install_client(monkeypatch, host)
    scripted([simple_plan(), "An answer.", passing_critique()])
    try:
        async with running(api_settings, FakeRetrievalService(), MemoryLedger()) as (_, http):
            t0 = time.monotonic()
            r = await http.post("/query", json={"question": "What is LoRA?", "stream": False})
            elapsed = time.monotonic() - t0
            # Measured; the shutdown flush's own 10 s bound against a dead stub is not what
            # this test is about, and would only slow the suite.
            monkeypatch.setattr(lf, "flush", lambda: None)
        assert r.status_code == 200
        assert r.json()["answer"].startswith("An answer.")
        assert r.json()["trace_id"], "tracing was on; the run still has its trace id"
        assert elapsed < LATENCY_BUDGET_S, (
            f"/query took {elapsed:.1f}s against a {backend}: export is on the request path"
        )
    finally:
        shut_down_before_the_stub_closes(installed)


def shut_down_before_the_stub_closes(installed: Installed) -> None:
    """Stop the exporter while its stub is still up, and prove it stopped.

    The first version started `client.shutdown()` in a fire-and-forget daemon thread and
    returned; the fixtures then closed the stubs, and the exporter's worker threads kept
    retrying against dead ports into later tests — six threads still alive two seconds after
    this file finished, logging `Connection refused … retrying`, hidden by output capture
    (D-057). `client.shutdown()` alone is not enough: the SDK flushes and stops its own queue
    threads but does not shut down a tracer provider it was handed, and the span processor
    belongs to that provider. Shutdown is not on the request path, so waiting for it here
    costs the suite a second or two and measures nothing it should not.
    """

    def shutdown() -> None:
        installed.client.shutdown()  # type: ignore[attr-defined]
        installed.provider.shutdown()  # type: ignore[attr-defined]
        # Langfuse 4.14's shutdown also leaves its prompt-cache consumer running (idle, no
        # network). Private path; if the SDK moves it this raises rather than passing quietly.
        installed.client._resources.prompt_cache._task_manager.shutdown()  # type: ignore[attr-defined]

    worker = threading.Thread(target=shutdown)
    worker.start()
    worker.join(timeout=15)
    assert not worker.is_alive(), "shutdown did not finish within 15 s"
    deadline = time.monotonic() + 5
    while any(t.is_alive() for t in installed.threads) and time.monotonic() < deadline:
        time.sleep(0.05)
    alive = sorted(t.name for t in installed.threads if t.is_alive())
    assert not alive, f"threads the client started outlived it: {alive}"
