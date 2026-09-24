"""Concurrent requests do not see each other's state.

Two properties the service depends on, asserted rather than commented:

1. **Request scope** (AUDIT §4.14). v2.1 kept per-request scoping as mutable attributes on a
   shared agent, so two concurrent requests could run with each other's paper filter. v3
   carries it in ``RequestOptions`` inside graph state. Here many requests with distinct
   filters and ``top_k`` run through one app at once, with the fakes sleeping at random so
   the runs genuinely interleave, and each must retrieve with — and be answered from — its
   own options only.
2. **Trace identity.** ``trace_id`` lives in ``AgentState``, not a module global. Here the
   real Langfuse client runs with an in-memory span exporter — real OpenTelemetry context
   propagation, no network, nothing written to any store (CLAUDE.md §8.10) — and each
   response's ``trace_id`` must be the trace its own graph run executed inside, and the
   exported root span for it must carry its own thread id.

The interleaving is measured, not assumed: a test whose requests happened to run one after
another would pass while proving nothing, so the peak overlap is asserted too.
"""

from __future__ import annotations

import asyncio
import random
import re
from pathlib import Path
from typing import Any

import pytest

from src.agent import llm as llm_module
from src.agent.nodes import critique as critique_node
from src.agent.nodes import generate as generate_node
from src.agent.nodes import plan as plan_node
from src.agent.state import Critique, Plan, Usage
from src.api.ledger import MemoryLedger
from src.config import Settings
from src.observability import langfuse as lf
from src.retrieval.service import RetrievalResult
from tests.api_harness import running
from tests.fakes import make_chunk

N = 12


def paper(i: int) -> str:
    return f"2605.{30000 + i}"


def question(i: int) -> str:
    return f"What does paper {paper(i)} say about attention, case {i}?"


class Recorder:
    """Fakes for retrieval and every model helper, recording what each run saw."""

    def __init__(self) -> None:
        self.retrievals: dict[str, tuple[int, frozenset[str] | None]] = {}
        self.trace_seen: dict[str, str] = {}
        self.in_flight = 0
        self.peak = 0

    async def _jitter(self) -> None:
        await asyncio.sleep(random.uniform(0.0, 0.02))

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | None = None,
        use_arxiv: bool = False,
        section_filter: str | None = None,
    ) -> RetrievalResult:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self._jitter()
            self.retrievals[query] = (top_k, allowed_paper_ids)
            (pid,) = allowed_paper_ids or {"none"}
            hit = make_chunk(chunk_id=f"{pid}_0001", paper_id=pid, title=f"Paper {pid}")
            return RetrievalResult(
                hits=[(hit, 0.9, "dense")], used_sparse=False, used_arxiv=False, latency_ms=1.0
            )
        finally:
            self.in_flight -= 1

    async def call_structured(
        self, schema: type, system: str, user: str, max_attempts: int = 2, model: Any = None
    ) -> tuple[Any, Usage]:
        await self._jitter()
        usage = Usage(input_tokens=10, output_tokens=2, llm_calls=1)
        if schema is Plan:
            return Plan(kind="simple", sub_questions=[]), usage
        if schema is Critique:
            return Critique(verdict="pass"), usage
        return schema(in_scope=True, reason="ok"), usage

    async def call_text(self, system: str, user: str, model: Any = None) -> tuple[str, Usage]:
        await self._jitter()
        asked = re.match(r"Question: (.*?)\n", user)
        assert asked is not None
        q = asked.group(1)
        # Inside the graph run: the trace this run is actually executing in.
        self.trace_seen[q] = lf.current_trace_id()
        # Answer from the passage this run retrieved, so a cross-wired context shows up.
        cited = re.search(r'chunk_id="([^"]+)"', user)
        return f"Answer to [{q}] from [{cited.group(1) if cited else '-'}]", Usage(llm_calls=1)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    for module in (plan_node, critique_node, llm_module):
        monkeypatch.setattr(module, "call_structured", rec.call_structured, raising=False)
    monkeypatch.setattr(generate_node, "call_text", rec.call_text, raising=False)
    monkeypatch.setattr("src.agent.nodes.validate_input.call_structured", rec.call_structured)
    return rec


@pytest.fixture
def api_settings(settings: Settings, tmp_path: Path) -> Settings:
    settings.checkpoint_db = tmp_path / "threads.sqlite"
    settings.api.per_ip_burst = 1000
    settings.api.per_ip_per_minute = 6000
    settings.api.max_concurrent_queries = N
    settings.api.daily_notional_ceiling_usd = 1000.0
    return settings


async def fire(client: Any) -> list[dict[str, Any]]:
    async def one(i: int) -> dict[str, Any]:
        r = await client.post(
            "/query",
            json={
                "question": question(i),
                "stream": i % 2 == 0,  # half streamed, half JSON: both paths under load
                "paper_ids": [paper(i)],
                "top_k": 1 + i % 10,
            },
        )
        assert r.status_code == 200, r.text
        if i % 2 == 0:
            import json

            final = [b for b in r.text.split("\n\n") if b.startswith("event: final")]
            assert len(final) == 1
            return dict(json.loads(final[0].split("data: ", 1)[1]))
        return dict(r.json())

    return list(await asyncio.gather(*(one(i) for i in range(N))))


class TestRequestScope:
    async def test_concurrent_requests_keep_their_own_options(
        self, api_settings: Settings, recorder: Recorder
    ) -> None:
        async with running(api_settings, recorder, MemoryLedger()) as (_, client):
            responses = await fire(client)

        assert recorder.peak > 1, "requests never overlapped; this test proved nothing"
        for i, body in enumerate(responses):
            q = question(i)
            top_k, allowed = recorder.retrievals[q]
            assert top_k == 1 + i % 10
            assert allowed == frozenset({paper(i)})
            assert body["answer"].startswith(f"Answer to [{q}] from [{paper(i)}_0001]")
            assert [s["paper_id"] for s in body["sources"]] == [paper(i)]
        assert len({b["thread_id"] for b in responses}) == N


class TestTraceIdentity:
    @pytest.fixture
    def exporter(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        client = Langfuse(
            public_key="pk-test-concurrency",
            secret_key="sk-test-concurrency",
            host="http://127.0.0.1:9",  # never contacted: spans go to the exporter above
            tracer_provider=TracerProvider(),
            span_exporter=exporter,
        )
        monkeypatch.setattr(lf, "_CLIENT", client)
        monkeypatch.setattr(lf, "_CLIENT_TRIED", True)
        # The LangChain handler is not under test and would build its own client.
        monkeypatch.setattr(lf, "callback_handler", lambda: None)
        yield exporter
        client.shutdown()

    async def test_each_response_carries_the_trace_its_run_executed_in(
        self, api_settings: Settings, recorder: Recorder, exporter: Any
    ) -> None:
        async with running(api_settings, recorder, MemoryLedger()) as (_, client):
            responses = await fire(client)
        lf.flush()

        assert recorder.peak > 1, "requests never overlapped; this test proved nothing"
        ids = [b["trace_id"] for b in responses]
        assert all(ids), "tracing was on, so every response must carry a trace id"
        assert len(set(ids)) == N
        for i, body in enumerate(responses):
            assert body["trace_id"] == recorder.trace_seen[question(i)]

        # The exported root span for each trace belongs to that response's thread.
        from langfuse._client.attributes import LangfuseOtelSpanAttributes as Attr

        sessions: dict[str, str] = {}
        for span in exporter.get_finished_spans():
            session = (span.attributes or {}).get(Attr.TRACE_SESSION_ID)
            if span.name == "query" and session:
                sessions[format(span.context.trace_id, "032x")] = str(session)
        for body in responses:
            assert sessions[body["trace_id"]] == body["thread_id"]

    async def test_a_sampled_out_request_reports_no_trace_id(
        self, api_settings: Settings, recorder: Recorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        exporter = InMemorySpanExporter()
        # An explicit provider, because Langfuse applies `sample_rate` only to a provider it
        # creates itself — and doing that here would register a process-global provider
        # that outlives the test. `sampling_not_in_effect` covers the production path.
        client = Langfuse(
            public_key="pk-test-sampling",
            secret_key="sk-test-sampling",
            host="http://127.0.0.1:9",
            tracer_provider=TracerProvider(sampler=TraceIdRatioBased(0.0)),
            span_exporter=exporter,
        )
        monkeypatch.setattr(lf, "_CLIENT", client)
        monkeypatch.setattr(lf, "_CLIENT_TRIED", True)
        monkeypatch.setattr(lf, "callback_handler", lambda: None)
        try:
            async with running(api_settings, recorder, MemoryLedger()) as (_, http):
                r = await http.post("/query", json={"question": question(0), "stream": False})
            lf.flush()
            assert r.json()["trace_id"] == ""
            assert exporter.get_finished_spans() == ()
        finally:
            client.shutdown()


class TestSamplingGuard:
    def test_a_dropped_sample_rate_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from pydantic import SecretStr

        from src.observability.config import ObservabilitySettings

        class FakeClient:
            def __init__(self, provider: TracerProvider) -> None:
                self._resources = type("R", (), {"tracer_provider": provider})()

        sampled = ObservabilitySettings(
            langfuse_public_key=SecretStr("pk"),
            langfuse_secret_key=SecretStr("sk"),
            langfuse_sample_rate=0.25,
        )
        monkeypatch.setattr(lf, "get_observability_settings", lambda: sampled)

        monkeypatch.setattr(lf, "get_client", lambda: FakeClient(TracerProvider()))
        assert "not in effect" in (lf.sampling_not_in_effect() or "")

        honoured = TracerProvider(sampler=TraceIdRatioBased(0.25))
        monkeypatch.setattr(lf, "get_client", lambda: FakeClient(honoured))
        assert lf.sampling_not_in_effect() is None
