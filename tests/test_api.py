"""The HTTP service: endpoints, limits, the daily ceiling, threads, and failure mapping.

Every limit here gets a case that makes it fire (CLAUDE.md §8.8b): a limiter that has only
ever admitted requests has not been shown to limit anything.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from src.agent import llm as llm_module
from src.agent.nodes import critique as critique_node
from src.agent.nodes import generate as generate_node
from src.agent.nodes import plan as plan_node
from src.agent.nodes.validate_input import ScopeVerdict
from src.api.app import startup_refusals
from src.api.ledger import (
    LedgerUnavailableError,
    MemoryLedger,
    check_ledger_is_durable,
    seconds_until_utc_midnight,
    utc_day,
)
from src.api.limits import PerClientLimiter, client_address, token_matches
from src.config import Settings
from src.observability.config import ObservabilitySettings
from tests.api_harness import running, sse_events
from tests.fakes import FakeRetrievalService, ScriptedLLM, passing_critique, simple_plan


@pytest.fixture
def api_settings(settings: Settings, tmp_path: Path) -> Settings:
    # Never the developer's .checkpoints/: tests must not write to a store anything reads.
    settings.checkpoint_db = tmp_path / "threads.sqlite"
    settings.api.per_ip_burst = 100
    settings.api.per_ip_per_minute = 600
    # The shared fixture's per-request budget is a generous $1; the ceiling must admit it.
    settings.api.daily_notional_ceiling_usd = 100.0
    return settings


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    def install(script: list[object]) -> ScriptedLLM:
        fake = ScriptedLLM(script)
        for module in (plan_node, critique_node, llm_module):
            monkeypatch.setattr(module, "call_structured", fake.call_structured, raising=False)
        monkeypatch.setattr(generate_node, "call_text", fake.call_text, raising=False)
        monkeypatch.setattr("src.agent.nodes.validate_input.call_structured", fake.call_structured)
        return fake

    return install


def ask(question: str = "What is LoRA?", **extra: Any) -> dict[str, Any]:
    return {"question": question, "stream": False, **extra}


# ── Liveness, readiness, metrics ──────────────────────────────────────────────


class TestProbes:
    async def test_health_and_ready(self, api_settings: Settings) -> None:
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            assert (await client.get("/health")).json() == {"status": "ok"}
            ready = await client.get("/ready")
            assert ready.status_code == 200
            body = ready.json()
            assert body["status"] == "ready"
            assert body["ledger"] == "memory"
            assert body["spent_today_notional_usd"] == 0.0

    async def test_ready_is_503_and_queries_refused_when_the_index_failed_to_load(
        self, api_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.retrieval.service import RetrievalService

        def missing(*a: Any, **k: Any) -> Any:
            raise FileNotFoundError("FAISS index not found; run `make index`")

        monkeypatch.setattr(RetrievalService, "load", missing)
        async with running(api_settings, None) as (_, client):
            ready = await client.get("/ready")
            query = await client.post("/query", json=ask())
            health = await client.get("/health")
        assert ready.status_code == 503
        assert ready.json()["status"] == "failed"
        assert "make index" in ready.json()["error"]
        assert query.status_code == 503
        assert health.status_code == 200  # alive, not ready: the two probes differ

    async def test_metrics_mounts_the_existing_registry(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([simple_plan(), "An answer.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            await client.post("/query", json=ask())
            text = (await client.get("/metrics")).text
            assert "arxiv_agent_requests_total" in text
            assert "arxiv_agent_request_latency_seconds" in text


# ── The query path ────────────────────────────────────────────────────────────


class TestQuery:
    async def test_json_response_carries_answer_trace_and_guardrail(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([simple_plan(), "LoRA adds low-rank adapters.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            r = await client.post("/query", json=ask())
        assert r.status_code == 200
        body = r.json()
        assert body["answer"].startswith("LoRA adds low-rank adapters.")
        assert len(body["thread_id"]) == 32
        assert body["guardrail_blocked"] is False
        assert body["truncated"] is False
        assert r.headers["x-request-id"] == body["request_id"]
        # "LoRA" is on the keyword list: decided without the model, so deterministic.
        assert body["guardrail"]["stage"] == "keyword_fastpath"
        assert body["guardrail"]["deterministic"] is True
        assert body["guardrail"]["note"] is None
        # Never $0 by default (D-046): unverified until a provider record supplies it.
        assert body["usage"]["billed_cost_usd"] is None
        assert body["usage"]["billed_cost_basis"].startswith("unverified")
        assert body["sources"][0]["paper_id"] == "2605.30350"

    async def test_a_classifier_pass_is_labelled_nondeterministic(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted(
            [
                ScopeVerdict(in_scope=True, reason="about optimisers"),
                simple_plan(),
                "A.",
                passing_critique(),
            ]
        )
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            body = (
                await client.post("/query", json=ask("How do adaptive optimisers work?"))
            ).json()
        assert body["guardrail"] == {
            "decision": "passed",
            "stage": "scope_classifier",
            "reason": "about optimisers",
            "deterministic": False,
            "note": body["guardrail"]["note"],
        }
        assert "pinned sampling" in body["guardrail"]["note"]
        assert "does not guarantee determinism" in body["guardrail"]["note"]

    async def test_a_classifier_refusal_is_legible(self, api_settings: Settings, scripted) -> None:
        """The D-029 case: refused by the classifier, and the caller is told so and why."""
        scripted([ScopeVerdict(in_scope=False, reason="clinical medicine, not ML")])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            q = "What is the age of the female patient described in the clinical case example?"
            body = (await client.post("/query", json=ask(q))).json()
        assert body["guardrail_blocked"] is True
        assert body["guardrail"]["decision"] == "refused"
        assert body["guardrail"]["stage"] == "scope_classifier"
        assert body["guardrail"]["reason"] == "clinical medicine, not ML"
        assert body["guardrail"]["deterministic"] is False
        assert body["sources"] == []

    async def test_a_deterministic_refusal_says_so(self, api_settings: Settings, scripted) -> None:
        scripted([])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            body = (await client.post("/query", json=ask("ab"))).json()
        assert body["guardrail_blocked"] is True
        assert body["guardrail"]["stage"] == "input_validation"
        assert body["guardrail"]["deterministic"] is True

    async def test_sse_streams_progress_then_one_final_event(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([simple_plan(), "Streamed answer.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            r = await client.post("/query", json={"question": "What is LoRA?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r.text)
        nodes = [data["node"] for kind, data in events if kind == "node"]
        assert nodes[0] == "validate_input" and nodes[-1] == "finalize"
        assert {"plan", "retrieve", "generate", "critique"} <= set(nodes)
        kind, final = events[-1]
        assert kind == "final"
        assert final["answer"].startswith("Streamed answer.")
        assert sum(1 for k, _ in events if k == "final") == 1

    async def test_streaming_runs_the_graph_once(self, api_settings: Settings, scripted) -> None:
        """The CLI's old --stream ran the graph twice. One question, one set of calls."""
        fake = scripted([simple_plan(), "Once.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            await client.post("/query", json={"question": "What is LoRA?"})
        assert fake.count("text") == 1
        assert fake.count("Plan") == 1

    async def test_a_tripped_budget_returns_a_flagged_partial_answer(
        self, api_settings: Settings, scripted
    ) -> None:
        api_settings.budget.max_llm_calls = 1
        scripted([simple_plan(), "Partial.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            body = (await client.post("/query", json=ask())).json()
        assert body["truncated"] is True
        assert "LLM call cap" in body["truncation_reason"]
        assert "partial" in body["answer"]

    async def test_unknown_fields_and_bad_paper_ids_are_rejected(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            assert (await client.post("/query", json=ask(use_arxiv=True))).status_code == 422
            bad = await client.post("/query", json=ask(paper_ids=["../../etc"]))
            assert bad.status_code == 422
            assert (await client.post("/query", json=ask(top_k=50))).status_code == 422

    async def test_provider_quota_errors_map_to_503_and_return_the_reservation(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([simple_plan(), RuntimeError("429 RESOURCE_EXHAUSTED: quota"), passing_critique()])
        ledger = MemoryLedger()
        async with running(api_settings, FakeRetrievalService(), ledger) as (_, client):
            r = await client.post("/query", json=ask())
            assert r.status_code == 503
            assert r.json()["error"] == "model_quota_exhausted"
            assert r.headers["retry-after"] == "30"
            # Settled to what the checkpoint shows was spent, not left at the reservation.
            assert await ledger.spent(utc_day()) < api_settings.budget.max_notional_cost_usd


# ── Per-IP limiting ───────────────────────────────────────────────────────────


class TestPerIpLimit:
    async def test_the_bucket_fires_with_retry_after(
        self, api_settings: Settings, scripted
    ) -> None:
        api_settings.api.per_ip_burst = 2
        api_settings.api.per_ip_per_minute = 1
        scripted([simple_plan(), "A.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            codes = [(await client.post("/query", json=ask())).status_code for _ in range(3)]
            third = await client.post("/query", json=ask())
            metrics = (await client.get("/metrics")).text
        assert codes[:2] == [200, 200]
        assert codes[2] == 429
        assert third.json()["error"] == "rate_limited"
        assert int(third.headers["retry-after"]) >= 1
        assert 'outcome="rejected"' in metrics
        assert 'category="per_ip_rate_limit"' in metrics

    async def test_a_forged_forwarded_for_does_not_change_the_key(
        self, api_settings: Settings, scripted
    ) -> None:
        api_settings.api.per_ip_burst = 1
        api_settings.api.per_ip_per_minute = 1
        scripted([simple_plan(), "A.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            first = await client.post("/query", json=ask(), headers={"X-Forwarded-For": "1.1.1.1"})
            second = await client.post("/query", json=ask(), headers={"X-Forwarded-For": "2.2.2.2"})
        assert first.status_code == 200
        assert second.status_code == 429

    def test_trusted_hops_take_the_entry_our_proxy_appended(self) -> None:
        # The client wrote "6.6.6.6"; our one proxy appended the real peer "198.51.100.4".
        header = "6.6.6.6, 198.51.100.4"
        assert client_address("10.0.0.2", header, trusted_hops=1) == "198.51.100.4"
        assert client_address("10.0.0.2", header, trusted_hops=0) == "10.0.0.2"
        # Fewer entries than trusted hops: the header cannot be trusted; use the peer.
        assert client_address("10.0.0.2", "", trusted_hops=2) == "10.0.0.2"

    def test_the_bucket_refills(self) -> None:
        limiter = PerClientLimiter(per_minute=60, burst=1)
        assert limiter.acquire("a", now=0.0) == (True, 0.0)
        allowed, wait = limiter.acquire("a", now=0.1)
        assert not allowed and 0.8 < wait <= 1.0
        assert limiter.acquire("a", now=1.2)[0]
        assert limiter.acquire("b", now=0.1)[0]  # per client

    def test_the_bucket_table_is_bounded(self) -> None:
        limiter = PerClientLimiter(per_minute=1, burst=1, max_clients=3)
        for i in range(10):
            limiter.acquire(f"c{i}", now=0.0)
        assert len(limiter._buckets) == 3

    async def test_the_loadcheck_token_bypasses_the_bucket_but_not_the_ceiling(
        self, api_settings: Settings, scripted
    ) -> None:
        from pydantic import SecretStr

        api_settings.api.per_ip_burst = 1
        api_settings.api.per_ip_per_minute = 1
        api_settings.loadcheck_token = SecretStr("s3cret")
        scripted([simple_plan(), "A.", passing_critique()])
        headers = {"X-Loadcheck-Token": "s3cret"}
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            codes = [
                (await client.post("/query", json=ask(), headers=headers)).status_code
                for _ in range(3)
            ]
            plain = await client.post("/query", json=ask())  # drains the burst of 1
            wrong = await client.post("/query", json=ask(), headers={"X-Loadcheck-Token": "no"})
            api_settings.api.daily_notional_ceiling_usd = 0.0
            capped = await client.post("/query", json=ask(), headers=headers)
        assert codes == [200, 200, 200]
        assert plain.status_code == 200
        assert wrong.status_code == 429
        assert capped.status_code == 429
        assert capped.json()["error"] == "daily_cost_ceiling"

    def test_an_unset_token_matches_nothing(self) -> None:
        assert not token_matches("", "")
        assert not token_matches("x", "")
        assert token_matches("x", "x")


# ── Daily cost ceiling ────────────────────────────────────────────────────────


class TestDailyCeiling:
    async def test_exhausted_ceiling_is_429_until_utc_midnight(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([])
        ledger = MemoryLedger()
        await ledger.reserve(utc_day(), 0.49, ceiling=1.0)  # the day is nearly spent
        api_settings.budget.max_notional_cost_usd = 0.025
        api_settings.api.daily_notional_ceiling_usd = 0.50
        async with running(api_settings, FakeRetrievalService(), ledger) as (_, client):
            r = await client.post("/query", json=ask())
            metrics = (await client.get("/metrics")).text
        assert r.status_code == 429
        assert r.json()["error"] == "daily_cost_ceiling"
        assert 0 < int(r.headers["retry-after"]) <= 86_400
        assert 'category="daily_cost_ceiling"' in metrics

    async def test_a_completed_request_settles_to_its_actual_cost(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([simple_plan(), "A.", passing_critique()])
        ledger = MemoryLedger()
        async with running(api_settings, FakeRetrievalService(), ledger) as (_, client):
            body = (await client.post("/query", json=ask())).json()
        assert await ledger.spent(utc_day()) == pytest.approx(body["usage"]["notional_cost_usd"])

    async def test_concurrent_requests_cannot_jointly_pass_the_ceiling(self) -> None:
        """Reserve-then-settle: 50 racing reservations, room for exactly 4."""
        ledger = MemoryLedger()
        results = await asyncio.gather(
            *(ledger.reserve("d", 0.025, ceiling=0.1) for _ in range(50))
        )
        assert sum(ok for ok, _ in results) == 4
        assert await ledger.spent("d") == pytest.approx(0.1)

    async def test_the_ledger_failing_refuses_the_query(
        self, api_settings: Settings, scripted
    ) -> None:
        class DownLedger(MemoryLedger):
            kind = "down"

            async def reserve(self, day: str, amount: float, ceiling: float) -> tuple[bool, float]:
                raise LedgerUnavailableError("connection refused")

            async def spent(self, day: str) -> float:
                raise LedgerUnavailableError("connection refused")

        fake = scripted([simple_plan(), "A.", passing_critique()])
        async with running(api_settings, FakeRetrievalService(), DownLedger()) as (_, client):
            r = await client.post("/query", json=ask())
            ready = await client.get("/ready")
        assert r.status_code == 503
        assert r.json()["error"] == "cost_ledger_unavailable"
        assert fake.calls == []  # fails closed: nothing ran
        assert ready.status_code == 503

    async def test_sqlite_ledger_survives_a_restart(self, tmp_path: Path) -> None:
        from src.api.ledger import SqliteLedger

        first = SqliteLedger(tmp_path / "ledger.sqlite")
        assert (await first.reserve("d", 0.3, ceiling=0.5))[0]
        await first.settle("d", -0.1)
        await first.close()
        second = SqliteLedger(tmp_path / "ledger.sqlite")  # a new process, same file
        assert await second.spent("d") == pytest.approx(0.2)
        assert (await second.reserve("d", 0.4, ceiling=0.5))[0] is False
        await second.close()

    def test_midnight_countdown(self) -> None:
        now = dt.datetime(2026, 9, 23, 23, 59, 0, tzinfo=dt.UTC)
        assert seconds_until_utc_midnight(now) == 60


class TestUpstashLedger:
    """Against a stub of Upstash's REST pipeline, so the protocol handling is exercised."""

    @staticmethod
    def ledger(handler: Any) -> Any:
        import httpx

        from src.api.ledger import UpstashLedger

        led = UpstashLedger("https://stub.upstash.io", "tok")
        led._client = httpx.AsyncClient(
            base_url="https://stub.upstash.io",
            headers=led._client.headers,
            transport=httpx.MockTransport(handler),
        )
        return led

    async def test_reserve_and_refund_on_overshoot(self) -> None:
        import json

        import httpx

        store: dict[str, float] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer tok"
            out = []
            for cmd in json.loads(request.content):
                if cmd[0] == "INCRBYFLOAT":
                    store[cmd[1]] = store.get(cmd[1], 0.0) + float(cmd[2])
                    out.append({"result": str(store[cmd[1]])})
                elif cmd[0] == "GET":
                    v = store.get(cmd[1])
                    out.append({"result": None if v is None else str(v)})
                else:
                    out.append({"result": 1})
            return httpx.Response(200, json=out)

        led = self.ledger(handler)
        assert (await led.reserve("d", 0.3, ceiling=0.5))[0]
        admitted, total = await led.reserve("d", 0.3, ceiling=0.5)
        assert not admitted and total == pytest.approx(0.3)
        assert await led.spent("d") == pytest.approx(0.3)  # the overshoot was given back

    async def test_a_200_carrying_a_command_error_is_not_a_write(self) -> None:
        """D-026, fourth instance: assert the effect, not the status code."""
        import httpx

        led = self.ledger(lambda r: httpx.Response(200, json=[{"error": "WRONGTYPE"}, {}]))
        with pytest.raises(LedgerUnavailableError):
            await led.reserve("d", 0.1, ceiling=1.0)

    async def test_unreachable_raises(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with pytest.raises(LedgerUnavailableError):
            await self.ledger(handler).spent("d")


# ── Concurrency gate ──────────────────────────────────────────────────────────


class TestConcurrencyGate:
    async def test_a_full_gate_is_503_and_returns_the_reservation(
        self, api_settings: Settings, scripted
    ) -> None:
        api_settings.api.max_concurrent_queries = 1
        api_settings.api.queue_timeout_s = 0.05
        release = asyncio.Event()

        class Slow(FakeRetrievalService):
            async def retrieve(self, *a: Any, **k: Any) -> Any:
                await release.wait()
                return await super().retrieve(*a, **k)

        scripted([simple_plan(), "A.", passing_critique()])
        ledger = MemoryLedger()
        async with running(api_settings, Slow(), ledger) as (_, client):
            first = asyncio.create_task(client.post("/query", json=ask()))
            await asyncio.sleep(0.05)
            second = await client.post("/query", json=ask())
            # Only the in-flight request's reservation is held.
            held = await ledger.spent(utc_day())
            release.set()
            assert (await first).status_code == 200
        assert second.status_code == 503
        assert second.json()["error"] == "busy"
        assert held == pytest.approx(api_settings.budget.max_notional_cost_usd)


# ── Threads ───────────────────────────────────────────────────────────────────


class TestThreads:
    async def test_a_thread_resumes_from_its_checkpoint(
        self, api_settings: Settings, scripted
    ) -> None:
        scripted([simple_plan(), "First.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            tid = (await client.post("/query", json=ask())).json()["thread_id"]
            view = (await client.get(f"/threads/{tid}")).json()
            assert [t["role"] for t in view["turns"]] == ["user", "assistant"]

            second = await client.post(f"/threads/{tid}/query", json=ask("What is RLHF?"))
            assert second.status_code == 200
            assert second.json()["thread_id"] == tid
            view = (await client.get(f"/threads/{tid}")).json()
        assert [t["content"] for t in view["turns"] if t["role"] == "user"] == [
            "What is LoRA?",
            "What is RLHF?",
        ]

    async def test_a_new_turn_gets_its_own_budget(self, api_settings: Settings, scripted) -> None:
        """D-013 through the API: the second turn must not inherit the first turn's spend."""
        scripted([simple_plan(), "A.", passing_critique()])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            first = (await client.post("/query", json=ask())).json()
            tid = first["thread_id"]
            second = (await client.post(f"/threads/{tid}/query", json=ask())).json()
        assert second["usage"]["llm_calls"] == first["usage"]["llm_calls"]

    async def test_unknown_and_malformed_threads(self, api_settings: Settings, scripted) -> None:
        scripted([])
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            missing = "0" * 32
            assert (await client.get(f"/threads/{missing}")).status_code == 404
            r = await client.post(f"/threads/{missing}/query", json=ask())
            assert r.status_code == 404
            assert (await client.get("/threads/not-a-thread")).status_code == 422


# ── Refusing to start ─────────────────────────────────────────────────────────


class TestStartupRefusals:
    def test_a_deployed_memory_ledger_is_refused(self, settings: Settings) -> None:
        settings.api.deployed = True
        settings.api.ledger = "memory"
        assert "reset on every restart" in (check_ledger_is_durable(settings) or "")

    def test_a_deployed_sqlite_ledger_off_a_volume_is_refused(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings.api.deployed = True
        settings.api.ledger = "sqlite"
        settings.api.ledger_path = Path("/app/.checkpoints/ledger.sqlite")
        monkeypatch.setattr("src.api.ledger.os.path.ismount", lambda p: str(p) == "/")
        assert "not a mounted volume" in (check_ledger_is_durable(settings) or "")
        monkeypatch.setattr(
            "src.api.ledger.os.path.ismount", lambda p: str(p) in {"/", "/app/.checkpoints"}
        )
        assert check_ledger_is_durable(settings) is None

    def test_local_development_is_not_refused(self, settings: Settings) -> None:
        assert check_ledger_is_durable(settings) is None

    def test_a_ceiling_below_one_reservation_is_refused(self, settings: Settings) -> None:
        settings.budget.max_notional_cost_usd = 0.025
        settings.api.daily_notional_ceiling_usd = 0.01
        assert any("no query could ever be admitted" in p for p in startup_refusals(settings))

    async def test_the_app_will_not_start_misconfigured(self, api_settings: Settings) -> None:
        api_settings.api.deployed = True
        with pytest.raises(RuntimeError, match="reset on every restart"):
            async with running(api_settings, FakeRetrievalService()):
                pass

    def test_seeded_langfuse_keys_are_refused_against_a_remote_host(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import SecretStr

        from src.observability import config as obs_config

        seeded = ObservabilitySettings(
            langfuse_public_key=SecretStr("pk-lf-1a1a1a1a-2b2b-4c4c-8d8d-3e3e3e3e3e3e"),
            langfuse_secret_key=SecretStr("sk-lf-anything"),
            langfuse_host="https://cloud.langfuse.com",
        )
        monkeypatch.setattr(obs_config, "get_observability_settings", lambda: seeded)
        assert any("seeded" in p for p in startup_refusals(settings))


class TestModelPacer:
    def test_off_by_default_so_the_eval_path_is_unpaced(self) -> None:
        from src.agent.llm import model_rate_limiter

        assert model_rate_limiter(0.0) is None

    def test_a_query_worth_of_calls_is_not_paced_from_idle(self) -> None:
        """The first version made every call after the first wait 5 s with one user."""
        import time

        from src.agent.llm import MODEL_CALL_BURST, model_rate_limiter

        model_rate_limiter.cache_clear()
        limiter = model_rate_limiter(12.0)
        t0 = time.monotonic()
        for _ in range(MODEL_CALL_BURST):
            assert limiter.acquire(blocking=False)
        assert time.monotonic() - t0 < 0.5
        assert not limiter.acquire(blocking=False)  # the burst is bounded
        model_rate_limiter.cache_clear()

    def test_the_container_window_leaves_headroom_under_the_quota(self) -> None:
        """Burst + rate is the most calls any 60 s window admits; it must sit *below* 15."""
        import re
        from pathlib import Path

        from src.agent.llm import MODEL_CALL_BURST

        dockerfile = (Path(__file__).parent.parent / "infra" / "Dockerfile").read_text()
        match = re.search(r"AGENT_REQUESTS_PER_MINUTE=(\d+)", dockerfile)
        assert match is not None
        assert MODEL_CALL_BURST + int(match.group(1)) < 15  # documented free-tier RPM


class TestPinnedClassifier:
    def test_the_scope_classifier_asks_for_a_pinned_model(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio as aio

        from src.agent.nodes import validate_input as vi
        from src.agent.state import initial_state

        seen: dict[str, Any] = {}

        def fake_model(**kwargs: Any) -> str:
            seen.update(kwargs)
            return "pinned-model"

        async def fake_call(schema: type, **kwargs: Any) -> tuple[Any, Any]:
            from src.agent.state import Usage

            seen["model_arg"] = kwargs.get("model")
            return schema(in_scope=True, reason="ok"), Usage()

        monkeypatch.setattr(vi, "get_chat_model", fake_model)
        monkeypatch.setattr(vi, "call_structured", fake_call)
        aio.run(vi.validate_input(initial_state("How do optimisers work?", "t")))
        assert seen == {"pinned": True, "model_arg": "pinned-model"}

        seen.clear()
        settings.pin_scope_classifier = False  # reproduces the Phase 4 configuration
        aio.run(vi.validate_input(initial_state("How do optimisers work?", "t")))
        assert seen == {"model_arg": None}

    def test_pinned_adds_seed_and_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.agent import llm

        captured: dict[str, Any] = {}

        def fake_init(model: str, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "m"

        monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
        llm.get_chat_model.cache_clear()
        llm.get_chat_model(pinned=True)
        llm.get_chat_model.cache_clear()
        assert captured["seed"] == 0 and captured["top_k"] == 1


class TestSingleWorker:
    def test_a_second_worker_cannot_start(self, settings: Settings, tmp_path: Path) -> None:
        from src.api.app import hold_single_worker_lock

        settings.api.deployed = True
        settings.api.worker_lock_path = tmp_path / "worker.lock"
        first = hold_single_worker_lock(settings)
        try:
            with pytest.raises(RuntimeError, match="exactly one process"):
                hold_single_worker_lock(settings)  # a separate open(): what a 2nd worker does
        finally:
            first.close()
        again = hold_single_worker_lock(settings)  # released on close
        again.close()

    def test_not_asserted_outside_a_deployed_container(self, settings: Settings) -> None:
        from src.api.app import hold_single_worker_lock

        assert hold_single_worker_lock(settings) is None


class TestHopVerificationHeader:
    async def test_the_client_key_ignores_a_forged_forwarded_for(
        self, api_settings: Settings
    ) -> None:
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            plain = (await client.get("/health")).headers["x-client-key"]
            forged = (
                await client.get("/health", headers={"X-Forwarded-For": "198.51.100.66"})
            ).headers["x-client-key"]
        assert plain == forged

    async def test_ready_reports_the_loaded_index_checksum(self, api_settings: Settings) -> None:
        async with running(api_settings, FakeRetrievalService()) as (_, client):
            body = (await client.get("/ready")).json()
        assert "index_sha256" in body and "trace_sample_rate" in body


class TestAdmissionIsNotCancellable:
    """The deployed load check leaked reservations: a client that disconnected while queued at
    the gate cancelled the handler between reserve and settle. 83 dropped connections exhausted
    the $0.50 day with one query served (DECISIONS D-048)."""

    def service(self, settings: Settings, ledger: MemoryLedger, slots: int) -> object:
        from src.api.app import ServiceState
        from src.api.limits import ConcurrencyGate, PerClientLimiter

        settings.api.daily_notional_ceiling_usd = 1.0
        return ServiceState(
            settings=settings,
            ledger=ledger,
            limiter=PerClientLimiter(600, 100),
            gate=ConcurrencyGate(slots, timeout_s=5.0),
        )

    async def test_a_client_that_leaves_while_queued_returns_its_reservation(
        self, settings: Settings
    ) -> None:
        from src.api.app import _admit

        ledger = MemoryLedger()
        svc = self.service(settings, ledger, slots=1)
        await svc.gate.acquire()  # type: ignore[attr-defined]  # the slot is taken
        waiting = asyncio.create_task(_admit(svc, "d", 0.025))  # type: ignore[arg-type]
        await asyncio.sleep(0.05)
        assert await ledger.spent("d") == pytest.approx(0.025)  # reserved while queued
        waiting.cancel()  # the client disconnects
        with pytest.raises(asyncio.CancelledError):
            await waiting
        svc.gate.release()  # type: ignore[attr-defined]  # the slot frees up
        for _ in range(50):
            await asyncio.sleep(0.01)
            if await ledger.spent("d") == 0.0:
                break
        assert await ledger.spent("d") == pytest.approx(0.0), "the reservation leaked"
        assert svc.gate.in_flight == 0  # type: ignore[attr-defined]  # and so did the slot

    async def test_an_orderly_admission_holds_both_until_the_run(self, settings: Settings) -> None:
        from src.api.app import _admit

        ledger = MemoryLedger()
        svc = self.service(settings, ledger, slots=1)
        await _admit(svc, "d", 0.025)  # type: ignore[arg-type]
        assert await ledger.spent("d") == pytest.approx(0.025)
        assert svc.gate.in_flight == 1  # type: ignore[attr-defined]


class TestUpstashFailsClosed:
    """An unreachable or quota-exhausted Upstash must refuse queries, never serve uncounted."""

    @staticmethod
    def ledger(handler: Any) -> Any:
        import httpx

        from src.api.ledger import UpstashLedger

        led = UpstashLedger("https://stub.upstash.io", "tok")
        led._client = httpx.AsyncClient(
            base_url="https://stub.upstash.io",
            headers=led._client.headers,
            transport=httpx.MockTransport(handler),
        )
        return led

    @pytest.mark.parametrize(
        "failure",
        ["unreachable", "quota_429", "command_error_200", "malformed_result", "timeout"],
    )
    async def test_the_query_is_refused_and_nothing_runs(
        self, api_settings: Settings, scripted, failure: str
    ) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            if failure == "unreachable":
                raise httpx.ConnectError("connection refused")
            if failure == "timeout":
                raise httpx.ReadTimeout("upstash did not answer")
            if failure == "quota_429":
                # Upstash's answer when the plan's request limit is used up.
                return httpx.Response(429, json={"error": "ERR max requests limit exceeded."})
            if failure == "command_error_200":
                return httpx.Response(200, json=[{"error": "ERR max daily request limit"}, {}])
            return httpx.Response(200, json=[{"result": "not-a-number"}, {"result": 1}])

        fake = scripted([simple_plan(), "A.", passing_critique()])
        async with running(api_settings, FakeRetrievalService(), self.ledger(handler)) as (
            _,
            client,
        ):
            r = await client.post("/query", json=ask())
            ready = await client.get("/ready")
        assert r.status_code == 503, r.text
        assert r.json()["error"] == "cost_ledger_unavailable"
        assert fake.calls == [], "the model was called without the ceiling being counted"
        assert ready.status_code == 503

    async def test_a_malformed_settlement_is_raised_not_swallowed(self) -> None:
        """D-054: settlement never parsed INCRBYFLOAT's answer, so a 200 with a malformed result
        was taken as a write. It now raises, and `_settle` logs it."""
        import httpx

        from src.api.ledger import LedgerUnavailableError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"result": "not-a-number"}, {"result": 1}])

        led = self.ledger(handler)
        with pytest.raises(LedgerUnavailableError, match="non-numeric"):
            await led.settle("2026-09-25", 0.004)
        await led.close()

    async def test_a_well_formed_settlement_passes(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[{"result": "0.029"}, {"result": 1}])

        led = self.ledger(handler)
        await led.settle("2026-09-25", 0.004)
        await led.close()
