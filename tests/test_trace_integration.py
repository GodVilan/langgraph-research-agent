"""One real trace, end to end, against a live local Langfuse.

Marked ``integration`` and excluded from `make check`. Run it with `make test-integration`
after `make langfuse-up`.

**Why this exists.** `tests/conftest.py` now disables observability for the whole suite,
which was the right fix for the contamination behind D-021 — but it left nothing exercising
the tracing path at all. That matters more here than it usually would: every one of the
Phase 3 tracing bugs was found by running the agent and reading the trace, never by a test.
Two Langfuse traps, a root span that silently split into two traces, and checkpoint models
deserialising as `dict` were all invisible to a suite full of injected recorders, because a
recorder faithfully records calls that the real backend would have rejected.

So the suite gets one test that talks to the real thing. It asserts the trace arrives and
carries the metadata keys `make budget` and `make reconcile-cost` depend on — the keys whose
absence is what "unattributed trace" means. It deliberately does not assert on cost: no
model call is made, so the usage block is zeroed by construction.

This test writes a trace to whatever Langfuse it is pointed at. It refuses to run unless the
host is local, so it cannot pollute a deployed project the way the suite once polluted the
development one.
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.state import GuardrailEvent, RetrievalEvent, Usage
from src.observability.config import ObservabilitySettings

pytestmark = pytest.mark.integration

# The keys `scripts/budget_from_traces.py` and `scripts/reconcile_cost.py` read. A trace
# missing any of them is what those scripts count as "unattributed".
REQUIRED_USAGE_KEYS = {
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "cost_usd_billed",
    "cost_usd_notional",
}
REQUIRED_METADATA_KEYS = {"usage", "guardrail_events", "guardrail_summary", "retrieval"}


# Its own Langfuse environment, never `development`.
#
# This test writes real traces, and `make budget` reads the same store. Left in
# `development` these two traces would sit alongside genuine agent runs and be counted as
# real spend — which is a smaller version of exactly the contamination that produced D-021.
# The budget table groups by environment, so a separate name keeps them visibly apart
# instead of silently mixed in.
TEST_ENVIRONMENT = "integration-test"


@pytest.fixture
def live_settings() -> ObservabilitySettings:
    """Real observability settings, bypassing the suite-wide isolation on purpose."""
    settings = ObservabilitySettings(_env_file=".env")  # type: ignore[call-arg]
    if not settings.langfuse_enabled:
        pytest.skip("Langfuse is not configured; run `make langfuse-up` and set the key pair")
    if not settings.host_is_local:
        pytest.fail(
            f"refusing to write a test trace to {settings.langfuse_host!r} — this test is "
            f"only ever run against a local stack"
        )
    return settings.model_copy(update={"langfuse_environment": TEST_ENVIRONMENT})


@pytest.fixture
def client(live_settings: ObservabilitySettings, monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.observability import langfuse as lf

    monkeypatch.setattr(lf, "get_observability_settings", lambda: live_settings)
    # The module memoises its client, so both sentinels have to be cleared or an earlier
    # test's disabled client survives into this one and every assertion below passes
    # against nothing.
    monkeypatch.setattr(lf, "_CLIENT", None, raising=False)
    monkeypatch.setattr(lf, "_CLIENT_TRIED", False, raising=False)

    created = lf.get_client()
    if created is None:
        pytest.skip(f"could not reach Langfuse at {live_settings.langfuse_host}")

    # `get_client` constructs lazily and does not connect, so a stopped stack produced a
    # client that looked fine and then failed at the first request — the test errored where
    # it should have skipped. An unreachable backend is a missing precondition, not a
    # failure of the code under test.
    from datetime import UTC, datetime, timedelta

    try:
        created.api.trace.list(from_timestamp=datetime.now(UTC) - timedelta(minutes=1), limit=1)
    except Exception as exc:
        pytest.skip(f"Langfuse at {live_settings.langfuse_host} is not answering: {exc}")
    return created


def fetch_trace(client: Any, thread_id: str, attempts: int = 20) -> Any:
    """Poll for the trace. Langfuse ingests asynchronously through a worker queue."""
    since = datetime.now(UTC) - timedelta(minutes=5)
    for _ in range(attempts):
        response = client.api.trace.list(from_timestamp=since, limit=100)
        for trace in getattr(response, "data", []) or []:
            if getattr(trace, "session_id", None) == thread_id:
                return trace
        time.sleep(1.5)
    return None


class TestOneRealTraceArrives:
    def test_trace_carries_the_metadata_the_cost_scripts_read(self, client: Any) -> None:
        from src.observability import langfuse as lf

        thread_id = f"integration-{uuid.uuid4().hex[:12]}"
        state: dict[str, Any] = {
            "answer": "A traced answer.",
            "refused": False,
            "truncated": False,
            "truncation_reason": None,
            "refinement_count": 0,
            "critique": None,
            "plan": [],
            "tool_calls": [],
            "usage": Usage(
                input_tokens=955,
                output_tokens=112,
                reasoning_tokens=0,
                cost_usd=0.0,
                notional_cost_usd=0.0005665,
                llm_calls=3,
            ),
            "guardrail_events": [
                GuardrailEvent(
                    kind="injection",
                    severity="block",
                    node="retrieve",
                    chunk_id="2605.30148_0021",
                    detail="delimiter escape",
                )
            ],
            "retrieval_events": [
                RetrievalEvent(
                    query="integration probe",
                    retriever="dense",
                    k=5,
                    n_hits=5,
                    latency_ms=12.0,
                    hit_chunk_ids=["2605.30148_0021"],
                )
            ],
        }

        with lf.trace_run(
            name="query",
            thread_id=thread_id,
            question="integration probe",
            model="gemini-3.5-flash-lite",
            prompt_version="v2",
        ) as root:
            assert root is not None, "trace_run yielded None against a configured Langfuse"
            lf.record_state(root, state)

        client.flush()

        trace = fetch_trace(client, thread_id)
        assert trace is not None, (
            f"no trace with session_id={thread_id} arrived within the polling window"
        )

        # One root, not two. The pre-fix integration split every query into a `query` trace
        # and a separate `LangGraph` trace, and only the orphan carried the cost (D-021).
        assert trace.name == "query"

        metadata = trace.metadata or {}
        assert set(metadata) >= REQUIRED_METADATA_KEYS, (
            f"missing {REQUIRED_METADATA_KEYS - set(metadata)} — a trace without these is "
            f"what `make budget` counts as unattributed"
        )
        assert set(metadata["usage"]) >= REQUIRED_USAGE_KEYS
        assert metadata["usage"]["input_tokens"] == 955
        assert metadata["usage"]["cost_usd_notional"] == pytest.approx(0.0005665)

        assert metadata["guardrail_summary"]["block"] == 1
        assert metadata["retrieval"][0]["hit_chunk_ids"] == ["2605.30148_0021"]
        assert "guardrail:block" in (trace.tags or [])

        # Kept out of `development`, so this trace can never be counted as agent spend.
        assert trace.environment == TEST_ENVIRONMENT

    def test_this_trace_is_classified_as_priced(self, client: Any) -> None:
        """Closes the loop: the reconciliation must see a real trace as real.

        Asserting only that metadata arrived would not catch a shape the cost scripts
        cannot read, which is the failure that actually cost a phase.
        """
        from scripts.reconcile_cost import classify
        from src.config import PRICING
        from src.observability import langfuse as lf

        thread_id = f"integration-{uuid.uuid4().hex[:12]}"
        usage = Usage(input_tokens=1000, output_tokens=200, notional_cost_usd=0.0008)

        with lf.trace_run(
            name="query",
            thread_id=thread_id,
            question="classification probe",
            model="gemini-3.5-flash-lite",
            prompt_version="v2",
        ) as root:
            lf.record_state(root, {"answer": "x", "usage": usage})

        client.flush()
        trace = fetch_trace(client, thread_id)
        assert trace is not None

        groups = classify([trace], PRICING["gemini-3.5-flash-lite"])

        assert len(groups["priced"]) == 1
        assert len(groups["unpriced"]) == 0
        assert len(groups["no_usage"]) == 0
