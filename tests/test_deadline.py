"""Every model call is bounded by the request's remaining wall-clock budget (D-063).

Before this, the 120 s budget was checked only between graph steps, so one call — Gemini
retrying a 503 inside a single `ainvoke` — ran requests to 128 to 339 s and could hold both
concurrency slots for minutes (D-061, D-062). The fakes below reproduce both shapes: a call that
never answers, and a call inside a real `tenacity` retry loop (the library google-genai uses)
that keeps getting 503s. Each must end at the deadline with a truncated partial answer, free its
slot, keep its full reservation on the ledger, and leave a usage-log row marked cancelled.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import tenacity

from src.agent import llm as llm_module
from src.api.ledger import MemoryLedger, utc_day
from src.config import Settings
from tests.api_harness import running
from tests.fakes import FakeRetrievalService
from tests.test_api import api_settings  # noqa: F401  (fixture)

BUDGET_S = 1.0
# Generous against the budget (the bound should land within milliseconds of it), and far below
# how long an unbounded fake runs: the mutation check removes the bound and must time out here.
TEST_TIMEOUT_S = 10.0


class Fake503Error(Exception):
    """What google-genai raises for `503 UNAVAILABLE — high demand`."""

    code = 503


class _Structured:
    def __init__(self, model: Any) -> None:
        self.model = model

    async def ainvoke(self, messages: Any) -> Any:
        return await self.model.ainvoke(messages)


class HangingModel:
    """A model call that never returns."""

    model = "fake-hanging"

    def __init__(self) -> None:
        self.started = 0
        # Set only in a test's teardown. With the bound, every call is cancelled long before;
        # without it (the mutation check), this lets hung calls end so the test *fails on its
        # timeout* instead of hanging the app's shutdown forever.
        self.release = asyncio.Event()

    async def ainvoke(self, messages: Any) -> Any:
        self.started += 1
        await self.release.wait()

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> _Structured:
        return _Structured(self)


class RetryingModel:
    """A model call that is one `tenacity.AsyncRetrying` loop over attempts that answer 503,
    as google-genai's `_async_request` wraps `_async_request_once` — six attempts, backoff
    between them, the whole loop inside one await."""

    model = "fake-503-retrying"

    def __init__(self) -> None:
        self.attempts = 0

    async def _attempt(self) -> Any:
        self.attempts += 1
        await asyncio.sleep(0.05)  # the request itself
        raise Fake503Error("503 UNAVAILABLE: This model is currently experiencing high demand.")

    async def ainvoke(self, messages: Any) -> Any:
        retrying = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(6),
            # google-genai's shape at half scale: exponential from 0.5 s (theirs: from 1 s,
            # base 2, capped at 60 s). Six attempts take ~15.5 s — longer than a request, as a
            # real 503 storm is — and the third attempt would start ~1.6 s in, after the cut.
            wait=tenacity.wait_exponential(multiplier=0.5, max=60),
            retry=tenacity.retry_if_exception_type(Fake503Error),
            reraise=True,
        )
        return await retrying(self._attempt)

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> _Structured:
        return _Structured(self)


class ImmediateTimeoutModel:
    """A call that fails at once with its own TimeoutError — an HTTP read timeout, say — long
    before the deadline. An ordinary failure, not a deadline cut."""

    model = "fake-own-timeout"

    def __init__(self) -> None:
        self.started = 0

    async def ainvoke(self, messages: Any) -> Any:
        self.started += 1
        raise TimeoutError("read timed out")

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> _Structured:
        return _Structured(self)


def use_model(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    monkeypatch.setattr(llm_module, "get_chat_model", lambda *a, **k: model)


def usage_rows(settings: Settings) -> list[dict[str, Any]]:
    path = settings.usage_log
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ── The wrapper ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("make", [HangingModel, RetryingModel], ids=["hangs", "503-retries"])
async def test_a_call_ends_at_the_deadline_and_is_recorded_as_cancelled(
    make: type, settings: Settings
) -> None:
    model = make()
    token = llm_module.CALL_DEADLINE.set(time.monotonic() + BUDGET_S)
    started = time.monotonic()
    try:
        with pytest.raises(llm_module.CallDeadlineExceededError) as caught:
            await asyncio.wait_for(
                llm_module.call_text("s", "u", model=model),  # type: ignore[arg-type]
                timeout=TEST_TIMEOUT_S,
            )
    finally:
        llm_module.CALL_DEADLINE.reset(token)
    elapsed = time.monotonic() - started
    assert elapsed < BUDGET_S + 0.5, f"returned after {elapsed:.2f} s against a {BUDGET_S} s budget"
    assert caught.value.usage.cancelled_calls == 1 and caught.value.usage.llm_calls == 1
    (row,) = usage_rows(settings)
    assert row["cancelled"] is True and row["usage_known"] is False
    assert row["input_tokens"] is None and row["output_tokens"] is None  # unknown, never 0


async def test_the_retry_loop_itself_stops_not_just_the_await() -> None:
    """Cancellation reaches into tenacity's backoff sleep: no attempt starts after the cut."""
    model = RetryingModel()
    token = llm_module.CALL_DEADLINE.set(time.monotonic() + BUDGET_S)
    try:
        with pytest.raises(llm_module.CallDeadlineExceededError):
            await llm_module.call_text("s", "u", model=model)  # type: ignore[arg-type]
    finally:
        llm_module.CALL_DEADLINE.reset(token)
    attempts_at_cut = model.attempts
    assert 1 <= attempts_at_cut < 6
    await asyncio.sleep(1.5)  # past when the next attempt would have started (~1.6 s)
    assert model.attempts == attempts_at_cut, "the retry loop kept going after the deadline"


async def test_no_call_starts_once_the_budget_is_spent(settings: Settings) -> None:
    model = HangingModel()
    token = llm_module.CALL_DEADLINE.set(time.monotonic() - 0.01)
    try:
        with pytest.raises(llm_module.CallDeadlineExceededError):
            await asyncio.wait_for(
                llm_module.call_text("s", "u", model=model),  # type: ignore[arg-type]
                timeout=TEST_TIMEOUT_S,
            )
    finally:
        llm_module.CALL_DEADLINE.reset(token)
    assert model.started == 0
    assert usage_rows(settings) == []  # no call, so nothing to record


async def test_outside_a_request_calls_are_not_bounded() -> None:
    """Eval tooling runs without a deadline; the bound applies only inside `run_query`."""
    assert llm_module.CALL_DEADLINE.get() is None


# ── Through the API ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("make", [HangingModel, RetryingModel], ids=["hangs", "503-retries"])
async def test_stuck_requests_truncate_free_their_slots_and_keep_their_reservation(
    make: type,
    api_settings: Settings,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_settings.budget.max_wall_clock_s = BUDGET_S
    api_settings.api.max_concurrent_queries = 2
    # A third request must wait for a slot; with the bound it gets one after ~1 s.
    api_settings.api.queue_timeout_s = TEST_TIMEOUT_S
    model = make()
    use_model(monkeypatch, model)
    ledger = MemoryLedger()
    reserved = api_settings.budget.max_notional_cost_usd
    question = {"question": "How does deep learning handle label noise?", "stream": False}

    async with running(api_settings, FakeRetrievalService(), ledger) as (_, client):
        started = time.monotonic()
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*(client.post("/query", json=question) for _ in range(3))),
                timeout=TEST_TIMEOUT_S,
            )
        finally:
            if isinstance(model, HangingModel):
                model.release.set()  # see HangingModel.release
        elapsed = time.monotonic() - started
        spent = await ledger.spent(utc_day())

    # Three requests through two slots: the third ran only because the first two freed theirs.
    assert [r.status_code for r in responses] == [200, 200, 200], [r.text for r in responses]
    assert elapsed < 2 * BUDGET_S + 1.5, f"{elapsed:.2f} s for 3 requests through 2 slots"
    for r in responses:
        body = r.json()
        assert body["truncated"] is True
        assert "wall-clock deadline exceeded" in body["truncation_reason"]
        assert "partial" in body["answer"]
        assert body["usage"]["cancelled_calls"] == 1
        assert any(e["kind"] == "model_call_deadline" for e in body["guardrail_events"])
    # Each request keeps its full reservation: a cancelled call's cost is unknown, never $0.
    assert spent == pytest.approx(3 * reserved)
    cancelled = [row for row in usage_rows(api_settings) if row["cancelled"]]
    assert len(cancelled) == 3 and all(row["input_tokens"] is None for row in cancelled)


# ── A call's own TimeoutError is not a deadline cut ─────────────────────────


async def test_a_call_s_own_timeout_propagates_as_an_ordinary_failure(settings: Settings) -> None:
    """Under `wait_for`, any TimeoutError looked like the deadline. Only an expired budget is."""
    model = ImmediateTimeoutModel()
    token = llm_module.CALL_DEADLINE.set(time.monotonic() + BUDGET_S)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError) as caught:
            await llm_module.call_text("s", "u", model=model)  # type: ignore[arg-type]
    finally:
        llm_module.CALL_DEADLINE.reset(token)
    assert not isinstance(caught.value, llm_module.CallDeadlineExceededError)
    assert str(caught.value) == "read timed out"
    assert time.monotonic() - started < BUDGET_S / 2  # failed at once, long before the deadline
    assert [r for r in usage_rows(settings) if r.get("cancelled")] == []


async def test_a_call_s_own_timeout_is_not_reported_or_settled_as_a_deadline(
    api_settings: Settings,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_settings.budget.max_wall_clock_s = BUDGET_S
    use_model(monkeypatch, ImmediateTimeoutModel())
    ledger = MemoryLedger()
    question = {"question": "How does deep learning handle label noise?", "stream": False}

    async with running(api_settings, FakeRetrievalService(), ledger) as (_, client):
        r = await asyncio.wait_for(client.post("/query", json=question), timeout=TEST_TIMEOUT_S)
        spent = await ledger.spent(utc_day())

    # An ordinary run failure: not a truncated answer, no wall-clock reason, no cancelled call.
    assert r.status_code == 500, r.text
    body = r.json()
    assert body["error"] == "internal_error" and "TimeoutError" in body["detail"]
    assert "wall-clock" not in r.text and "truncated" not in body
    # Settled from the checkpoint like any failed run, not held at the full reservation the
    # way a cancelled call is: nothing priced was spent before the failing call.
    assert spent == pytest.approx(0.0)
    assert spent < api_settings.budget.max_notional_cost_usd
    assert [row for row in usage_rows(api_settings) if row.get("cancelled")] == []


def test_the_bound_wraps_every_model_call_site() -> None:
    """`call_text` and `call_structured` are the only paths to a model (test_usage_log.py);
    both must await through `_bounded`."""
    import inspect

    for fn in (llm_module.call_text, llm_module.call_structured):
        source = inspect.getsource(fn)
        assert "_bounded(" in source, fn.__name__
        assert ".ainvoke(" not in source, f"{fn.__name__} calls a model outside the bound"


def test_this_file_does_not_touch_the_real_usage_log(settings: Settings) -> None:
    assert settings.usage_log != Path(__file__).resolve().parent.parent / ".usage/llm_usage.jsonl"
