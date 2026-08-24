"""What the construction limiter retries, and what it refuses to.

A fifteen-minute drafting run at 15 RPM is ~180 calls. Dying at call 150 on a dropped
connection discards everything before it — but retrying a schema failure three times just
turns one broken item into three wasted calls and the same failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.ratelimit import RateLimiter, _classify, limited


class TestClassification:
    @pytest.mark.parametrize(
        "message",
        ["429 RESOURCE_EXHAUSTED", "Error 429: quota", "RESOURCE_EXHAUSTED for metric"],
    )
    def test_quota_errors_are_recognised(self, message: str) -> None:
        assert _classify(RuntimeError(message)) == "quota"

    @pytest.mark.parametrize(
        "message", ["ReadError", "ConnectTimeout", "Server disconnected", "503 unavailable"]
    )
    def test_transient_network_errors_are_recognised(self, message: str) -> None:
        assert _classify(RuntimeError(message)) == "transient"

    def test_the_exception_type_name_counts_too(self) -> None:
        """`httpx.ReadError` carries an empty message; only its class name identifies it."""

        class ReadError(Exception):
            pass

        assert _classify(ReadError()) == "transient"

    @pytest.mark.parametrize(
        "message",
        ["validation error for CritiqueVerdict", "invalid api key", "404 not found"],
    )
    def test_everything_else_propagates(self, message: str) -> None:
        assert _classify(RuntimeError(message)) is None


class TestRetryBehaviour:
    async def test_a_transient_failure_is_retried_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("evals.ratelimit.asyncio.sleep", _no_sleep)
        calls = {"n": 0}

        async def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("ReadError")
            return "ok"

        assert await limited(flaky) == "ok"
        assert calls["n"] == 3

    async def test_a_schema_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One broken item must not become four identical failures."""
        monkeypatch.setattr("evals.ratelimit.asyncio.sleep", _no_sleep)
        calls = {"n": 0}

        async def broken() -> str:
            calls["n"] += 1
            raise ValueError("validation error for DraftedQuestion")

        with pytest.raises(ValueError, match="validation error"):
            await limited(broken)
        assert calls["n"] == 1

    async def test_retries_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("evals.ratelimit.asyncio.sleep", _no_sleep)
        calls = {"n": 0}

        async def always_down() -> str:
            calls["n"] += 1
            raise RuntimeError("ReadError")

        with pytest.raises(RuntimeError, match="ReadError"):
            await limited(always_down, retries=3)
        assert calls["n"] == 3

    async def test_quota_waits_longer_than_a_dropped_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Quota needs a full window; a dropped connection needs a moment."""
        waits: list[float] = []

        async def record(seconds: float) -> None:
            waits.append(seconds)

        monkeypatch.setattr("evals.ratelimit.asyncio.sleep", record)

        for message in ("429 RESOURCE_EXHAUSTED", "ReadError"):
            # Both loop variables bound as defaults: the closure outlives the iteration.
            async def flaky(msg: str = message, seen: list[int] = []) -> str:  # noqa: B006
                seen.append(1)
                if len(seen) == 1:
                    raise RuntimeError(msg)
                return "ok"

            await limited(flaky)

        # `waits` also contains the limiter's own spacing sleeps, so assert the two backoff
        # values are present rather than assuming these are the only sleeps that happened.
        assert 60.0 in waits, "a quota failure should wait a full window"
        assert 5.0 in waits, "a dropped connection should wait a moment"


class TestSpacing:
    async def test_the_limiter_spaces_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        waits: list[float] = []

        async def record(seconds: float) -> None:
            waits.append(seconds)

        monkeypatch.setattr("evals.ratelimit.asyncio.sleep", record)
        limiter = RateLimiter(rpm=60)  # one per second

        await limiter.acquire()
        await limiter.acquire()

        assert any(w > 0 for w in waits)


async def _no_sleep(seconds: float) -> None:
    return None
