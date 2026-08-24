"""Rate limiting for eval construction, deliberately *not* for the agent.

The Gemini free tier allows **15 requests per minute** for `gemini-3.5-flash-lite`. Found
the way these things are always found: a batch of necessity checks died mid-run with
`RESOURCE_EXHAUSTED` and a 49-second retry hint.

That matters more than it looks for Phase 4 arithmetic. One multi-hop candidate costs four
calls — one to draft, two single-paper sufficiency checks, one joint — so twenty candidates
is eighty calls and at least five and a half minutes of pure waiting. A 100-item eval run
repeated for a variance estimate is an hour of wall clock before any judging happens.

**This limiter is not installed in the agent's own path**, and that is a deliberate choice
rather than an oversight. Phase 4 reports p50/p95 latency; a limiter in
`src/agent/llm.py` would inject sleep into the measured path and quietly turn a latency
metric into a measurement of the rate limiter. Construction tooling can afford to wait,
the thing being measured cannot.

v2.1 had a token-bucket limiter and v3 does not; that gap is now visible rather than
theoretical (BACKLOG).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

# The documented free-tier ceiling, minus a little headroom. The quota is enforced per
# project per model, so concurrent processes share it — running two drafters at once will
# still trip it.
FREE_TIER_RPM = 15
SAFE_RPM = 12


class RateLimiter:
    """Simple spacing limiter: at most ``rpm`` calls per rolling minute."""

    def __init__(self, rpm: int = SAFE_RPM) -> None:
        self._interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


_LIMITER = RateLimiter()


# Errors that are worth trying again: the request never produced an answer, and the same
# request may well succeed. Kept as a narrow list rather than a bare `except Exception`,
# because a schema failure is *not* transient and retrying one three times just turns one
# broken item into three wasted calls before the same failure.
QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "429")
TRANSIENT_MARKERS = (
    "ReadError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "RemoteProtocolError",
    "Server disconnected",
    "503",
    "502",
)


def _classify(exc: Exception) -> str | None:
    """'quota', 'transient', or None for an error that should propagate."""
    text = f"{type(exc).__name__}: {exc}"
    if any(marker in text for marker in QUOTA_MARKERS):
        return "quota"
    if any(marker in text for marker in TRANSIENT_MARKERS):
        return "transient"
    return None


async def limited[T](call: Callable[[], Awaitable[T]], *, retries: int = 4) -> T:
    """Run one model call under the shared limiter, retrying quota and network failures.

    A 429 is retried because the provider tells us exactly how long to wait, and failing a
    fifteen-minute construction run on one throttle would be perverse. **Transient network
    errors are retried for the same reason** — an `httpx.ReadError` at call 150 of 180
    would otherwise discard every call before it. That was not hypothetical: one appeared
    partway through the first full draft, and only the provider SDK's own internal retry
    saved the run.

    Everything else propagates. A quota error is transient and a schema error is not;
    treating them alike is how a broken item ends up silently skipped after three
    identical failures.
    """
    for attempt in range(1, retries + 1):
        await _LIMITER.acquire()
        try:
            return await call()
        except Exception as exc:
            kind = _classify(exc)
            if kind is None or attempt == retries:
                raise
            # Quota needs a full window; a dropped connection needs a moment.
            backoff = 60.0 * attempt if kind == "quota" else 5.0 * attempt
            log.warning(
                "%s failure (attempt %d/%d); waiting %.0fs: %s",
                kind,
                attempt,
                retries,
                backoff,
                str(exc)[:120],
            )
            await asyncio.sleep(backoff)
    raise RuntimeError("unreachable")
