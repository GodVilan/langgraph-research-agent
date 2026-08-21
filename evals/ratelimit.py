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


async def limited[T](call: Callable[[], Awaitable[T]], *, retries: int = 3) -> T:
    """Run one model call under the shared limiter, retrying on quota exhaustion.

    A 429 is retried rather than raised because the provider tells us exactly how long to
    wait, and failing a two-hour construction run on one throttle would be perverse. Any
    other error propagates: a quota error is transient, a schema error is not, and treating
    them alike is how a broken item ends up silently skipped.
    """
    for attempt in range(1, retries + 1):
        await _LIMITER.acquire()
        try:
            return await call()
        except Exception as exc:
            if "RESOURCE_EXHAUSTED" not in str(exc) and "429" not in str(exc):
                raise
            if attempt == retries:
                raise
            backoff = 60.0 * attempt
            log.warning("quota exhausted (attempt %d/%d); waiting %.0fs", attempt, retries, backoff)
            await asyncio.sleep(backoff)
    raise RuntimeError("unreachable")
