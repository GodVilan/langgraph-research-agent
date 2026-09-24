"""Per-client rate limiting and admission control.

Two limits that protect different things, and one that is not a limit at all:

* **per-IP token bucket** — one client cannot take every slot. In memory, per process: it
  smooths bursts, and losing it on restart costs one burst, not money.
* **concurrency gate** — how many queries run the graph at once. The model quota, not the
  CPU, bounds throughput here (Gemini free tier: 15 requests/minute, ~4 calls per query),
  so extra in-flight queries would only wait inside the provider client's retry loop where
  nothing reports it. They wait here instead, visibly and with a timeout.
* the daily cost ceiling is neither: it must survive restarts and lives in ``ledger.py``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from collections import OrderedDict
from dataclasses import dataclass

# Buckets for this many distinct clients; the least recently seen is evicted first. Bounds
# memory against a caller rotating source addresses, at the cost of forgetting a client that
# has been quiet while 10,000 others were not — which is a client that has refilled anyway.
MAX_TRACKED_CLIENTS = 10_000


def client_address(peer: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """The address to rate-limit on.

    ``X-Forwarded-For`` is a list each proxy *appends* to, so everything left of the entries
    our own proxies added was written by the client and can say anything. With
    ``trusted_hops = n`` the client is the n-th entry from the right; with 0 the header is
    ignored entirely and the socket peer is used. Trusting the leftmost entry — the common
    mistake — would let any caller choose their own rate-limit key per request.
    """
    if trusted_hops > 0 and forwarded_for:
        entries = [e.strip() for e in forwarded_for.split(",") if e.strip()]
        if len(entries) >= trusted_hops:
            return entries[-trusted_hops]
    return peer or "unknown"


def client_key(address: str) -> str:
    """A stable, non-reversible label for logs, so request logs do not store raw IPs."""
    return hashlib.sha256(address.encode("utf-8")).hexdigest()[:12]


def token_matches(presented: str | None, expected: str) -> bool:
    """Constant-time comparison; an unset expected token matches nothing."""
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


@dataclass
class _Bucket:
    tokens: float
    updated: float


class PerClientLimiter:
    """Token bucket per client address."""

    def __init__(
        self, per_minute: float, burst: int, max_clients: int = MAX_TRACKED_CLIENTS
    ) -> None:
        self.rate = per_minute / 60.0
        self.burst = float(burst)
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def acquire(self, client: str, now: float | None = None) -> tuple[bool, float]:
        """Take one token. Returns ``(allowed, retry_after_s)``.

        Synchronous and lock-free on purpose: there is no ``await`` between reading and
        writing a bucket, so on one event loop the read-modify-write cannot interleave.
        """
        now = time.monotonic() if now is None else now
        bucket = self._buckets.pop(client, None) or _Bucket(tokens=self.burst, updated=now)
        bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
        bucket.updated = now
        self._buckets[client] = bucket
        while len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0.0
        wait = (1.0 - bucket.tokens) / self.rate if self.rate > 0 else 60.0
        return False, wait


class ConcurrencyGate:
    """At most ``limit`` queries in the graph; the rest wait up to ``timeout_s``."""

    def __init__(self, limit: int, timeout_s: float) -> None:
        self.limit = limit
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(limit)
        self.in_flight = 0

    async def acquire(self) -> bool:
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.timeout_s)
        except TimeoutError:
            return False
        self.in_flight += 1
        return True

    def release(self) -> None:
        self.in_flight -= 1
        self._sem.release()
