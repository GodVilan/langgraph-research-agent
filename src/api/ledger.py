"""The global daily cost ceiling, and where its running total lives.

D-013 made every budget in the graph strictly per-request: ``usage`` resets at the start of
each query so a thread cannot inherit its previous turn's spend. That was right for the
per-request guards and leaves nothing that can bound a *day*. This is that accumulator —
deliberately separate from graph state, as D-013 said it would have to be.

**Reserve, then settle.** A request's cost is known only after it runs, so checking the
total before admitting it would let N concurrent requests all see room for one and all
proceed. Instead each request atomically reserves the per-request ceiling
(``BudgetLimits.max_notional_cost_usd``) before it runs and settles the difference
afterwards. The day's committed-plus-reserved total can then never pass the ceiling, however
many requests arrive at once. The one overshoot left is inside a single request: the graph
checks its budget between nodes, so a request can exceed its own ceiling by the last call it
made before the check (docs/SERVING.md states the bound).

**Persistence is the whole point.** An in-memory total resets on every restart, and on a
host that sleeps when idle, every morning is a restart: that is a ceiling that never binds.
So ``memory`` is for tests and local development only, and the service refuses to start a
deployed container on it (``check_ledger_is_durable``). ``sqlite`` is durable exactly when
its file is on a mounted volume, and that is checked, not assumed. ``upstash`` is a Redis
over HTTPS, for hosts with no persistent disk at all.

**Fails closed.** A ledger that cannot be reached raises, and the API refuses the query
rather than serving it uncounted — the same rule the scope guard follows (AUDIT §4.2).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Protocol

import httpx

from src.config import Settings

log = logging.getLogger(__name__)

# Keys outlive their day by one, so a request that reserved at 23:59:59 UTC can still
# settle against the day it was charged to.
KEY_TTL_S = 2 * 24 * 3600


class LedgerUnavailableError(RuntimeError):
    """The ledger could not be read or written. The request must be refused."""


class Ledger(Protocol):
    kind: str

    async def reserve(self, day: str, amount: float, ceiling: float) -> tuple[bool, float]:
        """Atomically add ``amount`` if the total stays within ``ceiling``.

        Returns ``(admitted, total)``: the total after the reservation when admitted, the
        unchanged total when refused.
        """
        ...

    async def settle(self, day: str, delta: float) -> None:
        """Adjust a reservation to what the request actually spent (``delta`` may be < 0)."""
        ...

    async def spent(self, day: str) -> float: ...

    async def close(self) -> None: ...


def utc_day(now: dt.datetime | None = None) -> str:
    return (now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d")


def seconds_until_utc_midnight(now: dt.datetime | None = None) -> int:
    now = now or dt.datetime.now(dt.UTC)
    tomorrow = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


# ── Backends ──────────────────────────────────────────────────────────────────


class MemoryLedger:
    """Tests and local development. Resets on restart, so never a deployed ceiling."""

    kind = "memory"

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, day: str, amount: float, ceiling: float) -> tuple[bool, float]:
        async with self._lock:
            total = self._totals.get(day, 0.0)
            if total + amount > ceiling:
                return False, total
            self._totals[day] = total + amount
            return True, total + amount

    async def settle(self, day: str, delta: float) -> None:
        async with self._lock:
            self._totals[day] = max(0.0, self._totals.get(day, 0.0) + delta)

    async def spent(self, day: str) -> float:
        return self._totals.get(day, 0.0)

    async def close(self) -> None:
        return None


class SqliteLedger:
    """One row per UTC day. ``BEGIN IMMEDIATE`` makes check-and-add one atomic step."""

    kind = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: object | None = None
        self._lock = asyncio.Lock()

    async def _db(self) -> object:
        if self._conn is None:
            import aiosqlite

            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(self.path), isolation_level=None)
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS daily_cost (day TEXT PRIMARY KEY, usd REAL NOT NULL)"
            )
            self._conn = conn
        return self._conn

    async def reserve(self, day: str, amount: float, ceiling: float) -> tuple[bool, float]:
        import aiosqlite

        async with self._lock:
            try:
                conn = await self._db()
                assert isinstance(conn, aiosqlite.Connection)
                await conn.execute("BEGIN IMMEDIATE")
                try:
                    cur = await conn.execute("SELECT usd FROM daily_cost WHERE day = ?", (day,))
                    row = await cur.fetchone()
                    total = float(row[0]) if row else 0.0
                    if total + amount > ceiling:
                        await conn.execute("ROLLBACK")
                        return False, total
                    await conn.execute(
                        "INSERT INTO daily_cost (day, usd) VALUES (?, ?) "
                        "ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd",
                        (day, amount),
                    )
                    await conn.execute("COMMIT")
                    return True, total + amount
                except BaseException:
                    await conn.execute("ROLLBACK")
                    raise
            except aiosqlite.Error as exc:
                raise LedgerUnavailableError(f"sqlite ledger at {self.path}: {exc}") from exc

    async def settle(self, day: str, delta: float) -> None:
        import aiosqlite

        async with self._lock:
            try:
                conn = await self._db()
                assert isinstance(conn, aiosqlite.Connection)
                await conn.execute(
                    "UPDATE daily_cost SET usd = MAX(0, usd + ?) WHERE day = ?", (delta, day)
                )
            except aiosqlite.Error as exc:
                raise LedgerUnavailableError(f"sqlite ledger at {self.path}: {exc}") from exc

    async def spent(self, day: str) -> float:
        import aiosqlite

        try:
            conn = await self._db()
            assert isinstance(conn, aiosqlite.Connection)
            cur = await conn.execute("SELECT usd FROM daily_cost WHERE day = ?", (day,))
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0
        except aiosqlite.Error as exc:
            raise LedgerUnavailableError(f"sqlite ledger at {self.path}: {exc}") from exc

    async def close(self) -> None:
        import aiosqlite

        if isinstance(self._conn, aiosqlite.Connection):
            await self._conn.close()
        self._conn = None


class UpstashLedger:
    """Redis over HTTPS (Upstash's REST API) — httpx only, no new dependency.

    ``INCRBYFLOAT`` is atomic on the server, so the reservation is: add, look at the new
    total, and give the amount back if it overshot. A request racing another near the
    boundary can be refused by a total that is momentarily inflated by a reservation about
    to be returned — a spurious refusal, which is the safe direction for a cost guard.
    """

    kind = "upstash"

    def __init__(self, url: str, token: str, prefix: str = "arxiv-agent:daily-notional") -> None:
        if not url or not token:
            raise LedgerUnavailableError(
                "API__LEDGER=upstash needs UPSTASH_REDIS_REST_URL and _TOKEN"
            )
        self._client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        self._prefix = prefix

    def _key(self, day: str) -> str:
        return f"{self._prefix}:{day}"

    async def _pipeline(self, *commands: list[str]) -> list[object]:
        try:
            response = await self._client.post("/pipeline", json=list(commands))
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LedgerUnavailableError(f"upstash ledger: {type(exc).__name__}: {exc}") from exc
        # Assert the effect, not the status (D-026, fourth instance): a 200 whose body
        # carries a per-command error is not a successful write.
        if not isinstance(body, list) or len(body) != len(commands):
            raise LedgerUnavailableError(f"upstash ledger: unexpected response {str(body)[:200]}")
        results: list[object] = []
        for entry in body:
            if not isinstance(entry, dict) or "error" in entry:
                raise LedgerUnavailableError(f"upstash ledger: command failed: {entry}")
            results.append(entry.get("result"))
        return results

    async def reserve(self, day: str, amount: float, ceiling: float) -> tuple[bool, float]:
        key = self._key(day)
        total_raw, _ = await self._pipeline(
            ["INCRBYFLOAT", key, repr(amount)], ["EXPIRE", key, str(KEY_TTL_S)]
        )
        total = float(str(total_raw))
        if total > ceiling:
            await self._pipeline(["INCRBYFLOAT", key, repr(-amount)])
            return False, total - amount
        return True, total

    async def settle(self, day: str, delta: float) -> None:
        if delta == 0.0:
            return
        key = self._key(day)
        await self._pipeline(["INCRBYFLOAT", key, repr(delta)], ["EXPIRE", key, str(KEY_TTL_S)])

    async def spent(self, day: str) -> float:
        (raw,) = await self._pipeline(["GET", self._key(day)])
        return float(str(raw)) if raw is not None else 0.0

    async def close(self) -> None:
        await self._client.aclose()


# ── Construction and the durability guard ─────────────────────────────────────


def mount_point(path: Path) -> Path:
    """The mount point that holds ``path`` (the path need not exist yet)."""
    # Walked lexically: the ledger file and its directory may not exist until first write,
    # and a missing component is simply not a mount point.
    current = path.resolve()
    while not os.path.ismount(current) and current != current.parent:
        current = current.parent
    return current


def check_ledger_is_durable(settings: Settings) -> str | None:
    """Refuse a ledger that would forget the day's spend on restart, in a deployed container.

    A mechanism rather than an instruction (CLAUDE.md §8.9): a README line saying "use a
    persistent ledger" would be read once and then a sleeping free-tier host would reset
    the ceiling every time it woke. Returns an error string, or None.
    """
    api = settings.api
    if not api.deployed:
        return None
    if api.ledger == "memory":
        return (
            "API__LEDGER=memory in a deployed container: the daily cost ceiling would reset "
            "on every restart, and a host that sleeps when idle restarts every time it wakes. "
            "Set API__LEDGER=upstash, or API__LEDGER=sqlite on a mounted volume "
            "(docs/SERVING.md)."
        )
    if api.ledger == "sqlite" and mount_point(api.ledger_path) == Path("/"):
        return (
            f"API__LEDGER=sqlite at {api.ledger_path}, which is on the container's own "
            f"filesystem, not a mounted volume: it is discarded on restart. Mount a volume "
            f"and point API__LEDGER_PATH into it, or use API__LEDGER=upstash."
        )
    return None


def build_ledger(settings: Settings) -> Ledger:
    api = settings.api
    if api.ledger == "sqlite":
        return SqliteLedger(api.ledger_path)
    if api.ledger == "upstash":
        return UpstashLedger(
            settings.upstash_redis_rest_url,
            settings.upstash_redis_rest_token.get_secret_value(),
        )
    return MemoryLedger()
