"""The HTTP service.

One process, one compiled graph, one embedding model in memory, shared by every request.
Nothing request-specific lives on a shared object: scoping travels in ``RequestOptions``
inside graph state (the AUDIT §4.14 fix) and the trace id travels in ``AgentState`` (not a
module global), so concurrent requests cannot see each other's. That is asserted by
``tests/test_api_concurrency.py``, not by this comment.

Order of checks on a query, cheapest and least trusting first:

1. **ready** — the index and model have loaded (503 otherwise);
2. **per-IP bucket** — 429 with ``Retry-After``;
3. **daily cost ceiling** — reserve the per-request ceiling in the durable ledger; 429 until
   UTC midnight when exhausted, 503 if the ledger cannot be reached (fails closed);
4. **concurrency gate** — wait up to ``queue_timeout_s`` for a slot, else 503.

Only then does the graph run. Its own per-request ceilings still apply inside, and a
tripped one returns a partial answer with ``truncated: true`` — never a silent stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Path, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.agent.graph import Graph, build_graph, sqlite_checkpointer
from src.agent.runner import new_thread_id, run_config, run_query
from src.agent.state import AgentState, RequestOptions, Usage
from src.api.ledger import (
    Ledger,
    LedgerUnavailableError,
    build_ledger,
    check_ledger_is_durable,
    seconds_until_utc_midnight,
    utc_day,
)
from src.api.limits import (
    ConcurrencyGate,
    PerClientLimiter,
    client_address,
    client_key,
    token_matches,
)
from src.api.schemas import (
    THREAD_ID_PATTERN,
    ErrorBody,
    QueryRequest,
    QueryResponse,
    ThreadView,
    Turn,
)
from src.config import Settings, get_settings
from src.observability import langfuse as lf
from src.observability import metrics

log = logging.getLogger("arxiv_agent.api")

KEEPALIVE_S = 15.0
LOADCHECK_HEADER = "x-loadcheck-token"


@dataclass
class ServiceState:
    settings: Settings
    ledger: Ledger
    limiter: PerClientLimiter
    gate: ConcurrencyGate
    graph: Graph | None = None
    status: str = "loading"  # loading | ready | failed
    load_error: str | None = None
    chunk_count: int = 0
    index_sha256: str = ""
    thread_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    # Settles started for clients that left mid-admission; held so they are not collected.
    orphaned_settles: set[asyncio.Future[None]] = field(default_factory=set)
    # Unique per process, reported by /ready. A cold-start measurement waits for a *new* boot
    # id: after a restart the old container answered /ready for seconds, and the first
    # measurement timed the wrong process (DECISIONS D-048).
    boot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class RejectedError(Exception):
    """A request refused before the graph ran. Carries its HTTP status."""

    def __init__(
        self, status: int, error: str, detail: str, retry_after_s: int | None = None
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.body = ErrorBody(error=error, detail=detail, retry_after_s=retry_after_s)

    def response(self) -> JSONResponse:
        headers = {}
        if self.body.retry_after_s is not None:
            headers["Retry-After"] = str(self.body.retry_after_s)
        return JSONResponse(self.body.json_dict(), status_code=self.status, headers=headers)


def _reject_metric(category: str) -> None:
    # Existing series only (the brief: /metrics mounts the registry, no new definitions). A
    # rejection is a request received and turned away by a guard, so it is both.
    metrics.requests_total.labels(outcome="rejected").inc()
    metrics.guardrail_triggers_total.labels(category=category, severity="block", node="api").inc()


def startup_refusals(settings: Settings) -> list[str]:
    """Configuration a deployed service must refuse to start with. Empty means start."""
    from src.observability.config import get_observability_settings
    from src.observability.langfuse import sampling_not_in_effect

    problems = [
        p
        for p in (
            get_observability_settings().check_not_deployed_with_seeded_keys(),
            check_ledger_is_durable(settings),
            sampling_not_in_effect(),
        )
        if p
    ]
    # Every query reserves the per-request ceiling before it runs, so a daily ceiling below
    # it admits nothing, ever — a service that 429s every request while reporting healthy.
    if settings.api.daily_notional_ceiling_usd < settings.budget.max_notional_cost_usd:
        problems.append(
            f"API__DAILY_NOTIONAL_CEILING_USD (${settings.api.daily_notional_ceiling_usd}) is "
            f"below the per-request reservation BUDGET__MAX_NOTIONAL_COST_USD "
            f"(${settings.budget.max_notional_cost_usd}); no query could ever be admitted."
        )
    if settings.api.deployed and not settings.google_api_key.get_secret_value():
        problems.append("GOOGLE_API_KEY is not set; every query would fail at the first call.")
    return problems


def hold_single_worker_lock(settings: Settings) -> Any:
    """Take the container's single-worker lock, or refuse to start. None when not deployed.

    ``fcntl.flock`` locks belong to the open file description, so a second worker process —
    ``uvicorn --workers 2``, or a second uvicorn in the same container — cannot take it.
    """
    if not settings.api.deployed:
        return None
    import fcntl

    path = settings.api.worker_lock_path
    handle = open(path, "a+")  # noqa: SIM115 — held for the process's lifetime on purpose
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"another worker holds {path}: this service must run as exactly one process per "
            f"container, because its rate limits and concurrency gate are in-process"
        ) from exc
    return handle


def create_app(
    settings: Settings | None = None,
    *,
    retrieval: Any = None,
    ledger: Ledger | None = None,
) -> FastAPI:
    """Build the app. ``retrieval`` and ``ledger`` are injectable so tests run without a
    1.3 GB model or a network ledger; production loads both from settings."""
    s = settings or get_settings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        problems = startup_refusals(s)
        if problems:
            for p in problems:
                log.error("refusing to start: %s", p)
            raise RuntimeError("; ".join(problems))

        # One process per container, asserted: the per-IP buckets, the concurrency gate and
        # the model pacer are in-process, so a second worker would double every limit
        # silently. An exclusive, non-blocking lock that the second worker cannot take.
        worker_lock = hold_single_worker_lock(s)

        svc = ServiceState(
            settings=s,
            ledger=ledger or build_ledger(s),
            limiter=PerClientLimiter(s.api.per_ip_per_minute, s.api.per_ip_burst),
            gate=ConcurrencyGate(s.api.max_concurrent_queries, s.api.queue_timeout_s),
        )
        app.state.svc = svc
        metrics.build_info.labels(
            version=_release(), model=s.model_name(), prompt_version=RequestOptions().prompt_version
        ).set(1)

        async with contextlib.AsyncExitStack() as stack:
            saver = await stack.enter_async_context(sqlite_checkpointer(s.checkpoint_db))
            # The model and index load in the background so /health answers at once and
            # /ready says "loading" rather than the port being closed for a minute.
            loader = asyncio.create_task(_load(svc, saver, retrieval))
            try:
                yield
            finally:
                loader.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await loader
                await svc.ledger.close()
                # The one flush the server does: pending spans on the way out, off the loop
                # and bounded, so a dead backend cannot hold shutdown open.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.to_thread(lf.flush), timeout=10.0)
                if worker_lock is not None:
                    worker_lock.close()

    app = FastAPI(
        title="arXiv Agent v3",
        version=_release(),
        summary="Graph-orchestrated research agent over a fixed 150-paper arXiv ML corpus.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        svc: ServiceState | None = getattr(request.app.state, "svc", None)
        hops = svc.settings.api.trusted_proxy_hops if svc else 0
        address = client_address(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            hops,
        )
        log.info(
            json.dumps(
                {
                    "event": "http",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.monotonic() - started) * 1000, 1),
                    "client": client_key(address),
                }
            )
        )
        # The key the per-IP limiter uses for this caller (a truncated hash, never the
        # address). Returned so the proxy hop count can be verified from outside: a forged
        # X-Forwarded-For must not change it (`make smoke-live`).
        response.headers["X-Client-Key"] = client_key(address)
        return response

    @app.get("/")
    async def index() -> dict[str, Any]:
        return {
            "service": "arXiv Agent v3",
            "corpus": "150 arXiv cs.LG papers (fixed; the agent does not search the web)",
            "endpoints": {
                "POST /query": "ask a question; SSE by default, JSON with stream=false",
                "POST /threads/{thread_id}/query": "continue a checkpointed thread",
                "GET /threads/{thread_id}": "read a thread's transcript",
                "GET /health": "liveness",
                "GET /ready": "readiness, ledger and ceiling state",
                "GET /metrics": "Prometheus exposition",
                "GET /docs": "OpenAPI",
            },
            "example": (
                "curl -sN -X POST <base>/query -H 'Content-Type: application/json' "
                '-d \'{"question": "What is LoRA?", "stream": false}\''
            ),
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        svc = _svc(request)
        body: dict[str, Any] = {
            "status": svc.status,
            "boot_id": svc.boot_id,
            "model": svc.settings.model_name(),
            "embedding_model": svc.settings.retrieval.embedding_model,
            "chunks": svc.chunk_count,
            "index_sha256": svc.index_sha256,
            "trace_sample_rate": lf.effective_sample_rate(),
            "workers_asserted_single": svc.settings.api.deployed,
            "ledger": svc.ledger.kind,
            "daily_notional_ceiling_usd": svc.settings.api.daily_notional_ceiling_usd,
            "in_flight": svc.gate.in_flight,
        }
        if svc.load_error:
            body["error"] = svc.load_error
        try:
            body["spent_today_notional_usd"] = round(await svc.ledger.spent(utc_day()), 6)
        except LedgerUnavailableError as exc:
            body["status"] = "ledger_unavailable"
            body["error"] = str(exc)
        return JSONResponse(body, status_code=200 if body["status"] == "ready" else 503)

    @app.get("/metrics")
    async def prometheus() -> Response:
        payload, content_type = metrics.exposition()
        return Response(payload, media_type=content_type)

    @app.post("/query", response_model=QueryResponse)
    async def query(body: QueryRequest, request: Request) -> Response:
        return await _handle(request, body, new_thread_id(), existing=False)

    @app.post("/threads/{thread_id}/query", response_model=QueryResponse)
    async def thread_query(
        body: QueryRequest,
        request: Request,
        thread_id: str = Path(pattern=THREAD_ID_PATTERN),
    ) -> Response:
        return await _handle(request, body, thread_id, existing=True)

    @app.get("/threads/{thread_id}", response_model=ThreadView)
    async def thread_view(
        request: Request, thread_id: str = Path(pattern=THREAD_ID_PATTERN)
    ) -> Response:
        svc = _svc(request)
        if svc.graph is None:
            return _not_ready(svc).response()
        values = await _thread_values(svc, thread_id)
        if values is None:
            return _no_thread(thread_id).response()
        turns = [
            Turn(role="user" if m.type == "human" else "assistant", content=str(m.content))
            for m in values.get("messages") or []
            if m.type in {"human", "ai"}
        ]
        view = ThreadView(
            thread_id=thread_id,
            turns=turns,
            last_answer=values.get("answer"),
            last_guardrail_blocked=bool(values.get("refused")),
            last_truncated=bool(values.get("truncated")),
        )
        return JSONResponse(view.model_dump())

    return app


# ── The query path ────────────────────────────────────────────────────────────


async def _handle(request: Request, body: QueryRequest, thread_id: str, existing: bool) -> Response:
    svc = _svc(request)
    s = svc.settings
    request_id: str = request.state.request_id
    reserved = s.budget.max_notional_cost_usd
    day = utc_day()

    try:
        options = RequestOptions(
            top_k=min(body.top_k, s.api.max_top_k),
            allowed_paper_ids=body.validated_paper_ids(),
            use_arxiv=False,
        )
    except ValueError as exc:
        return RejectedError(422, "invalid_request", str(exc)).response()

    try:
        if svc.graph is None:
            raise _not_ready(svc)
        _check_client(svc, request)
        if existing and await _thread_values(svc, thread_id) is None:
            raise _no_thread(thread_id)
        lock = svc.thread_locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise RejectedError(409, "thread_busy", "a turn is already running on this thread", 2)
        await _admit(svc, day, reserved)
    except RejectedError as rejected:
        return rejected.response()

    # From here the request holds a ledger reservation and a gate slot; `run()` returns both.

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def emit(event: dict[str, Any]) -> None:
        await queue.put(event)

    async def run() -> QueryResponse:
        """Owns the reservation, the slot and the thread lock, whatever the client does.

        A client that disconnects mid-stream does not cancel this: the run completes, its
        checkpoint stays coherent, and the ledger is settled with what it actually spent.
        """
        started = time.monotonic()
        actual = reserved
        assert svc.graph is not None
        try:
            async with svc.thread_locks[thread_id]:
                state = await run_query(
                    svc.graph,
                    body.question,
                    thread_id,
                    options,
                    settings=s,
                    on_event=emit if body.stream else None,
                    # Never flush on the request path; spans export in the background.
                    flush=False,
                )
            actual = _notional(state)
            response = QueryResponse.from_state(
                state, request_id, (time.monotonic() - started) * 1000
            )
            _log_run(request_id, response)
            return response
        except BaseException:
            actual = await _checkpointed_notional(svc, thread_id, fallback=reserved)
            raise
        finally:
            svc.gate.release()
            lock = svc.thread_locks.get(thread_id)
            if lock is not None and not lock.locked():
                svc.thread_locks.pop(thread_id, None)
            await _settle(svc, day, reserved, actual)
            await queue.put(None)

    task = asyncio.create_task(run())
    # A streaming client that disconnects never awaits the task; its outcome is logged by
    # the run itself, and this keeps asyncio from reporting an unretrieved exception.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())

    if not body.stream:
        try:
            # Shielded: a client that hangs up must not cancel a run holding a reservation.
            response = await asyncio.shield(task)
        except Exception as exc:
            return _run_failed(exc, request_id).response()
        return JSONResponse(response.model_dump())

    return StreamingResponse(
        _sse(queue, task, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(
    queue: asyncio.Queue[dict[str, Any] | None],
    task: asyncio.Task[QueryResponse],
    request_id: str,
) -> AsyncIterator[str]:
    """Progress and tokens as they happen, then one authoritative ``final`` event.

    Tokens are the draft being generated. If the critic sends the run back for another
    round, a second draft streams too; the ``final`` event's ``answer`` is the one that
    counts, including any truncation note.
    """
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_S)
        except TimeoutError:
            # Waiting on the model pacer can take a while; a comment line keeps proxies from
            # closing an idle connection and is ignored by SSE clients.
            yield ": keepalive\n\n"
            continue
        if item is None:
            break
        kind = item.pop("event")
        yield f"event: {kind}\ndata: {json.dumps(item)}\n\n"

    try:
        response = await task
    except Exception as exc:
        rejected = _run_failed(exc, request_id)
        payload = {"status": rejected.status, **rejected.body.json_dict()}
        yield f"event: error\ndata: {json.dumps(payload)}\n\n"
        return
    yield f"event: final\ndata: {response.model_dump_json()}\n\n"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _svc(request: Request) -> ServiceState:
    svc: ServiceState = request.app.state.svc
    return svc


async def _load(svc: ServiceState, saver: Any, retrieval: Any) -> None:
    try:
        if retrieval is None:
            from src.retrieval.service import RetrievalService

            retrieval = await asyncio.to_thread(RetrievalService.load, svc.settings)
        svc.chunk_count = _chunk_count(retrieval)
        svc.index_sha256 = await asyncio.to_thread(_index_sha256, svc.settings)
        svc.graph = build_graph(retrieval, checkpointer=saver, settings=svc.settings)
        svc.status = "ready"
        log.info(
            "ready: %d chunks indexed, index sha256 %s, trace sample rate in effect %s",
            svc.chunk_count,
            svc.index_sha256[:12],
            lf.effective_sample_rate(),
        )
    except Exception as exc:
        svc.status = "failed"
        svc.load_error = f"{type(exc).__name__}: {exc}"
        log.exception("startup load failed")


def _index_sha256(settings: Settings) -> str:
    """The checksum of the index file actually loaded, for comparison with INDEX.sha256."""
    import hashlib

    path = settings.index_dir / f"{settings.retrieval.index_name}.faiss"
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk_count(retrieval: Any) -> int:
    with contextlib.suppress(Exception):
        return int(retrieval.dense.store._index.ntotal)
    return 0


async def _admit(svc: ServiceState, day: str, reserved: float) -> None:
    """Reserve the ceiling, then wait for a gate slot — as one task no client can cancel.

    The deployed load check found the leak this closes: a client that disconnected while its
    request waited at the gate cancelled the handler between taking the $0.025 reservation and
    returning it. 83 dropped connections exhausted the $0.50 day with one query served — and on
    a public endpoint that is a way to deny service by opening and dropping connections
    (DECISIONS D-048). Admission now runs in its own task, awaited through `asyncio.shield`: if
    the client goes away, the task still finishes, and a done-callback hands back whatever it
    took. Raises `RejectedError` (ceiling, ledger, busy) with nothing held.
    """
    s = svc.settings

    async def admission() -> None:
        await _reserve(svc, day, reserved)
        if not await svc.gate.acquire():
            await _settle(svc, day, reserved, actual=0.0)
            _reject_metric("busy")
            raise RejectedError(
                503,
                "busy",
                f"{s.api.max_concurrent_queries} queries are already running and none finished "
                f"within {s.api.queue_timeout_s:.0f}s. The model's free-tier quota limits this "
                f"service to a few queries a minute.",
                retry_after_s=10,
            )

    task = asyncio.ensure_future(admission())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:

        def give_back(t: asyncio.Future[None]) -> None:
            # The client left. If admission succeeded, nothing will run: return the slot and
            # the reservation. If it was rejected, it already holds nothing.
            if not t.cancelled() and t.exception() is None:
                svc.gate.release()
                settle = asyncio.ensure_future(_settle(svc, day, reserved, actual=0.0))
                svc.orphaned_settles.add(settle)
                settle.add_done_callback(svc.orphaned_settles.discard)

        if task.done():
            give_back(task)
        else:
            task.add_done_callback(give_back)
        raise


def _check_client(svc: ServiceState, request: Request) -> None:
    s = svc.settings
    if token_matches(request.headers.get(LOADCHECK_HEADER), s.loadcheck_token.get_secret_value()):
        return  # the load check bypasses the per-IP bucket only; ceiling and gate still apply
    address = client_address(
        request.client.host if request.client else None,
        request.headers.get("x-forwarded-for"),
        s.api.trusted_proxy_hops,
    )
    allowed, wait = svc.limiter.acquire(address)
    if not allowed:
        _reject_metric("per_ip_rate_limit")
        raise RejectedError(
            429,
            "rate_limited",
            f"at most {s.api.per_ip_per_minute:g} queries a minute per client "
            f"(burst {s.api.per_ip_burst})",
            retry_after_s=max(1, int(wait + 0.999)),
        )


async def _reserve(svc: ServiceState, day: str, amount: float) -> None:
    ceiling = svc.settings.api.daily_notional_ceiling_usd
    try:
        admitted, total = await svc.ledger.reserve(day, amount, ceiling)
    except LedgerUnavailableError as exc:
        log.error("ledger unavailable; refusing: %s", exc)
        _reject_metric("cost_ledger_unavailable")
        raise RejectedError(
            503,
            "cost_ledger_unavailable",
            "the daily cost ledger cannot be reached, and a query that cannot be counted is "
            "not served",
            retry_after_s=30,
        ) from exc
    if not admitted:
        _reject_metric("daily_cost_ceiling")
        raise RejectedError(
            429,
            "daily_cost_ceiling",
            f"today's notional cost ceiling (${ceiling:.2f}, UTC day {day}) is reached: "
            f"${total:.4f} committed or reserved. It resets at 00:00 UTC.",
            retry_after_s=seconds_until_utc_midnight(),
        )


async def _settle(svc: ServiceState, day: str, reserved: float, actual: float) -> None:
    try:
        await svc.ledger.settle(day, actual - reserved)
    except LedgerUnavailableError as exc:
        # The reservation stays counted — an over-count, the safe direction for a ceiling.
        log.error("could not settle %.6f against the ledger: %s", actual - reserved, exc)


def _notional(state: AgentState) -> float:
    usage = state.get("usage")
    return usage.notional_cost_usd if isinstance(usage, Usage) else 0.0


async def _checkpointed_notional(svc: ServiceState, thread_id: str, fallback: float) -> float:
    """What a failed run had spent by its last checkpoint.

    Calls inside the node that failed are not in it; for the common failure — the provider
    rejecting a call on quota — those calls were not billed either. When no state can be
    read, the full reservation stays counted.
    """
    with contextlib.suppress(Exception):
        values = await _thread_values(svc, thread_id)
        if values is not None:
            return _notional(values)  # type: ignore[arg-type]
    return fallback


async def _thread_values(svc: ServiceState, thread_id: str) -> dict[str, Any] | None:
    assert svc.graph is not None
    snapshot = await svc.graph.aget_state(run_config(thread_id, svc.settings))  # type: ignore[arg-type]
    values = snapshot.values if snapshot is not None else None
    return dict(values) if values else None


def _not_ready(svc: ServiceState) -> RejectedError:
    detail = svc.load_error or "the index and embedding model are still loading"
    return RejectedError(503, f"not_{svc.status}", detail, retry_after_s=15)


def _no_thread(thread_id: str) -> RejectedError:
    return RejectedError(
        404,
        "thread_not_found",
        f"no checkpoint for thread {thread_id}. Threads live on this instance's disk and do "
        f"not survive a restart unless it is backed by a volume (docs/SERVING.md).",
    )


def _run_failed(exc: BaseException, request_id: str) -> RejectedError:
    text = f"{type(exc).__name__}: {exc}"
    if "RESOURCE_EXHAUSTED" in text or "429" in text or "quota" in text.lower():
        log.warning("request %s: model provider quota exhausted: %s", request_id, text[:300])
        return RejectedError(
            503,
            "model_quota_exhausted",
            "the model provider's free-tier quota rejected the request; try again shortly",
            retry_after_s=30,
        )
    log.error("request %s failed: %s", request_id, text[:500])
    return RejectedError(500, "internal_error", f"the run failed ({type(exc).__name__}); see logs")


def _log_run(request_id: str, response: QueryResponse) -> None:
    log.info(
        json.dumps(
            {
                "event": "run",
                "request_id": request_id,
                "thread_id": response.thread_id,
                "trace_id": response.trace_id,
                "guardrail_blocked": response.guardrail_blocked,
                "guardrail_stage": response.guardrail.stage,
                "truncated": response.truncated,
                "llm_calls": response.usage.llm_calls,
                "notional_usd": response.usage.notional_cost_usd,
                "ms": response.latency_ms,
            }
        )
    )


def _release() -> str:
    from src.observability.config import get_observability_settings

    return get_observability_settings().release
