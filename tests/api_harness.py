"""Drive the FastAPI app in-process, with its lifespan, against fakes.

``httpx.ASGITransport`` does not run the lifespan, and the lifespan is where the checkpointer
opens and the graph is built, so it is entered by hand here. The retrieval service and the
model helpers are fakes (tests/fakes.py); nothing loads a 1.3 GB model or calls a provider.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI

from src.api.app import create_app
from src.api.ledger import Ledger, MemoryLedger
from src.config import Settings


@contextlib.asynccontextmanager
async def running(
    settings: Settings,
    retrieval: Any,
    ledger: Ledger | None = None,
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    app = create_app(settings, retrieval=retrieval, ledger=ledger or MemoryLedger())
    async with app.router.lifespan_context(app):
        for _ in range(200):
            if app.state.svc.status != "loading":
                break
            await asyncio.sleep(0.01)
        transport = httpx.ASGITransport(app=app, client=("203.0.113.7", 4321))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client


def sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body into (event, data) pairs, skipping keepalive comments."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.strip().split("\n\n"):
        kind, data = "message", ""
        for line in block.splitlines():
            if line.startswith("event: "):
                kind = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if data:
            events.append((kind, json.loads(data)))
    return events
