"""Live arXiv search and PDF fetch.

Ported from v2.1 ``rag/sources/arxiv_fetcher.py``. Two deliberate changes:

* async throughout, with ``asyncio.sleep`` for arXiv's rate policy. v2.1 called
  ``time.sleep`` inside tool functions (AUDIT §4.10), which under an async worker blocks
  the event loop and every concurrent request with it.
* an explicit timeout on every outbound call. v2.1 set one on the PDF download but none on
  anything else (AUDIT §4.9).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from src.config import DEFAULT_CHUNK
from src.retrieval.chunker import Chunk, chunks_from_bytes

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")

USER_AGENT = "arxiv-agent-v3/0.1 (research tool)"
SEARCH_TIMEOUT_S = 20.0
PDF_TIMEOUT_S = 30.0
PDF_MAX_BYTES = 25 * 1024 * 1024
ARXIV_POLITE_DELAY_S = 3.0


def normalise_id(query: str) -> str | None:
    m = _ID_RE.search(query)
    return m.group(1) if m else None


async def search_arxiv(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search arXiv and return metadata only (no PDF download).

    The ``arxiv`` library is synchronous, so it runs in a worker thread rather than
    blocking the loop.
    """

    def _search() -> list[dict[str, Any]]:
        import arxiv

        client = arxiv.Client(num_retries=2, delay_seconds=ARXIV_POLITE_DELAY_S)
        arxiv_id = normalise_id(query)
        search = (
            arxiv.Search(id_list=[arxiv_id])
            if arxiv_id
            else arxiv.Search(
                query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
            )
        )
        out: list[dict[str, Any]] = []
        for paper in client.results(search):
            out.append(
                {
                    "paper_id": paper.get_short_id().split("v")[0],
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "abstract": paper.summary.replace("\n", " ")[:500],
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "source": "arxiv",
                }
            )
        return out

    try:
        return await asyncio.wait_for(asyncio.to_thread(_search), timeout=SEARCH_TIMEOUT_S)
    except TimeoutError:
        log.warning("arXiv search timed out after %.0fs: %r", SEARCH_TIMEOUT_S, query[:60])
        return []
    except Exception as exc:
        log.warning("arXiv search failed: %s", exc)
        return []


async def fetch_paper_chunks(meta: dict[str, Any], chunk_size: int = DEFAULT_CHUNK) -> list[Chunk]:
    """Download a paper's PDF and chunk it. Returns [] on any failure."""
    pdf_url = meta.get("pdf_url", "")
    if not pdf_url:
        return []

    try:
        async with httpx.AsyncClient(
            timeout=PDF_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        ) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            data = resp.content
    except Exception as exc:
        log.warning("PDF download failed for %s: %s", meta.get("paper_id"), exc)
        return []

    if len(data) > PDF_MAX_BYTES:
        log.warning("PDF for %s exceeds %d bytes; skipping", meta.get("paper_id"), PDF_MAX_BYTES)
        return []

    # PyMuPDF parsing is CPU-bound; keep it off the event loop.
    return await asyncio.to_thread(
        chunks_from_bytes,
        data,
        str(meta["paper_id"]),
        str(meta["title"]),
        list(meta.get("authors", [])),
        "arxiv",
        chunk_size,
    )
