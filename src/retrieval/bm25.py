"""BM25 sparse retrieval. Ported from v2.1 ``rag/retrieval/bm25.py``.

Tokenisation (``text.lower().split()``) is FROZEN — it is part of the retrieval contract
the v2.1 comparison rests on (docs/DECISIONS.md D-002).
"""

from __future__ import annotations

import logging
import time

from src.retrieval.chunker import Chunk

log = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        t0 = time.monotonic()
        self._chunks = chunks
        self._bm25 = BM25Okapi([c.text.lower().split() for c in chunks])
        self.build_time_s = round(time.monotonic() - t0, 3)
        log.info("BM25 built in %.3fs (%d docs)", self.build_time_s, len(chunks))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[tuple[Chunk, float]] = []
        for i in order:
            if scores[i] <= 0:
                continue
            chunk = self._chunks[i]
            if allowed_paper_ids is None or chunk.paper_id in allowed_paper_ids:
                results.append((chunk, float(scores[i])))
                if len(results) >= top_k:
                    break
        return results
