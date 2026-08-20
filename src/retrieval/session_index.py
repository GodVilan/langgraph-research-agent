"""In-memory index for papers fetched live from arXiv during a run.

Ported from v2.1 ``rag/sources/session_index.py`` with its persistence layer removed. The
SQLite table and the FAISS-file rewrite on every added paper (AUDIT §4.18) existed to
survive Streamlit reruns; graph state is checkpointed instead, so the session index is now
just a process-local cache of embedded chunks.

It is deliberately process-global and additive. A paper fetched by one request becomes
visible to later ones, which is a cache hit rather than a leak — every chunk in it is
public arXiv text. Access is guarded by a lock because FastAPI will drive it concurrently
in Phase 5.
"""

from __future__ import annotations

import logging
import threading

from src.retrieval.chunker import Chunk
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


class SessionIndex:
    def __init__(self, emb_model: EmbeddingModel) -> None:
        self._emb = emb_model
        self._store = VectorStore(dim=emb_model.dim)
        self._chunk_ids: set[str] = set()
        self._papers: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()

    def add_chunks(self, chunks: list[Chunk], metadata: dict[str, object]) -> int:
        """Embed and add chunks. Returns the number actually added.

        Does not mutate ``metadata`` — v2.1 wrote ``paper_id`` back into the caller's dict
        (AUDIT §4.17).
        """
        paper_id = str(metadata.get("paper_id") or metadata.get("title") or "unknown")

        with self._lock:
            new = [c for c in chunks if c.chunk_id not in self._chunk_ids]
            if not new:
                return 0
            embeddings = self._emb.encode([c.text for c in new], show_progress=False)
            self._store.add(embeddings, new)
            self._chunk_ids.update(c.chunk_id for c in new)
            self._papers[paper_id] = {**metadata, "paper_id": paper_id}

        log.info("Session index: added %d chunks for %s", len(new), paper_id)
        return len(new)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._store.size == 0:
                return []
            store = self._store
        return store.search(
            self._emb.encode_query(query), top_k=top_k, allowed_paper_ids=allowed_paper_ids
        )

    def has_paper(self, paper_id: str) -> bool:
        with self._lock:
            return paper_id in self._papers

    def list_papers(self) -> list[dict[str, object]]:
        with self._lock:
            return list(self._papers.values())

    @property
    def chunk_count(self) -> int:
        return self._store.size

    def is_empty(self) -> bool:
        return self._store.size == 0
