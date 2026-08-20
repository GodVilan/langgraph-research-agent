"""Retrieval service — component wiring plus the deterministic routing policy.

The components below are ported unchanged from v2.1. The *policy* is new, and is the
substance of the rebuild: v2.1 let the model choose between ``search_corpus``,
``keyword_search``, and ``fetch_arxiv`` by emitting a JSON action, and then spent ~90 lines
of loop-guard machinery stopping it from choosing badly (AUDIT §2.2, MIGRATION_MAP §5.1).
Here the route is a rule:

1. dense (corpus + session, merged by ``SearchRouter``) — always;
2. sparse BM25 — only when dense returns fewer than ``min(top_k,
   sparse_fallback_threshold)`` hits;
3. live arXiv — only when both come up short *and* the request opted in.

Changing the *policy* is the change Phase 4 measures. Changing the *components* would
invalidate the comparison, so they stay frozen.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from src.config import INDEX_NAME, RetrievalSettings, Settings, get_settings
from src.retrieval.arxiv_client import fetch_paper_chunks, search_arxiv
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.chunker import Chunk, load_chunks
from src.retrieval.dense import DenseRetriever
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.router import SearchRouter
from src.retrieval.session_index import SessionIndex
from src.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)

MAX_ARXIV_PAPERS_PER_CALL = 2


class RetrievalResult:
    """Hits from one routed retrieval, with the per-retriever breakdown preserved."""

    __slots__ = ("hits", "latency_ms", "used_arxiv", "used_sparse")

    def __init__(
        self,
        hits: list[tuple[Chunk, float, str]],
        used_sparse: bool,
        used_arxiv: bool,
        latency_ms: float,
    ) -> None:
        self.hits = hits
        self.used_sparse = used_sparse
        self.used_arxiv = used_arxiv
        self.latency_ms = latency_ms


class RetrievalService:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        session: SessionIndex,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.session = session
        self.router = SearchRouter(dense, session)
        self.cfg = settings or get_settings().retrieval

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, settings: Settings | None = None) -> RetrievalService:
        """Load the corpus, the prebuilt index, and BM25 from disk."""
        s = settings or get_settings()
        chunks = load_chunks(s.chunks_path, classify_on_load=s.retrieval.classify_sections_on_load)
        log.info("Loaded %d chunks from %s", len(chunks), s.chunks_path)

        emb = EmbeddingModel()
        index_file = Path(s.index_dir) / f"{INDEX_NAME}.faiss"
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. Run `make index` to build it "
                f"from the committed chunks."
            )
        dense = DenseRetriever(emb, VectorStore.load(s.index_dir, name=INDEX_NAME))
        return cls(dense, BM25Retriever(chunks), SessionIndex(emb), s.retrieval)

    # ── Policy ────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | None = None,
        use_arxiv: bool = False,
        section_filter: str | None = None,
    ) -> RetrievalResult:
        t0 = time.monotonic()

        # "Under-delivered" is relative to what was asked for. A fixed threshold would fire
        # the fallback unconditionally whenever top_k is below it — with top_k=3 and a
        # threshold of 5, dense can never return 5 hits, so sparse would run on every
        # single query and the "fallback" would not be a fallback at all.
        enough = min(top_k, self.cfg.sparse_fallback_threshold)

        dense_hits = await asyncio.to_thread(
            self.router.search, query, top_k, None, allowed_paper_ids
        )
        hits: list[tuple[Chunk, float, str]] = [(c, s, "dense") for c, s in dense_hits]

        used_sparse = False
        if len(hits) < enough:
            used_sparse = True
            sparse_hits = await asyncio.to_thread(
                self.bm25.retrieve, query, top_k, allowed_paper_ids
            )
            hits.extend((c, s, "sparse") for c, s in sparse_hits)

        used_arxiv = False
        if use_arxiv and len({c.chunk_id for c, _, _ in hits}) < enough:
            used_arxiv = True
            hits.extend(await self._fetch_and_index(query, top_k, allowed_paper_ids))

        if section_filter and self.cfg.enable_section_filter:
            hits = [h for h in hits if h[0].section_type == section_filter]

        return RetrievalResult(
            hits=self._merge(hits, top_k),
            used_sparse=used_sparse,
            used_arxiv=used_arxiv,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    @staticmethod
    def _merge(hits: list[tuple[Chunk, float, str]], top_k: int) -> list[tuple[Chunk, float, str]]:
        """Dedupe by ``chunk_id`` and truncate to ``top_k``, preserving retriever order.

        Dense and BM25 scores live on different scales — cosine similarity in [-1, 1]
        against unbounded Okapi scores — so sorting the union by raw score would be
        meaningless ranking dressed up as a number. Instead the order is: dense hits in
        their own rank order first, then whatever sparse and arXiv added that dense missed.
        Truncating to ``top_k`` keeps the contract identical to v2.1's ``search_corpus``,
        which returned exactly ``top_k`` passages.
        """
        seen: set[str] = set()
        merged: list[tuple[Chunk, float, str]] = []
        for retriever in ("dense", "sparse", "arxiv"):
            for chunk, score, tag in hits:
                if tag != retriever or chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                merged.append((chunk, score, tag))
                if len(merged) >= top_k:
                    return merged
        return merged

    async def _fetch_and_index(
        self, query: str, top_k: int, allowed_paper_ids: frozenset[str] | None
    ) -> list[tuple[Chunk, float, str]]:
        """Fetch up to two papers from arXiv, index them, then re-query the session index."""
        papers = await search_arxiv(query, max_results=MAX_ARXIV_PAPERS_PER_CALL)
        for meta in papers[:MAX_ARXIV_PAPERS_PER_CALL]:
            if self.session.has_paper(str(meta["paper_id"])):
                continue
            chunks = await fetch_paper_chunks(meta)
            if chunks:
                await asyncio.to_thread(self.session.add_chunks, chunks, meta)

        session_hits = await asyncio.to_thread(
            self.session.retrieve, query, top_k, allowed_paper_ids
        )
        return [(c, s, "arxiv") for c, s in session_hits]

    # ── Corpus metadata ───────────────────────────────────────────────────────

    def corpus_paper_ids(self) -> set[str]:
        settings = get_settings()
        with open(settings.metadata_path) as f:
            return {str(p["paper_id"]) for p in json.load(f)}
