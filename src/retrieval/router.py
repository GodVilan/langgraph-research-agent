"""Unified dense search across the static corpus and the live session index.

Ported from v2.1 ``rag/sources/source_router.py``. The merge rule — dedupe by
``chunk_id`` keeping the best score, sort descending, truncate to ``top_k`` — is FROZEN
(docs/DECISIONS.md D-002).

One deletion: v2.1's ``section_type`` filter branch. AUDIT §4.15 found it had no call
site anywhere in the repository and could not have worked on the shipped corpus. Section
filtering now lives in ``RetrievalService`` behind a config flag that defaults to off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from src.retrieval.chunker import Chunk

log = logging.getLogger(__name__)


class SupportsRetrieve(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | None = None,
    ) -> list[tuple[Chunk, float]]: ...


@dataclass(frozen=True)
class SourceConfig:
    use_corpus: bool = True
    use_session: bool = True


class SearchRouter:
    """Fans a query out to the corpus retriever and the session index, then merges."""

    def __init__(
        self,
        corpus_retriever: SupportsRetrieve | None,
        session_index: SupportsRetrieve | None = None,
    ) -> None:
        self._corpus = corpus_retriever
        self._session = session_index

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_cfg: SourceConfig | None = None,
        allowed_paper_ids: frozenset[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        cfg = source_cfg or SourceConfig()
        all_results: list[tuple[Chunk, float]] = []

        if cfg.use_corpus and self._corpus is not None:
            all_results.extend(
                self._corpus.retrieve(query, top_k=top_k, allowed_paper_ids=allowed_paper_ids)
            )

        if cfg.use_session and self._session is not None:
            all_results.extend(
                self._session.retrieve(query, top_k=top_k, allowed_paper_ids=allowed_paper_ids)
            )

        # Dedupe: keep the best score per chunk_id.
        seen: dict[str, tuple[Chunk, float]] = {}
        for chunk, score in all_results:
            if chunk.chunk_id not in seen or score > seen[chunk.chunk_id][1]:
                seen[chunk.chunk_id] = (chunk, score)

        merged = sorted(seen.values(), key=lambda x: -x[1])
        return merged[:top_k]
