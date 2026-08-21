"""FAISS-backed vector store.

Ported from v2.1 ``rag/retrieval/vector_store.py``. ``IndexFlatIP`` and the search
semantics are FROZEN (docs/DECISIONS.md D-002).

Trust boundary: ``load`` unpickles the chunk sidecar. The index and its ``*_meta.pkl`` are
trusted build artifacts produced by ``make index`` from committed chunks; they are never
fetched at runtime from anywhere else. See AUDIT §4.12 and docs/DECISIONS.md D-005.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import EMBEDDING_DIM
from src.observability.tracing import span
from src.retrieval.chunker import Chunk

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

# When filtering by paper id, FAISS returns this many candidates before post-filtering.
# AUDIT §4.16: scoping to a small paper set can return fewer than top_k results because
# none of that paper's chunks ranked in the global top N. Named and documented here rather
# than buried as a literal, so the recall ceiling is visible; Phase 4 measures its cost.
FILTERED_CANDIDATE_POOL = 200


class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        import faiss

        self.dim = dim
        # Typed as Any because `load` swaps in whatever concrete Index faiss.read_index
        # returns; the store only ever calls the common add/search/ntotal surface.
        self._index: Any = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        import numpy as np

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings/chunks length mismatch: {embeddings.shape[0]} != {len(chunks)}"
            )
        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._chunks.extend(chunks)
        log.debug("VectorStore: %d vectors total", self._index.ntotal)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        import numpy as np

        if self._index.ntotal == 0:
            return []
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]

        search_k = min(
            self._index.ntotal,
            FILTERED_CANDIDATE_POOL if allowed_paper_ids is not None else top_k,
        )
        with span(
            "faiss.search",
            **{
                "faiss.ntotal": self._index.ntotal,
                "faiss.search_k": search_k,
                "faiss.filtered": allowed_paper_ids is not None,
            },
        ):
            scores, indices = self._index.search(query_vec, search_k)

        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if allowed_paper_ids is None or chunk.paper_id in allowed_paper_ids:
                results.append((chunk, float(score)))
                if len(results) >= top_k:
                    break
        return results

    def save(self, directory: Path, name: str = "index") -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / f"{name}.faiss"))
        with open(directory / f"{name}_meta.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    @classmethod
    def load(cls, directory: Path, name: str = "index") -> VectorStore:
        import faiss

        directory = Path(directory)
        index = faiss.read_index(str(directory / f"{name}.faiss"))
        with open(directory / f"{name}_meta.pkl", "rb") as f:
            chunks: list[Chunk] = pickle.load(f)
        store = cls(dim=index.d)
        store._index = index
        store._chunks = chunks
        log.info("Loaded %d vectors (dim=%d)", index.ntotal, index.d)
        return store

    @property
    def size(self) -> int:
        return int(self._index.ntotal)
