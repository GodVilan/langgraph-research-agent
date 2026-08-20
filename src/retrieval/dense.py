"""Dense BGE/FAISS retriever. Ported from v2.1 ``rag/retrieval/dense.py``."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import DEFAULT_CHUNK, INDEX_NAME
from src.retrieval.chunker import Chunk
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, emb_model: EmbeddingModel, store: VectorStore) -> None:
        self.emb_model = emb_model
        self.store = store

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        index_dir: Path,
        chunk_size: int = DEFAULT_CHUNK,
        force_rebuild: bool = False,
        emb_model: EmbeddingModel | None = None,
    ) -> DenseRetriever:
        index_name = f"BGE_cs{chunk_size}"
        index_file = Path(index_dir) / f"{index_name}.faiss"

        if index_file.exists() and not force_rebuild:
            log.info("Loading index from %s", index_dir)
            return cls(emb_model or EmbeddingModel(), VectorStore.load(index_dir, name=index_name))

        model = emb_model or EmbeddingModel()
        embeddings = model.encode([c.text for c in chunks], show_progress=True)
        store = VectorStore(dim=model.dim)
        store.add(embeddings, chunks)
        store.save(index_dir, name=index_name)
        return cls(model, store)

    @classmethod
    def load(cls, index_dir: Path, emb_model: EmbeddingModel | None = None) -> DenseRetriever:
        return cls(emb_model or EmbeddingModel(), VectorStore.load(index_dir, name=INDEX_NAME))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        q_vec = self.emb_model.encode_query(query)
        return self.store.search(q_vec, top_k=top_k, allowed_paper_ids=allowed_paper_ids)
