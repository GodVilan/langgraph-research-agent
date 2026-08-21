"""BGE embedding model wrapper.

Ported from v2.1 ``rag/retrieval/embeddings.py``. Model, normalisation, and the BGE query
instruction prefix are FROZEN (docs/DECISIONS.md D-002).

Deliberate change: v2.1's pickle-backed encode cache is dropped. It was a second pickle
load path (AUDIT §4.12) whose only user was index construction, which already caches its
output as the FAISS file itself.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import BGE_QUERY_PREFIX, EMBEDDING_MODEL, get_settings, resolve_device
from src.observability.tracing import span

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or resolve_device(settings.device)

        token = settings.hf_token.get_secret_value() or None
        log.info("Loading %s on %s", model_name, self.device)
        self._model = SentenceTransformer(model_name, device=self.device, token=token)
        # Renamed in sentence-transformers 6; keep the old name as a fallback so the
        # ported code runs on either major version.
        get_dim = getattr(
            self._model, "get_embedding_dimension", self._model.get_sentence_embedding_dimension
        )
        dim = get_dim()
        if dim is None:
            raise RuntimeError(f"{model_name} reported no embedding dimension")
        self.dim: int = dim
        log.info("Embedding dim = %d", self.dim)

    def encode(
        self,
        texts: list[str],
        normalise: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        import numpy as np

        with span(
            "embedding.encode",
            **{
                "embedding.model": self.model_name,
                "embedding.device": self.device,
                "embedding.n_texts": len(texts),
            },
        ):
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalise,
            ).astype(np.float32)

        if normalise:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = (embeddings / np.maximum(norms, 1e-10)).astype(np.float32)

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query with the BGE retrieval instruction prefix. FROZEN."""
        import numpy as np

        with span(
            "embedding.encode_query",
            **{"embedding.model": self.model_name, "embedding.device": self.device},
        ):
            vec = self._model.encode(
                [BGE_QUERY_PREFIX + query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(norms, 1e-10)
