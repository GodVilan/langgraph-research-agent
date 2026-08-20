"""Retrieval components and the routing policy.

The component tests are ported from v2.1 ``tests/test_retrieval.py`` — they were the one
genuinely useful part of that suite — and extended. They exist to prove the ported
components behave identically, since the v2.1-vs-v3 comparison rests on that.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.chunker import Chunk, classify_section, recursive_chunk, token_count
from src.retrieval.router import SearchRouter, SourceConfig
from src.retrieval.service import RetrievalService
from src.retrieval.vector_store import VectorStore


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            "paper1_0000",
            "paper1",
            "Transformer Attention",
            ["A"],
            "The attention mechanism is very powerful.",
            6,
            0,
            "corpus",
        ),
        Chunk(
            "paper2_0000",
            "paper2",
            "LoRA low rank",
            ["B"],
            "Low-Rank adaptation makes model training efficient.",
            6,
            0,
            "arxiv",
        ),
        Chunk(
            "paper3_0000",
            "paper3",
            "Catastrophic Forgetting",
            ["C"],
            "Continual learning models suffer from catastrophic forgetting.",
            7,
            0,
            "corpus",
        ),
    ]


class FakeRetriever:
    def __init__(self, chunks: list[Chunk], base: float = 0.9) -> None:
        self.chunks, self.base = chunks, base

    def retrieve(self, query, top_k=5, allowed_paper_ids=None):
        pool = self.chunks
        if allowed_paper_ids is not None:
            pool = [c for c in pool if c.paper_id in allowed_paper_ids]
        return [(c, self.base - 0.1 * i) for i, c in enumerate(pool[:top_k])]


class TestBM25:
    def test_exact_term_match(self, chunks: list[Chunk]) -> None:
        results = BM25Retriever(chunks).retrieve("attention", top_k=1)
        assert len(results) == 1
        assert results[0][0].chunk_id == "paper1_0000"

    def test_paper_id_scoping(self, chunks: list[Chunk]) -> None:
        results = BM25Retriever(chunks).retrieve(
            "attention mechanism adaptation", top_k=5, allowed_paper_ids=frozenset({"paper1"})
        )
        assert all(c.paper_id == "paper1" for c, _ in results)

    def test_zero_score_hits_are_excluded(self, chunks: list[Chunk]) -> None:
        assert BM25Retriever(chunks).retrieve("xylophone quokka", top_k=5) == []


class TestVectorStore:
    def test_add_and_exact_match_search(self, chunks: list[Chunk]) -> None:
        store = VectorStore(dim=4)
        store.add(np.eye(3, 4, dtype=np.float32), chunks)
        results = store.search(np.array([1.0, 0, 0, 0], dtype=np.float32), top_k=1)
        assert results[0][0].chunk_id == "paper1_0000"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_paper_id_scoping(self, chunks: list[Chunk]) -> None:
        store = VectorStore(dim=4)
        store.add(np.eye(3, 4, dtype=np.float32), chunks)
        results = store.search(
            np.array([1.0, 1.0, 0, 0], dtype=np.float32),
            top_k=5,
            allowed_paper_ids=frozenset({"paper2"}),
        )
        assert all(c.paper_id == "paper2" for c, _ in results)

    def test_empty_store_returns_nothing(self) -> None:
        assert VectorStore(dim=4).search(np.zeros(4, dtype=np.float32)) == []

    def test_length_mismatch_raises_rather_than_asserting(self, chunks: list[Chunk]) -> None:
        """v2.1 used a bare `assert`, which vanishes under `python -O`."""
        with pytest.raises(ValueError, match="length mismatch"):
            VectorStore(dim=4).add(np.eye(2, 4, dtype=np.float32), chunks)


class TestSearchRouter:
    def test_merges_and_sorts_across_sources(self, chunks: list[Chunk]) -> None:
        router = SearchRouter(FakeRetriever([chunks[0]], 0.9), FakeRetriever([chunks[1]], 0.95))
        results = router.search("q", top_k=5)
        assert [c.chunk_id for c, _ in results] == ["paper2_0000", "paper1_0000"]

    def test_dedupes_on_chunk_id_keeping_the_best_score(self, chunks: list[Chunk]) -> None:
        router = SearchRouter(FakeRetriever([chunks[0]], 0.5), FakeRetriever([chunks[0]], 0.95))
        results = router.search("q", top_k=5)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(0.95)

    def test_scoping_applies_across_both_sources(self, chunks: list[Chunk]) -> None:
        router = SearchRouter(
            FakeRetriever([chunks[0], chunks[2]]), FakeRetriever([chunks[1]], 0.95)
        )
        results = router.search("q", top_k=5, allowed_paper_ids=frozenset({"paper1", "paper2"}))
        ids = {c.paper_id for c, _ in results}
        assert ids == {"paper1", "paper2"}

    def test_source_toggles(self, chunks: list[Chunk]) -> None:
        router = SearchRouter(FakeRetriever([chunks[0]]), FakeRetriever([chunks[1]], 0.95))
        corpus_only = router.search("q", source_cfg=SourceConfig(use_session=False))
        assert [c.paper_id for c, _ in corpus_only] == ["paper1"]

    def test_truncates_to_top_k(self, chunks: list[Chunk]) -> None:
        router = SearchRouter(FakeRetriever(chunks), None)
        assert len(router.search("q", top_k=2)) == 2


class TestChunker:
    def test_short_text_is_one_chunk(self) -> None:
        assert recursive_chunk("a short sentence", chunk_size=512) == ["a short sentence"]

    def test_long_text_splits(self) -> None:
        pieces = recursive_chunk(" ".join(["word"] * 2000), chunk_size=100, overlap=10)
        assert len(pieces) > 1
        assert all(token_count(p) <= 200 for p in pieces)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Abstract. We present a model.", "abstract"),
            ("1 Introduction. Prior work has shown", "introduction"),
            ("Our method uses a novel approach", "methodology"),
            ("Experiments on the CIFAR dataset", "experiments"),
            ("Conclusion. We showed that", "conclusion"),
            ("The quick brown fox jumped over", "general"),
        ],
    )
    def test_section_classification(self, text: str, expected: str) -> None:
        assert classify_section(text) == expected

    def test_classifier_precision_is_known_to_be_imperfect(self) -> None:
        """Documents the limitation rather than pretending it is not there.

        This is why the section filter ships disabled until Phase 4 can measure it
        (AUDIT §4.15).
        """
        methodology_text = "Our proposed method, as stated in the abstract, uses attention."
        assert classify_section(methodology_text) == "abstract"  # wrong, and known to be

    def test_load_repopulates_section_type(self, tmp_path) -> None:
        """The committed corpus predates the field; without this every chunk is 'general'."""
        import json

        from src.retrieval.chunker import load_chunks

        path = tmp_path / "chunks.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "chunk_id": "p_0000",
                        "paper_id": "p",
                        "title": "T",
                        "authors": [],
                        "text": "Conclusion. Future work will explore larger corpora.",
                        "token_count": 8,
                        "chunk_index": 0,
                        "source": "corpus",
                    }
                ]
            )
        )
        assert load_chunks(path, classify_on_load=False)[0].section_type == "general"
        assert load_chunks(path, classify_on_load=True)[0].section_type == "conclusion"


class TestRoutingPolicy:
    """The policy is new, unlike the components. It is what Phase 4 measures."""

    @staticmethod
    def build(dense_hits: list[Chunk], sparse_hits: list[Chunk]) -> RetrievalService:
        from src.config import RetrievalSettings

        class FakeBM25:
            def retrieve(self, query, top_k=5, allowed_paper_ids=None):
                return [(c, 1.5) for c in sparse_hits]

        service = RetrievalService.__new__(RetrievalService)
        service.dense = None  # type: ignore[assignment]
        service.bm25 = FakeBM25()  # type: ignore[assignment]
        service.session = None  # type: ignore[assignment]
        service.router = SearchRouter(FakeRetriever(dense_hits), None)
        service.cfg = RetrievalSettings(sparse_fallback_threshold=3)
        return service

    async def test_sparse_is_skipped_when_dense_delivers(self, chunks: list[Chunk]) -> None:
        service = self.build(dense_hits=chunks, sparse_hits=[])
        result = await service.retrieve("q", top_k=5)
        assert result.used_sparse is False

    async def test_sparse_fires_when_dense_under_delivers(self, chunks: list[Chunk]) -> None:
        service = self.build(dense_hits=chunks[:1], sparse_hits=chunks[1:])
        result = await service.retrieve("q", top_k=5)
        assert result.used_sparse is True
        assert len(result.hits) == 3

    async def test_arxiv_is_never_reached_without_opt_in(self) -> None:
        service = self.build(dense_hits=[], sparse_hits=[])
        result = await service.retrieve("q", top_k=5, use_arxiv=False)
        assert result.used_arxiv is False

    async def test_results_are_deduped_across_retrievers(self, chunks: list[Chunk]) -> None:
        service = self.build(dense_hits=chunks[:1], sparse_hits=chunks[:1])
        result = await service.retrieve("q", top_k=5)
        assert len({c.chunk_id for c, _, _ in result.hits}) == len(result.hits)

    async def test_section_filter_is_inert_while_disabled(self, chunks: list[Chunk]) -> None:
        service = self.build(dense_hits=chunks, sparse_hits=[])
        assert service.cfg.enable_section_filter is False
        result = await service.retrieve("q", top_k=5, section_filter="methodology")
        assert len(result.hits) == 3, "the filter must not apply while the flag is off"

    async def test_sparse_does_not_fire_when_top_k_is_below_the_threshold(
        self, chunks: list[Chunk]
    ) -> None:
        """A fixed threshold would make the "fallback" run on every query.

        With top_k=3 and sparse_fallback_threshold=5, dense can never return 5 hits, so an
        absolute comparison fires BM25 unconditionally. The trigger is min(top_k, threshold).
        """
        service = self.build(dense_hits=chunks, sparse_hits=[])
        service.cfg = service.cfg.model_copy(update={"sparse_fallback_threshold": 5})
        result = await service.retrieve("q", top_k=3)
        assert result.used_sparse is False
        assert len(result.hits) == 3

    async def test_results_are_truncated_to_top_k(self, chunks: list[Chunk]) -> None:
        """v2.1's search_corpus returned exactly top_k; the merged route must match."""
        service = self.build(dense_hits=chunks[:1], sparse_hits=chunks[1:])
        result = await service.retrieve("q", top_k=2)
        assert len(result.hits) == 2

    async def test_dense_hits_are_ordered_ahead_of_sparse(self, chunks: list[Chunk]) -> None:
        """Dense cosine scores and unbounded Okapi scores are not comparable, so the merge
        orders by retriever rather than sorting the union by raw score."""
        service = self.build(dense_hits=chunks[:1], sparse_hits=chunks[1:])
        result = await service.retrieve("q", top_k=5)
        tags = [tag for _, _, tag in result.hits]
        assert tags == ["dense", "sparse", "sparse"]
        assert result.hits[0][1] < result.hits[1][1], (
            "the sparse hit outscores the dense one numerically, and must still rank below it"
        )
