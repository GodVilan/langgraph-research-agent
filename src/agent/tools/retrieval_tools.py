"""Retrieval tools bound to a ``RetrievalService``."""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.agent.state import RetrievedChunk
from src.retrieval.chunker import Chunk
from src.retrieval.service import RetrievalService


class SearchCorpusArgs(BaseModel):
    query: str = Field(description="Natural-language search query.")
    top_k: int = Field(default=5, ge=1, le=20, description="How many passages to return.")


class KeywordSearchArgs(BaseModel):
    query: str = Field(description="Keywords or an exact phrase for BM25 lookup.")
    top_k: int = Field(default=5, ge=1, le=20)


class FetchArxivArgs(BaseModel):
    query: str = Field(description="A topic query or an arXiv id such as '2106.09685'.")
    max_results: int = Field(default=2, ge=1, le=3)


class CompareArgs(BaseModel):
    """Three typed fields, replacing v2.1's pipe-delimited single string."""

    topic_a: str = Field(description="First topic or method.")
    topic_b: str = Field(description="Second topic or method.")
    aspect: str = Field(default="comparison", description="The axis to compare along.")
    top_k: int = Field(default=6, ge=2, le=20)


def to_retrieved(
    chunk: Chunk, score: float, retriever: Annotated[str, "dense|sparse|arxiv"]
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        title=chunk.title,
        text=chunk.text,
        score=score,
        source=chunk.source,
        section_type=chunk.section_type,
        retriever=retriever,
    )


def build_retrieval_tools(service: RetrievalService) -> list[StructuredTool]:
    """Build the tool list bound to a service instance.

    Returned as ``StructuredTool``s so they carry a JSON schema for the trace and for
    Phase 5's API documentation, even though dispatch is by rule rather than by the model.
    """

    async def search_corpus(query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Dense semantic search over the paper corpus and any live-fetched papers."""
        result = await service.retrieve(query, top_k=top_k)
        return [to_retrieved(c, s, r) for c, s, r in result.hits]

    async def keyword_search(query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Sparse BM25 keyword search. Best for exact method names, authors, and ids."""
        hits = service.bm25.retrieve(query, top_k=top_k)
        return [to_retrieved(c, s, "sparse") for c, s in hits]

    async def fetch_arxiv(query: str, max_results: int = 2) -> list[RetrievedChunk]:
        """Search arXiv live, download the papers, index them, and return matching passages."""
        result = await service.retrieve(query, top_k=max_results * 3, use_arxiv=True)
        return [to_retrieved(c, s, r) for c, s, r in result.hits if r == "arxiv"]

    async def compare_papers(
        topic_a: str, topic_b: str, aspect: str = "comparison", top_k: int = 6
    ) -> list[RetrievedChunk]:
        """Retrieve passages for two topics side by side along a named aspect."""
        half = max(1, top_k // 2)
        left = await service.retrieve(f"{topic_a} {aspect}", top_k=half)
        right = await service.retrieve(f"{topic_b} {aspect}", top_k=half)
        return [to_retrieved(c, s, r) for c, s, r in (*left.hits, *right.hits)]

    return [
        StructuredTool.from_function(
            coroutine=search_corpus, name="search_corpus", args_schema=SearchCorpusArgs
        ),
        StructuredTool.from_function(
            coroutine=keyword_search, name="keyword_search", args_schema=KeywordSearchArgs
        ),
        StructuredTool.from_function(
            coroutine=fetch_arxiv, name="fetch_arxiv", args_schema=FetchArxivArgs
        ),
        StructuredTool.from_function(
            coroutine=compare_papers, name="compare_papers", args_schema=CompareArgs
        ),
    ]
