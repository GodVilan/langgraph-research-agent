"""Test doubles.

The whole graph is driven against these, so node-transition and termination tests run in
milliseconds without loading a 1.3 GB embedding model, a 22 MB index, or making a network
call.
"""

from __future__ import annotations

from typing import Any

from src.agent.state import Critique, Plan, Usage
from src.retrieval.chunker import Chunk
from src.retrieval.service import RetrievalResult


def make_chunk(
    chunk_id: str = "2605.30350_0001",
    paper_id: str = "2605.30350",
    title: str = "A Paper About Things",
    text: str = "Transformers use attention over token sequences.",
    section_type: str = "general",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=title,
        authors=["A. Author"],
        text=text,
        token_count=len(text.split()),
        chunk_index=int(chunk_id.rsplit("_", 1)[-1]),
        source="corpus",
        section_type=section_type,
    )


class FakeRetrievalService:
    """Returns a fixed hit set and counts calls."""

    def __init__(
        self,
        hits: list[tuple[Chunk, float, str]] | None = None,
        fail: bool = False,
    ) -> None:
        self.hits = hits if hits is not None else [(make_chunk(), 0.91, "dense")]
        self.fail = fail
        self.calls: list[str] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        allowed_paper_ids: frozenset[str] | None = None,
        use_arxiv: bool = False,
        section_filter: str | None = None,
    ) -> RetrievalResult:
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("retrieval exploded")
        return RetrievalResult(
            hits=list(self.hits), used_sparse=False, used_arxiv=False, latency_ms=1.0
        )


class ScriptedLLM:
    """Feeds predetermined results to the node LLM helpers via monkeypatch.

    Results are routed **by requested schema**, not by call order. Order-based scripting
    is brittle here: whether ``validate_input`` calls the model at all depends on the
    keyword fast path, so a question containing "compare" and one containing "LoRA" have
    different call sequences for identical graph behaviour.

    Each queue is consumed in order; the last entry repeats once exhausted, which is what
    lets a loop test script one critique and have it returned on every round. A
    ``BaseException`` entry is raised instead of returned.
    """

    def __init__(self, script: list[Any]) -> None:
        self.queues: dict[str, list[Any]] = {}
        self.calls: list[tuple[str, str]] = []
        for item in script:
            self.queues.setdefault(self._key_for(item), []).append(item)

    @staticmethod
    def _key_for(item: Any) -> str:
        if isinstance(item, str):
            return "text"
        if isinstance(item, BaseException):
            return "error"
        return type(item).__name__

    def _next(self, key: str) -> Any:
        queue = self.queues.get(key)
        if not queue:
            errors = self.queues.get("error")
            if errors:
                raise errors[0]
            raise AssertionError(
                f"ScriptedLLM has nothing queued for {key!r}; queued: {sorted(self.queues)}"
            )
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item

    async def call_structured(
        self, schema: type, system: str, user: str, max_attempts: int = 2, model: Any = None
    ) -> tuple[Any, Usage]:
        name = schema.__name__
        self.calls.append((name, user[:60]))
        usage = Usage(input_tokens=100, output_tokens=20, llm_calls=1)
        # Scope defaults to in-scope unless a test explicitly queues a verdict, so tests
        # about graph flow do not have to care whether the keyword fast path fired.
        if name == "ScopeVerdict" and not self.queues.get(name):
            return schema(in_scope=True, reason="default"), usage
        return self._next(name), usage

    async def call_text(self, system: str, user: str, model: Any = None) -> tuple[str, Usage]:
        self.calls.append(("text", user[:60]))
        return self._next("text"), Usage(input_tokens=200, output_tokens=80, llm_calls=1)

    def count(self, key: str) -> int:
        return sum(1 for name, _ in self.calls if name == key)


def simple_plan() -> Plan:
    return Plan(kind="simple", sub_questions=[])


def complex_plan(*questions: str) -> Plan:
    return Plan(kind="complex", sub_questions=list(questions))


def passing_critique() -> Critique:
    return Critique(verdict="pass")


def retry_critique(*hints: str) -> Critique:
    return Critique(verdict="retry", grounded=False, search_hints=list(hints), gaps=["thin"])
