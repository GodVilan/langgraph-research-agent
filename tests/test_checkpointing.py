"""Checkpointing and resume.

This is the honest justification for the rewrite, so it gets tested rather than asserted.
v2.1 persisted *completed turns* across 911 lines of hand-rolled SQLite; a run that died
mid-flight lost everything. ``SqliteSaver`` persists graph state after every node.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import llm as llm_module
from src.agent.graph import build_graph
from src.agent.nodes import critique as critique_node
from src.agent.nodes import generate as generate_node
from src.agent.nodes import plan as plan_node
from src.agent.runner import run_config, run_query
from src.config import Settings
from tests.fakes import FakeRetrievalService, ScriptedLLM, passing_critique, simple_plan


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    def install(script: list[object]) -> ScriptedLLM:
        fake = ScriptedLLM(script)
        for module in (plan_node, critique_node, llm_module):
            monkeypatch.setattr(module, "call_structured", fake.call_structured, raising=False)
        monkeypatch.setattr(generate_node, "call_text", fake.call_text, raising=False)
        monkeypatch.setattr("src.agent.nodes.validate_input.call_structured", fake.call_structured)
        return fake

    return install


@pytest.fixture
async def saver(tmp_path: Path):
    """Async saver: every node is a coroutine, and the sync SqliteSaver raises
    NotImplementedError on the async checkpoint API."""
    from src.agent.graph import sqlite_checkpointer

    async with sqlite_checkpointer(tmp_path / "threads.sqlite") as s:
        yield s


class TestCheckpointing:
    async def test_state_is_persisted_after_the_run(
        self, scripted, saver, settings: Settings
    ) -> None:
        scripted([simple_plan(), "An answer.", passing_critique()])
        graph = build_graph(FakeRetrievalService(), checkpointer=saver, settings=settings)  # type: ignore[arg-type]

        await run_query(graph, "What is LoRA?", "thread-a", settings=settings)
        snapshot = await graph.aget_state(run_config("thread-a", settings))

        assert snapshot.values["answer"] == "An answer."
        assert snapshot.values["thread_id"] == "thread-a"

    async def test_a_checkpoint_exists_after_every_node(
        self, scripted, saver, settings: Settings
    ) -> None:
        """The capability v2.1 lacks: state exists mid-flight, not only at turn end."""
        scripted([simple_plan(), "An answer.", passing_critique()])
        graph = build_graph(FakeRetrievalService(), checkpointer=saver, settings=settings)  # type: ignore[arg-type]

        await run_query(graph, "What is LoRA?", "thread-b", settings=settings)
        history = [h async for h in graph.aget_state_history(run_config("thread-b", settings))]

        # A checkpoint exists *before* each node runs, which is what makes mid-flight
        # resume possible. `next` names the node that checkpoint would resume into.
        assert len(history) >= 6
        pending = {node for snap in history for node in snap.next}
        assert {"validate_input", "plan", "retrieve", "generate", "critique", "finalize"} <= pending

    async def test_threads_are_isolated(self, scripted, saver, settings: Settings) -> None:
        scripted([simple_plan(), "An answer.", passing_critique()])
        graph = build_graph(FakeRetrievalService(), checkpointer=saver, settings=settings)  # type: ignore[arg-type]

        await run_query(graph, "What is LoRA?", "thread-1", settings=settings)
        await run_query(graph, "What is RLHF?", "thread-2", settings=settings)

        assert (await graph.aget_state(run_config("thread-1", settings))).values["question"] == (
            "What is LoRA?"
        )
        assert (await graph.aget_state(run_config("thread-2", settings))).values["question"] == (
            "What is RLHF?"
        )

    async def test_resume_does_not_duplicate_messages(
        self, scripted, saver, settings: Settings
    ) -> None:
        """The reason ``messages`` uses ``add_messages`` and not ``operator.add``.

        Re-invoking on the same thread must not double the transcript.
        """
        scripted([simple_plan(), "An answer.", passing_critique()])
        graph = build_graph(FakeRetrievalService(), checkpointer=saver, settings=settings)  # type: ignore[arg-type]
        config = run_config("thread-c", settings)

        await run_query(graph, "What is LoRA?", "thread-c", settings=settings)
        first = list((await graph.aget_state(config)).values["messages"])

        # Resume from the persisted checkpoint with no new input.
        await graph.ainvoke(None, config=config)
        second = list((await graph.aget_state(config)).values["messages"])

        assert [m.id for m in second] == [m.id for m in first]

    async def test_request_scope_survives_the_checkpoint(
        self, scripted, saver, settings: Settings
    ) -> None:
        """Request scope lives in state, so resume restores it — the AUDIT §4.14 fix."""
        from src.agent.state import RequestOptions

        scripted([simple_plan(), "An answer.", passing_critique()])
        graph = build_graph(FakeRetrievalService(), checkpointer=saver, settings=settings)  # type: ignore[arg-type]
        options = RequestOptions(allowed_paper_ids=frozenset({"2605.30350"}), top_k=3)

        await run_query(graph, "What is LoRA?", "thread-d", request=options, settings=settings)
        restored = (await graph.aget_state(run_config("thread-d", settings))).values["request"]

        assert restored.allowed_paper_ids == frozenset({"2605.30350"})
        assert restored.top_k == 3


class TestSerdeAllowlist:
    """Guards a failure that is invisible at write time.

    Without the allowlist, LangGraph deserialises every state model back as a plain
    ``dict``. Writing a checkpoint still succeeds; the break only surfaces on resume, when
    a node does ``state["request"].prompt_version`` and gets ``AttributeError``.
    """

    def test_every_state_model_round_trips_as_itself(self) -> None:
        from src.agent.graph import checkpoint_serde
        from src.agent.state import (
            Critique,
            GuardrailEvent,
            Plan,
            RequestOptions,
            RetrievalEvent,
            RetrievedChunk,
            Source,
            SubQuestion,
            ToolCallRecord,
            Usage,
        )

        serde = checkpoint_serde()
        samples: list[object] = [
            RequestOptions(top_k=3, allowed_paper_ids=frozenset({"2605.30350"})),
            Usage(input_tokens=5),
            Critique(),
            RetrievedChunk(chunk_id="c_0001", paper_id="c", title="t", text="x", score=0.1),
            Source(paper_id="p", title="t", score=0.5),
            SubQuestion(text="q"),
            GuardrailEvent(kind="k"),
            RetrievalEvent(query="q", retriever="dense", k=1, n_hits=1, latency_ms=1.0),
            ToolCallRecord(name="n"),
            Plan(),
        ]
        for obj in samples:
            restored = serde.loads_typed(serde.dumps_typed(obj))  # type: ignore[attr-defined]
            assert type(restored) is type(obj), (
                f"{type(obj).__name__} degraded to {type(restored).__name__}"
            )

    def test_allowlist_covers_every_model_defined_in_state(self) -> None:
        """Reflection-derived, so a newly added state model cannot fall off the list."""
        import inspect

        from pydantic import BaseModel

        from src.agent import state as state_module
        from src.agent.graph import state_model_allowlist

        defined = {
            name
            for name, obj in inspect.getmembers(state_module, inspect.isclass)
            if issubclass(obj, BaseModel) and obj.__module__ == state_module.__name__
        }
        assert {cls for _, cls in state_model_allowlist()} == defined

    def test_frozenset_scope_survives_serialisation(self) -> None:
        from src.agent.graph import checkpoint_serde
        from src.agent.state import RequestOptions

        serde = checkpoint_serde()
        original = RequestOptions(allowed_paper_ids=frozenset({"a", "b"}))
        restored = serde.loads_typed(serde.dumps_typed(original))  # type: ignore[attr-defined]
        assert restored.allowed_paper_ids == frozenset({"a", "b"})
