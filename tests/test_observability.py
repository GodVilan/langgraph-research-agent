"""Observability tests.

Two things get tested here, and the first matters more than the second:

1. **Observability never breaks the agent.** A tracing backend that is unreachable,
   misconfigured, or absent must degrade to a no-op. An agent that fails because its
   telemetry failed is worse than an agent with no telemetry.
2. The metrics and trace payload carry what the Phase 3 brief requires, so a reviewer
   looking at a trace can answer "what did this cost" and "what did the guardrails do".
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from src.agent.state import (
    Critique,
    GuardrailEvent,
    RetrievalEvent,
    SubQuestion,
    ToolCallRecord,
    Usage,
    initial_state,
)
from src.observability import langfuse as lf
from src.observability import metrics as m
from src.observability.config import ObservabilitySettings


def sample_state() -> dict[str, object]:
    state = dict(initial_state("What is LoRA?", "thread-obs"))
    state.update(
        answer="LoRA is a low-rank adapter. [2605.1_0001]",
        plan=[SubQuestion(text="What is LoRA?")],
        usage=Usage(
            input_tokens=7295,
            output_tokens=465,
            reasoning_tokens=120,
            cost_usd=0.0,
            notional_cost_usd=0.00335,
            llm_calls=3,
            tool_calls=1,
        ),
        critique=Critique(verdict="pass"),
        retrieval_events=[
            RetrievalEvent(
                query="What is LoRA?",
                retriever="dense",
                k=5,
                n_hits=5,
                latency_ms=52.0,
                hit_chunk_ids=["2605.1_0001", "2605.1_0002"],
            )
        ],
        tool_calls=[ToolCallRecord(name="search_corpus", args={"query": "LoRA"}, ok=True)],
        guardrail_events=[
            GuardrailEvent(kind="scope_keyword_fastpath", severity="info", node="validate_input"),
            GuardrailEvent(
                kind="injection_instruction_override",
                severity="block",
                node="retrieve",
                chunk_id="2605.9_0001",
                detail="ignore_previous",
            ),
            GuardrailEvent(kind="injection_quarantine", severity="block", node="retrieve"),
        ],
    )
    return state


class TestDegradesToNoOp:
    """The property that matters most: telemetry failure must not be agent failure."""

    def test_client_is_none_without_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(lf, "_CLIENT", None)
        monkeypatch.setattr(lf, "_CLIENT_TRIED", False)
        # Explicit empty keys. `ObservabilitySettings()` reads .env, so on a machine with
        # Langfuse actually running this would assert the opposite of what it means.
        monkeypatch.setattr(
            lf,
            "get_observability_settings",
            lambda: ObservabilitySettings(
                langfuse_public_key=SecretStr(""), langfuse_secret_key=SecretStr("")
            ),
        )
        assert lf.get_client() is None
        assert lf.callback_handler() is None

    def test_trace_run_yields_none_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(lf, "get_client", lambda: None)
        with lf.trace_run(
            name="query", thread_id="t", question="q", model="m", prompt_version="v2"
        ) as root:
            assert root is None

    def test_trace_run_survives_a_client_that_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Exploding:
            def start_as_current_observation(self, **_: object) -> object:
                raise RuntimeError("langfuse is down")

        monkeypatch.setattr(lf, "get_client", lambda: Exploding())
        with lf.trace_run(
            name="query", thread_id="t", question="q", model="m", prompt_version="v2"
        ) as root:
            assert root is None

    def test_record_state_on_none_root_is_a_no_op(self) -> None:
        lf.record_state(None, sample_state())

    def test_record_state_swallows_a_broken_span(self) -> None:
        class Broken:
            def update(self, **_: object) -> None:
                raise RuntimeError("span already ended")

            def update_trace(self, **_: object) -> None:
                raise RuntimeError("span already ended")

            def create_event(self, **_: object) -> None:
                raise RuntimeError("span already ended")

        lf.record_state(Broken(), sample_state())

    def test_flush_is_safe_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(lf, "get_client", lambda: None)
        lf.flush()

    def test_spans_work_without_a_configured_provider(self) -> None:
        from src.observability.tracing import span

        with span("test.span", **{"test.attr": 1}) as sp:
            assert sp is not None

    def test_span_reraises_the_body_but_not_the_tracer(self) -> None:
        from src.observability.tracing import span

        with pytest.raises(ValueError, match="body failed"), span("test.span"):
            raise ValueError("body failed")


class TestTracePayload:
    """What a reviewer needs the trace to answer."""

    def test_state_is_attached_with_cost_guardrails_and_retrieval(self) -> None:
        captured: dict[str, object] = {}

        class Recorder:
            def update(self, **kwargs: object) -> None:
                captured.setdefault("update", []).append(kwargs)  # type: ignore[union-attr]

            def update_trace(self, **kwargs: object) -> None:
                captured.setdefault("trace", []).append(kwargs)  # type: ignore[union-attr]

            def create_event(self, **kwargs: object) -> None:
                captured.setdefault("events", []).append(kwargs)  # type: ignore[union-attr]

        lf.record_state(Recorder(), sample_state())
        metadata = captured["update"][-1]["metadata"]  # type: ignore[index]

        assert metadata["usage"]["cost_usd_billed"] == 0.0
        assert metadata["usage"]["cost_usd_notional"] == 0.00335
        assert metadata["usage"]["thinking_tokens"] == 120
        assert metadata["retrieval"][0]["hit_chunk_ids"] == ["2605.1_0001", "2605.1_0002"]
        assert metadata["tool_calls"][0]["name"] == "search_corpus"
        assert metadata["guardrail_summary"] == {"block": 2, "warn": 0}
        assert metadata["critique_verdict"] == "pass"

    def test_billed_and_notional_cost_stay_separate(self) -> None:
        """Collapsing them would read zero forever on the free tier, or imply spend."""
        captured: list[dict[str, object]] = []

        class Recorder:
            def update(self, **kwargs: object) -> None:
                captured.append(kwargs)

            def update_trace(self, **kwargs: object) -> None: ...
            def create_event(self, **kwargs: object) -> None: ...

        lf.record_state(Recorder(), sample_state())
        usage_update = next(c for c in captured if "cost_details" in c)
        assert usage_update["cost_details"] == {"total": 0.0}
        metadata = captured[-1]["metadata"]
        assert metadata["usage"]["cost_usd_notional"] > 0  # type: ignore[index]

    def test_block_and_warn_events_are_emitted_info_is_not(self) -> None:
        events: list[dict[str, object]] = []

        class Recorder:
            def update(self, **_: object) -> None: ...
            def update_trace(self, **_: object) -> None: ...

            def create_event(self, **kwargs: object) -> None:
                events.append(kwargs)

        lf.record_state(Recorder(), sample_state())
        names = {e["name"] for e in events}
        assert "guardrail:injection_instruction_override" in names
        assert "guardrail:scope_keyword_fastpath" not in names, "info events are noise"

    def test_trace_is_tagged_for_filtering(self) -> None:
        state = sample_state()
        state["truncated"] = True
        tags = lf._trace_tags(state, state["guardrail_events"])  # type: ignore[arg-type]
        assert "truncated" in tags
        assert "guardrail:block" in tags


class TestMetrics:
    def test_exposition_renders(self) -> None:
        payload, content_type = m.exposition()
        assert b"arxiv_agent_requests_total" in payload
        assert "openmetrics" in content_type or "text/plain" in content_type

    def test_record_run_populates_the_required_metrics(self) -> None:
        m.record_run(sample_state(), elapsed_s=4.2)
        payload = m.exposition()[0].decode()

        assert 'arxiv_agent_requests_total{outcome="answered"}' in payload
        assert 'arxiv_agent_tokens_total{kind="thinking"}' in payload
        assert 'arxiv_agent_cost_usd_total{kind="notional"}' in payload
        assert 'arxiv_agent_cost_usd_total{kind="billed"}' in payload
        assert "arxiv_agent_request_latency_seconds_bucket" in payload

    def test_guardrail_counter_separates_warn_from_block(self) -> None:
        """A warn withholds nothing; a block removes a passage. One counter would mislead."""
        state = sample_state()
        state["guardrail_events"] = [
            GuardrailEvent(kind="injection_output_control", severity="warn", node="retrieve"),
            GuardrailEvent(kind="injection_delimiter_escape", severity="block", node="retrieve"),
        ]
        m.record_run(state, elapsed_s=1.0)
        payload = m.exposition()[0].decode()
        assert 'severity="warn"' in payload
        assert 'severity="block"' in payload

    def test_refused_and_truncated_are_distinct_outcomes(self) -> None:
        refused = sample_state()
        refused["refused"] = True
        m.record_run(refused, elapsed_s=0.3)

        truncated = sample_state()
        truncated["truncated"] = True
        m.record_run(truncated, elapsed_s=9.0)

        payload = m.exposition()[0].decode()
        assert 'outcome="refused"' in payload
        assert 'outcome="truncated"' in payload

    def test_errors_are_labelled_by_type(self) -> None:
        m.record_error("GraphRecursionError", elapsed_s=2.0)
        payload = m.exposition()[0].decode()
        assert 'error_type="GraphRecursionError"' in payload

    def test_missing_usage_metadata_is_surfaced(self) -> None:
        state = sample_state()
        state["usage"] = Usage(llm_calls=1, missing_usage_metadata=2)
        m.record_run(state, elapsed_s=1.0)
        assert "arxiv_agent_missing_usage_metadata_total" in m.exposition()[0].decode()
