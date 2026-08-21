"""Cost accounting: the two figures must reconcile, and the table must refuse to lie.

These tests exist because `make budget` published a blended rate of $0.17 per 1M tokens
against a $0.30/$2.50 rate card. No token mix can do that. The cause was not a wrong rate —
it was the token columns summing 187 synthetic traces from the test suite while the cost
column summed only the 6 real runs that had been priced. See DECISIONS D-021.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage

from scripts.budget_from_traces import check_blended_rate, summarise
from scripts.reconcile_cost import (
    FIXTURE,
    classify,
    find_duplicate_roots,
    load_fixture,
    notional_for,
)
from src.agent.llm import usage_from_message
from src.config import PRICING, Settings

RATES = PRICING["gemini-3.5-flash-lite"]
BASE = datetime(2026, 8, 20, 17, 51, tzinfo=UTC)


def trace(
    *,
    tid: str = "t",
    name: str = "query",
    usage: dict[str, Any] | None = None,
    langfuse_cost: float = 0.0,
    question: str | None = None,
    at: datetime | None = None,
    environment: str = "development",
) -> SimpleNamespace:
    """A stand-in for a Langfuse trace carrying only the fields the scripts read."""
    metadata: dict[str, Any] = {}
    if usage is not None:
        metadata["usage"] = usage
    return SimpleNamespace(
        id=tid,
        name=name,
        timestamp=at or BASE,
        session_id=None,
        environment=environment,
        metadata=metadata,
        total_cost=langfuse_cost,
        input={"question": question} if question else None,
    )


def priced_usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """Usage as a real run records it — tokens priced at the configured rates."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": 0,
        "cost_usd_billed": 0.0,
        "cost_usd_notional": notional_for(input_tokens, output_tokens, RATES),
    }


def synthetic_usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """Usage as ``tests/fakes.py`` records it — round token counts, nothing priced."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": 0,
        "cost_usd_billed": 0.0,
        "cost_usd_notional": 0.0,
    }


class TestPopulationsAreNotMerged:
    def test_synthetic_tokens_stay_out_of_the_token_columns(self) -> None:
        """The exact shape of the bug: many fake traces, few real ones, one table."""
        traces = [trace(tid=f"real{i}", usage=priced_usage(955, 112)) for i in range(6)]
        traces += [trace(tid=f"fake{i}", usage=synthetic_usage(400, 120)) for i in range(121)]

        rows = summarise(traces)["development"]

        assert rows["priced"] == 6
        assert rows["unpriced"] == 121
        # 6 x 955, not 6 x 955 + 121 x 400.
        assert rows["input"] == 6 * 955
        assert rows["output"] == 6 * 112

    def test_traces_without_usage_metadata_are_counted_but_not_summed(self) -> None:
        rows = summarise([trace(usage=None, langfuse_cost=0.00594)])["development"]

        assert rows["no_usage"] == 1
        assert rows["input"] == 0
        # Langfuse's own estimate never leaks into our columns.
        assert rows["notional"] == 0

    def test_a_trace_contributing_tokens_also_contributes_cost(self) -> None:
        """The invariant that makes the blended rate meaningful."""
        traces = [
            trace(tid="a", usage=priced_usage(1000, 100)),
            trace(tid="b", usage=synthetic_usage(9999, 9999)),
            trace(tid="c", usage=None),
        ]
        row = summarise(traces)["development"]

        assert (row["input"] + row["output"] > 0) == (row["notional"] > 0)


class TestBlendedRateInvariant:
    def test_a_consistent_table_passes(self) -> None:
        rows = summarise([trace(usage=priced_usage(29287, 1477))])

        assert check_blended_rate(rows) is None

    def test_the_impossible_blend_is_caught(self) -> None:
        """Cost from one population, tokens from another — below the input floor."""
        rows = {
            "development": {
                "input": 100_787.0,
                "output": 22_817.0,
                "notional": 0.01248,
                "priced": 6.0,
                "unpriced": 0.0,
                "no_usage": 0.0,
                "traces": 214.0,
                "thinking": 0.0,
                "billed": 0.0,
            }
        }
        error = check_blended_rate(rows)

        assert error is not None
        assert "outside" in error

    def test_an_empty_window_is_not_an_error(self) -> None:
        assert check_blended_rate({}) is None

    @pytest.mark.parametrize(
        ("input_tokens", "output_tokens"),
        [(1, 0), (0, 1), (10_000, 1), (1, 10_000), (5587, 533)],
    )
    def test_any_real_token_mix_lands_inside_the_rate_card(
        self, input_tokens: int, output_tokens: int
    ) -> None:
        """The invariant is a property of the arithmetic, not of one observed run."""
        rows = summarise([trace(usage=priced_usage(input_tokens, output_tokens))])

        assert check_blended_rate(rows) is None


class TestReconciliation:
    def test_recompute_matches_what_the_agent_stored(self) -> None:
        """The three-way check: the calculator and the rate table cannot drift apart."""
        message = AIMessage(
            content="x",
            usage_metadata={"input_tokens": 955, "output_tokens": 112, "total_tokens": 1067},
        )
        settings = Settings(
            google_api_key="k",  # type: ignore[arg-type]
            agent_model="google_genai:gemini-3.5-flash-lite",
        )
        stored = usage_from_message(message, settings).notional_cost_usd

        assert stored == pytest.approx(notional_for(955, 112, RATES))

    def test_classify_splits_the_three_populations(self) -> None:
        groups = classify(
            [
                trace(tid="a", usage=priced_usage(955, 112)),
                trace(tid="b", usage=synthetic_usage(400, 120)),
                trace(tid="c", usage=None),
            ],
            RATES,
        )

        assert [len(groups[k]) for k in ("priced", "unpriced", "no_usage")] == [1, 1, 1]

    def test_duplicate_roots_are_paired(self) -> None:
        """The pre-fix symptom: two roots per query, cost on one, metadata on the other."""
        question = "What is catastrophic forgetting?"
        pairs = find_duplicate_roots(
            [
                trace(tid="q", name="query", question=question, at=BASE),
                trace(
                    tid="g",
                    name="LangGraph",
                    question=question,
                    at=BASE + timedelta(milliseconds=15),
                    langfuse_cost=0.00398,
                ),
            ]
        )

        assert len(pairs) == 1
        assert pairs[0][0]["id"] == "q"
        assert pairs[0][1]["cost"] == pytest.approx(0.00398)

    def test_two_separate_runs_of_one_question_are_not_paired(self) -> None:
        """Asking the same thing twice is not the double-trace bug."""
        question = "What is LoRA?"
        pairs = find_duplicate_roots(
            [
                trace(tid="q", name="query", question=question, at=BASE),
                trace(
                    tid="g",
                    name="LangGraph",
                    question=question,
                    at=BASE + timedelta(minutes=5),
                ),
            ]
        )

        assert pairs == []


class TestD021IsReproducibleFromCommittedEvidence:
    """The 214-trace window that D-021 was derived from, frozen and asserted.

    `make langfuse-reset` cleared the live store to give Phase 4 a clean window. Without
    this fixture that would have left a documented finding whose evidence no longer exists —
    exactly what AUDIT §5 catalogues in v2.1, whose README reported `MRR@5 = 0.990` from a
    dataset deleted in the same commit. A finding that cannot be re-derived is an assertion,
    not a result.
    """

    @pytest.fixture
    def traces(self) -> list[Any]:
        return load_fixture(FIXTURE)

    def test_the_fixture_is_committed_and_verifies_its_checksum(self, traces: list[Any]) -> None:
        assert FIXTURE.exists(), f"{FIXTURE} is the evidence for D-021 and must be committed"
        assert len(traces) == 214

    def test_a_tampered_fixture_is_refused(self, tmp_path: Path) -> None:
        """An edited fixture is not evidence, so it must fail loudly rather than reconcile."""
        fixture = json.loads(FIXTURE.read_text())
        fixture["traces"][0]["total_cost"] = 99.0
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(fixture))

        with pytest.raises(SystemExit, match="checksum"):
            load_fixture(tampered)

    def test_the_three_populations_are_as_documented(self, traces: list[Any]) -> None:
        groups = classify(traces, RATES)

        assert len(groups["priced"]) == 6
        assert len(groups["unpriced"]) == 187
        assert len(groups["no_usage"]) == 21

    def test_three_derivations_agree_exactly(self, traces: list[Any]) -> None:
        """The proof that the notional calculator was never the problem."""
        priced = classify(traces, RATES)["priced"]

        stored = sum(r["stored_notional"] for r in priced)
        recomputed = sum(r["recomputed_notional"] for r in priced)
        langfuse = sum(r["langfuse_cost"] for r in priced)

        assert stored == pytest.approx(0.01248, abs=5e-6)
        assert recomputed == pytest.approx(stored, abs=1e-9)
        assert langfuse == pytest.approx(stored, abs=5e-6)

    def test_the_blend_on_the_priced_population_is_possible(self, traces: list[Any]) -> None:
        priced = classify(traces, RATES)["priced"]
        tokens = sum(r["input_tokens"] + r["output_tokens"] for r in priced)
        blended = sum(r["stored_notional"] for r in priced) / tokens * 1_000_000

        assert blended == pytest.approx(0.4056, abs=5e-4)
        assert RATES.notional_input_usd <= blended <= RATES.notional_output_usd

    def test_the_blend_over_all_traces_is_the_impossible_one(self, traces: list[Any]) -> None:
        """Reproduces the bug itself: the merge that review caught, from the real data."""
        groups = classify(traces, RATES)
        every_trace = groups["priced"] + groups["unpriced"]
        tokens = sum(r["input_tokens"] + r["output_tokens"] for r in every_trace)
        cost = sum(r["stored_notional"] for r in every_trace)

        blended = cost / tokens * 1_000_000

        assert blended < RATES.notional_input_usd, (
            "the pre-fix table's blended rate must land below the input floor — that "
            "impossibility is what exposed the population merge"
        )

    def test_the_langfuse_gap_is_entirely_duplicate_roots(self, traces: list[Any]) -> None:
        """$0.02776 - $0.01248 = $0.01528, and every cent of it is a double-counted run."""
        priced = classify(traces, RATES)["priced"]
        duplicates = find_duplicate_roots(traces)

        assert len(duplicates) == 5
        orphan_cost = sum(b["cost"] for _, b in duplicates)
        dashboard_total = sum(float(t.total_cost) for t in traces)

        assert orphan_cost == pytest.approx(0.01528, abs=5e-6)
        assert dashboard_total == pytest.approx(
            sum(r["langfuse_cost"] for r in priced) + orphan_cost, abs=5e-6
        )
        assert dashboard_total == pytest.approx(0.02776, abs=5e-6)

    def test_most_of_the_window_was_test_traffic(self, traces: list[Any]) -> None:
        """The dominant cause, and the reason conftest now disables observability."""
        groups = classify(traces, RATES)
        synthetic = len(groups["unpriced"]) / len(traces)

        assert synthetic > 0.85
