"""The scorer's view carries full chunk text, and the judge will see the same string.

A judge given ids where the human saw text cannot evaluate grounding, and every grounding
disagreement would be an information artifact. So the view is asserted to contain the text of
every gold and every retrieved chunk, and to be split at the point the rubric's Q1 requires —
question and answer first, nothing about the expected behaviour until after.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.run_set import ItemRun, RetrievedRecord, status_of
from evals.scoring import before_gold, view, with_gold

RUN = ItemRun(
    item_id="sp-999",
    stratum="single_paper_factual",
    question="What error rate is reported?",
    gold_answer="23.4 percent",
    gold_chunk_ids=[],
    gold_support="strong",
    answer="The paper reports 23.4 percent error.",
    retrieved=[
        RetrievedRecord(
            chunk_id="2605.00000_0001",
            paper_id="2605.00000",
            score=0.9,
            text="Our method reaches 23.4 percent error on the held-out split.",
        )
    ],
    status="completed",
)


class TestScorerView:
    def test_q1_view_reveals_neither_stratum_nor_gold(self) -> None:
        first = before_gold(RUN)

        assert "23.4 percent error" not in first.split("AGENT ANSWER")[0]
        assert "STRATUM" not in first
        assert "GOLD" not in first
        assert "expected behaviour" not in first

    def test_retrieved_chunk_text_is_shown_not_only_ids(self) -> None:
        rest = with_gold(RUN)

        assert "2605.00000_0001" in rest
        assert "held-out split" in rest
        assert "grounding is judged against these" in rest

    def test_the_full_view_is_the_two_halves_in_rubric_order(self) -> None:
        full = view(RUN)

        assert full.index("AGENT ANSWER") < full.index("STRATUM") < full.index("RETRIEVED CHUNKS")


class TestRunStatus:
    def test_a_block_event_is_a_guardrail_hit(self) -> None:
        assert status_of({"guardrail_events": [{"severity": "block"}]}, None) == "guardrail_blocked"

    def test_truncation_is_reported_as_such(self) -> None:
        assert status_of({"truncated": True, "guardrail_events": []}, None) == "truncated"

    def test_an_exception_is_an_error_not_a_completion(self) -> None:
        assert status_of({}, "RuntimeError: x") == "error"

    def test_a_warn_event_does_not_count_as_a_block(self) -> None:
        assert status_of({"guardrail_events": [{"severity": "warn"}]}, None) == "completed"
