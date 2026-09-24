"""The judge sees what the human saw, and its label is computed, never chosen.

Same shape as the orphan guard: a property of the code, asserted, rather than a promise in a
report. If the judge prompt drifted from the scorer's view, every disagreement would be an
information artifact; if the judge returned a label, the rubric would exist twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.judge import (
    ARMS,
    JudgeVerdict,
    needs_q3,
    q1_messages,
    q3_messages,
    request_body,
    to_score,
)
from evals.rubric import Behaviour, Outcome, render, render_q1
from evals.scoring import before_gold, view
from tests.test_scoring_view import RUN


class TestStageOneIsBlind:
    """Q2's table maps expected x observed onto a label. A judge shown the stratum before Q1
    can read `hallucinated_refusal` off a lookup without judging what the agent did — and that
    is the 17-item refusal pair the experiment exists to measure. The human decided Q1 blind;
    this asserts the judge cannot do otherwise."""

    def test_stage_one_carries_neither_stratum_nor_gold_nor_chunks(self) -> None:
        payload = json.dumps(q1_messages(RUN))

        # "held-out split" occurs only in the retrieved chunk; the agent's own answer may
        # legitimately repeat the gold figure, so the figure is not on this list.
        for forbidden in (
            "STRATUM",
            "GOLD",
            "expected behaviour",
            "RETRIEVED",
            "single_paper_factual",
            "held-out split",
        ):
            assert forbidden not in payload, forbidden

    def test_stage_one_is_the_pre_gold_view_and_the_q1_rubric_only(self) -> None:
        system, user = q1_messages(RUN)

        assert system["content"] == render_q1()
        assert before_gold(RUN) in user["content"]
        assert "## Q2" not in system["content"] and "## Q3" not in system["content"]

    def test_the_q1_rubric_is_a_prefix_slice_of_the_full_rubric(self) -> None:
        """One rubric, not two: the Q1 section is cut from the same text the human read."""
        assert render_q1().split("## Q1", 1)[1].strip() in render()


class TestStageTwoIsTheScorerView:
    def test_the_user_message_embeds_the_full_view_verbatim(self) -> None:
        user = q3_messages(RUN, Behaviour.ANSWER)[1]["content"]

        assert view(RUN) in user
        assert "behaviour was: **answer**" in user

    def test_the_system_message_is_the_rubric_verbatim(self) -> None:
        assert q3_messages(RUN, Behaviour.ANSWER)[0]["content"] == render()

    def test_no_outcome_label_is_requested_from_the_judge(self) -> None:
        user = q3_messages(RUN, Behaviour.ANSWER)[1]["content"].split("ITEM ")[0]

        assert "outcome label" in user

    def test_stage_two_runs_only_where_the_cli_asked_q3(self) -> None:
        assert needs_q3(RUN, Behaviour.ANSWER)
        assert not needs_q3(RUN, Behaviour.REFUSE)
        unanswerable = RUN.model_copy(update={"stratum": "unanswerable_topic"})
        assert not needs_q3(unanswerable, Behaviour.ANSWER)

    def test_batch_arms_request_json_and_carry_their_effort(self) -> None:
        body = request_body(ARMS["luna-medium"], RUN, "q3", Behaviour.ANSWER)

        assert body["response_format"] == {"type": "json_object"}
        assert body["reasoning_effort"] == "medium"
        assert body["model"] == "gpt-5.6-luna"
        json.dumps(body)


class TestJudgeLabelsAreComputed:
    def test_an_omitted_q3_answer_is_never_read_as_yes(self) -> None:
        verdict = JudgeVerdict(behaviour=Behaviour.ANSWER, fact_matches=None, grounded=None)

        assert to_score(ARMS["luna-low"], RUN, verdict).outcome is Outcome.WRONG_ANSWER

    def test_a_refusal_on_an_answerable_item_is_a_hallucinated_refusal(self) -> None:
        verdict = JudgeVerdict(behaviour=Behaviour.REFUSE)

        assert to_score(ARMS["oss120b"], RUN, verdict).outcome is Outcome.HALLUCINATED_REFUSAL

    def test_a_grounded_correct_answer(self) -> None:
        verdict = JudgeVerdict(behaviour=Behaviour.ANSWER, fact_matches=True, grounded=True)

        assert to_score(ARMS["luna-low"], RUN, verdict).outcome is Outcome.CORRECT_ANSWER

    def test_batch_prices_are_half_the_standard_card(self) -> None:
        """docs/BUDGET.md: standard $0.20 / $1.20; batch is exactly half."""
        for arm_id in ("luna-low", "luna-medium"):
            assert ARMS[arm_id].price_in_per_m == 0.10
            assert ARMS[arm_id].price_out_per_m == 0.60


class TestASubmitCannotReportSuccessWithoutHappening:
    """A rejected submit must fail loudly, not log optimistically.

    Two submissions issued against a deactivated account 401'd, wrote no batch file, and left
    the driver log reading "SUBMITTED" for work that did not exist — `section_filter` and
    `v21` would have been silently absent from the results with nothing in the log to show it.
    That is a step reporting success without having happened (D-026's class), not an outage.
    The receipt is now written *and verified readable*, and a submit with no batch_id raises.
    """

    def test_a_rejected_submit_raises_instead_of_writing_a_receipt(self, tmp_path: Path) -> None:
        import typer

        from evals.judge import write_receipt

        receipt = tmp_path / "batch_luna-low_q1.json"
        # What a 401'd submit leaves behind: no id, because nothing was queued.
        with pytest.raises(typer.BadParameter, match="no batch_id"):
            write_receipt(receipt, {"arm": "luna-low", "stage": "q1", "requests": 69})

        assert not receipt.exists(), "a failed submit must leave no receipt to collect against"

    def test_a_successful_submit_leaves_a_readable_receipt(self, tmp_path: Path) -> None:
        import json as _json

        from evals.judge import write_receipt

        receipt = tmp_path / "batch_luna-low_q1.json"
        write_receipt(receipt, {"arm": "luna-low", "stage": "q1", "batch_id": "batch_abc"})

        assert _json.loads(receipt.read_text())["batch_id"] == "batch_abc"

    def test_an_unwritable_receipt_is_not_reported_as_submitted(self, tmp_path: Path) -> None:
        """The receipt is the handle a later collect needs; unverifiable means unsubmitted."""
        import typer

        from evals.judge import write_receipt

        target = tmp_path / "batch.json"
        target.write_text("{}", encoding="utf-8")
        target.chmod(0o444)
        try:
            with pytest.raises((typer.BadParameter, PermissionError, OSError)):
                write_receipt(target, {"batch_id": "batch_xyz"})
        finally:
            target.chmod(0o644)
