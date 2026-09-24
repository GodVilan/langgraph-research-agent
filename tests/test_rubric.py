"""The rubric is a function, and the function is exhaustive.

Every (stratum, behaviour) pair must map to exactly one outcome, every outcome must be
reachable, and every outcome must feed a named metric. A rubric with an unreachable label or
an unlabelled case would leave the human and the judges free to disagree on a case the rubric
never decided — and that disagreement would be scored as judge error.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.rubric import METRIC_OF, Behaviour, Outcome, expected_behaviour, outcome_for, render
from evals.schema import Stratum


class TestRubricIsExhaustive:
    def test_every_stratum_behaviour_pair_is_decided(self) -> None:
        for stratum, behaviour in itertools.product(Stratum, Behaviour):
            needs_q3 = (
                behaviour is Behaviour.ANSWER and expected_behaviour(stratum) is Behaviour.ANSWER
            )
            for fact, grounded in (
                [(True, True), (True, False), (False, True)] if needs_q3 else [(None, None)]
            ):
                assert outcome_for(stratum, behaviour, fact, grounded) in Outcome

    def test_every_outcome_is_reachable(self) -> None:
        reached = set()
        for stratum, behaviour in itertools.product(Stratum, Behaviour):
            if behaviour is Behaviour.ANSWER and expected_behaviour(stratum) is Behaviour.ANSWER:
                for fact, grounded in [(True, True), (True, False), (False, True)]:
                    reached.add(outcome_for(stratum, behaviour, fact, grounded))
            else:
                reached.add(outcome_for(stratum, behaviour, None, None))
        assert reached == set(Outcome)

    def test_every_outcome_feeds_a_metric(self) -> None:
        assert set(METRIC_OF) == set(Outcome)

    def test_an_answered_answerable_item_needs_the_third_question(self) -> None:
        with pytest.raises(ValueError):
            outcome_for(Stratum.SINGLE_PAPER, Behaviour.ANSWER, None, None)

    @pytest.mark.parametrize(
        ("stratum", "behaviour", "expected"),
        [
            (Stratum.UNANSWERABLE_TOPIC, Behaviour.REFUSE, Outcome.CORRECT_REFUSAL),
            (Stratum.UNANSWERABLE_ATTRIBUTE, Behaviour.ANSWER, Outcome.HALLUCINATED_ANSWER),
            (Stratum.SINGLE_PAPER, Behaviour.REFUSE, Outcome.HALLUCINATED_REFUSAL),
            (Stratum.AMBIGUOUS, Behaviour.ANSWER, Outcome.SILENT_DISAMBIGUATION),
            (Stratum.AMBIGUOUS, Behaviour.REFUSE, Outcome.HALLUCINATED_REFUSAL),
            (Stratum.UNANSWERABLE_ATTRIBUTE, Behaviour.CLARIFY, Outcome.EVASIVE_CLARIFICATION),
            (Stratum.UNANSWERABLE_TOPIC, Behaviour.CLARIFY, Outcome.EVASIVE_CLARIFICATION),
        ],
    )
    def test_the_four_named_cases_and_their_neighbours(
        self, stratum: Stratum, behaviour: Behaviour, expected: Outcome
    ) -> None:
        assert outcome_for(stratum, behaviour, None, None) is expected

    def test_the_rendered_rubric_names_every_label(self) -> None:
        """docs/RUBRIC.md is generated from this; a label the prose never mentions is one the
        human cannot apply."""
        text = render()
        for outcome in Outcome:
            assert outcome.value in text, outcome


class TestSampleIsSeededAndDocumented:
    def test_the_draw_is_deterministic_and_matches_its_quota(self) -> None:
        from evals.judge_sample import QUOTA, draw
        from evals.schema import EvalSet

        evalset = EvalSet.read(Path("evals/datasets/phase4.json"))
        first, second = draw(evalset), draw(evalset)

        assert [i.item_id for i in first] == [i.item_id for i in second]
        counts: dict[Stratum, int] = {}
        for item in first:
            counts[item.stratum] = counts.get(item.stratum, 0) + 1
        assert counts == QUOTA
        assert Stratum.MULTI_HOP not in counts
