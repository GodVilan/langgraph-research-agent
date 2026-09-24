"""Schema invariants that keep the eval set honest.

The two unanswerable sub-strata measure different capabilities and must never be merged into
one figure; the schema is where that is made structurally hard rather than remembered.
See docs/EVALS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.schema import (
    AbsenceShape,
    AnchorClass,
    EvalItem,
    EvalSet,
    MachineCheck,
    Provenance,
    Stratum,
    Verification,
)


def item(**overrides: object) -> EvalItem:
    base: dict[str, object] = {
        "item_id": "x-001",
        "stratum": Stratum.SINGLE_PAPER,
        "question": "What accuracy is reported?",
        "gold_chunk_ids": ["2605.30148_0021"],
        "gold_answer": "23.4 percent",
        "provenance": Provenance(generator_model="hand", source_paper_ids=["2605.30148"]),
    }
    return EvalItem(**{**base, **overrides})  # type: ignore[arg-type]


class TestTheTwoRefusalStrataStayApart:
    def test_they_are_distinct_reporting_groups(self) -> None:
        assert (
            Stratum.UNANSWERABLE_TOPIC.reporting_group
            != Stratum.UNANSWERABLE_ATTRIBUTE.reporting_group
        )
        assert Stratum.UNANSWERABLE_TOPIC.expects_refusal
        assert Stratum.UNANSWERABLE_ATTRIBUTE.expects_refusal

    def test_an_attribute_item_must_name_its_anchor_paper(self) -> None:
        with pytest.raises(ValueError, match="anchored to a real paper"):
            item(
                stratum=Stratum.UNANSWERABLE_ATTRIBUTE,
                gold_chunk_ids=[],
                absent_term="gsm8k",
                absence_shape=AbsenceShape.BENCHMARK,
                anchor_class=AnchorClass.STRONG,
                anchor_paper_id="",
            )

    def test_a_topic_item_must_not_have_one(self) -> None:
        """Anchoring a topic item would silently turn it into an attribute item."""
        with pytest.raises(ValueError, match="no anchor paper"):
            item(
                stratum=Stratum.UNANSWERABLE_TOPIC,
                gold_chunk_ids=[],
                absent_term="curriculum learning",
                anchor_paper_id="2605.30148",
            )

    def test_an_attribute_item_must_declare_its_absence_shape(self) -> None:
        """Without this, eleven items collapse into one question asked eleven times."""
        with pytest.raises(ValueError, match="absence_shape is required"):
            item(
                stratum=Stratum.UNANSWERABLE_ATTRIBUTE,
                gold_chunk_ids=[],
                absent_term="gsm8k",
                anchor_paper_id="2605.30148",
            )

    def test_counts_report_the_sub_strata_separately(self) -> None:
        evalset = EvalSet(
            name="t",
            corpus_sha256="abc",
            items=[
                item(
                    item_id="t1",
                    stratum=Stratum.UNANSWERABLE_TOPIC,
                    gold_chunk_ids=[],
                    absent_term="curriculum learning",
                ),
                item(
                    item_id="a1",
                    stratum=Stratum.UNANSWERABLE_ATTRIBUTE,
                    gold_chunk_ids=[],
                    absent_term="gsm8k",
                    anchor_paper_id="2605.30148",
                    anchor_class=AnchorClass.STRONG,
                    absence_shape=AbsenceShape.BENCHMARK,
                ),
            ],
        )
        counts = evalset.counts()

        assert counts["unanswerable_topic"] == 1
        assert counts["unanswerable_attribute"] == 1
        # There is deliberately no combined "unanswerable" key to reach for.
        assert "unanswerable" not in counts


class TestItemInvariants:
    def test_an_unanswerable_item_cannot_carry_gold_chunks(self) -> None:
        with pytest.raises(ValueError, match="not unanswerable"):
            item(
                stratum=Stratum.UNANSWERABLE_TOPIC,
                absent_term="curriculum learning",
                gold_chunk_ids=["2605.30148_0021"],
            )

    def test_an_answerable_item_needs_gold_chunks(self) -> None:
        with pytest.raises(ValueError, match="at least one gold chunk"):
            item(gold_chunk_ids=[])

    def test_multi_hop_requires_two_source_papers(self) -> None:
        with pytest.raises(ValueError, match="at least two papers"):
            item(
                stratum=Stratum.MULTI_HOP,
                provenance=Provenance(generator_model="hand", source_paper_ids=["2605.30148"]),
            )

    def test_verification_defaults_to_unchecked(self) -> None:
        """Nothing may default to verified; an unverified item is inadmissible."""
        checks = item().verification

        assert not checks.human_verified
        assert not checks.absence_verified_against_corpus
        assert not checks.single_paper_sufficiency_checked


class TestFreezing:
    def test_round_trip_preserves_the_checksum(self, tmp_path: Path) -> None:
        original = EvalSet(name="t", corpus_sha256="abc", items=[item()])
        path = tmp_path / "set.json"
        original.write(path)

        assert EvalSet.read(path).sha256 == original.sha256

    def test_an_edited_set_fails_its_checksum(self, tmp_path: Path) -> None:
        evalset = EvalSet(name="t", corpus_sha256="abc", items=[item()])
        path = tmp_path / "set.json"
        evalset.write(path)
        path.write_text(path.read_text().replace("What accuracy", "What speed"))

        with pytest.raises(ValueError, match="failed its checksum"):
            EvalSet.read(path)

    def test_absence_shapes_are_counted_for_the_balance_check(self) -> None:
        evalset = EvalSet(
            name="t",
            corpus_sha256="abc",
            items=[
                item(
                    item_id=f"a{n}",
                    stratum=Stratum.UNANSWERABLE_ATTRIBUTE,
                    gold_chunk_ids=[],
                    absent_term="x",
                    anchor_paper_id="2605.30148",
                    anchor_class=AnchorClass.STRONG,
                    absence_shape=shape,
                )
                for n, shape in enumerate([AbsenceShape.BENCHMARK, AbsenceShape.ABLATION])
            ],
        )

        assert evalset.absence_shape_counts() == {"benchmark": 1, "ablation": 1}


class TestDisclosureOrder:
    """The human label must be independent of the machine verdict.

    Showing "necessity: PASS" before the decision turns the agreement rate into
    agreement-with-the-checker. Multi-hop and unanswerable-attribute have no other
    validation, so that number is only worth having if it was reached independently.
    """

    def _ruled(self, *, human: bool, machine: bool) -> EvalItem:
        return item(
            verification=Verification(
                human_accepted=human,
                human_verified=human,
                machine_checks=[MachineCheck(name="multi_hop_necessity", passed=machine)],
            )
        )

    def test_an_unruled_item_has_no_agreement_verdict(self) -> None:
        """ "Not yet seen" must not read as agreement or disagreement."""
        assert item().verification.agrees is None

    def test_a_ruled_item_with_no_checks_has_no_agreement_verdict(self) -> None:
        checks = Verification(human_accepted=True, human_verified=True)

        assert checks.agrees is None

    @pytest.mark.parametrize(
        ("human", "machine", "expected"),
        [(True, True, True), (False, False, True), (True, False, False), (False, True, False)],
    )
    def test_agreement_is_computed_from_both_labels(
        self, human: bool, machine: bool, expected: bool
    ) -> None:
        assert self._ruled(human=human, machine=machine).verification.agrees is expected

    def test_a_disagreement_names_the_check_responsible(self) -> None:
        """ "3 disagreements" is not actionable; "the necessity check disagreed" is."""
        names = [
            c.name for c in self._ruled(human=True, machine=False).verification.disagreeing_checks()
        ]

        assert names == ["multi_hop_necessity"]

    def test_rejection_is_distinguishable_from_not_yet_seen(self) -> None:
        rejected = self._ruled(human=False, machine=True).verification
        unseen = item().verification

        assert rejected.human_accepted is False
        assert unseen.human_accepted is None
        assert not rejected.human_verified and not unseen.human_verified

    def test_agreement_is_reported_per_stratum_never_pooled(self) -> None:
        evalset = EvalSet(
            name="t",
            corpus_sha256="abc",
            items=[
                item(
                    item_id="f1",
                    verification=Verification(
                        human_accepted=True, machine_checks=[MachineCheck(name="g", passed=True)]
                    ),
                ),
                item(
                    item_id="m1",
                    stratum=Stratum.MULTI_HOP,
                    provenance=Provenance(generator_model="g", source_paper_ids=["a", "b"]),
                    verification=Verification(
                        human_accepted=True,
                        machine_checks=[MachineCheck(name="multi_hop_necessity", passed=False)],
                    ),
                ),
            ],
        )
        report = evalset.agreement_by_stratum()

        assert report["single_paper_factual"] == {"ruled": 1, "agree": 1, "disagree": 0}
        assert report["multi_hop"] == {"ruled": 1, "agree": 0, "disagree": 1}
        assert evalset.disagreements() == [("m1", "multi_hop", ["multi_hop_necessity"])]


class TestCrossStratumIndependence:
    """One paper feeding several strata makes those metrics correlated.

    A quirk in that paper — an unusual results table, odd phrasing — then surfaces as
    several apparently independent findings. Disjointness is achievable at 150 papers
    (5,598 of 6,456 multi-hop pairs avoid the attribute anchors), so it is enforced rather
    than accepted; `shared_papers` exists to record any overlap that becomes unavoidable.
    """

    def _set(self, *items: EvalItem) -> EvalSet:
        return EvalSet(name="t", corpus_sha256="abc", items=list(items))

    def _attr(self, paper: str) -> EvalItem:
        return item(
            item_id=f"a-{paper}",
            stratum=Stratum.UNANSWERABLE_ATTRIBUTE,
            gold_chunk_ids=[],
            absent_term="gsm8k",
            anchor_paper_id=paper,
            anchor_class=AnchorClass.STRONG,
            absence_shape=AbsenceShape.BENCHMARK,
            provenance=Provenance(generator_model="g", source_paper_ids=[paper]),
        )

    def _hop(self, a: str, b: str) -> EvalItem:
        return item(
            item_id=f"m-{a}-{b}",
            stratum=Stratum.MULTI_HOP,
            provenance=Provenance(generator_model="g", source_paper_ids=[a, b]),
        )

    def test_disjoint_strata_report_no_overlap(self) -> None:
        evalset = self._set(self._attr("p1"), self._hop("p2", "p3"))

        assert evalset.cross_stratum_overlap() == {}

    def test_a_shared_paper_is_detected_and_named(self) -> None:
        evalset = self._set(self._attr("p1"), self._hop("p1", "p2"))

        overlap = evalset.cross_stratum_overlap()

        assert overlap == {("multi_hop", "unanswerable_attribute"): {"p1"}}

    def test_papers_by_stratum_counts_anchors_and_sources(self) -> None:
        evalset = self._set(self._attr("p1"), self._hop("p2", "p3"))

        assert evalset.papers_by_stratum() == {
            "unanswerable_attribute": {"p1"},
            "multi_hop": {"p2", "p3"},
        }

    def test_shared_papers_defaults_empty_and_is_recordable(self) -> None:
        """Enforced disjointness today; recordable if 150 papers ever stop allowing it."""
        assert item().shared_papers == []
        assert item(shared_papers=["p1"]).shared_papers == ["p1"]


class TestDecisionCapture:
    """The verification CLI must record the decision that was actually made.

    A session recorded 22 accepts and 2 rejects against an intent of roughly 17 and 7. Notes
    pasted as several lines were consumed one line per subsequent prompt, so every answer
    after a paste landed one slot early — decisions stored as notes, notes read as
    decisions. It turned a 28% disagreement rate into 8%, on the one number the exercise
    exists to produce.
    """

    def test_an_empty_answer_is_not_a_decision(self) -> None:
        from evals.verify_cli import VALID_DECISIONS

        assert "" not in VALID_DECISIONS

    def test_only_the_five_keys_are_decisions(self) -> None:
        from evals.verify_cli import VALID_DECISIONS

        assert set(VALID_DECISIONS) == {"a", "e", "r", "s", "q"}

    def test_free_text_is_not_silently_truncated_to_a_key(self) -> None:
        """`answer[:1]` turned "reject this" into "r" and "absolutely not" into "a"."""
        from evals.verify_cli import VALID_DECISIONS

        for text in ("reject this", "absolutely not", "REJECT", "accept?"):
            assert text.strip().lower() not in VALID_DECISIONS

    def test_notes_have_an_explicit_terminator(self) -> None:
        """Without one, a multi-line paste runs into the next prompt."""
        from evals.verify_cli import NOTES_TERMINATOR

        assert NOTES_TERMINATOR == "."


class TestGoldAnswerIsRequired:
    """Every multi-hop item reached human review with "(none recorded)".

    Without a gold answer the judge has nothing to grade against, and the gold chunks cannot
    be checked for containing the fact they supposedly support — the item is unusable in
    both directions. It now fails at construction instead of at review.
    """

    def test_an_answerable_item_without_a_gold_answer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a gold answer"):
            item(gold_answer="")

    def test_whitespace_does_not_count_as_an_answer(self) -> None:
        with pytest.raises(ValueError, match="needs a gold answer"):
            item(gold_answer="   ")

    def test_multi_hop_is_held_to_the_same_rule(self) -> None:
        with pytest.raises(ValueError, match="needs a gold answer"):
            item(
                stratum=Stratum.MULTI_HOP,
                gold_answer="",
                provenance=Provenance(generator_model="g", source_paper_ids=["a", "b"]),
            )

    def test_refusal_items_need_no_gold_answer(self) -> None:
        """Their expected behaviour is a refusal; there is no answer to record."""
        built = item(
            stratum=Stratum.UNANSWERABLE_TOPIC,
            gold_chunk_ids=[],
            gold_answer="",
            absent_term="curriculum learning",
        )

        assert built.gold_answer == ""
