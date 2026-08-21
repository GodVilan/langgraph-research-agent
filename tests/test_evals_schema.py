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
    EvalItem,
    EvalSet,
    Provenance,
    Stratum,
)


def item(**overrides: object) -> EvalItem:
    base: dict[str, object] = {
        "item_id": "x-001",
        "stratum": Stratum.SINGLE_PAPER,
        "question": "What accuracy is reported?",
        "gold_chunk_ids": ["2605.30148_0021"],
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
                    absence_shape=shape,
                )
                for n, shape in enumerate([AbsenceShape.BENCHMARK, AbsenceShape.ABLATION])
            ],
        )

        assert evalset.absence_shape_counts() == {"benchmark": 1, "ablation": 1}
