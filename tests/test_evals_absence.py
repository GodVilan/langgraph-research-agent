"""Absence verification against the real corpus.

These run against all 5,401 chunks because that is the entire point: absence checked
anywhere smaller than where the agent searches is the false-absence hazard (docs/EVALS.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.absence import (
    load_corpus,
    paper_handles,
    verify_attribute_absent,
    verify_attribute_absent_in,
    verify_topic_absent,
)

pytestmark = pytest.mark.slow  # reads the 16 MiB chunk file


class TestTopicAbsence:
    @pytest.mark.parametrize("term", ["AlphaFold", "click-through rate"])
    def test_genuinely_absent_topics_pass(self, term: str) -> None:
        result = verify_topic_absent(term)

        assert result.absent
        assert result.chunks_scanned == load_corpus().n_chunks

    @pytest.mark.parametrize("term", ["curriculum learning", "capsule networks"])
    def test_topics_absent_as_a_string_but_present_as_a_concept_are_caught(self, term: str) -> None:
        """Hazard 1, one level down: absence checked as a string, asked as a concept.

        The corpus never writes "curriculum learning" and does discuss training examples
        ordered "easy to hard"; it never writes "capsule networks" and does discuss
        "dynamic routing". Both were certified absent by a literal check and both would have
        scored a correct answer as a failure to refuse. Screening now covers every surface
        form of a term, which cut the topic stratum from 4 to 2.
        """
        result = verify_topic_absent(term)

        assert not result.absent
        assert result.mentioning_chunk_ids

    @pytest.mark.parametrize(
        "term", ["mixture-of-experts", "active learning", "knowledge graph", "meta-learning"]
    )
    def test_topics_absent_from_abstracts_but_present_in_the_body_are_caught(
        self, term: str
    ) -> None:
        """The 86% false-absence rate, asserted term by term.

        Each of these looks absent when screening the 150 abstracts. Each is discussed in
        the corpus body. An item built on any of them would expect a refusal while the
        corpus holds the answer — scoring a correct answer as a hallucination.
        """
        result = verify_topic_absent(term)

        assert not result.absent
        assert result.mentioning_chunk_ids

    def test_the_scan_covers_every_chunk(self) -> None:
        assert verify_topic_absent("curriculum learning").chunks_scanned == 5401


class TestAttributeAbsence:
    def test_a_term_the_anchor_paper_uses_is_not_absent(self) -> None:
        result = verify_attribute_absent("2605.30350", "imagenet")

        assert not result.absent
        assert "its own text" in result.reason

    def test_a_term_absent_from_the_anchor_and_uncited_elsewhere_passes(self) -> None:
        result = verify_attribute_absent("2605.30348", "gsm8k")

        assert result.absent
        assert result.chunks_scanned == 5401

    def test_a_paraphrased_form_in_the_anchor_paper_blocks_the_item(self) -> None:
        """The exact defect: "batch size" absent, "global batch 16" present in Appendix A."""
        result = verify_attribute_absent("2605.30237", "batch size")

        assert not result.absent
        assert "global batch" in result.reason

    def test_an_unknown_paper_is_rejected_rather_than_assumed_absent(self) -> None:
        result = verify_attribute_absent("9999.99999", "gsm8k")

        assert not result.absent
        assert "no paper" in result.reason


class TestHandles:
    def test_every_paper_has_at_least_its_arxiv_id(self) -> None:
        handles = paper_handles()

        assert len(handles) == 150
        assert all(pid in hs for pid, hs in handles.items())

    def test_ambiguous_method_names_are_not_used_as_handles(self) -> None:
        """`Gram` matches "n-gram", "program", "diagram" — it eliminated two valid pairs."""
        for pid, hs in paper_handles().items():
            assert "Gram" not in hs, f"{pid} uses an ambiguous handle"

    def test_only_a_minority_of_papers_have_a_distinctive_name(self) -> None:
        """A real limit on the cross-citation check, asserted so it stays visible."""
        named = sum(1 for hs in paper_handles().values() if len(hs) > 1)

        assert 30 <= named <= 60


class TestTheCrossCitationRuleCanActuallyFire:
    """Prove the check is *clean*, not *inert*.

    On the real corpus the cross-citation rule eliminates 0 of 979 candidate pairs, because
    all 150 papers were published on the same afternoon and cannot cite one another. That
    number alone cannot distinguish "nothing to catch" from "catches nothing" — the same
    defect as a workflow that cannot fire and a test that skips its own failure
    (DECISIONS D-023). So a chunk that should trip it is injected, and it must trip.
    """

    ANCHOR = "2605.30075"
    OTHER = "2605.30123"

    def _corpus(self, extra: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {"chunk_id": "a_0", "paper_id": self.ANCHOR, "text": "We study federated aggregation."},
            {"chunk_id": "b_0", "paper_id": self.OTHER, "text": "Unrelated related work."},
            *extra,
        ]

    def test_absent_when_nothing_attributes_the_term(self) -> None:
        result = verify_attribute_absent_in(
            self._corpus([]),
            {self.ANCHOR: frozenset({self.ANCHOR, "Q-ANCHOR"})},
            self.ANCHOR,
            "imagenet",
        )

        assert result.absent

    def test_fires_when_another_paper_attributes_the_benchmark_by_method_name(self) -> None:
        """The exact scenario the rule exists for, and the one the corpus cannot produce."""
        citing = {
            "chunk_id": "b_1",
            "paper_id": self.OTHER,
            "text": "Prior work: Q-ANCHOR reports 71.2% top-1 on ImageNet under this protocol.",
        }
        result = verify_attribute_absent_in(
            self._corpus([citing]),
            {self.ANCHOR: frozenset({self.ANCHOR, "Q-ANCHOR"})},
            self.ANCHOR,
            "imagenet",
        )

        assert not result.absent
        assert result.mentioning_chunk_ids == ["b_1"]
        assert "Q-ANCHOR" in result.reason

    def test_fires_when_another_paper_attributes_it_by_arxiv_id(self) -> None:
        """The weak-anchor path: 105 of 150 papers have only this."""
        citing = {
            "chunk_id": "b_2",
            "paper_id": self.OTHER,
            "text": f"As reported in {self.ANCHOR}, ImageNet accuracy reaches 71.2%.",
        }
        result = verify_attribute_absent_in(
            self._corpus([citing]),
            {self.ANCHOR: frozenset({self.ANCHOR})},
            self.ANCHOR,
            "imagenet",
        )

        assert not result.absent

    def test_does_not_fire_on_an_unrelated_mention_of_the_benchmark(self) -> None:
        """Another paper using ImageNet itself says nothing about the anchor's score."""
        unrelated = {
            "chunk_id": "b_3",
            "paper_id": self.OTHER,
            "text": "We evaluate our own method on ImageNet and report 68.4%.",
        }
        result = verify_attribute_absent_in(
            self._corpus([unrelated]),
            {self.ANCHOR: frozenset({self.ANCHOR, "Q-ANCHOR"})},
            self.ANCHOR,
            "imagenet",
        )

        assert result.absent

    def test_the_real_corpus_uses_the_same_code_path(self) -> None:
        """Guards against the injected-corpus variant drifting from the shipped one."""
        injected = verify_attribute_absent_in(
            [
                {"chunk_id": c["chunk_id"], "paper_id": c["paper_id"], "text": c["text"]}
                for c in load_corpus().chunks
            ],
            dict(paper_handles()),
            "2605.30348",
            "gsm8k",
        )

        assert injected.absent == verify_attribute_absent("2605.30348", "gsm8k").absent
