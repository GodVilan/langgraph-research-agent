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
    verify_topic_absent,
)

pytestmark = pytest.mark.slow  # reads the 16 MiB chunk file


class TestTopicAbsence:
    @pytest.mark.parametrize("term", ["curriculum learning", "capsule networks"])
    def test_genuinely_absent_topics_pass(self, term: str) -> None:
        result = verify_topic_absent(term)

        assert result.absent
        assert result.chunks_scanned == load_corpus().n_chunks

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
