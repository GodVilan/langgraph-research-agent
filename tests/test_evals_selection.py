"""The attribute-stratum selection must be regenerable and balanced.

A set whose items were chosen by an unrecorded process cannot be re-derived, which is the
failure AUDIT §5 catalogues in v2.1. Seeded, stratified, and documented in
evals/select_attributes.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.schema import AbsenceShape, AnchorClass
from evals.select_attributes import (
    SEED,
    TARGET,
    Candidate,
    enumerate_candidates,
    select,
    shape_counts,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def candidates() -> list[Candidate]:
    return enumerate_candidates()


class TestSelectionIsRegenerable:
    def test_the_same_seed_gives_the_same_items(self, candidates: list[Candidate]) -> None:
        first = select(candidates, TARGET, SEED)
        second = select(candidates, TARGET, SEED)

        assert [(c.paper_id, c.term) for c in first] == [(c.paper_id, c.term) for c in second]

    def test_a_different_seed_gives_different_items(self, candidates: list[Candidate]) -> None:
        """Otherwise the seed is decorative and the selection is really just sort order."""
        assert select(candidates, TARGET, SEED) != select(candidates, TARGET, 999)

    def test_enumeration_is_canonically_ordered(self, candidates: list[Candidate]) -> None:
        assert [c.sort_key for c in candidates] == sorted(c.sort_key for c in candidates)


class TestSelectionIsBalanced:
    def test_it_fills_the_target(self, candidates: list[Candidate]) -> None:
        assert len(select(candidates, TARGET, SEED)) == TARGET

    def test_every_shape_is_represented(self, candidates: list[Candidate]) -> None:
        """11 items over 6 shapes is one item measured 11 times if a shape dominates."""
        counts = shape_counts(select(candidates, TARGET, SEED))

        assert set(counts) == {s.value for s in AbsenceShape}
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_no_paper_anchors_two_items(self, candidates: list[Candidate]) -> None:
        """Two absences from one paper share its retrieval behaviour; not independent."""
        chosen = select(candidates, TARGET, SEED)

        assert len({c.paper_id for c in chosen}) == len(chosen)

    def test_strong_anchors_are_preferred(self, candidates: list[Candidate]) -> None:
        """The cross-citation check can only recognise 45 of 150 papers."""
        chosen = select(candidates, TARGET, SEED)

        assert all(c.anchor_class is AnchorClass.STRONG for c in chosen)

    def test_weak_anchors_are_used_only_after_strong_ones_run_out(self) -> None:
        """Falling back is allowed; doing so silently is not — anchor_class records it."""
        synthetic = [
            Candidate("p1", "flops", AbsenceShape.COMPUTE, AnchorClass.WEAK),
            Candidate("p2", "flops", AbsenceShape.COMPUTE, AnchorClass.STRONG),
        ]
        chosen = select(sorted(synthetic, key=lambda c: c.sort_key), target=1, seed=SEED)

        assert chosen[0].anchor_class is AnchorClass.STRONG

    def test_a_shortfall_is_returned_rather_than_padded(self) -> None:
        chosen = select(
            [Candidate("p1", "flops", AbsenceShape.COMPUTE, AnchorClass.STRONG)],
            target=11,
            seed=SEED,
        )

        assert len(chosen) == 1
