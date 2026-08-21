"""The corpus constraint on eval-set construction, pinned as assertions.

150 cs.LG papers from a single afternoon is a narrow slice, and the Phase 4 plan asks for
four strata that slice may not be able to fill. These tests record what the measurement
found so that a later change to the corpus, or an optimistic assumption about it, has to
argue with something.

The one worth reading is `test_absence_must_be_checked_against_full_text`. Screening
candidate unanswerable topics against abstracts marks topics absent that the body text
discusses at length — so a set generated that way is mostly broken items that look fine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.corpus_diversity import (
    load,
    multi_hop,
    single_paper_factual,
    unanswerable_attributes,
    unanswerable_topics,
)

pytestmark = pytest.mark.slow  # reads the 16 MiB chunk file


@pytest.fixture(scope="module")
def corpus() -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    return load()


def test_single_paper_factual_stratum_is_fillable(corpus: tuple) -> None:
    """55 items need 55 anchorable papers; there are far more."""
    result = single_paper_factual(corpus[1])

    assert result["eligible"] >= result["target"]
    assert result["weak"] < 20


def test_multi_hop_stratum_is_fillable(corpus: tuple) -> None:
    """20 items need 20 genuinely distinct bridgeable pairs."""
    result = multi_hop(corpus[0])

    assert result["pairs_4plus"] >= result["target"]
    assert result["papers_involved"] > 100


def test_absence_must_be_checked_against_full_text(corpus: tuple) -> None:
    """The trap: most topics missing from abstracts are discussed in the body.

    This is why unanswerable items cannot be generated from abstracts. If this ratio ever
    drops, the corpus has changed and the unanswerable construction should be revisited.
    """
    result = unanswerable_topics(*corpus)

    assert result["absent_from_abstracts"] > len(result["absent_from_full_text"]) * 4
    assert len(result["false_absences"]) >= 20


def test_the_unanswerable_stratum_does_not_fill_from_absent_topics(corpus: tuple) -> None:
    """Recorded as a shortfall rather than padded — the target is 15, the supply is ~4."""
    result = unanswerable_topics(*corpus)

    assert len(result["absent_from_full_text"]) < result["target"]


def test_absent_attributes_fill_it_instead(corpus: tuple) -> None:
    """Facts absent from papers that are present: abundant, and a sharper probe.

    These land retrieval on the anchor paper's own chunks at high similarity, which is the
    no-score-floor failure the stratum exists to expose.
    """
    result = unanswerable_attributes(corpus[1])

    assert result["absent_pair_combinations"] > 500
    # A benchmark used by some papers and not others is what makes the item unambiguous.
    assert 0 < result["papers_using_each"]["imagenet"] < 150
