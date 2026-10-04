"""The landing screenshot keeps only a draw that cites a source the page lists."""

from __future__ import annotations

import pytest

from scripts.capture_landing import kept

SOURCES = ["2605.29525", "2605.30103"]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"answer": "21.92% [2605.29525_0010].", "sources": SOURCES}, ["2605.29525"]),
        # Abbreviated ids, as the free-tier model sometimes writes them, match by paper suffix.
        ({"answer": "21.92% [29525_0010].", "sources": SOURCES}, ["2605.29525"]),
        ({"answer": "The passages do not say.", "sources": SOURCES}, []),
        ({"answer": "21.92% [2605.99999_0001].", "sources": SOURCES}, []),
        ({"answer": None, "error": "Too many questions", "sources": []}, []),
        ({"answer": "x [2605.29525_0010]", "error": "late error", "sources": SOURCES}, []),
    ],
)
def test_only_a_draw_citing_a_listed_source_is_kept(
    result: dict[str, object], expected: list[str]
) -> None:
    assert kept(result) == expected
