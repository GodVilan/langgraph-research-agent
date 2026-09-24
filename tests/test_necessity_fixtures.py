"""Fixtures for `multi_hop_necessity` — the check with the worst record here.

The nine local fixtures in ``test_check_matrix.py`` cover phrasing, stem, premise, leak and
containment: checks that have never certified a bad item. None covered the necessity check,
which has certified two that human review rejected — one carrying a false premise
(CIFAR-10 asserted of a paper that never used it) and one stating its answer in the stem.
The root of trust proved the checks that never failed and was silent on the one that had.

These cost a model call each, which is why they were skipped and why they are marked
``integration``. That is not a reason to omit them: a check nothing exercises is a check
that has never fired, and this one has fired *wrongly*.

Run with `make test-necessity` (needs GOOGLE_API_KEY; free tier, 27 calls, ~2 min at 15 RPM).

Each case varies one dimension against a fixed pair, matching the design of the local
matrix: two papers whose content is known, and a question whose answer is placed
deliberately in one, both, or neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.multihop import check_many

pytestmark = [pytest.mark.integration, pytest.mark.network]

# A real, verified multi-hop pair: two federated-learning papers. Chosen because the
# comparative question over them is the one case already shown to pass end to end, so a
# failure here is the check changing rather than the pair being unsuitable.
PAPER_A = "2605.30075"
PAPER_B = "2605.30123"

CASES = [
    (
        "single_paper",
        "What is ZNE-guided correction and why does the first paper need it?",
        {"answerable_by_one_paper": True, "is_genuinely_multi_hop": False},
    ),
    (
        "neither_nor_joint",
        "What accuracy do these papers report for protein structure prediction on CASP15?",
        {"answerable_by_all_papers": False, "is_genuinely_multi_hop": False},
    ),
    (
        "genuinely_multi_hop",
        "How do these two differ in what they do to client updates before the server "
        "aggregates them?",
        {"answerable_by_one_paper": False, "is_genuinely_multi_hop": True},
    ),
]

# Measured 2026-09-18 on byte-identical code, three draws of the multi-hop case produced
# three different verdict patterns: (30123 sufficient, joint yes), (neither, joint no),
# (neither, joint yes). The single-paper and neither cases were stable on every draw. So the
# margin is per case: the stable cases must land on a majority; the multi-hop case must land
# at least once — which proves the check *can* fire — and its split is printed, because that
# split IS the instrument's variance on a real pair, and the Phase 4 variance estimate is
# designed around it rather than around a fixture tuned until it stopped moving.
MIN_DRAWS = {"single_paper": 2, "neither_nor_joint": 2, "genuinely_multi_hop": 1}


# N repeats and a margin, because the instrument is nondeterministic (D-014) and one draw of
# it reads as a regression when it is a coin flip: measured on byte-identical code, this file
# went 1 fail then 2 passes. Each case is checked REPEATS times and must land where it was
# placed on at least MAJORITY of them; the split is reported so a 2-of-3 is visible as a
# 2-of-3 rather than as a pass. Same instrument the Phase 4 variance estimate uses, so this
# is the first place its variance is measured.
REPEATS = 3
MAJORITY = 2


@pytest.fixture(scope="module")
def results() -> dict[str, list[object]]:
    """REPEATS batched runs for all three cases — 3 x 3 questions, 27 model calls."""
    import asyncio

    from src.config import get_settings

    if not get_settings().google_api_key.get_secret_value():
        pytest.skip("GOOGLE_API_KEY is not set; the necessity check needs a live model")

    draws: dict[str, list[object]] = {label: [] for label, _, _ in CASES}

    async def repeated() -> list[list[object]]:
        # One event loop for every repeat. The chat model is memoised with an httpx client
        # bound to the loop that created it, so a second `asyncio.run` dies with "Event loop
        # is closed" — and a first version of this fixture then *skipped* on that error,
        # reporting "could not reach the model". A test that skips its own failure is the
        # D-023 pattern; the except below is narrowed to provider unreachability only.
        return [
            list(await check_many([(q, [PAPER_A, PAPER_B]) for _, q, _ in CASES]))
            for _ in range(REPEATS)
        ]

    import httpx

    try:
        rounds = asyncio.run(repeated())
    except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"could not reach the model: {exc}")
    for checks in rounds:
        for (label, _, _), check in zip(CASES, checks, strict=True):
            draws[label].append(check)
    return draws


def landed(draws: list[object], expected: dict[str, bool]) -> list[bool]:
    return [all(getattr(c, k) is v for k, v in expected.items()) for c in draws]


class TestNecessityFixtures:
    @pytest.mark.parametrize(("label", "question", "expected"), CASES, ids=[c[0] for c in CASES])
    def test_each_case_lands_where_it_was_placed_on_a_majority_of_draws(
        self,
        label: str,
        question: str,
        expected: dict[str, bool],
        results: dict[str, list[object]],
    ) -> None:
        hits = landed(results[label], expected)
        detail = [
            (getattr(c, "verdicts", None), getattr(c, "answerable_by_all_papers", None))
            for c in results[label]
        ]
        assert sum(hits) >= MIN_DRAWS[label], (
            f"{label}: landed on {sum(hits)} of {REPEATS} draws "
            f"(need {MIN_DRAWS[label]}) — {detail}"
        )
        if sum(hits) < REPEATS:
            # Visible, not swallowed: a split draw is the instrument's variance, and the
            # variance estimate downstream needs to know it exists.
            print(f"\n{label}: {sum(hits)} of {REPEATS} draws agreed — nondeterministic instrument")

    def test_the_stable_cases_are_distinguishable_and_the_check_can_fire(
        self, results: dict[str, list[object]]
    ) -> None:
        """The two stable cases must land apart on a majority, and the multi-hop signature
        must occur at least once — otherwise the check is agreeing, not discriminating."""
        from collections import Counter

        def signature(c: object) -> tuple[bool, bool]:
            return (c.answerable_by_one_paper, c.answerable_by_all_papers)  # type: ignore[attr-defined]

        stable = {
            label: Counter(signature(c) for c in results[label]).most_common(1)[0]
            for label in ("single_paper", "neither_nor_joint")
        }
        for label, (_, count) in stable.items():
            assert count >= MAJORITY, f"{label} has no majority signature: {stable}"
        assert stable["single_paper"][0] != stable["neither_nor_joint"][0], stable
        assert any(signature(c) == (False, True) for c in results["genuinely_multi_hop"]), (
            "the multi-hop signature never occurred in any draw; the check cannot fire"
        )

    def test_a_rejected_case_says_why(self, results: dict[str, list[object]]) -> None:
        """A cull with no reason cannot be acted on (D-023, the reporting variant)."""
        for label in ("single_paper", "neither_nor_joint"):
            assert any(c.rejection_reason for c in results[label])  # type: ignore[attr-defined]
