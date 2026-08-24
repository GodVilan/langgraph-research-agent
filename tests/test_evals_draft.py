"""The drafter's guard rails, and the cull report that diagnoses it.

The first drafting pass culled 7 of 8 candidates, all for `neither_nor_joint`. Pooled, that
reads as corpus scarcity and invites padding the stratum from weaker pairs. Broken out by
reason it reads as a prompt bug — which it was — so the breakdown is a permanent field
rather than a one-off observation. See docs/EVALS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.draft import (
    BANNED_PATTERNS,
    Candidate,
    ConstructionReport,
    CullReason,
    banned_phrases_in,
)


class TestBannedPhrasing:
    """Enforced after generation, not merely requested in the prompt.

    The model complied with "requires both papers" and still produced eight research
    proposals. A prompt instruction is a request; this is the check.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "How can the aggregation scheme be integrated into the other setting?",
            "What synthesis of the two approaches would work best?",
            "How could these methods be combined?",
            "How might the second paper's weighting be applied to the first?",
            "Which approach will generalise better?",
            "What potential improvements do these suggest?",
        ],
    )
    def test_invention_inviting_questions_are_caught(self, question: str) -> None:
        assert banned_phrases_in(question)

    @pytest.mark.parametrize(
        "question",
        [
            "How do these two differ in what they do to client updates before aggregation?",
            "What different evaluation setups do the two papers report for the same task?",
            "Which of the two reports a memory ceiling, and what does the other report?",
        ],
    )
    def test_comparative_questions_pass(self, question: str) -> None:
        """The shape that survived the necessity check on the real corpus."""
        assert banned_phrases_in(question) == []

    def test_the_reason_names_the_construction(self) -> None:
        found = banned_phrases_in("How can these be integrated?")

        assert "how can" in found
        assert "integrate" in found

    def test_every_pattern_has_a_name(self) -> None:
        assert all(name and pattern for name, pattern in BANNED_PATTERNS.items())


class TestCullReportIsNeverPooled:
    def _report(self, *reasons: CullReason) -> ConstructionReport:
        return ConstructionReport(
            candidates=[Candidate(f"q{n}", ["a", "b"], r) for n, r in enumerate(reasons)]
        )

    def test_counts_are_broken_out_by_reason(self) -> None:
        report = self._report(
            CullReason.KEPT, CullReason.NEITHER_NOR_JOINT, CullReason.SINGLE_PAPER
        )

        assert report.by_reason == {"kept": 1, "neither_nor_joint": 1, "single_paper": 1}

    def test_cull_rate_counts_only_non_kept(self) -> None:
        report = self._report(CullReason.KEPT, CullReason.NEITHER_NOR_JOINT)

        assert report.cull_rate == pytest.approx(0.5)

    def test_the_observed_failure_is_diagnosed_as_a_prompt_bug(self) -> None:
        """The real 7-of-8 run: every cull neither_nor_joint, none single_paper."""
        report = self._report(CullReason.KEPT, *[CullReason.NEITHER_NOR_JOINT] * 7)

        assert "prompt bug" in report.diagnosis()
        assert "weaker pairs" in report.diagnosis()

    def test_single_paper_culls_are_diagnosed_as_ordinary_calibration(self) -> None:
        """The shape that means the prompt is fixed and the residue is difficulty."""
        report = self._report(CullReason.KEPT, CullReason.SINGLE_PAPER, CullReason.SINGLE_PAPER)

        assert "Healthy" in report.diagnosis()

    def test_the_observed_pilot_reads_as_healthy(self) -> None:
        """The real post-fix pilot: 4 kept, 1 single_paper, 0 neither_nor_joint.

        An earlier diagnosis keyed on ratios called this "mixed" — the cleanest result the
        check can produce. Presence of a reason is the signal, not its share.
        """
        report = self._report(*[CullReason.KEPT] * 4, CullReason.SINGLE_PAPER)

        assert "Healthy" in report.diagnosis()

    def test_a_residue_of_unanswerable_questions_is_called_out(self) -> None:
        """Below the 50% threshold but non-zero: worth inspecting, not worth panicking."""
        report = self._report(*[CullReason.KEPT] * 8, CullReason.NEITHER_NOR_JOINT)

        assert "neither_nor_joint remain" in report.diagnosis()

    def test_banned_phrasing_is_its_own_reason(self) -> None:
        """Distinct from the necessity culls: it means the prompt leaked, not the pairing."""
        report = self._report(*[CullReason.BANNED_PHRASING] * 3, CullReason.KEPT)

        assert report.by_reason["banned_phrasing"] == 3
        assert "wording rules" in report.diagnosis()

    def test_an_empty_report_does_not_divide_by_zero(self) -> None:
        assert ConstructionReport().cull_rate == 0.0
        assert ConstructionReport().diagnosis() == "Nothing drafted."

    def test_render_shows_every_reason(self) -> None:
        rendered = self._report(CullReason.KEPT, CullReason.SINGLE_PAPER).render()

        assert "kept" in rendered and "single_paper" in rendered
        assert "never pool" in rendered
