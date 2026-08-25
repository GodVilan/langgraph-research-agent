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

from evals.build_set import (
    MAX_PHRASE_OVERLAP,
    MAX_VERBATIM_RUN,
    leaks,
    longest_verbatim_run,
    phrase_overlap,
)
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


class TestPhraseOverlapMeasuresPhrases:
    """Verbatim phrase reuse, not word reuse — the rule EVALS.md actually states.

    The first implementation measured unigram bag overlap and culled 39 of 55 factual
    questions. Inspecting them showed the overlap was dominated by *unavoidable* proper
    nouns: "What is the expense per 1,000 evaluations for Gemini 3.1 Flash-Lite?" scored
    0.80 while being a correctly paraphrased question, because a model name has no synonym.
    Against real 380-token chunks the n-gram metric culls none of twelve, with longest
    verbatim runs of one to four words — so the drafter was never copying phrasing.

    Which raises the same question as every other quiet check (D-023): can it fire?
    """

    GOLD = (
        "Table 4 reports the cost per 1,000 evaluations for each model. The top-1 error "
        "rate achieved by the varied bound variant of LPA on CIFAR-100 is 23.4 percent, "
        "and we require round-to-nearest rounding mode for the tail value r throughout "
        "all of the experiments described in this section."
    )

    def test_a_verbatim_lift_is_caught(self) -> None:
        """The failure the check exists for: retrieval succeeding on string match."""
        lifted = "What is the top-1 error rate achieved by the varied bound variant of LPA?"

        assert leaks(lifted, self.GOLD)
        assert longest_verbatim_run(lifted, self.GOLD) > MAX_VERBATIM_RUN

    def test_a_necessary_entity_name_does_not_trigger_it(self) -> None:
        """You cannot ask what a paper reports on CIFAR-100 without writing CIFAR-100."""
        paraphrased = "Which dataset does the reported 23.4 percent figure refer to?"

        assert not leaks(paraphrased, self.GOLD)
        assert longest_verbatim_run(paraphrased, self.GOLD) <= MAX_VERBATIM_RUN

    def test_a_technical_term_with_no_synonym_does_not_trigger_it(self) -> None:
        """Nine words is six 4-grams, so one incidental match scores 0.167 against a
        threshold calibrated on 12-18 word questions. `leaks` declines to apply the ratio
        below `MIN_NGRAMS_FOR_RATIO` and lets the run-length check decide."""
        question = "Which rounding approach is required for the tail value?"

        assert not leaks(question, self.GOLD)
        assert phrase_overlap(question, self.GOLD) > MAX_PHRASE_OVERLAP  # the raw ratio does

    @pytest.mark.parametrize(
        "question",
        [
            "What is the maximum reduction in computational operations achieved by AsymVLM?",
            "What is the expense per one thousand assessments for Claude Haiku 4.5?",
            "What iteration budget was utilized for the majority of the training runs?",
        ],
    )
    def test_real_drafted_questions_pass(self, question: str) -> None:
        """Verbatim from the probe run — these are what the drafter actually produces."""
        assert not leaks(question, self.GOLD)

    def test_a_short_question_is_judged_on_run_length_alone(self) -> None:
        """Guards the guard: a short question that *does* lift must still be caught."""
        lifted_short = "the cost per 1,000 evaluations for each model?"

        assert not leaks("Which dataset is used?", self.GOLD)
        assert leaks(lifted_short, self.GOLD)

    def test_an_empty_question_does_not_divide_by_zero(self) -> None:
        assert phrase_overlap("", self.GOLD) == 0.0

    def test_the_run_length_is_measured_in_words(self) -> None:
        assert longest_verbatim_run("the cost per 1,000 evaluations for each model", self.GOLD) >= 7
        assert longest_verbatim_run("entirely unrelated wording here", self.GOLD) <= 1


class TestDuplicateDraws:
    """N draws from one prompt is not N items.

    Three ambiguous questions per topic, drawn from an identical prompt, produced two
    byte-identical pairs — the stratum reported n=10 while holding 8 distinct questions.
    Caught by validating the written set, not by anything in the drafting path, which is
    why the check now lives in the drafting path.
    """

    def _report(self, *reasons: CullReason) -> ConstructionReport:
        return ConstructionReport(
            candidates=[Candidate(f"q{n}", ["a"], r) for n, r in enumerate(reasons)]
        )

    def test_duplicates_are_their_own_reason(self) -> None:
        report = self._report(CullReason.KEPT, CullReason.DUPLICATE)

        assert report.by_reason["duplicate"] == 1

    def test_a_duplicate_draw_is_diagnosed_plainly(self) -> None:
        """Distinct from every other cull: the prompt is fine, the sampling is not."""
        diagnosis = self._report(*[CullReason.KEPT] * 8, CullReason.DUPLICATE).diagnosis()

        assert "duplicate draws" in diagnosis
        assert "fewer distinct items than it appears" in diagnosis

    def test_a_set_with_duplicates_is_not_reported_as_healthy(self) -> None:
        report = self._report(*[CullReason.KEPT] * 9, CullReason.DUPLICATE)

        assert "Healthy" not in report.diagnosis()


class TestDiagnosisDescribesWhatHappened:
    """A report must not explain a mechanism the stratum does not have.

    The healthy-case message said "the only culls are single_paper — those pairs are too
    close" and was printed verbatim against strata with no pairs and no culls at all.
    Correct verdict, false explanation (DECISIONS D-023, the reporting variant).
    """

    def _report(self, *reasons: CullReason) -> ConstructionReport:
        return ConstructionReport(
            candidates=[Candidate(f"q{n}", ["a"], r) for n, r in enumerate(reasons)]
        )

    def test_zero_culls_is_not_explained_as_single_paper(self) -> None:
        diagnosis = self._report(*[CullReason.KEPT] * 10).diagnosis()

        assert "Clean" in diagnosis
        assert "single_paper" not in diagnosis
        assert "pairs" not in diagnosis

    def test_benign_culls_still_get_the_calibration_explanation(self) -> None:
        diagnosis = self._report(*[CullReason.KEPT] * 4, CullReason.SINGLE_PAPER).diagnosis()

        assert "Healthy" in diagnosis
        assert "single_paper" in diagnosis


class TestCullReasonsAreGroupedSafely:
    """A mislabelled reason is worse than a missing one.

    `banned_phrasing: 39` appeared in a construction report for 39 lexical-overlap
    rejections. It read as a check firing constantly while that check had never fired once,
    which retires the exact question the live probe exists to answer (DECISIONS D-023).
    """

    def test_an_unknown_reason_fails_loudly_rather_than_being_tallied(self) -> None:
        report = ConstructionReport(candidates=[Candidate("q", ["a"], "made_up")])  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="not a CullReason"):
            _ = report.by_reason

    def test_every_reason_the_builder_emits_is_a_member(self) -> None:
        """Guards against a new cull site inventing a string the report cannot group."""
        import evals.build_set as builder

        emitted = {
            name
            for name in dir(CullReason)
            if not name.startswith("_") and isinstance(getattr(CullReason, name), CullReason)
        }
        assert {"KEPT", "LEXICAL_OVERLAP", "DUPLICATE", "TOPIC_NOT_ABSENT"} <= emitted
        assert builder.CullReason is CullReason

    def test_topic_and_phrasing_culls_are_distinct_reasons(self) -> None:
        """A topic that turns out to be present is not a banned construction."""
        assert CullReason.TOPIC_NOT_ABSENT is not CullReason.BANNED_PHRASING
