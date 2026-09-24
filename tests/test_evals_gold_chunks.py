"""The gold-chunk check must be able to reject.

54 of 55 factual gold chunks holding is plausible and, on its own, unverified — the same
"has it ever fired" question as every other quiet check in the D-023 family. A gold-chunk
check that cannot fail would certify embedding-selected chunks forever, and Recall@k would
measure whether the dense retriever retrieves what the dense retriever picked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.verify_items import (
    answer_in_stem,
    asymmetric_premises,
    entities_in,
    false_premises,
    gold_chunks_support_jointly,
    premise_severity,
)

SUPPORTING = (
    "Table 3 reports top-1 accuracy across all settings. Our method reaches 23.4 percent "
    "error on the held-out split, compared with 27.9 for the strongest prior baseline."
)
UNRELATED = (
    "We describe the hardware configuration used throughout. Each run occupies a single "
    "accelerator with 80GB of memory, and the Shapley attribution is computed per client."
)


class TestGoldChunkCheckCanFail:
    """Against the check the builders run — not a single-chunk variant nothing called.

    These six tests spent a round proving that `gold_chunk_supports` could reject, while every
    builder and the gate called `gold_chunks_support_jointly`. The orphan guard in
    `test_no_orphan_checks.py` found it on its first run: a fixture and no call site (D-026).
    The single-chunk function is deleted; the property is asserted on the one that runs.
    """

    def test_a_chunk_containing_the_figure_is_accepted(self) -> None:
        ok, why = gold_chunks_support_jointly("23.4 percent error", [SUPPORTING])

        assert ok
        assert "every claim" in why

    def test_a_chunk_missing_the_figure_is_rejected(self) -> None:
        """The defect: gold chunks were hardware specs while the answer stated round counts."""
        ok, why = gold_chunks_support_jointly("23.4 percent error", [UNRELATED])

        assert not ok
        assert "23.4" in why

    def test_a_topically_similar_chunk_with_the_wrong_number_is_rejected(self) -> None:
        """Similarity is exactly what must not be sufficient here."""
        near_miss = "Table 3 reports top-1 accuracy. Our method reaches 31.8 percent error."
        ok, why = gold_chunks_support_jointly("23.4 percent error", [near_miss])

        assert not ok
        assert "23.4" in why

    def test_an_answer_with_no_figures_falls_back_to_words_and_says_so(self) -> None:
        ok, why = gold_chunks_support_jointly("the strongest prior baseline", [SUPPORTING])

        assert ok
        assert "weak" in why.lower()

    def test_an_answer_with_no_figures_and_no_shared_words_is_rejected(self) -> None:
        ok, _ = gold_chunks_support_jointly("quantum annealing schedules", [SUPPORTING])

        assert not ok

    def test_an_empty_answer_carries_nothing_checkable(self) -> None:
        ok, why = gold_chunks_support_jointly("", [SUPPORTING])

        assert not ok
        assert "nothing checkable" in why


class TestAnswerInStemCanFire:
    def test_a_stem_stating_its_own_figure_is_caught(self) -> None:
        caught, why = answer_in_stem("Is the reported error 23.4 percent?", "23.4 percent")

        assert caught
        assert "23.4" in why

    def test_an_ordinary_question_is_not(self) -> None:
        caught, _ = answer_in_stem("What error rate is reported on the held-out split?", "23.4")

        assert not caught


class TestPremiseChecksCanFire:
    PAPERS: ClassVar[list[str]] = [
        "we evaluate on ImageNet and report top-1",
        "we evaluate on MNIST only",
    ]

    def test_an_entity_in_no_paper_is_a_false_premise(self) -> None:
        assert false_premises("What do these report on CIFAR-10?", self.PAPERS) == ["CIFAR-10"]

    def test_an_entity_in_every_paper_is_not_flagged(self) -> None:
        assert asymmetric_premises("What about ImageNet?", ["uses ImageNet", "ImageNet too"]) == {}

    def test_an_entity_in_only_one_paper_is_asymmetric(self) -> None:
        """The union check missed this: present in the corpus, false of the pair."""
        assert asymmetric_premises("What about ImageNet?", self.PAPERS) == {"ImageNet": 1}

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What do the two papers report on ImageNet?", "false"),
            ("Which of the two papers evaluates on ImageNet?", "ok"),
            ("Regarding ImageNet, what is described?", "review"),
        ],
    )
    def test_severity_follows_what_the_question_asserts(self, question: str, expected: str) -> None:
        assert premise_severity(question, {"ImageNet": 1}) == expected

    def test_sentence_initial_capitals_are_not_entities(self) -> None:
        """Otherwise every question opens with a false premise."""
        assert entities_in("Which approach performs better?") == set()


class TestGoldChunksAreSelectedByTheAnswer:
    """Selection, not just the check, has to be answer-grounded.

    Adding a containment check on top of a question-similarity selector is not a fix: the
    selector picked chunks resembling the *question*, so they did not contain the answer's
    figures, and a pilot lost 5 of 5 items to a check correctly rejecting a selector that
    was still wrong. If gold chunks were selected by similarity, Recall@k would measure
    whether the retriever retrieves what the retriever picked.
    """

    INDEX: ClassVar[dict[str, list[dict[str, object]]]] = {
        "p1": [
            {"chunk_id": "p1_0", "text": "We describe the experimental setup and hardware."},
            {"chunk_id": "p1_1", "text": "Our method reaches 23.4 percent error on the split."},
            {"chunk_id": "p1_2", "text": "Related work reports 88.1 on a different task."},
        ]
    }

    def test_the_chunk_holding_the_answer_is_chosen(self) -> None:
        from evals.build_set import gold_chunks_for

        assert gold_chunks_for("23.4 percent error", "p1", self.INDEX)[0] == "p1_1"

    def test_a_figure_no_chunk_reports_yields_no_gold_chunk(self) -> None:
        """Usually the drafter citing a number the paper does not contain — a real finding."""
        from evals.build_set import gold_chunks_for

        assert gold_chunks_for("99.9 percent error", "p1", self.INDEX) == []

    def test_selection_ignores_question_similarity(self) -> None:
        """The setup chunk shares the question's wording and holds none of the answer."""
        from evals.build_set import gold_chunks_for

        chosen = gold_chunks_for("23.4", "p1", self.INDEX)

        assert "p1_0" not in chosen

    def test_every_selected_chunk_passes_the_containment_check(self) -> None:
        """Selector and check must agree, or one of them is decorative."""
        from evals.build_set import gold_chunks_for

        for chunk_id in gold_chunks_for("23.4 percent error", "p1", self.INDEX):
            text = next(c["text"] for c in self.INDEX["p1"] if c["chunk_id"] == chunk_id)
            assert gold_chunks_support_jointly("23.4 percent error", [str(text)])[0]


class TestMultiHopSupportIsJoint:
    """A multi-hop answer cites figures from several papers by construction.

    Requiring one chunk to contain all of them is a test no correct multi-hop item can pass;
    it rejected 5 of 5 otherwise-sound candidates. Support is a property of the gold *set*.
    """

    # The figures are deliberately rare ones. `0.1` and `0.5` were used here until document
    # frequency decided what counts as a claim, and they appear in 240 and 344 of 5,401
    # chunks — a chunk containing either is evidence of nothing, so the fixture was asserting
    # joint support over two non-claims. The property under test is unchanged; the numbers had
    # to become ones the corpus can actually distinguish.
    ANSWER = "Paper A reports a slot duration of 0.147 s, whereas paper B uses 0.482 s"

    def test_the_set_supports_what_no_single_chunk_does(self) -> None:

        chunks = ["the slot duration T is 0.147 s", "we adopt 0.482 s throughout"]

        assert gold_chunks_support_jointly(self.ANSWER, chunks)[0]
        assert not gold_chunks_support_jointly(self.ANSWER, [chunks[0]])[0]

    def test_a_missing_figure_still_fails(self) -> None:
        """Joint must not mean lenient."""
        from evals.verify_items import gold_chunks_support_jointly

        ok, why = gold_chunks_support_jointly(self.ANSWER, ["the slot duration T is 0.147 s"])

        assert not ok
        assert "0.482" in why

    def test_an_empty_gold_set_fails(self) -> None:
        from evals.verify_items import gold_chunks_support_jointly

        assert not gold_chunks_support_jointly(self.ANSWER, [])[0]

    def test_arxiv_ids_are_not_treated_as_figures(self) -> None:
        """ "Paper 2605.29913 reports 0.1 s" asserts one quantity, not two."""
        from evals.verify_items import numbers_in

        assert numbers_in("Paper 2605.29913 reports 0.1 s") == {"0.1"}

    def test_a_chunk_contributing_no_figure_is_named(self) -> None:
        """Gold chunks carried for no reason inflate Recall@k denominators."""
        from evals.verify_items import unsupported_chunks

        chunks = {"c1": "the slot duration T is 0.147 s", "c2": "hardware configuration details"}

        assert unsupported_chunks(self.ANSWER, chunks) == ["c2"]


class TestShortAnswersAreNotDiscarded:
    """Short answers are the most precise ones a factual question can have.

    A five-character word threshold left "five" and "Swin-Tiny" with no checkable token and
    rejected them as carrying nothing — while being correct, specific answers. A threshold
    tuned for prose discards exactly the answers a factual stratum wants.
    """

    @pytest.mark.parametrize(
        ("answer", "chunk"),
        [
            ("five", "we average results over five random seeds"),
            ("Swin-Tiny", "the Swin-Tiny backbone attains the highest score"),
            ("AUC", "we report AUC across all folds"),
        ],
    )
    def test_a_short_answer_present_in_the_chunk_is_supported(
        self, answer: str, chunk: str
    ) -> None:
        from evals.verify_items import gold_chunks_support_jointly

        assert gold_chunks_support_jointly(answer, [chunk])[0]

    def test_a_short_answer_absent_from_the_chunk_still_fails(self) -> None:
        from evals.verify_items import gold_chunks_support_jointly

        assert not gold_chunks_support_jointly("Swin-Tiny", ["a ResNet-50 backbone is used"])[0]

    def test_hyphenated_names_contribute_both_halves(self) -> None:
        from evals.verify_items import content_tokens

        assert content_tokens("Swin-Tiny") == {"swin", "tiny"}


class TestFiguresDecideTheStemCheck:
    """When an answer states a figure, the figure is the answer; the words are framing."""

    def test_shared_framing_words_are_not_a_leak(self) -> None:
        """ "At what epoch...?" / "epoch 25" gives away nothing — 25 is absent from the stem."""
        from evals.verify_items import answer_in_stem

        assert not answer_in_stem("At what epoch does it stop converging?", "epoch 25")[0]

    def test_a_figure_present_in_the_stem_is_still_a_leak(self) -> None:
        from evals.verify_items import answer_in_stem

        assert answer_in_stem("Is the error 23.4 percent?", "23.4 percent")[0]

    def test_a_figureless_answer_still_uses_word_overlap(self) -> None:
        from evals.verify_items import answer_in_stem

        assert answer_in_stem("Does it use the Swin transformer?", "Swin transformer")[0]


class TestTheCheckChainIsShared:
    """One chain, three callers. Divergence must be impossible, not merely tested for.

    The chain was reimplemented three times and diverged three times: multi-hop ran five
    checks, factual ran two and then four, and the behavioural matrix tested a fourth copy
    written in the test file — so the matrix passed green while `build_factual` was missing
    `banned_phrasing` and `false_premises`, and could never have caught it. A test of a
    reimplementation proves the reimplementation works.

    Asserting *call* rather than the presence of strings: the earlier version of this test
    looked for reason names in the builder's source and broke the moment the checks were
    correctly extracted into one function.
    """

    def test_every_caller_delegates_to_the_shared_chain(self) -> None:
        import inspect

        from evals import build_set, draft, verify_dataset

        for module, function in (
            (build_set, "build_factual"),
            (draft, "draft_multi_hop"),
            (verify_dataset, "check_item"),
        ):
            source = inspect.getsource(getattr(module, function))
            assert "classify_candidate" in source, (
                f"{module.__name__}.{function} does not use the shared chain; a private "
                f"copy is how the builders diverged before"
            )

    def test_the_chain_returns_none_only_for_a_clean_candidate(self) -> None:
        from evals.verify_items import classify_candidate

        clean = classify_candidate(
            question="What error rate is reported on the held-out split?",
            answer="23.4 percent",
            gold_texts=["our method reaches 23.4 percent error"],
            paper_texts=["our method reaches 23.4 percent error"],
            banned=[],
            leaked=False,
        )

        assert clean is None

    def test_the_chain_reports_a_reason_string_the_enum_accepts(self) -> None:
        """A reason the enum cannot parse is the mislabel defect (D-023, sixth)."""
        from evals.draft import CullReason
        from evals.verify_items import classify_candidate

        verdict = classify_candidate(
            question="Q?", answer="", gold_texts=[], paper_texts=[], banned=[], leaked=False
        )

        assert verdict is not None
        assert CullReason(verdict[0]) is CullReason.NO_GOLD_ANSWER
