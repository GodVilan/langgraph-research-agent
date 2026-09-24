"""Stratum x defect-class: every builder must reject every defect, with the right reason.

This replaces two tests that inspected builder source for the presence of a string. Those
proved a name appears in a function — not that the check runs, receives the right input, or
emits the right reason. They would pass against a check wired up backwards and fail against
a correct refactor, which is the wrong sensitivity in both directions.

Each case here injects a known-bad item of a known class and asserts the specific
``CullReason``. It is the same argument as every detector needing a case that makes it fire
(DECISIONS D-023), applied to the matrix rather than to one check at a time: a chain applied
to one stratum is not a chain, and that gap shipped once already — `build_factual` ran no
gold-containment and no answer-in-stem check while the chain was reported as governing the
whole set.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.build_set import leaks
from evals.draft import BANNED_PATTERNS, REMEDIES, CullReason, banned_phrases_in
from evals.verify_items import (
    asymmetric_premises,
    classify_candidate,
    conjunctive_discriminator,
    contrastive_premise,
    degenerate_clauses,
    entities_in,
    false_premises,
    premise_severity,
)

GOLD = (
    "Table 4 reports the cost per 1,000 evaluations. The top-1 error rate achieved by the "
    "varied bound variant of LPA on CIFAR-100 is 23.4 percent, and we require "
    "round-to-nearest rounding for the tail value r in every experiment we describe."
)
# Paper A carries every entity the well-formed cases name, so that a case meant to test a
# late check is not caught by the premise check first. The premise cases below deliberately
# name something these do not contain.
PAPERS = [
    "we evaluate the varied bound variant of LPA on ImageNet and report top-1 accuracy",
    "we evaluate on MNIST only",
]


def classify(question: str, answer: str, gold: list[str], papers: list[str]) -> CullReason:
    """Delegates to the shared chain the builders run — not a reimplementation of it.

    An earlier version of this helper *was* a reimplementation, and it passed while
    `build_factual` was missing `banned_phrasing` and `false_premises`. A matrix that tests
    its own copy of the logic proves the copy works. It can only speak for the builders if
    it runs their code.
    """
    verdict = classify_candidate(
        question=question,
        answer=answer,
        gold_texts=gold,
        paper_texts=papers,
        banned=banned_phrases_in(question),
        leaked=bool(gold) and leaks(question, gold[0]),
    )
    return CullReason(verdict[0]) if verdict else CullReason.KEPT


# (defect class, question, answer, gold chunks, papers, expected reason)
CASES: list[tuple[str, str, str, list[str], list[str], CullReason]] = [
    (
        "clean",
        "What error rate is reported on the held-out split?",
        "23.4 percent",
        [GOLD],
        PAPERS,
        CullReason.KEPT,
    ),
    (
        "banned phrasing",
        "How can the two approaches be integrated?",
        "23.4 percent",
        [GOLD],
        PAPERS,
        CullReason.BANNED_PHRASING,
    ),
    (
        "no gold answer",
        "What error rate is reported on the held-out split?",
        "",
        [GOLD],
        PAPERS,
        CullReason.NO_GOLD_ANSWER,
    ),
    (
        "answer in stem",
        "Is the reported error rate 23.4 percent?",
        "23.4 percent",
        [GOLD],
        PAPERS,
        CullReason.ANSWER_IN_STEM,
    ),
    (
        "false premise (in no paper)",
        "What do the two papers report on CIFAR-10?",
        "23.4 percent",
        [GOLD],
        PAPERS,
        CullReason.FALSE_PREMISE,
    ),
    (
        "false premise (asymmetric, universal form)",
        "What do the two papers report on ImageNet?",
        "23.4 percent",
        [GOLD],
        PAPERS,
        CullReason.FALSE_PREMISE,
    ),
    (
        "lexical leak",
        "What is the top-1 error rate achieved by the varied bound variant of LPA?",
        "23.4 percent",
        [GOLD],
        PAPERS,
        CullReason.LEXICAL_OVERLAP,
    ),
    (
        "gold unsupported",
        "What error rate is reported on the held-out split?",
        "31.8 percent",
        [GOLD],
        PAPERS,
        CullReason.GOLD_UNSUPPORTED,
    ),
    (
        "no gold chunk at all",
        "What error rate is reported on the held-out split?",
        "23.4 percent",
        [],
        PAPERS,
        CullReason.GOLD_UNSUPPORTED,
    ),
]


class TestDefectMatrix:
    @pytest.mark.parametrize(
        ("label", "question", "answer", "gold", "papers", "expected"),
        CASES,
        ids=[c[0] for c in CASES],
    )
    def test_each_defect_class_yields_its_own_reason(
        self,
        label: str,
        question: str,
        answer: str,
        gold: list[str],
        papers: list[str],
        expected: CullReason,
    ) -> None:
        assert classify(question, answer, gold, papers) is expected

    def test_a_discriminative_question_is_not_a_false_premise(self) -> None:
        """ "Which of the two uses ImageNet" asserts it of neither and asks."""
        reason = classify(
            "Which of the two papers evaluates on ImageNet?", "23.4 percent", [GOLD], PAPERS
        )

        assert reason is CullReason.KEPT

    def test_every_defect_class_is_distinguishable(self) -> None:
        """Two classes collapsing to one reason is the mislabel defect (D-023, sixth)."""
        reasons = [expected for *_, expected in CASES if expected is not CullReason.KEPT]

        assert len(set(reasons)) >= 5


TWO_PAPERS = ["2605.11111", "2605.22222"]

# Two papers that between them satisfy a conjunction neither satisfies alone. Hand-authored to
# the shape human review found in `mh-014`: one paper has the packet traces, the other has the
# four-phase protocol, and every earlier premise check passes because each named thing exists
# in *some* paper and each half of the answer is individually true.
SPLIT_PAPERS = [
    "we evaluate on CICIDS2017 packet captures collected from an enterprise network",
    "we describe a four-phase protocol and report numerical results on MNIST",
]


def classify_pair(
    question: str, answer: str, gold: dict[str, str], papers: list[str]
) -> CullReason:
    """The shared chain with the clause-scoped inputs a multi-hop item carries."""
    verdict = classify_candidate(
        question=question,
        answer=answer,
        gold_texts=list(gold.values()),
        paper_texts=papers,
        banned=banned_phrases_in(question),
        leaked=False,
        source_paper_ids=TWO_PAPERS,
        gold_by_paper={p: [t for g, t in gold.items() if g.startswith(p)] for p in TWO_PAPERS},
    )
    return CullReason(verdict[0]) if verdict else CullReason.KEPT


class TestMultiHopDefectMatrix:
    """The five classes human review found in items that had passed the strengthened chain.

    Every one of them is a *clause* defect: the answer is two halves joined by a contrast, and
    every check pooled their claims into one set. Pooling cannot tell "this paper supports its
    own claim" from "this paper's chunks contain the other paper's numbers".
    """

    def test_a_clause_naming_a_variable_without_a_value_is_degenerate(self) -> None:
        """`mh-004`: "reports a hidden dimension h" answers "what value" with a symbol."""
        reason = classify_pair(
            "What embedding dimension does each paper report?",
            "Paper 2605.11111 reports 23.4 for its encoder, whereas Paper 2605.22222 reports "
            "a hidden dimension h of the sparse autoencoder.",
            {"2605.11111_0001": GOLD, "2605.22222_0001": GOLD},
            ["reports 23.4", "a hidden dimension h"],
        )

        assert reason is CullReason.DEGENERATE_CLAUSE

    def test_a_clause_stating_only_commonplace_figures_is_reported_as_such(self) -> None:
        """`mh-001`: -90 dBm and 0.1 s are real values that nothing can ground.

        Depends on measured corpus frequency, deliberately: `100` appears in 581 of 5,401
        chunks, so a chunk containing it is evidence of almost nothing. The fixture asserts
        the *reason string*, because reporting this as "states no value" would name a defect
        the item does not have.
        """
        verdict = classify_candidate(
            question="What distinct batch sizes do the two papers report?",
            answer="Paper 2605.11111 reports 100, whereas Paper 2605.22222 reports 100.",
            gold_texts=[GOLD],
            paper_texts=["reports 100", "reports 100"],
            banned=[],
            leaked=False,
            source_paper_ids=TWO_PAPERS,
        )

        assert verdict is not None
        assert CullReason(verdict[0]) is CullReason.DEGENERATE_CLAUSE
        assert "too common in the corpus" in verdict[1]

    def test_a_conjunction_no_single_paper_satisfies_is_a_false_premise(self) -> None:
        """`mh-014`: which paper does A *while* doing B, when A and B live in different papers."""
        reason = classify_pair(
            "Which of the two studies evaluates on CICIDS2017 while reporting on MNIST?",
            "Paper 2605.11111 reports 23.4 and Paper 2605.22222 reports 31.8.",
            {"2605.11111_0001": GOLD},
            SPLIT_PAPERS,
        )

        assert reason is CullReason.CONJUNCTIVE_PREMISE

    def test_a_comparative_question_naming_both_papers_things_is_not_that(self) -> None:
        """The tightening rule. `mh-007` names benchmarks from both papers and is sound.

        Its second half is *about the other paper*, so the discriminating clause ends at the
        turn. A check that cut the question at its end instead would reject every well-formed
        comparative item in the stratum.
        """
        assert (
            conjunctive_discriminator(
                "Which of the two evaluates on MATH500, and what benchmarks does the other use?"
            )
            == ""
        )

    def test_a_paper_whose_gold_holds_only_the_other_clause_is_barren(self) -> None:
        """`mh-004`'s second defect: 30120's chunks were credited for 29628's figures."""
        reason = classify_pair(
            "What does each paper report?",
            "Paper 2605.11111 reports 23.4 percent, whereas Paper 2605.22222 reports 31.8 percent.",
            {"2605.11111_0001": GOLD, "2605.22222_0001": GOLD},
            ["reports 23.4", "reports 31.8"],
        )

        # GOLD carries 23.4 and not 31.8, so the second paper's gold supports only the first
        # paper's claim — which pooled attribution scored as a pass.
        assert reason is CullReason.GOLD_UNSUPPORTED

    def test_a_question_asserting_a_difference_the_answer_does_not_show(self) -> None:
        assert (
            contrastive_premise(
                "What distinct slot durations do the two papers report?",
                "Paper 2605.11111 reports 0.16 s, whereas Paper 2605.22222 reports 0.16 s.",
                TWO_PAPERS,
            )[0]
            == "false"
        )

    def test_a_genuine_difference_passes(self) -> None:
        assert (
            contrastive_premise(
                "What distinct slot durations do the two papers report?",
                "Paper 2605.11111 reports 0.16 s, whereas Paper 2605.22222 reports 0.48 s.",
                TWO_PAPERS,
            )[0]
            == "ok"
        )

    def test_an_answer_naming_no_paper_by_id_declines_rather_than_rejects(self) -> None:
        """`mh-003` names its papers descriptively. Rejecting that is a false rejection."""
        assert (
            degenerate_clauses(
                "The cytology study reports 1,634 patients, whereas the screening study "
                "reports 183,098.",
                TWO_PAPERS,
            )
            == {}
        )


class TestEveryReasonHasARemedy:
    """Exhaustive over possible reasons, not over the ones that happen to appear.

    The reporting layer failed four times, each because a chain of hand-written cases knew
    only the reasons existing when it was written. The next reason added would have been the
    fifth, so the map is asserted complete rather than maintained by attention.
    """

    def test_no_reason_lacks_a_remedy(self) -> None:
        missing = [r.value for r in CullReason if r is not CullReason.KEPT and r not in REMEDIES]

        assert not missing, f"{missing} would be reported without a remedy"

    def test_remedies_do_not_outlive_their_reasons(self) -> None:
        assert all(isinstance(r, CullReason) for r in REMEDIES)

    def test_kept_is_not_a_remedy(self) -> None:
        assert CullReason.KEPT not in REMEDIES

    @pytest.mark.parametrize("reason", [r for r in CullReason if r is not CullReason.KEPT])
    def test_each_remedy_says_something_actionable(self, reason: CullReason) -> None:
        assert len(REMEDIES[reason]) > 20

    def test_every_banned_pattern_is_named(self) -> None:
        """A pattern without a name produces a cull nobody can act on."""
        assert all(name and pattern for name, pattern in BANNED_PATTERNS.items())
        assert banned_phrases_in("How can these be integrated?") == ["integrate", "how can"]


class TestBoundaryMatchingInPremises:
    """Fixture #5's substring case, asserted directly rather than by accident.

    `CIFAR-10` against a passage saying `CIFAR-100` is a boundary trap, and the naive
    version passed it: `"cifar-10" in "cifar-100"` is true, so a question whose premise is
    false was not flagged. The same shape as `Gram` matching inside "n-gram" during absence
    verification — `absence._mentions` learned it, the premise check had not.
    """

    CIFAR100_PAPERS: ClassVar[list[str]] = [
        "we evaluate the varied bound variant on CIFAR-100 and report top-1",
        "we evaluate on MNIST only",
    ]

    def test_a_shorter_name_does_not_match_inside_a_longer_one(self) -> None:
        assert false_premises(
            "What do the two papers report on CIFAR-10?", self.CIFAR100_PAPERS
        ) == ["CIFAR-10"]

    def test_the_exact_name_still_matches(self) -> None:
        assert (
            false_premises("What do the two papers report on CIFAR-100?", self.CIFAR100_PAPERS)
            == []
        )

    @pytest.mark.parametrize(
        ("asked", "present"), [("A100", "A1000"), ("GPT-4", "GPT-40"), ("BERT", "BERTology")]
    )
    def test_the_boundary_holds_for_other_confusable_names(self, asked: str, present: str) -> None:
        assert false_premises(f"Which paper used {asked}?", [f"we used {present}", "none"]) == [
            asked
        ]

    def test_asymmetric_premises_uses_the_same_boundary(self) -> None:
        """Both premise checks, or the fix is half-applied — the shape of every gap so far."""
        assert asymmetric_premises("What about CIFAR-10?", self.CIFAR100_PAPERS) == {}
        assert asymmetric_premises("What about CIFAR-100?", self.CIFAR100_PAPERS) == {
            "CIFAR-100": 1
        }


class TestDiscriminativeFormWithAnAbsentEntity:
    """The shape that auto-passes `premise_severity` — caught, but by a different check.

    `premise_severity` returns "ok" for any discriminative form, so on its own it would pass
    "Which of the two evaluates on CIFAR-10?" when neither paper uses CIFAR-10. What saves
    it is `false_premises` running earlier in the chain. That is worth asserting rather than
    assuming: the protection lives in a different check from the one a reader would look at,
    and reordering the chain would silently remove it.
    """

    PAPERS: ClassVar[list[str]] = ["we evaluate on ImageNet only", "we evaluate on MNIST only"]

    def test_severity_alone_would_pass_it(self) -> None:
        assert premise_severity("Which of the two evaluates on CIFAR-10?", {}) == "ok"

    def test_the_chain_catches_it_anyway(self) -> None:
        assert (
            classify("Which of the two evaluates on CIFAR-10?", "23.4 percent", [GOLD], self.PAPERS)
            is CullReason.FALSE_PREMISE
        )

    def test_the_same_form_with_a_present_entity_is_kept(self) -> None:
        assert (
            classify("Which of the two evaluates on ImageNet?", "23.4 percent", [GOLD], self.PAPERS)
            is CullReason.KEPT
        )


class TestGenericVocabularyIsNotAPremise:
    """The premise check asserts *named* entities, not technical vocabulary.

    A question asking "how much RAM was used" against a paper reporting "256GB system
    memory" is a correct paraphrase — the one the construction rules demand — and flagging
    it penalises the behaviour being asked for. Substring matching hid this by matching
    `ram` inside "framework", "program" and "diagram", so tightening the boundary surfaced a
    false positive that had been passing by accident.

    Same shape as the unigram leakage metric rejecting "Gemini 3.1 Flash-Lite": a stricter
    check is an improvement only if what it newly rejects is actually wrong.
    """

    HARDWARE: ClassVar[list[str]] = [
        "experiments ran on a single 80GB accelerator with 256GB of system memory",
        "we use the same hardware throughout",
    ]

    @pytest.mark.parametrize("term", ["RAM", "GPU", "CPU", "FLOPS", "AUC", "SGD"])
    def test_generic_terms_are_not_treated_as_entities(self, term: str) -> None:
        assert term not in entities_in(f"How much {term} was used?")

    def test_a_paraphrased_hardware_question_is_not_a_false_premise(self) -> None:
        assert false_premises("How much RAM was used for the experiments?", self.HARDWARE) == []

    def test_a_genuine_named_entity_is_still_asserted(self) -> None:
        """The exclusion must not blunt the check it protects."""
        assert false_premises("What is reported on CIFAR-10?", self.HARDWARE) == ["CIFAR-10"]


class TestWhatCountsAsEvidence:
    """The rarity criterion, and the two rules it replaced.

    "A decimal, or three or more digits" was a guess at rarity and wrong by an order of
    magnitude: it admitted `100` (581 of 5,401 chunks) as a claim on the same footing as
    `183,098` (0). Three multi-hop items were grounded on figures of that kind — `0.1` matched
    pseudocode containing neither quantity the answer stated, `2.5` came from the model name
    "Qwen 2.5 7B", and `3.2` from "Llama 3.2" matched a *section number* in the other paper.
    """

    def test_the_ceiling_is_derived_from_the_coincidence_budget(self) -> None:
        """Not a tuned constant: 1 - (1 - p)**3 < 0.05 over 5,401 chunks."""
        from evals.verify_items import max_document_frequency

        assert max_document_frequency(5401) == 91

    def test_a_commonplace_figure_is_not_a_claim(self) -> None:
        from evals.verify_items import discriminating_figures

        assert "100" not in discriminating_figures("we train for 100 epochs")

    def test_a_rare_figure_is(self) -> None:
        from evals.verify_items import discriminating_figures

        assert "23.4" in discriminating_figures("the error rate is 23.4 percent")

    def test_a_parameter_scale_is_a_claim_of_its_own_kind(self) -> None:
        """`405B` yielded the bare figure 405 and `1B` yielded nothing at all, so an answer
        contrasting checkpoint sizes asserted nothing checkable about the smaller model."""
        from evals.verify_items import claims_in

        claims = claims_in("evaluating Llama-family 8B, 70B, and 405B checkpoints")

        assert claims["scales"] == {"8B", "70B", "405B"}
        assert "405" not in claims["figures"]

    def test_a_scale_does_not_match_inside_a_larger_one(self) -> None:
        from evals.matching import mentions_name

        assert not mentions_name("we evaluate the 21b checkpoint", "1b")

    def test_a_bare_common_figure_answer_is_weak_rather_than_uncheckable(self) -> None:
        """Four factual answers are a number and nothing else. Rejecting them as "nothing
        checkable" is the short-answer failure a second time."""
        from evals.verify_items import gold_chunks_support_jointly

        ok, why = gold_chunks_support_jointly("0.001", ["we use a learning rate of 0.001"])

        assert ok
        assert "WEAK" in why

    def test_and_still_fails_when_the_figure_is_absent(self) -> None:
        from evals.verify_items import gold_chunks_support_jointly

        ok, why = gold_chunks_support_jointly("0.001", ["we use a learning rate of 0.01"])

        assert not ok
        assert "0.001" in why


class TestStructuralIneligibility:
    """A reference entry is `[8] B. McMahan, …`; citation-dense prose is not one.

    Counting `[n]` markers barred 11.6% of the corpus, and the false positives were the most
    fact-dense chunks a paper has — results tables, methods prose, a license table. Three of
    them were named by a human as the correct gold for items the selector then got wrong.
    """

    PROSE: ClassVar[str] = (
        "We base our experiments on WSAC [14], which reduces the dimensionality from 1024 "
        "to 100. Later works add keyword guidance [16], a datastore [17], and LoRA [6]."
    )
    ENTRIES: ClassVar[str] = (
        "[8] B. McMahan, E. Moore, D. Ramage. Communication-efficient learning. AISTATS, "
        "2017. [13] T. Li, Z. He, Y. Li. Flat-LoRA: low-rank adaptation. ICML, 2024. "
        "[14] X. Wu and Y. Chen. Zero-shot audio captioning. ICASSP, 2023."
    )

    def test_citation_dense_prose_stays_eligible(self) -> None:
        from evals.verify_items import is_ineligible_gold

        assert is_ineligible_gold(self.PROSE) is None

    def test_a_reference_list_does_not(self) -> None:
        from evals.verify_items import is_ineligible_gold

        assert is_ineligible_gold(self.ENTRIES) == "reference list"

    def test_a_results_paragraph_citing_urls_stays_eligible(self) -> None:
        """The URL rule cost sp-009 the only chunk stating its answer, and no threshold
        separates the classes — a real bibliography runs 1.23 URL markers per 100 words and
        that paragraph runs 1.43."""
        from evals.verify_items import is_ineligible_gold

        pricing = (
            "GPT-5.4 nano costs $3.387 per 1,000 evaluations, see https://a.example/pricing, "
            "Gemini costs $4.157, https://b.example/pricing, and Claude costs 19.746 per "
            "1,000 evaluations, https://c.example/pricing."
        )

        assert is_ineligible_gold(pricing) is None


class TestGoldSetsAreMinimal:
    """Interchangeable evidence dilutes Recall@k, and no earlier check could see it.

    ``unsupported_chunks`` names chunks supporting *nothing*. Three chunks of one paper each
    supporting the same four benchmark names each support something, so all three survived —
    and Recall@k divides by the size of the gold set.
    """

    ANSWER = "Paper 2605.11111 reports 23.4 percent, whereas Paper 2605.22222 reports 31.8."
    PAPERS: ClassVar[list[str]] = ["2605.11111", "2605.22222"]

    def test_interchangeable_chunks_collapse_to_one(self) -> None:
        from evals.verify_items import minimal_gold

        chunks = {
            "2605.11111_0001": "the error rate is 23.4 percent",
            "2605.11111_0002": "we again report 23.4 percent on the split",
            "2605.22222_0001": "our method reaches 31.8",
        }

        assert minimal_gold(self.ANSWER, chunks, self.PAPERS) == [
            "2605.11111_0001",
            "2605.22222_0001",
        ]

    def test_every_contributing_paper_keeps_a_chunk(self) -> None:
        """Minimising must not drop a paper out of a multi-hop item."""
        from evals.verify_items import minimal_gold

        chunks = {
            "2605.11111_0001": "the error rate is 23.4 percent and the other reaches 31.8",
            "2605.22222_0001": "our method reaches 31.8",
        }

        assert "2605.22222_0001" in minimal_gold(self.ANSWER, chunks, self.PAPERS)

    def test_an_ineligible_chunk_is_never_kept(self) -> None:
        """The pruner kept a chunk the containment check then rejected, turning sp-009 weak."""
        from evals.verify_items import minimal_gold

        chunks = {
            "2605.11111_0001": TestStructuralIneligibility.ENTRIES + " 23.4 percent",
            "2605.11111_0002": "the error rate is 23.4 percent",
            "2605.22222_0001": "our method reaches 31.8",
        }

        assert "2605.11111_0001" not in minimal_gold(self.ANSWER, chunks, self.PAPERS)


class TestTheDroppedQualifier:
    """`mh-002`: claims come from the answer, so an answer that drops the question's qualifier
    yields a claim set that cannot ground it. Containment then confirms the entity's *name*
    while nothing checks the *role* the question assigned it."""

    QUESTION = (
        "Which of the two papers fine-tunes Qwen2.5 7B, and what model does the other paper "
        "use for its primary maze-trained setup?"
    )
    ANSWER = (
        "Paper 2605.11111 reports fine-tuning Qwen2.5 7B, while Paper 2605.22222 reports "
        "using Qwen3-4B-Instruct-2507 as its primary model."
    )

    def test_gold_that_names_the_entity_but_not_the_qualifier_is_rejected(self) -> None:
        from evals.verify_items import unaddressed_qualifiers

        gold = {
            "2605.11111": ["the base model is Qwen2.5 7B, fine-tuned using LoRA"],
            "2605.22222": ["Figure 30: Emotion PC1 extracted from Qwen3-4B-Instruct-2507"],
        }

        assert unaddressed_qualifiers(self.QUESTION, self.ANSWER, gold, TWO_PAPERS) == {
            "2605.22222": ["maze-trained"]
        }

    def test_gold_that_addresses_the_qualifier_passes(self) -> None:
        """Inflection-tolerant across the hyphen: "maze training" satisfies "maze-trained"."""
        from evals.verify_items import unaddressed_qualifiers

        gold = {
            "2605.11111": ["the base model is Qwen2.5 7B, fine-tuned using LoRA"],
            "2605.22222": ["models used as starting checkpoints for maze training: Qwen3-4B"],
        }

        assert unaddressed_qualifiers(self.QUESTION, self.ANSWER, gold, TWO_PAPERS) == {}

    def test_the_chain_reports_it_under_its_own_reason(self) -> None:
        reason = classify_pair(
            self.QUESTION,
            self.ANSWER,
            {
                "2605.11111_0001": "the base model is Qwen2.5 7B, fine-tuned using LoRA",
                "2605.22222_0001": "Figure 30: Emotion PC1 from Qwen3-4B-Instruct-2507 (4B)",
            },
            ["Qwen2.5 7B fine-tuned with LoRA", "Qwen3-4B-Instruct-2507 for maze training"],
        )

        assert reason is CullReason.UNADDRESSED_QUALIFIER
