"""The string-matching policy, asserted once for everything that matches strings.

Word boundaries were solved in `absence._mentions` after `Gram` matched inside "n-gram" —
and the fix did not travel. `false_premises` was still matching substrings months later,
passing all 93 items while `cifar-10` matched inside `cifar-100`. That is not a check that
could not fire; it is a **solved problem that failed to propagate**, and two correct
implementations in two places is a coincidence rather than a policy.

So there is one implementation, two named modes, and this file is the statement of the rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.matching import first_term_present, mentions_name, mentions_term


class TestTermMode:
    """ "Does this text discuss this concept?" — inflections match, digit suffixes do not."""

    @pytest.mark.parametrize(
        ("text", "term"),
        [
            ("we use minibatches of 32", "minibatch"),
            ("we run ablations on each component", "ablation"),
            ("the ablated variant", "ablate"),
            ("warmup steps", "warmup"),
        ],
    )
    def test_inflections_and_plurals_match(self, text: str, term: str) -> None:
        """A paper writing "minibatches" does report batch size."""
        assert mentions_term(text, term)

    @pytest.mark.parametrize(
        ("text", "term"),
        [("cifar-100 results", "cifar-10"), ("we used a1000 gpus", "a100"), ("gpt-40", "gpt-4")],
    )
    def test_a_digit_suffix_is_a_different_thing(self, text: str, term: str) -> None:
        """CIFAR-10 and CIFAR-100 are different datasets, whatever the prefix suggests."""
        assert not mentions_term(text, term)

    def test_a_hyphenated_compound_matches_deliberately(self) -> None:
        """Over-matching costs an item; under-matching produces a backwards one.

        In an absence check, saying "the corpus discusses this" when it may not merely
        rejects the item. Saying it does not when it does creates an item whose expected
        answer is a refusal while the corpus holds the answer.
        """
        assert mentions_term("n-gram overlap", "gram")

    def test_a_preceding_letter_still_blocks(self) -> None:
        assert not mentions_term("programgram", "gram")


class TestNameMode:
    """ "Does this text name exactly this thing?" — nothing may follow, no compounds."""

    def test_an_exact_name_matches(self) -> None:
        assert mentions_name("evaluated on cifar-10 we report", "cifar-10")

    @pytest.mark.parametrize(
        ("text", "name"),
        [
            ("cifar-100 results", "cifar-10"),
            ("we used a1000", "a100"),
            ("n-gram overlap", "gram"),
            ("we use minibatches", "minibatch"),
        ],
    )
    def test_near_matches_are_rejected(self, text: str, name: str) -> None:
        """A question naming Gram is not satisfied by a paper writing "n-gram"."""
        assert not mentions_name(text, name)

    def test_the_two_modes_genuinely_differ(self) -> None:
        """If they agreed everywhere, one of them would be unnecessary."""
        assert mentions_term("n-gram", "gram")
        assert not mentions_name("n-gram", "gram")
        assert mentions_term("ablations", "ablation")
        assert not mentions_name("ablations", "ablation")


class TestEveryMatchingSiteUsesThePolicy:
    """The propagation failure, prevented structurally rather than by attention."""

    def test_absence_and_premises_share_one_implementation(self) -> None:
        import inspect

        from evals import absence, verify_items

        assert "first_term_present" in inspect.getsource(absence._mentions_any)
        assert verify_items._names is mentions_name

    def test_first_term_present_reports_which_form_matched(self) -> None:
        """A cull that cannot say which surface form fired cannot be acted on."""
        assert first_term_present("we use minibatches", ("batch size", "minibatch")) == "minibatch"
        assert first_term_present("nothing relevant here", ("batch size", "minibatch")) is None

    def test_regex_entries_are_honoured(self) -> None:
        assert first_term_present("the ctr metric", (r"\bctr\b",)) == r"\bctr\b"
