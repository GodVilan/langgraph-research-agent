"""The one string-matching policy, stated once and asserted once.

Word boundaries were solved in ``absence._mentions`` after `Gram` matched inside "n-gram",
and the fix did not propagate: ``false_premises`` was still matching substrings months
later, passing all 93 items while `cifar-10` matched inside `cifar-100`. That is not a
check that could not fire — it is a **solved problem that failed to travel**, and the only
defence is one implementation everything shares rather than a convention each site
re-derives.

TWO MODES, AND WHY THEY DIFFER. Auditing the call sites showed the two existing
implementations were not simply one right and one wrong. They serve different questions:

* ``mentions_term`` — "does this text discuss this concept?" A paper writing "minibatches"
  *does* report batch size, and "ablations" *is* an ablation. Inflected and plural forms
  must match, so a trailing alphabetic character is allowed.

* ``mentions_name`` — "does this text name this specific thing?" `CIFAR-10` and `CIFAR-100`
  are different datasets, `A100` and `A1000` different hardware. Nothing may follow.

THE RULE THAT SEPARATES THEM. What follows the match decides:

    a trailing LETTER is inflection      -> allowed in term mode  ("minibatch" + "es")
    a trailing DIGIT is a different name -> forbidden in both     ("cifar-10" + "0")

So term mode is not "no right boundary" — the earlier implementation had none at all, which
is why it matched `cifar-10` inside `cifar-100`. It forbids a trailing digit and permits a
trailing letter. Both modes forbid a preceding alphanumeric.

Getting this wrong is dangerous in both directions, which is why the modes are named rather
than chosen per call site by whoever wrote it:

* over-matching in an absence check certifies a term as present that is not, discarding a
  valid item — recoverable, and visible as a shortfall;
* under-matching certifies a term absent that the corpus discusses, which produces a
  **backwards item**: the expected answer is a refusal, the corpus holds the answer, and a
  correct response is scored as a hallucination.
"""

from __future__ import annotations

import re
from functools import lru_cache

# A trailing digit always means a different name; a trailing letter may be inflection.
_TERM_SUFFIX = r"(?![0-9])"
_NAME_SUFFIX = r"(?![A-Za-z0-9])"

# The prefix differs too, and for the same danger-asymmetry reason.
#
# A hyphen before the match means the token is part of a compound: "gram" inside "n-gram",
# "batch" inside "mini-batch". Whether that should match depends on which way being wrong
# is worse.
#
#   term mode  — permits it. Over-matching says "the corpus discusses this" when it may
#                not, which *rejects* an unanswerable item. Recoverable, and visible as a
#                shortfall. Under-matching produces a backwards item, which is not.
#   name mode  — forbids it. A question naming `Gram` must not be satisfied by a paper
#                writing "n-gram": over-matching here lets a false premise through, which is
#                the failure the premise check exists to prevent.
#
# So the modes are not strict-versus-lax. Each errs in the direction that is safe for the
# question it answers.
_TERM_PREFIX = r"(?<![A-Za-z0-9])"
_NAME_PREFIX = r"(?<![A-Za-z0-9-])"


@lru_cache(maxsize=4096)
def _compiled(needle: str, prefix: str, suffix: str) -> re.Pattern[str]:
    return re.compile(prefix + re.escape(needle) + suffix, re.IGNORECASE)


def mentions_term(text: str, term: str) -> bool:
    """Does ``text`` discuss ``term`` as a concept? Inflections match, digit suffixes do not.

    "minibatches" matches "minibatch"; "cifar-100" does not match "cifar-10". A hyphenated
    compound *does* match ("n-gram" contains "gram") — deliberately, because over-matching
    only costs an item while under-matching produces a backwards one.
    """
    return _compiled(term, _TERM_PREFIX, _TERM_SUFFIX).search(text) is not None


def mentions_name(text: str, name: str) -> bool:
    """Does ``text`` name exactly ``name``? Nothing may follow.

    For datasets, models, benchmarks and systems, where `CIFAR-10` and `CIFAR-100` are
    different things and a near-match is a wrong match. A hyphenated compound does not
    match: a question naming `Gram` is not satisfied by a paper writing "n-gram".
    """
    return _compiled(name, _NAME_PREFIX, _NAME_SUFFIX).search(text) is not None


def first_term_present(text: str, terms: tuple[str, ...]) -> str | None:
    """The first of ``terms`` that ``text`` discusses, or None.

    Entries already written as regex (they start with ``\\b``) are honoured as regex, which
    is how a few paraphrase sets express alternations.
    """
    for term in terms:
        if term.startswith("\\b"):
            if re.search(term, text, re.IGNORECASE):
                return term
        elif mentions_term(text, term):
            return term
    return None
