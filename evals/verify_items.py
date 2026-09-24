"""Item-level checks that do not use embeddings.

Three defects found in human review, all of which let an item through that could not be
graded:

**Gold chunks were embedding-selected, not answer-verified.** ``best_chunk_for`` picked the
chunk sharing most content words with the *question*. If gold chunks are chosen by
similarity to the question, then Recall@k and MRR@k measure whether the dense retriever
retrieves what the dense retriever picked — near 1.0 by construction, and precisely the
artifact the eval exists to avoid. A gold chunk now has to contain the fact the gold answer
states, checked by literal grounding rather than similarity.

**Questions carried false premises.** An item asked what two papers report on CIFAR-10 when
one of them never used it, and passed the necessity check, because "can these papers answer
this?" is not the same question as "is this question true of these papers?".

**Questions stated their own answers.** A stem containing its answer grades every system as
correct regardless of retrieval.

Nothing here uses the embedding model. A check built from the component under test cannot
find that component's failures.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from evals.matching import mentions_name

# Figures are the strongest grounding signal: if the gold answer says 23.4, the passage that
# supports it contains 23.4. Matching on words alone would accept a chunk that discusses the
# right topic and states a different number.
NUMBER = re.compile(r"\d+(?:\.\d+)?")

# arXiv identifiers are digits and are not figures. A multi-hop answer naming its sources
# ("Paper 2605.29913 reports a slot duration of 0.1 s") would otherwise be checked for
# containing "2605.29913" as though it were a reported quantity.
ARXIV_ID = re.compile(r"\b\d{4}\.\d{4,5}\b")

# Names a question can assert: ALLCAPS acronyms, CamelCase methods, and alphanumerics with a
# digit (CIFAR-10, GPT-4, A100). Ordinary capitalised words are excluded — sentence-initial
# capitals would otherwise flood the premise check with false positives.
ENTITY = re.compile(r"\b(?:[A-Z]{2,}[-\w]*|[A-Z][a-z]+[A-Z]\w*|[A-Za-z]+-?\d+[\w-]*)\b")

# THE CRITERION, not a list of things that happened to cause false positives.
#
#   A NAMED ENTITY is a proper noun for a *specific* dataset, model, benchmark, or system —
#   something that could be swapped for a different one and change what the question means.
#   CIFAR-10, ImageNet, GSM8K, Qwen2, A100 as a specific accelerator.
#
#   GENERIC TECHNICAL VOCABULARY is not. RAM, GPU, FLOPS, AUC, SGD name a *category* of
#   thing. A paper reporting "256GB system memory" has answered a question about RAM, and
#   demanding the token verbatim penalises the paraphrase the construction rules require.
#
# The list below is that criterion applied, not the criterion itself. It was previously
# grown one entry at a time — GPU, CPU, FLOPS, AUC, RAM each added after causing a false
# positive — which encodes the ones that happened to be hit and leaves TPU, MACs, IoU, BLEU
# and F1 behaving identically and untested. Same shape as the hand-written `by_reason` chain
# that only knew the reasons existing when it was written.
#
# `make audit-entities` runs the premise check across every item and prints every entity it
# would assert, so a borderline term surfaces as a list to review rather than as one
# rejection at a time.
GENERIC_VOCABULARY = frozenset(
    {
        "ADAM",
        "ADAMW",
        "AI",
        "AND",
        "API",
        "AUC",
        "AUPR",
        "AUPRC",
        "AUROC",
        "BLEU",
        "CI",
        "CNN",
        "CPU",
        "CSV",
        "CV",
        "DRAM",
        "EM",
        "F1",
        "FLOP",
        "FLOPS",
        "FNR",
        "FOR",
        "FPR",
        "FPS",
        "GAN",
        "GB",
        "GPGPU",
        "GPU",
        "GRPC",
        "GRU",
        "HBM",
        "HDD",
        "HOW",
        "HTTP",
        "HTTPS",
        "IID",
        "IOU",
        "JSON",
        "KB",
        "KL",
        "LBFGS",
        "LLM",
        "LLMS",
        "LSTM",
        "MAC",
        "MACS",
        "MAE",
        "MAP",
        "MB",
        "MCMC",
        "METEOR",
        "ML",
        "MLE",
        "MLP",
        "MRR",
        "MS",
        "MSE",
        "NDCG",
        "NLP",
        "NPU",
        "OOD",
        "PAPER",
        "PAPERS",
        "PB",
        "PCA",
        "PPL",
        "PSNR",
        "QPS",
        "RAM",
        "REST",
        "RL",
        "RMSE",
        "RNN",
        "ROC",
        "ROM",
        "ROUGE",
        "SEM",
        "SGD",
        "SOTA",
        "SQL",
        "SSD",
        "SSIM",
        "SSL",
        "STD",
        "SVD",
        "TB",
        "THE",
        "TNR",
        "TOP-1",
        "TOP-5",
        "TPR",
        "TPU",
        "TWO",
        "VAE",
        "VLM",
        "VRAM",
        "WHAT",
        "WHICH",
        "WITH",
        "YAML",
    }
)

# Kept as the historical name; every reference goes through the criterion above.
STOP_ENTITIES = GENERIC_VOCABULARY


# Thousands separators. "2,819 traces (19,311 steps)" was yielding {"2", "819", "19", "311"}
# — four figures, none of which the paper reports — so items were rejected for not containing
# numbers that were never claimed. Commas between digit groups are removed before extraction.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def numbers_in(text: str) -> set[str]:
    """Figures a passage would have to contain, excluding arXiv identifiers.

    Thousands separators are normalised away, so "183,098" is one figure and not two.
    """
    return set(NUMBER.findall(_THOUSANDS.sub("", ARXIV_ID.sub(" ", text))))


def content_tokens(text: str) -> set[str]:
    """Checkable words in an answer, down to three characters.

    Hyphenated names split, so "Swin-Tiny" contributes both halves rather than nothing.
    """
    return {w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)}


def entities_in(text: str) -> set[str]:
    """Named entities a question asserts, case-folded to one canonical form.

    Folding matters: "top-1" and "Top-1" were counted as two distinct entities by
    `make audit-entities`, so excluding one would have left the other asserting silently.
    """
    seen: dict[str, str] = {}
    for candidate in ENTITY.findall(text):
        if candidate.upper() in GENERIC_VOCABULARY or len(candidate) <= 2:
            continue
        seen.setdefault(candidate.upper(), candidate)
    return set(seen.values())


# ── Claims, and why digits alone were not enough ─────────────────────────────────────────
#
# Containment was verified on figures. Multi-hop answers are frequently qualitative, and the
# figures they do carry are often small integers, so the check passed vacuously:
#
#   the figure "2"  appears in 3,772 of 5,401 chunks (70%)
#   the figure "16" appears in   821 of 5,401 chunks (15%)
#   the figure "23.4" appears in     6 of 5,401 chunks (0.1%)
#
# `mh-006` asserted a "lightweight 2D U-Net" against four chunks that were *entirely
# bibliography*, and passed because its only figure was `2`. Worse, the selector ranked
# candidates by how many of the answer's figures they contained, so a reference list — dense
# in bracketed numerals — was preferentially chosen. The check validated on digits and the
# selector optimised for digits, so the two failures reinforced each other.
#
# A claim is therefore a quoted span, a named entity, or a *discriminating* figure. And some
# chunks cannot support a claim about a paper's method at all, whatever they contain.

QUOTED = re.compile(r'"([^"]{3,80})"')

# Parameter and data scales — "405B", "1B", "7.65B", "8K". These are the single most common
# shape of claim in a model-comparison answer and neither ENTITY (which needs letters before
# the digits) nor NUMBER (which yields a bare "405") captured them, so an answer contrasting
# "8B, 70B, and 405B checkpoints" against "Llama 3.2 1B" asserted *nothing checkable* about
# the second paper. Its gold then passed on the figure `3.2`, which is the model's version.
SCALE = re.compile(r"\b\d+(?:\.\d+)?[BMK]\b")

# Reference lists, acknowledgements and prompt templates. These are structurally incapable of
# supporting an architecture or results claim, so they are ineligible as gold regardless of
# which tokens they happen to contain.
_BIBLIO = re.compile(r"\[\d{1,3}\]")

# WHAT FOLLOWS THE MARKER, not how many markers there are.
#
# Counting them flagged 504 chunks as reference lists and 121 more as citation lists — 11.6% of
# the corpus — and the false positives were the most fact-dense chunks a paper has. Three that
# a human named as correct gold, all structurally barred from ever being selected:
#
#   29628_0021  "reduce the dimensionality from 1024 to 100"     35 mid-sentence citations
#   30232_0100  "Language models used as starting checkpoints"    4 markers in a Citation column
#   29601_0047  "Claude Haiku 4.5 costs 19.746 per 1,000"         8 pricing URLs
#
# A reference entry reads `[8] B. McMahan, E. Moore, …` — the marker is followed by a capital
# beginning an author name. Prose reads `WSAC [14], which is`, and a license table reads
# `Apache 2.0 [30] huggingface.co/…`. That share is sharply bimodal here: 318 chunks at 0.0,
# 218 at 1.0, and 36 in between, so the cut sits in an empty trough rather than on judgement.
#
# THE URL RULE IS GONE, and measurement is why. It was retained one round longer than the
# bracket rule on the grounds that no observed item had been harmed by it; sp-009 then lost the
# only chunk stating 19.746 to it. Marker density cannot separate the two classes either — a
# real bibliography runs 1.23 URL markers per 100 words and that results paragraph runs 1.43 —
# so there is no threshold to retune, and a rule that cannot be made to discriminate is deleted
# rather than tuned. The cost: a bibliography keyed `[CJS12]` rather than `[8]` is no longer
# excluded. It would still have to support one of the answer's claims to be selected.
_BIBLIO_ENTRY = re.compile(r"\[\d{1,3}\]\s+(?=[A-Z])")
MIN_BIBLIO_MARKERS = 3
MIN_ENTRY_INITIAL_SHARE = 0.5

_TAGGY = re.compile(r"</?[A-Za-z_]{2,20}>")


def is_ineligible_gold(text: str) -> str | None:
    """Why this chunk cannot be a gold chunk, or None if it can.

    Structural ineligibility, not a scoring penalty: a bibliography does not support an
    architecture claim at any similarity, and four of five rejected multi-hop items had
    reference lists or an XML prompt template as gold.
    """
    flat = " ".join(text.split())
    markers = _BIBLIO.findall(flat)
    if len(markers) >= MIN_BIBLIO_MARKERS:
        share = len(_BIBLIO_ENTRY.findall(flat)) / len(markers)
        if share >= MIN_ENTRY_INITIAL_SHARE:
            return "reference list"
    if len(_TAGGY.findall(text)) >= 4:
        return "prompt or markup template"
    lowered = text.lower()
    if "acknowledg" in lowered and len(text) < 1500:
        return "acknowledgements"
    return None


# ── How rare a figure has to be to count as evidence ─────────────────────────────────────
#
# "A decimal, or three or more digits" was a *guess* at rarity, and it is wrong by an order of
# magnitude on this corpus. Measured document frequency over all 5,401 chunks:
#
#     100    581 chunks (10.8%)      405     38 chunks (0.7%)
#     3.2    375 chunks ( 6.9%)      768     28 chunks (0.5%)
#     2.5    352 chunks ( 6.5%)      23.4     6 chunks (0.1%)
#     0.1    240 chunks ( 4.4%)      183098   0 chunks (0.0%)
#
# So `100` and `0.1` were admitted as claims on the same footing as `183,098`, and three
# multi-hop items were grounded on them: `0.1` matched Algorithm 1 pseudocode containing
# neither quantity the answer states, `2.5` came from the model name "Qwen 2.5 7B", and `3.2`
# came from "Llama 3.2" and matched a *section number* in the other paper. Same defect as the
# figure `2` appearing in 70% of chunks, one notch less obvious.
#
# The threshold is derived, not chosen. A pruned gold set holds at most three chunks, so a
# figure is evidence only if the chance that *any* of them contains it coincidentally stays
# under 5%:
#
#     1 - (1 - p)**3 < 0.05   ->   p < 0.017   ->   df <= 91 of 5,401
#
# ``GOLD_SET_CEILING`` is the premise, not a preference, and ``evals/verify_dataset.py``
# asserts it against the artifact — a set whose gold grows past three invalidates this
# threshold, and that must fail a gate rather than quietly widen the coincidence budget.
#
# WHAT THIS COSTS, STATED PLAINLY. It is a *loosening* of claim extraction — 9 of the 46
# figures the set currently treats as claims stop being claims (100, 3.2, 2.5, 1000, 256, 0.6,
# 0.2, 0.1, 0.001), and `0.001` is a plausible learning rate rather than a commonplace. Fewer
# claims means easier attribution, which is the dangerous direction for an eval set. Three
# things offset it: parameter scales become claims (`SCALE` above, a tightening), attribution
# is now clause-scoped rather than pooled across the whole answer
# (``papers_contributing_nothing``, a tightening), and a clause left with no groundable value
# is a rejection rather than a free pass (``degenerate_clauses``).
COINCIDENCE_BUDGET = 0.05
GOLD_SET_CEILING = 3


def max_document_frequency(n_chunks: int) -> int:
    """The most chunks a figure may appear in and still count as evidence."""
    per_chunk = 1.0 - (1.0 - COINCIDENCE_BUDGET) ** (1.0 / GOLD_SET_CEILING)
    return int(per_chunk * n_chunks)


def bare_figures(text: str) -> set[str]:
    """Figures with scale tokens removed — "405B" is a scale claim, not the figure 405.

    One extractor for both the frequency table and the claim set. Counting them differently
    is how a ratio comes to describe two populations (D-021).
    """
    return numbers_in(SCALE.sub(" ", text))


@lru_cache(maxsize=1)
def figure_frequency() -> tuple[dict[str, int], int]:
    """Document frequency of every figure in the corpus, and the chunk count."""
    from evals.absence import load_corpus

    corpus = load_corpus()
    counts: Counter[str] = Counter()
    for chunk in corpus.chunks:
        counts.update(bare_figures(str(chunk["text"])))
    return dict(counts), corpus.n_chunks


def discriminating_figures(text: str) -> set[str]:
    """Figures rare enough to constitute a claim, excluding digits inside names.

    Digits that are part of a named entity are not figures at all. "Qwen3-4B-Instruct-2507"
    and "CICIDS2017" yielded 2507 and 2017 as reported quantities, so items were rejected
    for gold chunks not containing a model's version string. A version is part of the name,
    and the name is already checked as a name. Rarity handles the spaced form of the same
    thing ("Llama 3.2"), which no name pattern caught.
    """
    names = " ".join(entities_in(text)).lower()
    frequency, n_chunks = figure_frequency()
    ceiling = max_document_frequency(n_chunks)
    return {f for f in bare_figures(text) if f not in names and frequency.get(f, 0) <= ceiling}


def scales_in(text: str) -> set[str]:
    """Parameter and data scales the text states — "405B", "1B", "7.65B"."""
    return set(SCALE.findall(text))


def claims_in(gold_answer: str) -> dict[str, set[str]]:
    """What the answer asserts, by kind. Every claim must be attributable to a gold chunk."""
    return {
        "quoted": {q.strip() for q in QUOTED.findall(gold_answer) if len(q.strip()) >= 3},
        "entities": entities_in(gold_answer),
        "figures": discriminating_figures(gold_answer),
        "scales": scales_in(gold_answer),
    }


# ONE support test, four callers. `unattributable_claims`, `papers_contributing_nothing`,
# `unsupported_chunks` and the builder's selector each carried their own copy of "does this
# text support this claim", and adding a claim kind meant editing four places — the same shape
# as the four divergent check chains (D-024). They now all go through this.
def claims_supported_by(text: str, claims: dict[str, set[str]]) -> dict[str, set[str]]:
    """Which of ``claims`` this passage supports, by kind.

    Quoted spans match as substrings after whitespace normalisation, because the answer is
    quoting the paper. Entities and scales take name-mode matching, so `1B` does not match
    inside `21B` and CIFAR-10 does not match inside CIFAR-100. Figures match exactly.
    """
    flat = " ".join(text.split())
    lowered = flat.lower()
    return {
        "quoted": {q for q in claims["quoted"] if q.lower() in lowered},
        "entities": {e for e in claims["entities"] if mentions_name(lowered, e.lower())},
        "figures": claims["figures"] & discriminating_figures(flat),
        "scales": {s for s in claims["scales"] if mentions_name(lowered, s.lower())},
    }


def supports_any_claim(text: str, claims: dict[str, set[str]]) -> bool:
    """Whether a passage carries at least one of the answer's claims."""
    return any(claims_supported_by(text, claims).values())


def unattributable_claims(gold_answer: str, chunk_texts: list[str]) -> dict[str, list[str]]:
    """Claims no gold chunk supports, grouped by kind."""
    claims = claims_in(gold_answer)
    covered = claims_supported_by(" ".join(chunk_texts), claims)
    return {
        kind: sorted(values - covered[kind])
        for kind, values in claims.items()
        if values - covered[kind]
    }


def gold_chunks_support_jointly(gold_answer: str, chunk_texts: list[str]) -> tuple[bool, str]:
    """Whether the gold chunks *together* support every claim the answer makes.

    Claims, not digits. The figure-only version passed four of five rejected multi-hop items
    vacuously — one against four bibliography chunks, on the strength of the numeral "2".

    Support is a property of the gold *set*: a multi-hop answer cites facts from several
    papers, so requiring one chunk to carry all of them is a test no correct item can pass.
    What each chunk must do individually is contribute, which ``unsupported_chunks`` checks.
    """
    ineligible = [
        reason for text in chunk_texts if (reason := is_ineligible_gold(text)) is not None
    ]
    if ineligible and len(ineligible) == len(chunk_texts):
        return False, f"every gold chunk is structurally ineligible: {sorted(set(ineligible))}"

    eligible = [t for t in chunk_texts if is_ineligible_gold(t) is None]
    if not eligible:
        return False, "no eligible gold chunk"

    claims = claims_in(gold_answer)
    total = sum(len(v) for v in claims.values())
    if not total:
        # No quoted span, no named entity, no discriminating figure — but that does not make
        # the answer uncheckable. Short factual answers are the most precise a question can
        # have: "up to 54%", "75 years old", "agency-sa-goal", "five". Hard-failing them
        # rejected 14 of 50 factual items that a human had verified clean, which is the
        # tightening rule's exact failure mode — the strengthening was aimed at multi-hop
        # answers passing vacuously on the numeral "2", and it caught short answers instead.
        #
        # So the token fallback survives, and says it is weak. A weak pass is reported as
        # weak rather than promoted to a strong one or discarded.
        # A figure too common to be *evidence* is not thereby unstated. Four factual answers
        # are a bare number and nothing else — "0.001", "0.07", "1,000", "0.6/0.2/0.2" — and
        # once rarity decided what counts as a claim, all four had no claim and no word either,
        # so they failed as "nothing checkable at all". They are the most precise answers a
        # factual question can have, and this is the second time a strengthening aimed at
        # vacuous multi-hop support landed on short factual answers instead. Containment on a
        # common figure is weak evidence, not absent evidence, so it is required and labelled.
        loose = bare_figures(gold_answer)
        if loose:
            joined = " ".join(" ".join(t.split()) for t in eligible)
            present_figures = bare_figures(joined)
            absent = sorted(loose - present_figures)
            if absent:
                return False, f"the gold set lacks the figures the answer states: {absent[:4]}"
            return True, (
                f"WEAK: every figure the answer states appears in the gold set, but none is "
                f"rare enough in the corpus to be evidence on its own ({sorted(loose)[:4]})"
            )

        tokens = content_tokens(gold_answer)
        if not tokens:
            return False, "gold answer carries nothing checkable at all"
        joined = " ".join(" ".join(t.split()) for t in eligible).lower()
        share = len({t for t in tokens if t in joined}) / len(tokens)
        if share < 0.5:
            return False, f"only {share:.0%} of the answer's terms appear in the gold set"
        return True, (
            f"WEAK: no quote, entity or precise figure; {share:.0%} of the answer's terms "
            f"appear in the gold set"
        )

    missing = unattributable_claims(gold_answer, eligible)
    if missing:
        detail = "; ".join(f"{kind}: {vals}" for kind, vals in sorted(missing.items()))
        return False, f"no gold chunk supports {detail}"

    kinds = ", ".join(f"{len(v)} {k}" for k, v in sorted(claims.items()) if v)
    note = f" ({len(ineligible)} ineligible chunk(s) excluded)" if ineligible else ""
    return True, f"the gold set supports every claim ({kinds}){note}"


# ── Clauses: which half of the answer is about which paper ────────────────────────────────
#
# A multi-hop answer is two clauses joined by a contrast — "Paper A reports X, whereas Paper B
# reports Y" — and every check so far pooled their claims into one set. That pooling hid two
# defects at once in `mh-004`, whose answer reads:
#
#   "Paper 2605.29628 reports embedding dimensions of 1024 or 768 for the CLAP models, whereas
#    Paper 2605.30120 reports a hidden dimension h of the sparse autoencoder."
#
# The second clause states no value at all — it names the variable. And 30120's gold chunks
# were credited as contributing, because they happen to contain 1024 and 768, which are the
# *first* clause's figures. Pooled attribution cannot tell "this paper supports its own claim"
# from "this paper's chunks contain the other paper's numbers".
_PAPER_MENTION = re.compile(r"\b\d{4}\.\d{4,5}\b")


def clauses_by_paper(gold_answer: str, source_paper_ids: list[str]) -> dict[str, str]:
    """The span of the answer that speaks about each source paper.

    Split at paper-id mentions: each clause runs from the mention of its paper to the next
    mention or the end.

    **Empty when the answer names no paper by id**, and that is a declination, not a finding.
    `mh-003` refers to its papers descriptively — "the hematological malignancy cytology study
    … whereas the pancreatic cancer screening study" — and a first version of this read that as
    "the answer never mentions this paper" and rejected a sound item. An answer written that
    way has no clause boundary a string can find, so the clause-scoped checks decline; the
    coverage gap is recorded in docs/EVALS.md rather than paid for in false rejections.
    """
    wanted = set(source_paper_ids)
    marks = [
        (m.start(), m.group()) for m in _PAPER_MENTION.finditer(gold_answer) if m.group() in wanted
    ]
    if not marks:
        return {}
    clauses = {pid: "" for pid in source_paper_ids}
    for index, (start, pid) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(gold_answer)
        clauses[pid] += gold_answer[start:end]
    return clauses


def degenerate_clauses(gold_answer: str, source_paper_ids: list[str]) -> dict[str, str]:
    """Source papers whose clause of the answer asserts nothing checkable.

    ``mh-004``'s second clause names a quantity by its symbol and gives no value, so the
    question ("what specific embedding dimension values…") has no answer for that paper while
    reading as though it does. A judge grading against it would mark any value correct.

    Only applies to answers that name more than one paper: a single-paper answer has no clause
    structure, and its emptiness is already caught by ``gold_chunks_support_jointly``.

    Two outcomes, kept distinct because collapsing them *is* the mislabel defect: a clause
    carrying no quantity at all is degenerate, and one carrying quantities too common to ground
    is unverifiable. `mh-001` states -90 dBm and 0.1 seconds — real values that no containment
    check can confirm, since 0.1 appears in 240 of 5,401 chunks. Reporting that as "states no
    value" would name the wrong defect to whoever reads the report.
    """
    if len(source_paper_ids) < 2:
        return {}
    clauses = clauses_by_paper(gold_answer, source_paper_ids)
    if not clauses:
        return {}
    findings: dict[str, str] = {}
    for pid, text in clauses.items():
        if not text:
            findings[pid] = "the answer never mentions this paper"
        elif not any(claims_in(text).values()):
            loose = sorted(bare_figures(text))
            findings[pid] = (
                f"the clause states only values too common in the corpus to ground: {loose}"
                if loose
                else "the clause about this paper states no figure, scale, name or quotation"
            )
    return findings


# ── The dropped qualifier ─────────────────────────────────────────────────────────────────
#
# `mh-002` asks what the other paper uses "for its primary maze-trained setup". The answer says
# "as its primary model" — the qualifier is gone — and the retained gold is a figure caption
# about Emotion PC1 steering that grounds neither "maze-trained" nor "primary". Every check
# passed, and the reason is structural: **claims are extracted from the answer.** When the
# answer drops the question's discriminating qualifier, the selector optimises for a claim set
# that cannot ground it, and containment then confirms the entity's *name* while nothing
# checks the *role* the question assigned it.
#
# The qualifier has to come from the question, and the check has to run against the gold.
# What counts as a qualifier is a shape, not a list: a hyphenated compound in the question —
# "maze-trained", "four-phase", "zero-shot", "multi-vector" — is a specific descriptor the
# question uses to pick out one setup among several, and in every case human review has
# rejected on this ground it was one of these. Matching is inflection-tolerant across the
# hyphen ("maze-trained", "maze training", "mazetrained" all satisfy "maze-trained").
QUALIFIER = re.compile(r"\b[A-Za-z]{3,}-[A-Za-z]{3,}\b")
_QUESTION_TURN = re.compile(r",\s*(?:and|but)\s+(?:what|which|how)\b|;", re.IGNORECASE)


def question_clauses(question: str) -> list[str]:
    """The halves of a comparative question, split where it turns to the other paper."""
    parts = [p.strip() for p in _QUESTION_TURN.split(question) if p and p.strip()]
    return parts or [question]


def qualifiers_in(text: str) -> set[str]:
    return {q.lower() for q in QUALIFIER.findall(text)}


_INFLECTION = re.compile(r"(ing|ed|es|s)$")


def _qualifier_pattern(qualifier: str) -> re.Pattern[str]:
    """ "maze-trained" matches "maze training"; "fine-tuning" matches "fine-tuned".

    The tail is stemmed of its inflection, because the question and the paper conjugate
    independently — a first version flagged `fine-tuning` against gold reading "when
    fine-tuned using LoRA", which is the same act in a different tense.
    """
    head, tail = qualifier.split("-", 1)
    stemmed = _INFLECTION.sub("", tail)
    stem = stemmed if len(stemmed) >= 3 else tail
    return re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(head)}[-\s]?{re.escape(stem)}[a-z]*", re.IGNORECASE
    )


def unaddressed_qualifiers(
    question: str,
    gold_answer: str,
    gold_by_paper: dict[str, list[str]],
    source_paper_ids: list[str],
) -> dict[str, list[str]]:
    """Qualifiers the question puts on each paper that the paper's gold does not contain.

    The question's clauses are paired with the answer's paper clauses in order — the answer
    of a "which of the two … and what does the other …" question names the papers in the
    order the question asks about them. A question with a single clause applies its
    qualifiers to every paper. Any other shape declines, and says nothing.

    Runs against *gold*, not the whole paper: "maze" appears throughout 30232, and the
    defect is that the chunks chosen to ground the answer are not the ones that say it.
    """
    if len(source_paper_ids) < 2:
        return {}
    clauses = clauses_by_paper(gold_answer, source_paper_ids)
    if not clauses or not all(clauses.values()):
        return {}
    order = sorted(source_paper_ids, key=lambda pid: gold_answer.find(pid))
    halves = question_clauses(question)
    if len(halves) == 1:
        pairing = {pid: halves[0] for pid in order}
    elif len(halves) == len(order):
        pairing = dict(zip(order, halves, strict=True))
    else:
        return {}

    missing: dict[str, list[str]] = {}
    for pid, half in pairing.items():
        gold = " ".join(" ".join(t.split()) for t in gold_by_paper.get(pid, []))
        absent = sorted(q for q in qualifiers_in(half) if not _qualifier_pattern(q).search(gold))
        if absent:
            missing[pid] = absent
    return missing


def papers_contributing_nothing(
    gold_answer: str,
    gold_by_paper: dict[str, list[str]],
    source_paper_ids: list[str] | None = None,
) -> list[str]:
    """Source papers whose gold chunks support none of *their own clause's* claims.

    Joint attribution across the whole gold set was the wrong property to check. A multi-hop
    answer of the form "Paper A reports X, while Paper B reports Y" was passing when A's gold
    chunks supported nothing at all and B's chunks happened to contain enough tokens to cover
    both halves — `mh-005` (A's gold was a bibliography and a derivation) and `mh-011` (B's
    gold was two RMSE tables) both survived claim-level checking that way.

    Requiring every source paper to contribute at least one supported claim is what
    "multi-hop" structurally means: if one paper contributes nothing, the item is not
    multi-hop and its gold for that paper is unfounded.

    Scoped to the clause, not the answer. Checking against the whole answer's claim pool let
    `mh-004`'s second paper pass on the first paper's figures — a paper cannot earn its place
    in a multi-hop item by containing numbers that belong to the other one.
    """
    # Iterate the papers the item *claims to need*, not the keys that happen to be present.
    # Keying off `gold_by_paper` made a source paper with zero gold chunks invisible rather
    # than barren — `mh-011` passed with gold from one of its two papers, because the paper
    # contributing nothing contributed no dictionary key either. Same defect shape as the
    # others in this file: the check ran over the wrong domain.
    required = list(source_paper_ids) if source_paper_ids else list(gold_by_paper)
    clauses = clauses_by_paper(gold_answer, required) if len(required) > 1 else {}

    barren: list[str] = []
    for paper_id in sorted(required):
        # The clause when there is one, the whole answer for a single-paper item.
        claims = claims_in(clauses.get(paper_id) or gold_answer)
        if not any(claims.values()):
            barren.append(paper_id)
            continue
        eligible = [t for t in gold_by_paper.get(paper_id, []) if is_ineligible_gold(t) is None]
        if not eligible or not supports_any_claim(" ".join(eligible), claims):
            barren.append(paper_id)
    return barren


def unsupported_chunks(gold_answer: str, chunks: dict[str, str]) -> list[str]:
    """Gold chunks supporting no claim the answer makes — carried for no reason.

    Claims, not figures. This measured figures while selection scored quoted spans, named
    entities and figures, so a chunk chosen because it contained "CLAP" was reported as
    contributing nothing. Third instance of the same wrong-property defect in this file
    (D-023, ninth): the check ran, passed, and asked about the wrong thing.
    """
    claims = claims_in(gold_answer)
    if not any(claims.values()):
        return []
    return sorted(cid for cid, text in chunks.items() if not supports_any_claim(text, claims))


def minimal_gold(
    gold_answer: str, chunks: dict[str, str], source_paper_ids: list[str]
) -> list[str]:
    """The smallest gold set that still supports every claim the full set supports.

    Selecting the top three chunks per paper grew multi-hop gold from 3-4 chunks to 5-6, and
    the extra ones are *interchangeable*: three chunks of paper 30085 each supported the same
    four benchmark names. ``unsupported_chunks`` cannot see that — every one of them supports
    something — so redundancy survives a check designed to catch dead weight.

    This matters to the metric, not just to tidiness. Recall@k divides by the size of the gold
    set, so three interchangeable chunks turn one retrieval into a 1-of-3 credit and dilute
    MRR@k in the retriever's favour. A gold set has to be the evidence, not everything that
    resembles it.

    Greedy set cover over the claims, deterministic in chunk id so a rebuild is reproducible.
    Every paper whose clause carries claims keeps at least one chunk, so minimising cannot
    silently drop a paper out of a multi-hop item.
    """
    claims = claims_in(gold_answer)
    if not any(claims.values()) or not chunks:
        return sorted(chunks)

    # An ineligible chunk is not a candidate. Ignoring this let the pruner keep a chunk the
    # containment check would then reject as structurally ineligible, turning a strong item
    # weak — a check and a selector disagreeing about what counts as gold, which is the same
    # split that made the selector prefer bibliographies in the first place.
    chunks = {cid: text for cid, text in chunks.items() if is_ineligible_gold(text) is None}
    if not chunks:
        return []

    covered_by = {cid: claims_supported_by(text, claims) for cid, text in chunks.items()}

    def flat(support: dict[str, set[str]]) -> set[tuple[str, str]]:
        return {(kind, value) for kind, values in support.items() for value in values}

    reachable = set().union(*(flat(s) for s in covered_by.values())) if covered_by else set()
    chosen: list[str] = []
    outstanding = set(reachable)
    while outstanding:
        best = max(
            sorted(cid for cid in chunks if cid not in chosen),
            key=lambda cid: len(flat(covered_by[cid]) & outstanding),
        )
        gained = flat(covered_by[best]) & outstanding
        if not gained:
            break
        chosen.append(best)
        outstanding -= gained

    # A paper contributing to the answer must keep a chunk, even when another paper's chunk
    # already covered the claim it shares.
    clauses = clauses_by_paper(gold_answer, source_paper_ids) if len(source_paper_ids) > 1 else {}
    for paper_id in source_paper_ids:
        clause_claims = claims_in(clauses.get(paper_id) or gold_answer)
        if not any(clause_claims.values()) or any(cid.startswith(paper_id) for cid in chosen):
            continue
        supporting = [
            cid
            for cid in sorted(chunks)
            if cid.startswith(paper_id) and supports_any_claim(chunks[cid], clause_claims)
        ]
        if supporting:
            chosen.append(supporting[0])
    return sorted(chosen)


# Named entities take the exact-name mode: CIFAR-10 and CIFAR-100 are different datasets.
# This was a substring check until the boundary policy was centralised — `absence` had
# already solved it and the fix had not propagated (evals/matching.py).
_names = mentions_name


def false_premises(question: str, paper_texts: list[str]) -> list[str]:
    """Entities the question asserts that appear in *none* of its anchored papers."""
    haystack = " ".join(paper_texts).lower()
    return sorted(e for e in entities_in(question) if not _names(haystack, e.lower()))


# "What do the two report on X" asserts X of both papers. "Which of the two uses X" asserts
# it of neither and asks. The first is false when only one paper uses X; the second is the
# question working as intended. The discriminator is the question's form.
UNIVERSAL_FORMS = re.compile(
    r"\b(both|the two papers|these two papers|each (?:paper|study)|their respective|"
    r"do the two|do these two)\b",
    re.IGNORECASE,
)
DISCRIMINATIVE_FORMS = re.compile(
    r"\b(which of the two|which paper|which study|which one)\b", re.IGNORECASE
)


def universal_premises(question: str, paper_texts: list[str]) -> list[str]:
    """Entities a *discriminative* question names that every anchored paper contains.

    "Which of the two papers evaluates on MNIST?" presupposes that exactly one does. When
    both do, the question is false — and none of the other premise checks sees it:
    ``false_premises`` requires absence from all, ``asymmetric_premises`` requires presence
    in *some but not all*, and ``premise_severity`` passes discriminative forms
    unconditionally because that form was the safe one against asymmetry.

    Presence in *all* papers is the third case, and it is the one `mh-005` occupied: its own
    answer says "Paper A reports using MNIST … while Paper B reports evaluating on MNIST".
    Grounding passes it, because both halves are individually true; the falsehood is in the
    question's presupposition, not in the answer.
    """
    if not DISCRIMINATIVE_FORMS.search(question) or len(paper_texts) < 2:
        return []
    lowered = [t.lower() for t in paper_texts]
    return sorted(
        entity
        for entity in entities_in(question)
        if all(mentions_name(text, entity.lower()) for text in lowered)
    )


# ── The conjunctive discriminator ─────────────────────────────────────────────────────────
#
# `mh-014` asks "which of the two studies evaluates on public network packet traces *while*
# using a four-phase transmission protocol over wireless channels". One paper has the packet
# traces; the other has the four-phase protocol; neither has both. Every premise check passes:
# `false_premises` finds each named thing in *some* paper, `universal_premises` needs an entity
# in *all* papers, `premise_severity` waves discriminative forms through, and grounding passes
# because each half of the answer is individually true. The falsehood is in the conjunction.
#
# The conditions here are descriptive phrases — "a four-phase transmission protocol" — not
# named entities, so no string check can verify them against a paper. What is detectable
# locally is the *shape*; verifying it needs a model call, which `evals/multihop.py` makes.
# Detected and unverified is not the same as passing, so the shape forces review.
CONJUNCTIVE_JOINERS = re.compile(
    r"\b(while|whilst|as well as|at the same time as|simultaneously with|together with)\b",
    re.IGNORECASE,
)
# The discriminator ends where the question turns to the other paper.
_DISCRIMINATOR_END = re.compile(r",\s*(?:and|but)\b|;", re.IGNORECASE)


def conjunctive_discriminator(question: str) -> str:
    """The discriminating clause of a question that asks for two properties at once.

    Empty when the question is not discriminative, or asks for a single property. A
    comparative question naming benchmarks from both papers ("which evaluates on MATH500, and
    what does the other use instead?") is *not* this shape — its second half is about the
    other paper, so the clause is cut at the turn.
    """
    match = DISCRIMINATIVE_FORMS.search(question)
    if not match:
        return ""
    tail = question[match.start() :]
    end = _DISCRIMINATOR_END.search(tail)
    clause = tail[: end.start()] if end else tail
    return clause.strip() if CONJUNCTIVE_JOINERS.search(clause) else ""


def conjunctive_premises(question: str, paper_texts: list[str]) -> list[str]:
    """Named entities a conjunctive discriminator requires of one paper that no paper has all of.

    The half of the shape that *is* checkable without a model: when the conjunctive clause
    names two or more entities, some single paper must contain every one of them. If the
    entities are split across papers, the question presupposes a conjunction nothing satisfies.
    """
    clause = conjunctive_discriminator(question)
    if not clause or len(paper_texts) < 2:
        return []
    required = entities_in(clause)
    if len(required) < 2:
        return []
    lowered = [t.lower() for t in paper_texts]
    if any(all(mentions_name(text, e.lower()) for e in required) for text in lowered):
        return []
    return sorted(required)


# ── The false contrast ────────────────────────────────────────────────────────────────────
#
# "What *distinct* noise power values and slot durations do the two papers report" asserts that
# the values differ. `mh-001`'s own answer gives 0.1 seconds for both papers, so the question
# is false of the pair — and it is false in a way no premise check could see, because every
# entity it names is present in both papers and the answer's halves are each true.
CONTRASTIVE_FORMS = re.compile(
    r"\b(distinct|differ|differs|different|differing|contrast|unlike|whereas)\b", re.IGNORECASE
)


def contrastive_premise(
    question: str, gold_answer: str, source_paper_ids: list[str]
) -> tuple[str, str]:
    """'false', 'review' or 'ok' for a question asserting that two papers' values differ.

    Compares the figures and scales each clause of the answer attributes to its own paper.
    Identical sets are a hard failure: the answer exhibits no difference at all, so whatever
    the question asserts is distinct, is not. A partial overlap is routed to a human with the
    shared value named — papers may legitimately agree on one quantity and differ on another,
    and `mh-001` is exactly that case, so the reject is earned by the degenerate one only.
    """
    if not CONTRASTIVE_FORMS.search(question) or len(source_paper_ids) < 2:
        return "ok", ""
    clauses = clauses_by_paper(gold_answer, source_paper_ids)
    # Every stated value, not only the groundable ones. Whether two papers report the *same*
    # number is a property of the answer's text, and corpus rarity has nothing to do with it —
    # filtering by it hid `mh-001`'s shared 0.1 s slot duration entirely.
    values = {pid: bare_figures(text) | scales_in(text) for pid, text in clauses.items() if text}
    if len(values) < 2:
        return "ok", ""
    sets = list(values.values())
    if all(s and s == sets[0] for s in sets):
        return "false", (
            f"the question asserts a difference, and every paper's clause states the same "
            f"values {sorted(sets[0])}"
        )
    shared = set.intersection(*sets) if all(sets) else set()
    if shared:
        return (
            "review",
            f"the papers' clauses share {sorted(shared)} while the question asserts a difference",
        )
    return "ok", ""


def premise_severity(question: str, asymmetric: dict[str, int]) -> str:
    """'false', 'review', or 'ok' for a question naming entities only some papers use.

    Heuristic, and routed to a human rather than trusted: the form tells you what the
    question asserts, but not reliably enough to reject an item on its own.
    """
    if not asymmetric:
        return "ok"
    if DISCRIMINATIVE_FORMS.search(question):
        return "ok"
    if UNIVERSAL_FORMS.search(question):
        return "false"
    return "review"


def asymmetric_premises(question: str, paper_texts: list[str]) -> dict[str, int]:
    """Entities present in some anchored papers but not all, with the count that have them.

    Checking the *union* is not enough, and that is the case human review actually hit: a
    question asking what both papers report on CIFAR-10 passed because one of them used it.
    A question of that shape is false of the pair even though the entity exists in the
    corpus, and the necessity check cannot see it — "can these papers answer this?" is a
    different question from "is this true of these papers?".

    Reported rather than auto-failed: a comparative question may legitimately name something
    only one paper uses ("which of the two evaluates on CIFAR-10?"). The distinction needs a
    human, so it is surfaced to one.
    """
    lowered = [t.lower() for t in paper_texts]
    counts: dict[str, int] = {}
    for entity in entities_in(question):
        present = sum(1 for t in lowered if _names(t, entity.lower()))
        if 0 < present < len(lowered):
            counts[entity] = present
    return counts


def answer_in_stem(question: str, gold_answer: str) -> tuple[bool, str]:
    """Whether the question already contains its own answer.

    Every figure the answer states appearing in the stem means the item grades a system
    correct without it retrieving anything.
    """
    answer_numbers = numbers_in(gold_answer)
    if answer_numbers:
        # When the answer states a figure, the figure *is* the answer and the surrounding
        # words are framing. "At what epoch does it stop?" / "epoch 25" shares the word
        # "epoch" and gives nothing away — deciding on word overlap flagged it as
        # self-answering while the only thing asked for, 25, was absent from the stem.
        if answer_numbers <= numbers_in(question):
            return True, f"the stem already states {sorted(answer_numbers)[:3]}"
        return False, ""

    words = {w.lower() for w in re.findall(r"[A-Za-z]{5,}", gold_answer)}
    if words and len(words & {w.lower() for w in re.findall(r"[A-Za-z]{5,}", question)}) == len(
        words
    ):
        return True, "the stem contains every distinctive word of the answer"
    return False, ""


# ── The one chain ────────────────────────────────────────────────────────────────────────
#
# Every builder and the independent verifier call this. It exists because the chain was
# reimplemented three times and diverged three times: multi-hop ran five checks, factual ran
# two and then four, and the behavioural matrix tested a fourth copy written in the test
# file — so the matrix passed while `build_factual` was missing `banned_phrasing` and
# `false_premises`, and could never have caught it.
#
# A test of a reimplementation proves the reimplementation works. The only way a matrix can
# speak for the builders is if the builders and the matrix run the same code.


def classify_candidate(
    *,
    question: str,
    answer: str,
    gold_texts: list[str],
    paper_texts: list[str],
    banned: list[str],
    leaked: bool,
    source_paper_ids: list[str] | None = None,
    gold_by_paper: dict[str, list[str]] | None = None,
) -> tuple[str, str] | None:
    """The single ordered check chain. Returns (reason_value, detail), or None if clean.

    Ordered cheapest-first: local string checks before anything that costs a model call, so
    a candidate a regex can reject never reaches the necessity checks.

    ``banned`` and ``leaked`` are passed in rather than computed here because they need the
    drafter's own gold passage and the caller already has it; everything else is derived.

    ``source_paper_ids`` and ``gold_by_paper`` enable the clause-scoped checks. They are
    optional because a single-paper item has no clause structure — not because a multi-hop
    caller may omit them. ``papers_contributing_nothing`` was written to catch two rejected
    items, described in its own docstring as the fix, and called from nowhere at all; it is in
    the chain now, which is the only place a check counts as running.
    """
    if banned:
        return "banned_phrasing", ", ".join(banned)
    if not answer.strip():
        return "no_gold_answer", "drafter returned no answer"

    stem_leak, stem_why = answer_in_stem(question, answer)
    if stem_leak:
        return "answer_in_stem", stem_why

    if paper_texts:
        absent = false_premises(question, paper_texts)
        if absent:
            return "false_premise", f"named but in no source paper: {absent}"
        universal = universal_premises(question, paper_texts)
        if universal:
            return (
                "false_premise",
                f"asks which of the papers uses {universal}, but all of them do",
            )
        split = conjunctive_premises(question, paper_texts)
        if split:
            return (
                "conjunctive_premise",
                f"asks which paper satisfies {split} together, and no single paper does",
            )
        asymmetric = asymmetric_premises(question, paper_texts)
        if premise_severity(question, asymmetric) == "false":
            return (
                "false_premise",
                f"asserts of all papers what only some report: {asymmetric}",
            )

    papers = list(source_paper_ids or [])
    if len(papers) > 1:
        degenerate = degenerate_clauses(answer, papers)
        if degenerate:
            detail = "; ".join(f"{pid}: {why}" for pid, why in sorted(degenerate.items()))
            return "degenerate_clause", detail
        severity, why = contrastive_premise(question, answer, papers)
        if severity == "false":
            return "false_premise", why

    if leaked:
        return "lexical_overlap", "question reproduces gold phrasing"

    supported, why = gold_chunks_support_jointly(answer, gold_texts)
    if not gold_texts or not supported:
        return "gold_unsupported", why

    if gold_by_paper is not None and len(papers) > 1:
        barren = papers_contributing_nothing(answer, gold_by_paper, papers)
        if barren:
            return (
                "paper_contributes_nothing",
                f"gold for {barren} supports none of that paper's own claims",
            )
        dropped = unaddressed_qualifiers(question, answer, gold_by_paper, papers)
        if dropped:
            detail = "; ".join(f"{pid}: {quals}" for pid, quals in sorted(dropped.items()))
            return (
                "unaddressed_qualifier",
                f"the question's qualifier is absent from that paper's gold — {detail}",
            )
    return None
