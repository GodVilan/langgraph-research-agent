"""Verify that an unanswerable item is actually unanswerable.

This module exists because the first attempt at the unanswerable stratum was wrong in a way
that would not have shown up until the numbers were published. See
``docs/EVALS.md`` — "the false-absence hazard".

Two rules, both learned the same way:

1. **Screen against the full text, never the abstracts.** 24 of 28 topics that look absent
   from the 150 abstracts are discussed in the body. Related-work sections name nearly
   every term in machine learning at least once.

2. **Screen against the whole corpus, never the anchor paper alone.** "Paper X never used
   SQuAD" does not make "what did X score on SQuAD?" unanswerable — paper Y's related work
   may state X's SQuAD score. Then the corpus does contain the answer, the agent correctly
   retrieves it, and the eval scores a correct answer as a hallucination. The item would be
   silently backwards.

Both rules are the same mistake at different scopes: absence checked somewhere smaller than
where the agent is allowed to look.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from evals.matching import first_term_present, mentions_name
from src.config import get_settings

# Method names too generic to identify a paper. "Gram" matches "n-gram", "program", and
# "diagram"; matching it as a substring produced two false eliminations before word
# boundaries and this list were added.
AMBIGUOUS_NAMES = frozenset(
    {
        "gram",
        "state",
        "model",
        "score",
        "prompt",
        "agent",
        "graph",
        "token",
        "layer",
        "chain",
        "trace",
        "align",
        "scale",
        "flow",
        "core",
        "base",
        "unit",
        "node",
    }
)

# A name occurring this often across the corpus is a common word, whatever the title says.
MAX_NAME_OCCURRENCES = 400


@dataclass(frozen=True)
class Corpus:
    chunks: list[dict[str, object]]
    text_by_paper: dict[str, str] = field(default_factory=dict)
    lowered: str = ""

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)


@lru_cache(maxsize=1)
def load_corpus() -> Corpus:
    settings = get_settings()
    chunks = json.loads(settings.chunks_path.read_text(encoding="utf-8"))
    by_paper: dict[str, list[str]] = {}
    for chunk in chunks:
        by_paper.setdefault(str(chunk["paper_id"]), []).append(str(chunk["text"]))
    return Corpus(
        chunks=chunks,
        text_by_paper={pid: " ".join(t).lower() for pid, t in by_paper.items()},
        lowered=" ".join(str(c["text"]).lower() for c in chunks),
    )


@lru_cache(maxsize=1)
def paper_handles() -> dict[str, frozenset[str]]:
    """Strings that identify each paper if another paper refers to it.

    The arXiv id always; the method name from the title when it is distinctive enough to
    mean anything. Only 45 of 150 papers have a usable one, which is a real limit on rule 2
    and is stated in ``docs/EVALS.md`` rather than glossed over.
    """
    settings = get_settings()
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    corpus = load_corpus()

    handles: dict[str, frozenset[str]] = {}
    for paper in metadata:
        pid = str(paper["paper_id"])
        found = {pid}
        match = re.match(r"^([A-Z][A-Za-z0-9\-]{2,20})\s*[::]", str(paper["title"]))
        if match:
            name = match.group(1)
            if (
                len(name) >= 4
                and name.lower() not in AMBIGUOUS_NAMES
                and corpus.lowered.count(name.lower()) < MAX_NAME_OCCURRENCES
            ):
                found.add(name)
        handles[pid] = frozenset(found)
    return handles


# Absence is a property of the *concept*, not the string. A question asking what batch size
# a paper used is not made unanswerable by the paper writing "global batch 16" instead of
# "batch size" — the agent retrieves on meaning, and the corpus does contain the answer.
# This is Hazard 1 recurring one level down: absence checked at a finer granularity than the
# agent searches. Each term therefore carries the surface forms that mean the same thing,
# and *all* of them are screened across all 5,401 chunks.
TERM_PARAPHRASES: dict[str, tuple[str, ...]] = {
    "batch size": (
        "batch size",
        "batch-size",
        "minibatch",
        "mini-batch",
        "global batch",
        "batch of",
        "bsz",
        "per-device batch",
    ),
    "learning rate": (
        "learning rate",
        "learning-rate",
        "lr schedule",
        "step size",
        "base lr",
        "peak lr",
    ),
    "weight decay": ("weight decay", "l2 regularization", "l2 regularisation", "wd "),
    "warmup": ("warmup", "warm-up", "warm up", "linear ramp"),
    "flops": (
        "flops",
        "flop",
        "floating-point operations",
        "floating point operations",
        "macs",
        "multiply-accumulate",
        "petaflop",
        "teraflop",
    ),
    "gpu hours": (
        "gpu hours",
        "gpu-hours",
        "gpu time",
        "device hours",
        "a100 hours",
        "compute hours",
    ),
    "wall-clock": ("wall-clock", "wall clock", "wallclock", "elapsed time", "runtime"),
    "a100": ("a100", "a-100", "nvidia a100"),
    "throughput": ("throughput", "tokens per second", "samples per second", "images/sec", "qps"),
    "ablation": ("ablation", "ablate", "ablated", "leave-one-out", "component analysis"),
    "ablation study": ("ablation study", "ablation experiment", "ablations"),
    "leave-one-out": ("leave-one-out", "leave one out", "loo"),
    "random baseline": ("random baseline", "random chance", "chance level", "random guess"),
    "zero-shot baseline": ("zero-shot baseline", "zero shot baseline", "zeroshot baseline"),
    "majority class": ("majority class", "majority baseline", "most frequent class"),
    "wikitext": ("wikitext", "wiki-text", "wikitext-103", "wikitext103"),
    "librispeech": ("librispeech", "libri-speech", "libri speech"),
    "openwebtext": ("openwebtext", "open web text", "owt"),
    "laion": ("laion", "laion-5b", "laion-400m"),
    "pile": ("the pile", "pile dataset"),
    "imagenet": ("imagenet", "image-net", "ilsvrc", "in-1k", "in1k"),
    "cifar": ("cifar", "cifar-10", "cifar-100", "cifar10", "cifar100"),
    "mnist": ("mnist", "fashion-mnist", "fashionmnist"),
    "gsm8k": ("gsm8k", "gsm-8k", "grade school math"),
    "mmlu": ("mmlu", "massive multitask"),
    "humaneval": ("humaneval", "human-eval", "human eval"),
    "coco": ("coco", "ms-coco", "mscoco", "common objects in context"),
    "curriculum learning": (
        "curriculum learning",
        "curriculum-based",
        "easy-to-hard",
        "easy to hard",
        "curriculum schedule",
        "self-paced learning",
    ),
    "capsule networks": (
        "capsule network",
        "capsnet",
        "routing-by-agreement",
        "routing by agreement",
        "dynamic routing",
    ),
    "alphafold": ("alphafold", "alpha-fold", "protein structure prediction", "protein folding"),
    "click-through rate": ("click-through", "clickthrough", "ctr prediction", "\bctr\b"),
}


def paraphrases_of(term: str) -> tuple[str, ...]:
    """Surface forms that would answer a question about ``term``."""
    return TERM_PARAPHRASES.get(term.lower().strip(), (term,))


# Both delegate to the shared policy in `evals/matching.py`. They had two different
# hand-rolled boundaries here — one correct, one with no trailing boundary at all, so
# `cifar-10` matched inside `cifar-100` in the check gating every unanswerable item.
def _mentions_any(text: str, needles: tuple[str, ...]) -> str | None:
    """The first surface form the text discusses, or None."""
    return first_term_present(text, needles)


def _mentions(text: str, needle: str) -> bool:
    """Exact-name match, for identifying a paper by handle."""
    return mentions_name(text, needle)


@dataclass
class AbsenceResult:
    absent: bool
    reason: str
    chunks_scanned: int
    mentioning_chunk_ids: list[str] = field(default_factory=list)


def verify_topic_absent(term: str) -> AbsenceResult:
    """Rule 1: no surface form of the term appears in any chunk of the corpus."""
    corpus = load_corpus()
    forms = paraphrases_of(term)
    hits, found = [], None
    for c in corpus.chunks:
        hit = _mentions_any(str(c["text"]).lower(), forms)
        if hit:
            hits.append(str(c["chunk_id"]))
            found = found or hit
    if hits:
        return AbsenceResult(
            absent=False,
            reason=(
                f"{term!r} appears in {len(hits)} chunks (as {found!r}). It is discussed in "
                f"the corpus, so a refusal would be the wrong answer."
            ),
            chunks_scanned=corpus.n_chunks,
            mentioning_chunk_ids=hits[:10],
        )
    return AbsenceResult(
        absent=True,
        reason=(
            f"{term!r} and its {len(forms)} surface forms appear in none of the "
            f"{corpus.n_chunks} chunks"
        ),
        chunks_scanned=corpus.n_chunks,
    )


def verify_attribute_absent(anchor_paper_id: str, term: str) -> AbsenceResult:
    """Rule 2: absent from the anchor paper *and* not attributed to it anywhere else."""
    corpus = load_corpus()
    own = corpus.text_by_paper.get(anchor_paper_id)
    if own is None:
        return AbsenceResult(
            absent=False,
            reason=f"no paper {anchor_paper_id!r} in the corpus",
            chunks_scanned=corpus.n_chunks,
        )

    forms = paraphrases_of(term)
    own_hit = _mentions_any(own, forms)
    if own_hit:
        return AbsenceResult(
            absent=False,
            reason=(
                f"{anchor_paper_id} does report {term!r} in its own text, as {own_hit!r} — "
                f"the question is answerable and a refusal would be wrong"
            ),
            chunks_scanned=corpus.n_chunks,
        )

    # The part that the anchor-paper-only check misses.
    handles = paper_handles().get(anchor_paper_id, frozenset({anchor_paper_id}))
    for chunk in corpus.chunks:
        if str(chunk["paper_id"]) == anchor_paper_id:
            continue
        text = str(chunk["text"])
        if not _mentions_any(text.lower(), forms):
            continue
        for handle in handles:
            if _mentions(text, handle):
                return AbsenceResult(
                    absent=False,
                    reason=(
                        f"chunk {chunk['chunk_id']} (paper {chunk['paper_id']}) mentions both "
                        f"{term!r} and {handle!r} — another paper may report this figure for "
                        f"{anchor_paper_id}, which would make the item answerable"
                    ),
                    chunks_scanned=corpus.n_chunks,
                    mentioning_chunk_ids=[str(chunk["chunk_id"])],
                )

    return AbsenceResult(
        absent=True,
        reason=(
            f"{term!r} is absent from {anchor_paper_id} and is not attributed to it in any "
            f"of the other {corpus.n_chunks} chunks"
        ),
        chunks_scanned=corpus.n_chunks,
    )


def verify_attribute_absent_in(
    chunks: list[dict[str, object]],
    handles: dict[str, frozenset[str]],
    anchor_paper_id: str,
    term: str,
) -> AbsenceResult:
    """``verify_attribute_absent`` against a supplied corpus rather than the real one.

    Exists so the cross-citation rule can be tested by injecting a chunk that trips it.
    "0 of 979 eliminated" is either *verified clean* or *inert*, and nothing about the
    number itself distinguishes those two readings — a check that has never fired is
    indistinguishable from one that cannot (DECISIONS D-023).
    """
    own = " ".join(str(c["text"]) for c in chunks if str(c["paper_id"]) == anchor_paper_id).lower()
    if not own:
        return AbsenceResult(False, f"no paper {anchor_paper_id!r}", len(chunks))
    if _mentions(own, term.lower()):
        return AbsenceResult(
            False, f"{anchor_paper_id} does mention {term!r} in its own text", len(chunks)
        )

    anchor_handles = handles.get(anchor_paper_id, frozenset({anchor_paper_id}))
    for chunk in chunks:
        if str(chunk["paper_id"]) == anchor_paper_id:
            continue
        text = str(chunk["text"])
        if not _mentions(text.lower(), term.lower()):
            continue
        for handle in anchor_handles:
            if _mentions(text, handle):
                return AbsenceResult(
                    absent=False,
                    reason=(
                        f"chunk {chunk['chunk_id']} (paper {chunk['paper_id']}) mentions both "
                        f"{term!r} and {handle!r} — another paper may report this figure for "
                        f"{anchor_paper_id}, which would make the item answerable"
                    ),
                    chunks_scanned=len(chunks),
                    mentioning_chunk_ids=[str(chunk["chunk_id"])],
                )
    return AbsenceResult(
        absent=True,
        reason=f"{term!r} absent from {anchor_paper_id} and unattributed elsewhere",
        chunks_scanned=len(chunks),
    )


def candidate_pairs(benchmarks: list[str]) -> dict[str, int]:
    """How much full-corpus verification actually eliminates, on this corpus.

    Reported rather than assumed. The answer here is ~0%, and the reason is structural: all
    150 papers were published on the same afternoon, so no paper *can* cite another's
    results. The check still ships — it costs nothing, it is the discipline whose absence
    cost the topic stratum, and it becomes load-bearing the moment the corpus spans more
    than one day.
    """
    corpus = load_corpus()
    naive = eliminated = 0
    for pid in corpus.text_by_paper:
        for benchmark in benchmarks:
            result = verify_attribute_absent(pid, benchmark)
            if _mentions(corpus.text_by_paper[pid], benchmark.lower()):
                continue
            naive += 1
            if not result.absent:
                eliminated += 1
    return {"naive": naive, "eliminated": eliminated, "surviving": naive - eliminated}


def load_eval_corpus_checksum() -> str:
    path: Path = get_settings().corpus_checksum_path
    return path.read_text(encoding="utf-8").strip().split()[0] if path.exists() else ""
