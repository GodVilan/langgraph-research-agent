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


def _mentions(text: str, needle: str) -> bool:
    """Word-boundary match. Substring matching is what produced the false eliminations."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])", text) is not None


@dataclass
class AbsenceResult:
    absent: bool
    reason: str
    chunks_scanned: int
    mentioning_chunk_ids: list[str] = field(default_factory=list)


def verify_topic_absent(term: str) -> AbsenceResult:
    """Rule 1: the term appears nowhere in any chunk of the corpus."""
    corpus = load_corpus()
    hits = [
        str(c["chunk_id"]) for c in corpus.chunks if _mentions(str(c["text"]).lower(), term.lower())
    ]
    if hits:
        return AbsenceResult(
            absent=False,
            reason=(
                f"{term!r} appears in {len(hits)} chunks. It is discussed in the corpus, so "
                f"a refusal would be the wrong answer."
            ),
            chunks_scanned=corpus.n_chunks,
            mentioning_chunk_ids=hits[:10],
        )
    return AbsenceResult(
        absent=True,
        reason=f"{term!r} appears in none of the {corpus.n_chunks} chunks",
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

    if _mentions(own, term.lower()):
        return AbsenceResult(
            absent=False,
            reason=f"{anchor_paper_id} does mention {term!r} in its own text",
            chunks_scanned=corpus.n_chunks,
        )

    # The part that the anchor-paper-only check misses.
    handles = paper_handles().get(anchor_paper_id, frozenset({anchor_paper_id}))
    for chunk in corpus.chunks:
        if str(chunk["paper_id"]) == anchor_paper_id:
            continue
        text = str(chunk["text"])
        if not _mentions(text.lower(), term.lower()):
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
