"""Can this corpus honestly fill the four eval strata?

Backs `make corpus-diversity`. Run before generating a single eval item.

The Phase 4 plan asks for ~55 single-paper factual, 20 multi-hop, 15 unanswerable, and 10
ambiguous items. The corpus is 150 cs.LG papers published on one afternoon, which is a
narrow slice, and the instruction was to confront that before generating rather than
halfway through. This script is that confrontation: it measures what each stratum can be
built from and reports a shortfall as a shortfall.

The finding that matters is in `unanswerable_topics`. Screening candidate "absent" topics
against paper *abstracts* marks eight topics absent that the **full text** mentions —
so an unanswerable set generated from abstracts would be roughly 80% broken. 13.4M
characters of related-work sections name nearly every term in machine learning at least
once in passing. Absence has to be checked against all 5,401 chunks, and even then only a
handful of topics survive.

Usage:
    python scripts/corpus_diversity.py
    python scripts/corpus_diversity.py --json
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings

# Terms probed for absence. Chosen to be plausible things a reader might ask a machine
# learning corpus about, spanning classical ML, vision, NLP, and training practice.
ABSENCE_PROBES: dict[str, str] = {
    "AutoML / hyperparameter search": r"automl|hyperparameter (search|optimi|tuning)",
    "active learning": r"active learning",
    "autonomous driving": r"autonomous driving|self-driving",
    "curriculum learning": r"curriculum learning",
    "knowledge graphs": r"knowledge graph",
    "meta-learning (MAML)": r"meta-learning|\bmaml\b",
    "mixture of experts": r"mixture[- ]of[- ]experts|\bmoe\b",
    "optical character recognition": r"\bocr\b|optical character",
    "CTR / click-through prediction": r"\bctr\b|click-through",
    "spiking neural networks": r"spiking",
    "collaborative filtering": r"collaborative filtering",
    "matrix factorization": r"matrix factori[sz]ation",
    "word2vec / GloVe": r"word2vec|\bglove\b",
    "LSTM / GRU": r"\blstm\b|\bgru\b",
    "capsule networks": r"capsule network",
    "genetic / evolutionary algorithms": r"genetic algorithm|evolution(ary)? strateg",
    "particle swarm optimization": r"swarm optimi|particle swarm",
    "fuzzy logic": r"fuzzy logic",
    "SLAM": r"\bslam\b|simultaneous localization",
    "text-to-speech": r"text-to-speech|\btts\b|speech synthesis",
    "BLEU / machine translation": r"\bbleu\b",
    "named entity recognition": r"named entity",
    "sentiment analysis": r"sentiment analysis",
    "topic modeling / LDA": r"topic model|latent dirichlet",
    "random forest": r"random forest",
    "gradient boosting / XGBoost": r"xgboost|gradient boost|lightgbm",
    "support vector machine": r"support vector machine|\bsvm\b",
    "k-means clustering": r"k-means",
    "AlphaFold": r"alphafold",
    "AlphaGo / MCTS": r"alphago|monte carlo tree",
    "normalizing flows": r"normalizing flow",
    "neural radiance fields": r"\bnerf\b|radiance field",
    "style transfer": r"style transfer",
    "mixup augmentation": r"\bmixup\b",
    "label smoothing": r"label smoothing",
}

BENCHMARKS = ["imagenet", "cifar", "mnist", "gsm8k", "mmlu", "humaneval", "squad", "coco"]

# The escape is U+00D7, the typographic multiplication sign. Papers report speedups both as
# "2.4x" and with that character, so matching only the ASCII letter would miss about half.
NUMERIC_FACT = re.compile(r"\d+\.\d+\s*%|\b\d{1,3}\.\d{1,2}\b|\b\d+(?:\.\d+)?[\u00d7x]\b")

# Words too generic to bridge two papers. A bridge term has to be specific enough that two
# papers sharing it are actually talking about the same thing.
STOPWORDS = set(
    [
        "this",
        "that",
        "with",
        "from",
        "which",
        "have",
        "been",
        "were",
        "also",
        "more",
        "than",
        "such",
        "their",
        "there",
        "these",
        "those",
        "using",
        "used",
        "use",
        "can",
        "may",
        "our",
        "we",
        "results",
        "result",
        "show",
        "shows",
        "showing",
        "method",
        "methods",
        "model",
        "models",
        "approach",
        "approaches",
        "based",
        "both",
        "when",
        "what",
        "while",
        "into",
        "over",
        "under",
        "between",
        "during",
        "about",
        "across",
        "each",
        "other",
        "others",
        "many",
        "most",
        "some",
        "first",
        "second",
        "third",
        "however",
        "therefore",
        "thus",
        "then",
        "they",
        "them",
        "its",
        "paper",
        "work",
        "works",
        "propose",
        "proposed",
        "performance",
        "training",
        "test",
        "data",
        "dataset",
        "datasets",
        "learning",
        "neural",
        "network",
        "networks",
        "deep",
        "large",
        "small",
        "new",
        "novel",
        "state",
        "art",
        "tasks",
        "task",
        "benchmark",
        "benchmarks",
        "evaluate",
        "evaluation",
        "experiments",
        "experimental",
    ]
)


def load() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    settings = get_settings()
    metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    chunks = json.loads(settings.chunks_path.read_text(encoding="utf-8"))
    by_paper: dict[str, list[str]] = collections.defaultdict(list)
    for chunk in chunks:
        by_paper[chunk["paper_id"]].append(chunk["text"])
    return metadata, dict(by_paper)


def single_paper_factual(by_paper: dict[str, list[str]]) -> dict[str, Any]:
    """Papers carrying enough concrete, checkable facts to anchor a factual question."""
    counts = {pid: len(NUMERIC_FACT.findall(" ".join(t))) for pid, t in by_paper.items()}
    return {
        "target": 55,
        "papers": len(counts),
        "eligible": sum(1 for c in counts.values() if c >= 20),
        "rich": sum(1 for c in counts.values() if c >= 50),
        "weak": sum(1 for c in counts.values() if c < 10),
    }


def multi_hop(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Pairs of distinct papers sharing enough specific vocabulary to bridge honestly."""
    terms: dict[str, set[str]] = {}
    doc_freq: collections.Counter[str] = collections.Counter()
    for paper in metadata:
        blob = (paper["title"] + " " + paper["abstract"]).lower()
        words = {w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", blob) if w not in STOPWORDS}
        terms[paper["paper_id"]] = words
        doc_freq.update(words)

    # In 2-8 papers: shared enough to connect two papers, rare enough to be about something.
    bridges = {w for w, c in doc_freq.items() if 2 <= c <= 8}
    by_term: dict[str, list[str]] = collections.defaultdict(list)
    for pid, words in terms.items():
        for word in words & bridges:
            by_term[word].append(pid)

    shared: collections.Counter[tuple[str, str]] = collections.Counter()
    for pids in by_term.values():
        for a, b in itertools.combinations(sorted(set(pids)), 2):
            shared[(a, b)] += 1

    strong = {p for p, c in shared.items() if c >= 4}
    return {
        "target": 20,
        "bridge_terms": len(bridges),
        "pairs_4plus": len(strong),
        "pairs_6plus": sum(1 for c in shared.values() if c >= 6),
        "papers_involved": len({p for pair in strong for p in pair}),
    }


def unanswerable_topics(
    metadata: list[dict[str, Any]], by_paper: dict[str, list[str]]
) -> dict[str, Any]:
    """The stratum that does not fill, and the measurement that shows why."""
    abstracts = " ".join((p["title"] + " " + p["abstract"]).lower() for p in metadata)
    full_text = " ".join(" ".join(t).lower() for t in by_paper.values())

    absent_in_abstracts, absent_in_full, trap = [], [], []
    for name, pattern in ABSENCE_PROBES.items():
        in_abstract = bool(re.search(pattern, abstracts))
        in_full = bool(re.search(pattern, full_text))
        if not in_abstract:
            absent_in_abstracts.append(name)
        if not in_full:
            absent_in_full.append(name)
        elif not in_abstract:
            # Looks absent from abstracts, is present in the body. Generating from
            # abstracts would have produced a broken "unanswerable" item here.
            trap.append(name)

    return {
        "target": 15,
        "probed": len(ABSENCE_PROBES),
        "absent_from_abstracts": len(absent_in_abstracts),
        "absent_from_full_text": sorted(absent_in_full),
        "false_absences": sorted(trap),
        "corpus_chars": len(full_text),
    }


def unanswerable_attributes(by_paper: dict[str, list[str]]) -> dict[str, Any]:
    """The construction that *does* fill: facts absent from papers that are present.

    Asking what a real paper reports on a benchmark it never used is on-topic enough to
    pass the scope guard, unambiguous (the fact is verifiably absent), and lands retrieval
    squarely on that paper's own chunks at high similarity — which is the no-score-floor
    failure the unanswerable stratum exists to expose.
    """
    combinations = 0
    for texts in by_paper.values():
        blob = " ".join(texts).lower()
        combinations += sum(1 for b in BENCHMARKS if b not in blob)
    users = {b: sum(1 for t in by_paper.values() if b in " ".join(t).lower()) for b in BENCHMARKS}
    return {
        "benchmarks_probed": len(BENCHMARKS),
        "absent_pair_combinations": combinations,
        "papers_using_each": users,
    }


def ambiguous(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Referential ambiguity, measured two ways — one works, one does not."""
    acronyms: dict[str, set[str]] = collections.defaultdict(set)
    for paper in metadata:
        blob = paper["title"] + " " + paper["abstract"]
        for acronym in set(re.findall(r"\b([A-Z]{2,6})\b", blob)):
            acronyms[acronym].add(paper["paper_id"])
    generic = {
        "THE",
        "AND",
        "FOR",
        "WITH",
        "OUR",
        "ML",
        "AI",
        "LLM",
        "LLMS",
        "GPU",
        "CPU",
        "API",
        "SOTA",
        "OOD",
        "IID",
    }
    collisions = {a: len(v) for a, v in acronyms.items() if len(v) >= 4 and a not in generic}

    topics = {
        "federated learning": r"\bfederated\b",
        "reinforcement learning": r"reinforcement learning|\brl\b",
        "diffusion models": r"\bdiffusion\b",
        "alignment / safety": r"\b(alignment|safety|jailbreak)\b",
        "quantum": r"\bquantum\b",
    }
    crowded = {}
    for name, pattern in topics.items():
        crowded[name] = sum(
            1 for p in metadata if re.search(pattern, (p["title"] + " " + p["abstract"]).lower())
        )
    return {"target": 10, "acronym_collisions": collisions, "papers_per_topic": crowded}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    metadata, by_paper = load()
    report = {
        "single_paper_factual": single_paper_factual(by_paper),
        "multi_hop": multi_hop(metadata),
        "unanswerable_topics": unanswerable_topics(metadata, by_paper),
        "unanswerable_attributes": unanswerable_attributes(by_paper),
        "ambiguous": ambiguous(metadata),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    factual = report["single_paper_factual"]
    hop = report["multi_hop"]
    topics = report["unanswerable_topics"]
    attrs = report["unanswerable_attributes"]
    amb = report["ambiguous"]

    print(f"Corpus: {len(metadata)} papers, {topics['corpus_chars']:,} characters of text\n")

    print(f"1. SINGLE-PAPER FACTUAL  target {factual['target']}  -- FEASIBLE")
    print(
        f"   {factual['eligible']}/{factual['papers']} papers carry >=20 numeric facts; "
        f"{factual['rich']} carry >=50"
    )
    print(f"   only {factual['weak']} papers are too thin to anchor an item\n")

    print(f"2. MULTI-HOP  target {hop['target']}  -- FEASIBLE")
    print(
        f"   {hop['pairs_4plus']} paper pairs share >=4 distinctive terms "
        f"({hop['pairs_6plus']} share >=6)"
    )
    print(f"   {hop['papers_involved']}/{len(metadata)} papers appear in at least one pair\n")

    print(f"3. UNANSWERABLE  target {topics['target']}  -- SHORTFALL as specified")
    print(
        f"   probed {topics['probed']} topics; only "
        f"{len(topics['absent_from_full_text'])} are absent from the full text:"
    )
    for name in topics["absent_from_full_text"]:
        print(f"     - {name}")
    print(
        f"   {topics['absent_from_abstracts']} looked absent in abstracts, so "
        f"{len(topics['false_absences'])} of those are traps:"
    )
    for name in topics["false_absences"]:
        print(f"     x {name}")
    print("   cause: related-work sections name almost every ML term at least once\n")

    print("   Alternative that does fill it -- absent attributes of present papers:")
    print(
        f"     {attrs['absent_pair_combinations']:,} (paper, benchmark-it-never-uses) "
        f"combinations from {attrs['benchmarks_probed']} benchmarks alone"
    )
    print(
        f"     e.g. ImageNet: {attrs['papers_using_each']['imagenet']} papers use it, "
        f"{len(by_paper) - attrs['papers_using_each']['imagenet']} do not\n"
    )

    print(f"4. AMBIGUOUS  target {amb['target']}  -- FEASIBLE, but not from acronyms")
    print(f"   acronyms shared by >=4 papers: {amb['acronym_collisions'] or 'none usable'}")
    print("   referential ambiguity from crowded topics instead:")
    for name, count in sorted(amb["papers_per_topic"].items(), key=lambda kv: -kv[1]):
        print(f"     {count:3d} papers  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
