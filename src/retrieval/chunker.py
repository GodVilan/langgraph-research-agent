"""PDF text extraction and recursive chunking.

Ported from v2.1 ``rag/processing/chunker.py``. The chunking algorithm is FROZEN — the
committed FAISS index was built from its output, so any change invalidates the v2.1-vs-v3
comparison (docs/DECISIONS.md D-002).

One deliberate change: ``load_chunks`` re-runs ``classify_section`` on load. AUDIT §4.15
found that the committed chunk cache predates the ``section_type`` field, so all 5,401
chunks deserialise as ``"general"``. Classification is deterministic and text-only, so
recomputing it repopulates the field without re-embedding anything and leaves the index
bit-identical (MIGRATION_MAP §5.8 option 1).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import CHUNK_OVERLAP, DEFAULT_CHUNK

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    authors: list[str]
    text: str
    token_count: int
    chunk_index: int
    source: str = "corpus"  # corpus | arxiv | upload
    section_type: str = "general"


def classify_section(text: str) -> str:
    """Ordered substring match over the first 600 characters.

    Precision is unvalidated: a methodology chunk containing the phrase "in the abstract"
    classifies as ``abstract``. This is why the section filter ships disabled
    (``RetrievalSettings.enable_section_filter``) until Phase 4 can measure it.
    """
    text_lower = text.lower()[:600]
    if any(k in text_lower for k in ("abstract", "summary", "tldr", "tldr:")):
        return "abstract"
    if any(k in text_lower for k in ("introduction", "background", "motivation")):
        return "introduction"
    if any(
        k in text_lower
        for k in (
            "methodology",
            "method",
            "approach",
            "proposed architecture",
            "mathematical model",
        )
    ):
        return "methodology"
    if any(
        k in text_lower
        for k in ("evaluation", "experiment", "result", "benchmarks", "ablation", "dataset")
    ):
        return "experiments"
    if any(k in text_lower for k in ("conclusion", "future work", "discussion", "limitations")):
        return "conclusion"
    return "general"


def extract_text(pdf_path: str | Path) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        log.warning("Cannot read %s: %s", pdf_path, exc)
        return ""


def extract_text_from_bytes(data: bytes) -> str:
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as exc:
        log.warning("Cannot read PDF bytes: %s", exc)
        return ""


def clean_text(raw: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def token_count(text: str) -> int:
    """Whitespace token count. FROZEN — the committed chunks carry these values."""
    return len(text.split())


def recursive_chunk(
    text: str,
    chunk_size: int = DEFAULT_CHUNK,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """FROZEN. Ported verbatim from v2.1."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int) -> list[str]:
        if sep_idx >= len(separators):
            words = text.split()
            pieces, start = [], 0
            while start < len(words):
                pieces.append(" ".join(words[start : start + chunk_size]))
                start += chunk_size - overlap
            return pieces

        sep = separators[sep_idx]
        parts = text.split(sep)
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = token_count(part)
            if current_tokens + part_tokens <= chunk_size:
                current.append(part)
                current_tokens += part_tokens
            else:
                if current:
                    chunks.append(sep.join(current))
                if part_tokens > chunk_size:
                    chunks.extend(_split(part, sep_idx + 1))
                    current, current_tokens = [], 0
                else:
                    overlap_words = (
                        " ".join(sep.join(current).split()[-overlap:]) if current else ""
                    )
                    current = [overlap_words, part] if overlap_words else [part]
                    current_tokens = token_count(sep.join(current))

        if current:
            chunks.append(sep.join(current))

        return [c.strip() for c in chunks if c.strip()]

    if token_count(text) <= chunk_size:
        return [text.strip()]

    return _split(text, sep_idx=0)


def process_papers(
    metadata: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for paper in metadata:
        pid = str(paper["paper_id"])
        pdf_path = str(paper.get("pdf_path", ""))
        title = str(paper.get("title", pid))
        authors = list(paper.get("authors", []))

        if not pdf_path or not Path(pdf_path).exists():
            log.warning("PDF not found for %s", pid)
            continue

        text = clean_text(extract_text(pdf_path))
        if not text:
            continue

        for idx, piece in enumerate(recursive_chunk(text, chunk_size, overlap)):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{pid}_{idx:04d}",
                    paper_id=pid,
                    title=title,
                    authors=authors,
                    text=piece,
                    token_count=token_count(piece),
                    chunk_index=idx,
                    source="corpus",
                    section_type=classify_section(piece),
                )
            )

    log.info("Processed %d papers -> %d chunks", len(metadata), len(all_chunks))
    return all_chunks


def chunks_from_bytes(
    data: bytes,
    paper_id: str,
    title: str,
    authors: list[str],
    source: str = "upload",
    chunk_size: int = DEFAULT_CHUNK,
) -> list[Chunk]:
    """Process a PDF from bytes (live arXiv fetch)."""
    text = clean_text(extract_text_from_bytes(data))
    if not text:
        return []

    return [
        Chunk(
            chunk_id=f"{paper_id}_{idx:04d}",
            paper_id=paper_id,
            title=title,
            authors=authors,
            text=piece,
            token_count=token_count(piece),
            chunk_index=idx,
            source=source,
            section_type=classify_section(piece),
        )
        for idx, piece in enumerate(recursive_chunk(text, chunk_size))
    ]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)


def load_chunks(path: Path, classify_on_load: bool = True) -> list[Chunk]:
    """Load chunks, optionally repopulating ``section_type``.

    The committed ``chunks_512.json`` has no ``section_type`` key, so without
    ``classify_on_load`` every chunk silently defaults to ``"general"`` — the bug behind
    AUDIT §4.15. Reclassifying touches only the label; ``text`` is untouched, so the
    embeddings and the index are unaffected.
    """
    with open(path) as f:
        raw: list[dict[str, Any]] = json.load(f)

    chunks = [Chunk(**d) for d in raw]
    if classify_on_load:
        for chunk in chunks:
            chunk.section_type = classify_section(chunk.text)
    return chunks
