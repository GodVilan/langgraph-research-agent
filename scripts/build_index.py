"""Build the FAISS index from the committed chunks.

Backs `make index`. The index is a build artifact, not source: it is gitignored and
rebuilt from `data/chunks_512.json`, which *is* committed and checksummed.

Determinism note: BGE inference on MPS and on CPU do not produce bit-identical float32
output. `--device cpu` is therefore the reproducible path, and `make index-verify` uses it.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.retrieval.chunker import load_chunks
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_index")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the index exists")
    parser.add_argument(
        "--print-hashes", action="store_true", help="Print sha256 of the written artifacts"
    )
    args = parser.parse_args()

    settings = get_settings()
    index_name = settings.retrieval.index_name
    out_dir = args.out or settings.index_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / f"{index_name}.faiss"

    if index_file.exists() and not args.force:
        log.info("Index already exists at %s (use --force to rebuild)", index_file)
        if args.print_hashes:
            _print_hashes(out_dir)
        return 0

    log.info("Loading chunks from %s", settings.chunks_path)
    # Section labels do not affect embeddings; classify off to keep the build minimal.
    chunks = load_chunks(settings.chunks_path, classify_on_load=False)
    log.info("Loaded %d chunks", len(chunks))

    model = EmbeddingModel(device=args.device)
    log.info("Encoding on %s — this takes roughly 5 min on MPS, 15-20 min on CPU", model.device)

    t0 = time.monotonic()
    embeddings = model.encode([c.text for c in chunks], show_progress=True)
    log.info("Encoded %d chunks in %.1f min", len(chunks), (time.monotonic() - t0) / 60)

    store = VectorStore(dim=model.dim)
    store.add(embeddings, chunks)
    store.save(out_dir, name=index_name)
    log.info("Wrote %s (%d vectors, dim=%d)", index_file, store.size, model.dim)

    if args.print_hashes:
        _print_hashes(out_dir)
    return 0


def _print_hashes(out_dir: Path) -> None:
    for suffix in (".faiss", "_meta.pkl"):
        path = out_dir / f"{get_settings().retrieval.index_name}{suffix}"
        if path.exists():
            print(f"{sha256(path)}  {path}")


if __name__ == "__main__":
    raise SystemExit(main())
