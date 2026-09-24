"""Run v2.1 — a clone of published `main`, pinned — over the frozen set on the same generator.

Backs `make run-v21`. The comparison arm the hold-the-generator-constant rationale needs: v2.1
on `gemini-3.5-flash-lite`, same corpus (sha asserted), same 69 questions, persisted in the
same run-record shape as v3 so the judge and the metrics code read it unchanged.

What is and is not touched. The clone under `evals/baselines/arXiv-Agent` is published `main`
at the commit recorded in the run; nothing in it is edited. Two things are overridden at
runtime, from here, on the imported modules: `rag.config.GEMINI_MODEL` (v2.1 pins
`gemini-2.5-flash-lite`, retired — D-012) and the API-key name (v2.1 reads `GEMINI_API_KEY`).
Retrieval calls are wrapped, not replaced, so the chunk ids v2.1 actually retrieved are
recorded — v2.1's tools return formatted strings and keep no ids (AUDIT §4.19).

v2.1's `out_of_scope` short-circuit is recorded as `guardrail_blocked`, the same status v3's
input layer produces, because it is the same behaviour. A fresh agent is built per item:
v2.1's conversation memory is per instance and would otherwise carry across questions.

Timeboxed at six hours by the working agreement. If it does not finish, the record says how
far it got.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.absence import load_corpus, load_eval_corpus_checksum
from evals.run_set import (
    FREE_TIER_RPM,
    ItemRun,
    RetrievedRecord,
    RunRecord,
    load_record,
    run_path,
    save_record,
)
from evals.schema import EvalItem, EvalSet

CLONE = Path("evals/baselines/arXiv-Agent")
GENERATOR = "gemini-3.5-flash-lite"
TAG = "v21"


PINNED_COMMIT = "8d3e67f4cea716af65d2ce5e5fe8e46567f1fc58"  # the audited commit; published main


def pinned_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def assert_clone_pristine() -> str:
    """Refuse to run unless the clone is at the pinned commit with a clean tree.

    Not a D-023 instance but its own class (D-033): `ruff --fix`, configured for `evals/`,
    rewrote 38 files of this clone the first time it was run after the clone existed. A
    reformatted v2.1 still runs and still produces numbers, and nothing in the output would
    have shown that the baseline was a comparison against a modified system. The tool did
    exactly what it was configured to do to a directory that was never in scope. So the
    harness checks, every run, and the run record carries the commit it checked.
    """
    if not CLONE.exists():
        raise SystemExit(f"clone published main into {CLONE} first")
    head = pinned_commit()
    if head != PINNED_COMMIT:
        raise SystemExit(
            f"v2.1 clone is at {head[:7]}, not the pinned {PINNED_COMMIT[:7]}; refusing"
        )
    dirty = subprocess.run(
        ["git", "-C", str(CLONE), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "v2.1 clone has local modifications; refusing to run a baseline against a modified "
            f"system:\n{dirty[:800]}"
        )
    return head


def _env(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def build_v21() -> tuple[Any, Any, Any, Any, list[str]]:
    """Import the clone, override model and key at runtime, build its retrievers once."""
    os.environ.setdefault("GEMINI_API_KEY", _env("GOOGLE_API_KEY"))
    sys.path.insert(0, str(CLONE.resolve()))
    from rag import config as v21_config

    v21_config.GEMINI_MODEL = GENERATOR
    v21_config.GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

    from rag.processing.chunker import load_chunks
    from rag.retrieval.bm25 import BM25Retriever
    from rag.retrieval.dense import Retriever
    from rag.retrieval.embeddings import EmbeddingModel
    from rag.sources.session_index import SessionIndex
    from rag.sources.source_router import SourceRouter

    chunks = load_chunks(v21_config.DATA_DIR / f"chunks_{v21_config.DEFAULT_CHUNK}.json")
    dense = Retriever.build(
        chunks=chunks,
        chunk_size=v21_config.DEFAULT_CHUNK,
        index_dir=v21_config.RESULTS_DIR / "indices",
    )
    bm25 = BM25Retriever(chunks)
    session = SessionIndex(EmbeddingModel())
    router = SourceRouter(dense, session)

    # Record every chunk id the retrievers return, in order, without changing what they
    # return. This is the only way to get Recall@k out of v2.1.
    seen: list[str] = []
    real_search, real_bm25 = router.search, bm25.retrieve

    def search(*args: Any, **kwargs: Any) -> Any:
        out = real_search(*args, **kwargs)
        seen.extend(str(c.chunk_id) for c, _ in out)
        return out

    def retrieve(*args: Any, **kwargs: Any) -> Any:
        out = real_bm25(*args, **kwargs)
        seen.extend(str(c.chunk_id) for c, _ in out)
        return out

    router.search = search
    bm25.retrieve = retrieve
    return router, bm25, session, v21_config, seen


def run_one(
    item: EvalItem, router: Any, bm25: Any, session: Any, seen: list[str], texts: dict[str, str]
) -> ItemRun:
    from rag.agent import ReActAgent

    seen.clear()
    agent = ReActAgent(router, bm25, session)
    started = time.monotonic()
    error: str | None = None
    answer, oos, steps = None, False, 0
    try:
        response = agent.run(item.question, use_arxiv=False)
        answer, oos, steps = response.answer, bool(response.out_of_scope), response.total_steps
    except Exception as exc:  # recorded, never swallowed
        error = f"{type(exc).__name__}: {exc}"[:500]
    latency = time.monotonic() - started

    ordered: list[str] = []
    for cid in seen:
        if cid not in ordered:
            ordered.append(cid)
    return ItemRun(
        item_id=item.item_id,
        stratum=item.stratum.value,
        question=item.question,
        gold_answer=item.gold_answer,
        gold_chunk_ids=list(item.gold_chunk_ids),
        gold_support=item.gold_support.value if item.gold_support else None,
        answer=answer,
        guardrail_blocked=oos,
        guardrail_reason="v2.1 out_of_scope short-circuit" if oos else None,
        retrieved=[
            RetrievedRecord(chunk_id=c, paper_id=c.split("_")[0], score=0.0, text=texts.get(c, ""))
            for c in ordered
        ],
        usage={"llm_calls": steps},  # v2.1 exposes step count, not token usage
        latency_s=round(latency, 3),
        status="error" if error else ("guardrail_blocked" if oos else "completed"),
        error=error,
        thread_id="",
        ran_at=datetime.now(UTC).isoformat(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--only", default="")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--timebox-hours", type=float, default=6.0)
    args = parser.parse_args()

    evalset = EvalSet.read(args.set)
    commit = assert_clone_pristine()
    out = run_path(evalset.sha256, TAG)
    existing = None if args.fresh else load_record(out)
    record = existing or RunRecord(
        arm="v2.1",
        arm_config={
            "repo": "https://github.com/GodVilan/arXiv-Agent",
            "commit": commit,
            "clone_pristine_checked": "yes — HEAD at the pinned commit, tree clean, every run",
        },
        set_path=str(args.set),
        set_sha256=evalset.sha256,
        corpus_sha256=load_eval_corpus_checksum(),
        generator_model=GENERATOR,
        prompt_version="v2.1 published main",
        started_at=datetime.now(UTC).isoformat(),
    )
    wanted = {i.strip() for i in args.only.split(",") if i.strip()}
    todo = [
        i
        for i in evalset.items
        if (not wanted or i.item_id in wanted) and i.item_id not in record.items
    ]
    print(
        f"v2.1 @ {commit[:7]} on {GENERATOR}: {len(todo)} items to run, "
        f"timebox {args.timebox_hours}h"
    )

    texts = {str(c["chunk_id"]): str(c["text"]) for c in load_corpus().chunks}
    router, bm25, session, _, seen = build_v21()
    deadline = time.monotonic() + args.timebox_hours * 3600
    for n, item in enumerate(todo, start=1):
        if time.monotonic() > deadline:
            print(f"timebox reached after {n - 1} items; the record says so")
            break
        started = time.monotonic()
        run = run_one(item, router, bm25, session, seen, texts)
        record.items[item.item_id] = run
        save_record(out, record)
        calls = int(run.usage.get("llm_calls", 0) or 0)
        print(
            f"  [{n}/{len(todo)}] {item.item_id:8} {run.status:17} {calls:2d} steps  "
            f"{run.latency_s:6.1f}s"
        )
        budget_s = 60.0 * max(calls, 1) / FREE_TIER_RPM
        elapsed = time.monotonic() - started
        if budget_s > elapsed:
            time.sleep(budget_s - elapsed)
    print(f"\nwrote {out}; statuses {record.counts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(asyncio.to_thread(main)))
