"""Does `seed=0, top_k=1` make the *generator* deterministic? (Phase 5 review, G-2.)

D-014 concluded determinism was unrecoverable after trying temperature, reasoning effort and
thinking budget. The Phase 5 classifier probe (D-035) showed `seed=0, top_k=1` recovers it
for the scope classifier. Phase 4's outcome variance lives in generation, so the question
that matters is whether the same holds for the generator.

**Measure only.** Nothing here changes what ships: `top_k=1` is greedy decoding and would
change answer quality, so generator pinning would invalidate every Phase 4 metric. A positive
result goes to BACKLOG as a Phase 6 arm.

Two prompts, each run N times unpinned (production) and N times pinned:

* ``d014`` — D-014's own fixed prompt, verbatim, so the result is directly comparable with
  its table;
* ``generate`` — the real generation call: the v2 generate prompt over a *fixed* rendered
  context (the five chunks r1 retrieved for ``sp-001``), so every call sees identical input.

Gemini free tier, $0 billed. Writes ``evals/runs/generator_determinism.json``.

    make generator-determinism            # runs it (4N calls)
    make generator-determinism REPORT=1   # reprints the stored result
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT = REPO / "evals" / "runs" / "generator_determinism.json"
RUN = REPO / "evals" / "runs" / "v3_de699d68.json"
ITEM = "sp-001"


def generate_input() -> tuple[str, str]:
    from src.agent.nodes.generate import render_context
    from src.agent.prompts import load_prompt
    from src.agent.tools.retrieval_tools import to_retrieved
    from src.config import get_settings
    from src.retrieval.chunker import load_chunks

    item = json.loads(RUN.read_text(encoding="utf-8"))["items"][ITEM]
    ids = [r if isinstance(r, str) else r["chunk_id"] for r in item["retrieved"]][:5]
    by_id = {c.chunk_id: c for c in load_chunks(get_settings().chunks_path)}
    chunks = [to_retrieved(by_id[i], 1.0, "dense") for i in ids]
    system = load_prompt("generate", "v2")
    user = f"Question: {item['question']}\n\nRetrieved passages:\n{render_context(chunks)}"
    return system, user


async def run(n: int) -> dict[str, Any]:
    from evals.ratelimit import RateLimiter
    from scripts.determinism_probe import PROMPT, SYSTEM
    from src.agent.llm import call_text, get_chat_model
    from src.config import get_settings

    s = get_settings()
    if not s.google_api_key.get_secret_value():
        raise SystemExit("GOOGLE_API_KEY is not set")
    # Through the wrapper, so every call lands in the usage log (D-046).
    models = {"production": get_chat_model(), "pinned": get_chat_model(pinned=True)}
    prompts = {"d014": (SYSTEM, PROMPT), "generate": generate_input()}

    limiter = RateLimiter()
    result: dict[str, Any] = {
        "model": s.model_name(),
        "reasoning_effort": s.agent_reasoning_effort,
        "n": n,
        "probed_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "generate_item": ITEM,
        "cells": {},
    }
    for prompt_name, (system, user) in prompts.items():
        for config, model in models.items():
            outputs: list[str] = []
            for i in range(n):
                await limiter.acquire()
                text, _ = await call_text(system, user, model=model)
                outputs.append(text)
                print(f"{prompt_name}/{config} {i + 1}/{n}", flush=True)
            result["cells"][f"{prompt_name}/{config}"] = {
                "distinct": len(set(outputs)),
                "sha8": [hashlib.sha256(o.encode()).hexdigest()[:8] for o in outputs],
                "chars": [len(o) for o in outputs],
                "outputs": outputs,
            }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def report() -> None:
    r = json.loads(OUT.read_text(encoding="utf-8"))
    print(
        f"Generator determinism probe — {r['model']} @ reasoning_effort={r['reasoning_effort']}, "
        f"n={r['n']} calls per cell, probed {r['probed_utc']} (one day, one model version)\n"
    )
    for cell, v in r["cells"].items():
        verdict = "byte-identical" if v["distinct"] == 1 else "NOT identical"
        print(f"  {cell:<22} {v['distinct']} distinct of {r['n']}  ({verdict})  {v['sha8']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if not args.report:
        asyncio.run(run(args.runs))
    report()


if __name__ == "__main__":
    main()
