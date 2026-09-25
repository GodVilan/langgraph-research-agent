"""Reconcile the Gemini bill against every token count this project recorded — on tokens.

    make gemini-reconcile            # needs the local Langfuse (make langfuse-up)
    make gemini-reconcile REPORT=1   # reprint from evals/runs/gemini_reconcile.json

The bill (docs/billing/gemini.json, hand-entered from Google Cloud Billing) says what was
consumed. This says how much of it any instrumentation saw, source by source, and names the
rest. Three kinds of number, never mixed:

* **billed** — the provider's record;
* **measured** — token counts the project recorded (Usage in run artifacts and checkpoints,
  Langfuse traces, the D-021 fixture, the self-judge arm's receipt), each source counted once;
* **sized** — an unrecorded activity's *per-unit* prompt size, computed from the code and the
  corpus (characters / 4, an approximation). Sizes are shown beside the gap to test a
  hypothesis about what the gap is. They are **never added** to the measured total and never
  used to "explain away" part of it: the gap stays one named, unmeasured quantity.

Billed input is two SKUs — uncached and cached — and Gemini's ``prompt_token_count`` (what
LangChain reports as ``input_tokens``) includes the cached part, so the comparison is against
their sum.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BILLING = REPO / "docs" / "billing" / "gemini.json"
OUT = REPO / "evals" / "runs" / "gemini_reconcile.json"
RUNS = REPO / "evals" / "runs"
CHARS_PER_TOKEN = 4  # approximation for sizing only; never used for a measured figure

LOCAL_LANGFUSE = {
    "LANGFUSE_HOST": "http://localhost:3000",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-1a1a1a1a-2b2b-4c4c-8d8d-3e3e3e3e3e3e",
    "LANGFUSE_SECRET_KEY": "sk-lf-4f4f4f4f-5a5a-4b6b-8c7c-6d6d6d6d6d6d",
}


def billed() -> dict[str, int]:
    rec = json.loads(BILLING.read_text(encoding="utf-8"))
    out: dict[str, int] = defaultdict(int)
    for s in rec["skus"]:
        out[f"{s['model']}:{s['kind']}"] += int(s["tokens"])
    return dict(out)


def billing_period_end() -> str:
    return str(json.loads(BILLING.read_text(encoding="utf-8"))["period"]["to"])


def run_artifacts() -> list[dict[str, Any]]:
    """Run artifacts inside the bill's period. A run started after it is on another bill — and
    from 2026-09-25, on keys in no-billing projects — so counting it as "measured" would divide
    two populations (D-021). The first version globbed every run and did exactly that the day
    after the period closed: 658,690 tokens of a deploy-key run moved the gap from 76% to 74%."""
    end = billing_period_end()
    rows = []
    for path in sorted(RUNS.glob("v3_de699d68*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if str(data.get("started_at", ""))[:10] > end:
            continue
        items = data["items"].values()
        rows.append(
            {
                "source": f"run artifact {path.name}",
                "activity": "eval runs",
                "calls": sum(int(i["usage"].get("llm_calls", 0) or 0) for i in items),
                "input": sum(int(i["usage"].get("input_tokens", 0) or 0) for i in items),
                "output": sum(int(i["usage"].get("output_tokens", 0) or 0) for i in items),
                "tokens_recorded": any("input_tokens" in i["usage"] for i in items),
            }
        )
    return rows


def langfuse_local() -> list[dict[str, Any]]:
    """Environments not already covered by run artifacts. `development` in the local store
    is exactly the traced run plus the three pinned runs — counted from their artifacts."""
    os.environ.update(LOCAL_LANGFUSE)
    from src.observability.langfuse import get_client

    client = get_client()
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    page = 1
    while True:
        batch = client.api.trace.list(page=page, limit=100).data or []
        for t in batch:
            u = (t.metadata or {}).get("usage") or {}
            row = totals[getattr(t, "environment", None) or "?"]
            row[0] += 1
            row[1] += int(u.get("llm_calls", 0) or 0)
            row[2] += int(u.get("input_tokens", 0) or 0)
            row[3] += int(u.get("output_tokens", 0) or 0)
        if len(batch) < 100:
            break
        page += 1
    activity = {"integration-test": "tests (make test-integration)", "span-loss-probe": "probes"}
    return [
        {
            "source": f"local Langfuse, environment {env!r} ({v[0]} traces)",
            "activity": activity.get(env, env),
            "calls": v[1],
            "input": v[2],
            "output": v[3],
            "tokens_recorded": True,
            "duplicate_of_run_artifacts": env == "development",
        }
        for env, v in totals.items()
    ]


def d021_fixture() -> dict[str, Any]:
    traces = json.loads((REPO / "data" / "traces_d021.json").read_text(encoding="utf-8"))["traces"]
    real = [t for t in traces if (t.get("total_cost") or 0) > 0]
    return {
        "source": f"D-021 fixture, the {len(real)} priced (real) traces of 214; the rest are the "
        "test suite's synthetic runs, which never reached Gemini",
        "activity": "CLI/dev (Phase 3)",
        "calls": sum(int((t.get("usage") or {}).get("llm_calls", 0) or 0) for t in real),
        "input": sum(int((t.get("usage") or {}).get("input_tokens", 0) or 0) for t in real),
        "output": sum(int((t.get("usage") or {}).get("output_tokens", 0) or 0) for t in real),
        "tokens_recorded": True,
        "may_overlap": "CLI checkpoint store (at most this row's size)",
    }


async def cli_checkpoints() -> dict[str, Any]:
    from src.agent.graph import sqlite_checkpointer

    path = REPO / ".checkpoints" / "threads.sqlite"
    seen: set[tuple[str, str]] = set()
    calls = inp = out = 0
    async with sqlite_checkpointer(path) as saver:
        async for cp in saver.alist(None):
            vals = cp.checkpoint.get("channel_values", {})
            usage, answer = vals.get("usage"), vals.get("answer")
            key = (cp.config["configurable"]["thread_id"], str(vals.get("question")))
            if answer is None or usage is None or key in seen:
                continue
            seen.add(key)
            calls += usage.llm_calls
            inp += usage.input_tokens
            out += usage.output_tokens
    return {
        "source": f"CLI checkpoint store .checkpoints/threads.sqlite ({len(seen)} turns)",
        "activity": "CLI/dev (Phase 1)",
        "calls": calls,
        "input": inp,
        "output": out,
        "tokens_recorded": True,
    }


def self_judge() -> dict[str, Any]:
    d = json.loads((RUNS / "spend_flash-lite-self.json").read_text(encoding="utf-8"))
    return {
        "source": "judge arm flash-lite-self receipt (spend_flash-lite-self.json)",
        "activity": "judge experiment (Gemini arm)",
        "calls": None,
        "input": int(d["prompt_tokens"]),
        "output": int(d["completion_tokens"]),
        "tokens_recorded": True,
    }


def sized_unrecorded() -> list[dict[str, Any]]:
    """Per-unit prompt sizes of activities that recorded no tokens, from code and corpus."""
    from evals.multihop import paper_text

    chunks = json.loads((REPO / "data" / "chunks_512.json").read_text(encoding="utf-8"))
    papers = sorted({c["paper_id"] for c in chunks})
    at60 = statistics.mean(len(paper_text(p)) for p in papers)
    at30 = statistics.mean(len(paper_text(p, 30_000)) for p in papers)
    # One multi-hop candidate: draft (2 papers @30k) + single-paper sufficiency (2 @60k) +
    # joint (2 @60k) + presupposition (up to 2 @60k) — evals/draft.py, evals/multihop.py.
    per_candidate = (2 * at30 + 6 * at60) / CHARS_PER_TOKEN

    v21 = json.loads((RUNS / "v3_de699d68_v21.json").read_text(encoding="utf-8"))["items"]
    v21_calls = sum(int(i["usage"].get("llm_calls", 0) or 0) for i in v21.values())
    # Lower bound only: every call sees at least the item's retrieved text once, averaged over
    # its steps — the ReAct scratchpad re-sends observations, so the real figure is higher.
    v21_floor = (
        sum(sum(len(r.get("text", "")) for r in i.get("retrieved") or []) for i in v21.values())
        / CHARS_PER_TOKEN
    )

    return [
        {
            "activity": "eval-set construction (drafting + necessity/presupposition checks)",
            "unit": "one multi-hop candidate, ~8 full-text calls",
            "tokens_per_unit": round(per_candidate),
            "units_recorded": None,
            "why_unrecorded": "evals/build_set.py, draft.py, multihop.py discard the Usage "
            "call_structured returns",
        },
        {
            "activity": "multi-hop necessity fixtures (make test-necessity / test-all)",
            "unit": "one suite run: 3 cases x 3 repeats of the same full-text checks",
            "tokens_per_unit": round(9 * 4 * at60 / CHARS_PER_TOKEN),
            "units_recorded": None,
            "why_unrecorded": "tests do not record usage",
        },
        {
            "activity": "v2.1 baseline run (69 items, same key)",
            "unit": f"the whole run, {v21_calls} calls recorded; tokens not exposed by v2.1",
            "tokens_per_unit": round(v21_floor),
            "units_recorded": 1,
            "why_unrecorded": "v2.1's agent exposes step counts, not token usage; "
            "figure is a floor (retrieved text sent once per item)",
        },
        {
            "activity": "probes without artifacts (guardrail 280 calls, generator 20, D-014, "
            "injection-live, local container smoke queries)",
            "unit": "all of them",
            "tokens_per_unit": None,
            "units_recorded": None,
            "why_unrecorded": "probe scripts store verdicts, not usage",
        },
    ]


def build() -> dict[str, Any]:
    sources = run_artifacts() + langfuse_local() + [d021_fixture(), self_judge()]
    sources.append(asyncio.run(cli_checkpoints()))
    counted = [
        s for s in sources if s.get("tokens_recorded") and not s.get("duplicate_of_run_artifacts")
    ]
    b = billed()
    billed_input = (
        b["gemini-3.5-flash-lite:input_uncached"] + b["gemini-3.5-flash-lite:input_cached"]
    )
    billed_output = b["gemini-3.5-flash-lite:output"]
    measured_input = sum(s["input"] for s in counted)
    measured_output = sum(s["output"] for s in counted)
    by_activity: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for s in counted:
        by_activity[s["activity"]][0] += s["input"]
        by_activity[s["activity"]][1] += s["output"]
    return {
        "billed": {
            "input_incl_cached": billed_input,
            "cached": b["gemini-3.5-flash-lite:input_cached"],
            "output": billed_output,
            "gemini-2.5-flash-lite": {
                "input": b.get("gemini-2.5-flash-lite:input_uncached", 0),
                "output": b.get("gemini-2.5-flash-lite:output", 0),
            },
        },
        "sources": sources,
        "measured_by_activity": {
            k: {"input": v[0], "output": v[1]} for k, v in by_activity.items()
        },
        "measured": {"input": measured_input, "output": measured_output},
        "gap": {"input": billed_input - measured_input, "output": billed_output - measured_output},
        "sized_unrecorded": sized_unrecorded(),
        "possible_overlap_tokens": next(s["input"] for s in sources if "may_overlap" in s),
    }


def report(r: dict[str, Any]) -> None:
    b, m, g = r["billed"], r["measured"], r["gap"]
    print("Gemini 3.5 flash-lite, 2026-08-19 to 2026-09-24 — tokens, not dollars\n")
    print(f"{'':44}{'input (incl. cached)':>22}{'output':>12}")
    print(f"{'billed (Google Cloud Billing)':44}{b['input_incl_cached']:>22,}{b['output']:>12,}")
    for act, v in sorted(r["measured_by_activity"].items(), key=lambda kv: -kv[1]["input"]):
        print(f"  measured: {act:34}{v['input']:>22,}{v['output']:>12,}")
    print(f"{'measured, all sources':44}{m['input']:>22,}{m['output']:>12,}")
    pi, po = g["input"] / b["input_incl_cached"], g["output"] / b["output"]
    print(
        f"{'GAP — no instrumentation saw it':44}{g['input']:>22,}{g['output']:>12,}"
        f"   ({pi:.0%} of input, {po:.0%} of output)"
    )
    print(
        f"\n(measured sources may overlap by at most {r['possible_overlap_tokens']:,} input "
        f"tokens: Phase 3 CLI traces vs the CLI checkpoint store)"
    )
    print(
        "\nUnrecorded activities, sized per unit from code and corpus (chars/4 — an estimate, "
        "not added to anything above):"
    )
    for s in r["sized_unrecorded"]:
        size = f"{s['tokens_per_unit']:,} tokens" if s["tokens_per_unit"] else "not sized"
        print(f"  - {s['activity']}\n      per {s['unit']}: {size}  [{s['why_unrecorded']}]")
    mh = next(s for s in r["sized_unrecorded"] if s["activity"].startswith("eval-set"))
    print(
        f"\nThe gap equals ~{g['input'] / mh['tokens_per_unit']:.0f} multi-hop candidates' worth "
        f"of prompt ({g['input']:,} / {mh['tokens_per_unit']:,})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report(json.loads(OUT.read_text(encoding="utf-8")))
        return
    result = build()
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    report(result)


if __name__ == "__main__":
    main()
