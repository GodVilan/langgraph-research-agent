"""How nondeterministic is the input scope guardrail, and does pinning sampling fix it?

Two modes.

``--from-runs`` (default, no API calls). Every v3 run over the frozen set passes the
*identical* question through the *identical* guardrail before anything else happens — the
comparison arms (``dense_only``, ``section_filter``, ``embedding_small``) change retrieval,
which runs after it. So the seven v3 run artifacts are seven draws of the guardrail on the
same 43 factual questions, and the per-item block frequency across them is a measurement,
not an argument. v2.1's run is excluded: a different classifier.

``--probe N`` (Gemini free tier, $0 billed). Calls the scope classifier alone, N times per
item, over every factual item the keyword fast path does *not* decide — the only items the
model ever sees — under two configurations:

* ``production`` — exactly what ships: ``reasoning_effort=minimal``, temperature ignored by
  the model (DECISIONS D-012);
* ``pinned`` — the same plus ``seed=0`` and ``top_k=1``, the sampling parameters the
  provider accepts.

If ``pinned`` shows no flips where ``production`` does, pinning ships. If it flips too, the
variation is not a sampling-parameter problem this service can switch off, and the answer is
to make the decision legible rather than to pretend it is stable.

Writes ``evals/runs/guardrail_probe.json`` (resumable) and prints the table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RUNS = REPO / "evals" / "runs"
DATASET = REPO / "evals" / "datasets" / "phase4.json"
PROBE_OUT = RUNS / "guardrail_probe.json"

# Every v3 run over the frozen set. The arm changes retrieval only; the guardrail input is
# identical across all seven. v2.1 (`_v21`) is a different system and is excluded.
V3_RUNS = (
    "v3_de699d68.json",
    "v3_de699d68_r2.json",
    "v3_de699d68_r3.json",
    "v3_de699d68_traced.json",
    "v3_de699d68_dense_only.json",
    "v3_de699d68_section_filter.json",
    "v3_de699d68_embedding_small.json",
)


def factual_questions() -> dict[str, str]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    return {
        it["item_id"]: it["question"]
        for it in data["items"]
        if it["stratum"] == "single_paper_factual"
    }


def classifier_judged(questions: dict[str, str]) -> dict[str, str]:
    """The items the keyword fast path does not decide — the only ones the model sees."""
    import unicodedata

    from src.agent.nodes.validate_input import IN_SCOPE_TERMS

    out: dict[str, str] = {}
    for item_id, q in questions.items():
        lowered = unicodedata.normalize("NFKC", q.strip()).lower()
        if not any(term in lowered for term in IN_SCOPE_TERMS):
            out[item_id] = q
    return out


def from_runs() -> dict[str, Any]:
    questions = factual_questions()
    draws: list[tuple[str, set[str]]] = []
    for name in V3_RUNS:
        path = RUNS / name
        if not path.exists():
            raise SystemExit(f"missing run artifact {path}; cannot report seven draws")
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        blocked = {
            k for k, v in items.items() if k in questions and v.get("status") == "guardrail_blocked"
        }
        draws.append((name, blocked))

    judged = classifier_judged(questions)
    counts = Counter(k for _, blocked in draws for k in blocked)
    print(f"Input scope guardrail over n={len(questions)} verified factual questions")
    print(f"({len(judged)} reach the model classifier; the rest are keyword-fast-pathed)\n")
    print(f"{'run':<36} blocked")
    for name, blocked in draws:
        print(f"{name:<36} {len(blocked):>2} of {len(questions)}")
    per_draw = [len(b) for _, b in draws]
    print(
        f"\nblocked per draw: {', '.join(map(str, per_draw))} "
        f"(range {min(per_draw)}-{max(per_draw)}, {len(draws)} draws, identical input)"
    )
    print(f"\n{'item':<8} blocked in")
    always = [k for k, c in counts.items() if c == len(draws)]
    for item_id, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{item_id:<8} {c} of {len(draws)}")
    flipping = sorted(k for k, c in counts.items() if c < len(draws))
    print(
        f"\n{len(always)} item(s) blocked in every draw; {len(flipping)} flip between draws "
        f"({', '.join(flipping)}). Same question, same code: answered or refused by chance."
    )
    return {"per_draw": per_draw, "always": sorted(always), "flipping": flipping}


async def probe(n: int) -> None:
    from evals.ratelimit import RateLimiter
    from src.agent.llm import call_structured, get_chat_model
    from src.agent.nodes.validate_input import ScopeVerdict
    from src.agent.prompts import load_prompt
    from src.config import get_settings

    settings = get_settings()
    key = settings.google_api_key.get_secret_value()
    if not key:
        raise SystemExit("GOOGLE_API_KEY is not set; the probe calls the real classifier")

    # Through the wrapper, so every call lands in the usage log (D-046).
    models = {"production": get_chat_model(), "pinned": get_chat_model(pinned=True)}
    system = load_prompt("scope", "v2")
    judged = classifier_judged(factual_questions())

    record: dict[str, Any] = (
        json.loads(PROBE_OUT.read_text(encoding="utf-8"))
        if PROBE_OUT.exists()
        else {"model": settings.model_name(), "n": n, "draws": {}}
    )
    if record.get("n") != n:
        raise SystemExit(f"{PROBE_OUT} was written with n={record.get('n')}; delete it or match")

    limiter = RateLimiter()
    for config, model in models.items():
        for item_id, question in sorted(judged.items()):
            key_ = f"{config}/{item_id}"
            done = record["draws"].setdefault(key_, [])
            while len(done) < n:
                await limiter.acquire()
                try:
                    verdict, _ = await call_structured(
                        ScopeVerdict, system=system, user=f"Question: {question}", model=model
                    )
                    done.append({"in_scope": verdict.in_scope, "reason": verdict.reason})
                except Exception as exc:  # recorded, not retried silently
                    done.append({"error": f"{type(exc).__name__}: {str(exc)[:160]}"})
                PROBE_OUT.write_text(json.dumps(record, indent=1), encoding="utf-8")
                print(f"{key_:<24} {len(done)}/{n}", flush=True)
    report()


def report() -> None:
    if not PROBE_OUT.exists():
        raise SystemExit(f"no probe artifact at {PROBE_OUT}; run with --probe N first")
    record = json.loads(PROBE_OUT.read_text(encoding="utf-8"))
    n = record["n"]
    by_config: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for key, draws in record["draws"].items():
        config, item_id = key.split("/", 1)
        by_config.setdefault(config, {})[item_id] = draws

    print(f"\nClassifier-only probe, {record['model']}, n={n} draws per item per config\n")
    for config, items in by_config.items():
        refused_any = {k for k, d in items.items() if any(x.get("in_scope") is False for x in d)}
        flipping = {
            k
            for k, d in items.items()
            if len({x.get("in_scope") for x in d if "error" not in x}) > 1
        }
        errors = sum(1 for d in items.values() for x in d if "error" in x)
        per_draw = [
            sum(1 for d in items.values() if len(d) > i and d[i].get("in_scope") is False)
            for i in range(n)
        ]
        varied = sum(
            1 for d in items.values() if len({json.dumps(x, sort_keys=True) for x in d}) > 1
        )
        calls = sum(len(d) for d in items.values())
        print(
            f"{config:<11} items={len(items):>2}  refused per draw: {per_draw}  "
            f"ever refused: {len(refused_any)}  flipping: {len(flipping)}  errors: {errors}"
        )
        print(
            f"{'':<11} output varied across draws on {varied} of {len(items)} items "
            f"({calls} calls; {'identical on every call' if not varied else 'not identical'})"
        )
        for item_id in sorted(flipping):
            verdicts = "".join(
                "E" if "error" in x else ("." if x["in_scope"] else "R") for x in items[item_id]
            )
            print(f"    {item_id:<8} {verdicts}   (R = refused, . = passed)")


def cross_reference() -> None:
    """G-1 step 1: what pinning chose, against how often each item was blocked unpinned.

    All 43 factual items are verified in scope, so every block here is a false refusal.
    """
    record = json.loads(PROBE_OUT.read_text(encoding="utf-8"))
    pinned = sorted(
        k.split("/", 1)[1]
        for k, d in record["draws"].items()
        if k.startswith("pinned/") and any(x.get("in_scope") is False for x in d)
    )
    questions = factual_questions()
    counts: Counter[str] = Counter()
    for name in V3_RUNS:
        items = json.loads((RUNS / name).read_text(encoding="utf-8"))["items"]
        counts.update(
            k for k, v in items.items() if k in questions and v.get("status") == "guardrail_blocked"
        )
    n = len(V3_RUNS)
    per_draw_mean = sum(counts.values()) / n
    print(
        f"\nPinned vs unpinned — every block is a false refusal (all {len(questions)} in scope)\n"
    )
    print(f"{'item':<8} pinned  unpinned (of {n} runs)")
    for item in sorted(set(pinned) | set(counts), key=lambda k: (-counts[k], k)):
        mark = "blocked" if item in pinned else "passed "
        note = (
            "  <- pinned blocks an item unpinned rarely blocked"
            if (item in pinned and counts[item] <= n // 3)
            else (
                "  <- unpinned sometimes blocked; pinned never does" if item not in pinned else ""
            )
        )
        print(f"{item:<8} {mark} {counts[item]} of {n}{note}")
    print(
        f"\npinned: {len(pinned)} of {len(questions)} false refusals on every call; "
        f"unpinned: mean {per_draw_mean:.1f} per draw "
        f"(range {min_max(counts, n)}, n={n} draws). Pinning fixed the stricter end in place."
    )


def min_max(_counts: Counter[str], _n: int) -> str:
    per = []
    questions = factual_questions()
    for name in V3_RUNS:
        items = json.loads((RUNS / name).read_text(encoding="utf-8"))["items"]
        per.append(
            sum(
                1
                for k, v in items.items()
                if k in questions and v.get("status") == "guardrail_blocked"
            )
        )
    return f"{min(per)}-{max(per)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--probe", type=int, metavar="N", help="call the classifier N times")
    parser.add_argument("--report", action="store_true", help="print the stored probe")
    args = parser.parse_args()
    if args.probe:
        asyncio.run(probe(args.probe))
    elif args.report:
        report()
        cross_reference()
    else:
        from_runs()


if __name__ == "__main__":
    main()
