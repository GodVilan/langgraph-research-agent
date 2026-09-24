"""Run v3 over the frozen eval set and persist everything a scorer or a judge will need.

Backs `make run-set`. One record per item, written after every item so an interrupted run
resumes rather than restarts:

    question, gold answer, gold chunk ids, gold support class,
    the agent's answer, whether it refused and why,
    every retrieved chunk — id, paper, score AND full text — and the sources projection,
    guardrail events, truncation flag and reason, usage, wall-clock latency,
    a status: completed | guardrail_blocked | truncated | error

Retrieved chunk *text* is persisted, not only ids. The rubric checks grounding against what
the agent retrieved, so a scorer without the text cannot score grounding, and a judge without
it turns every grounding disagreement into an information artifact rather than a judgment.

The agent runs on the Gemini free tier — $0 OpenAI — and that tier is 15 requests a minute.
The rate limiter is deliberately not in the agent path (that path is where latency is
measured), so pacing happens *between* items here: after each query, sleep long enough that
the calls it made fit the per-minute budget. Per-item latency is measured inside the query and
excludes that sleep.

Usage:
    python -m evals.run_set                      # all 69, resuming from evals/runs/
    python -m evals.run_set --only sp-001,ua-002 # a subset
    python -m evals.run_set --fresh              # discard the checkpoint and rerun everything
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evals.absence import load_eval_corpus_checksum
from evals.schema import EvalItem, EvalSet

log = logging.getLogger(__name__)

RUN_DIR = Path("evals/runs")
FREE_TIER_RPM = 15
QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "Too Many Requests")


class RetrievedRecord(BaseModel):
    chunk_id: str
    paper_id: str
    score: float
    section_type: str = "general"
    text: str


class ItemRun(BaseModel):
    """Everything about one (item, answer) pair. The scorer and the judge see the same record."""

    item_id: str
    stratum: str
    question: str
    gold_answer: str
    gold_chunk_ids: list[str]
    gold_support: str | None
    answer: str | None = None
    # Named for what it is. The agent state's `refused` is set only by the input guardrail;
    # a generator that says "the passages do not contain this" leaves it False. Persisting it
    # under the state's name misled the first reader of this file — a refusal count that
    # counted only guardrail blocks — so the run record names the flag by its cause.
    guardrail_blocked: bool = False
    guardrail_reason: str | None = None
    retrieved: list[RetrievedRecord] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False
    truncation_reason: str | None = None
    # The Langfuse trace this item wrote to, "" when tracing was off. What `make push-scores`
    # attaches judge scores to, and the reason a traced run has to be judged on its own
    # answers: a different run's scores belong to different traces.
    trace_id: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    status: str  # completed | guardrail_blocked | truncated | error
    error: str | None = None
    thread_id: str = ""
    ran_at: str = ""


class RunRecord(BaseModel):
    """The run as an artifact: what was run, on what, with what, and every item's result."""

    arm: str = "v3"  # the shipping configuration, or a comparison arm from ARMS
    arm_config: dict[str, str] = Field(default_factory=dict)
    set_path: str
    set_sha256: str
    corpus_sha256: str
    generator_model: str
    prompt_version: str
    started_at: str
    # What a reader can regenerate this from, and the one thing they cannot. Corpus, set,
    # prompt version and every per-item record are pinned above; the judge that later scores
    # these answers has no pinnable weights revision from any provider (D-030b), and the
    # artifact says so rather than leaving the reader to discover it.
    pins: dict[str, str] = Field(
        default_factory=lambda: {
            "corpus": "corpus_sha256 above",
            "eval_set": "set_sha256 above",
            "agent_prompt": "prompt_version above",
            "per_item_record": "question, gold, agent answer, retrieved chunk text — all below",
            "judge_weights_revision": "NOT PINNED — no provider exposes one; see D-030",
        }
    )
    items: dict[str, ItemRun] = Field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for run in self.items.values():
            out[run.status] = out.get(run.status, 0) + 1
        return out


# ── Comparison arms ───────────────────────────────────────────────────────────────────────
#
# Each arm is one configuration difference from the shipping system, applied through the same
# settings the system reads, so the run record's `arm` field is the only thing that differs.
# `dense_only` separates the BM25 fallback's contribution from the orchestration delta (D-015).
# `section_filter` turns on the feature AUDIT §4.15 found dead in v2.1, with a stated rule for
# choosing the section, because nothing in the graph chooses one and the arm needs a policy
# to measure. The rule is the arm's, not the agent's, and it is written here.
ARMS: dict[str, dict[str, str]] = {
    "dense_only": {"RETRIEVAL__SPARSE_FALLBACK_THRESHOLD": "0"},
    "section_filter": {"RETRIEVAL__ENABLE_SECTION_FILTER": "true"},
    # The embedding comparison: the same graph over an index built with the small sibling of
    # the frozen model (384-dim vs 1024-dim, ~130 MB vs ~1.3 GB). Runs only because the
    # three-run spread on factual Recall@5 was 0.000 — narrower than the 0.05 rule set before
    # the spread was known (docs/EVALS.md). Index: `make index` with the same two variables.
    "embedding_small": {
        "RETRIEVAL__EMBEDDING_MODEL": "BAAI/bge-small-en",
        "RETRIEVAL__INDEX_NAME": "BGEsmall_cs512",
    },
}

_SECTION_RULE: list[tuple[tuple[str, ...], str]] = [
    (
        ("result", "accuracy", "score", "benchmark", "evaluat", "perform", "error rate"),
        "evaluation",
    ),
    (("architecture", "dimension", "layer", "method", "approach", "algorithm"), "methodology"),
]


def section_for(question: str) -> str | None:
    """The `section_filter` arm's rule: a keyword map from question to section type."""
    lowered = question.lower()
    for keys, section in _SECTION_RULE:
        if any(k in lowered for k in keys):
            return section
    return None


def apply_arm(arm: str) -> None:
    """Set the arm's configuration in the environment before settings are read."""
    import os

    for key, value in ARMS[arm].items():
        os.environ[key] = value


def run_path(set_sha: str, tag: str = "") -> Path:
    """The run record for a set; `tag` distinguishes repeat runs for the variance estimate."""
    return RUN_DIR / (f"v3_{set_sha[:8]}{'_' + tag if tag else ''}.json")


def load_record(path: Path) -> RunRecord | None:
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_record(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def status_of(state: dict[str, Any], error: str | None) -> str:
    if error:
        return "error"
    if any(e.get("severity") == "block" for e in state.get("guardrail_events", [])):
        return "guardrail_blocked"
    if state.get("truncated"):
        return "truncated"
    return "completed"


async def run_one(graph: Any, item: EvalItem, options: Any) -> ItemRun:
    from src.agent.runner import new_thread_id, run_query

    thread_id = new_thread_id()
    started = time.monotonic()
    error: str | None = None
    state: dict[str, Any] = {}
    try:
        final = await run_query(graph, item.question, thread_id, options)
        state = dict(final)
    except Exception as exc:  # recorded as an errored item, never swallowed
        error = f"{type(exc).__name__}: {exc}"[:500]
    latency = time.monotonic() - started

    def dump(x: Any) -> Any:
        return x.model_dump(mode="json") if hasattr(x, "model_dump") else x

    return ItemRun(
        item_id=item.item_id,
        stratum=item.stratum.value,
        question=item.question,
        gold_answer=item.gold_answer,
        gold_chunk_ids=list(item.gold_chunk_ids),
        gold_support=item.gold_support.value if item.gold_support else None,
        answer=state.get("answer"),
        guardrail_blocked=bool(state.get("refused", False)),
        guardrail_reason=state.get("refusal_reason"),
        retrieved=[
            RetrievedRecord(
                chunk_id=c.chunk_id,
                paper_id=c.paper_id,
                score=c.score,
                section_type=c.section_type,
                text=c.text,
            )
            for c in state.get("retrieved", [])
        ],
        sources=[dump(s) for s in state.get("sources", [])],
        guardrail_events=[dump(e) for e in state.get("guardrail_events", [])],
        truncated=bool(state.get("truncated", False)),
        truncation_reason=state.get("truncation_reason"),
        trace_id=str(state.get("trace_id", "") or ""),
        usage=dump(state.get("usage", {})) or {},
        latency_s=round(latency, 3),
        status=status_of(
            {**state, "guardrail_events": [dump(e) for e in state.get("guardrail_events", [])]},
            error,
        ),
        error=error,
        thread_id=thread_id,
        ran_at=datetime.now(UTC).isoformat(),
    )


def report(record: RunRecord) -> str:
    """Every number the run report states, regenerable from the artifact."""
    import statistics
    from collections import Counter

    runs = list(record.items.values())
    calls = [int(r.usage.get("llm_calls", 0) or 0) for r in runs]
    latency = sorted(r.latency_s for r in runs)
    notional = [float(r.usage.get("notional_cost_usd", 0.0) or 0.0) for r in runs]
    tokens = sum(
        int(r.usage.get("input_tokens", 0) or 0) + int(r.usage.get("output_tokens", 0) or 0)
        for r in runs
    )
    events = Counter((e.get("kind"), e.get("severity")) for r in runs for e in r.guardrail_events)
    blocked = [r.item_id for r in runs if r.status == "guardrail_blocked"]
    p95 = latency[max(0, int(0.95 * len(latency)) - 1)] if latency else 0.0
    fastpath = [
        r.item_id
        for r in runs
        if any(e.get("kind") == "scope_keyword_fastpath" for e in r.guardrail_events)
    ]
    classified = [r for r in runs if r.item_id not in fastpath]
    lines = [
        f"run of {record.set_path} (set {record.set_sha256[:8]}, corpus "
        f"{record.corpus_sha256[:8]}) on {record.generator_model}, prompt {record.prompt_version}",
        f"items {len(runs)}  statuses {dict(Counter(r.status for r in runs))}",
        f"errors {[r.item_id for r in runs if r.status == 'error']}",
        f"truncated {[(r.item_id, r.truncation_reason) for r in runs if r.truncated]}",
        f"guardrail-blocked {blocked}",
        f"guardrail events {dict(events)}",
        # Both denominators, always. "5 refusals" reads differently over 69 items than over
        # the 44 the classifier actually judged: the keyword fast path *accepts* a question
        # without the classifier, so a fast-path event is a question the classifier never saw.
        f"scope: keyword fast path accepted {len(fastpath)}/{len(runs)}; the LLM classifier "
        f"judged the other {len(classified)} and refused {len(blocked)} of them "
        f"({len(blocked) / len(classified):.1%} of judged, n={len(classified)}; "
        f"{len(blocked) / len(runs):.1%} of all, n={len(runs)})",
        f"llm calls: min {min(calls)} median {statistics.median(calls)} max {max(calls)} "
        f"total {sum(calls)}",
        f"latency s (excludes pacing sleeps): p50 {statistics.median(latency):.1f} p95 {p95:.1f} "
        f"max {max(latency):.1f}",
        f"tokens {tokens:,}; billed $0.0000 (free tier); notional ${sum(notional):.4f} total, "
        f"${statistics.median(notional):.4f}/query median, ${max(notional):.4f} max "
        f"(ceiling {0.05})",
    ]
    return "\n".join(lines)


def is_quota_error(run: ItemRun) -> bool:
    return bool(run.error) and any(m.lower() in (run.error or "").lower() for m in QUOTA_MARKERS)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=Path("evals/datasets/phase4.json"))
    parser.add_argument("--only", default="", help="comma-separated item ids")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--report", action="store_true", help="summarise the recorded run")
    parser.add_argument(
        "--tag", default="", help="repeat-run tag (r2, r3 …) for the variance estimate"
    )
    parser.add_argument(
        "--arm",
        default="",
        choices=["", *ARMS],
        help="a comparison arm; sets the configuration and the tag",
    )
    args = parser.parse_args()
    if args.arm:
        args.tag = args.tag or args.arm
        apply_arm(args.arm)

    if args.report:
        evalset = EvalSet.read(args.set)
        record = load_record(run_path(evalset.sha256, args.tag))
        if record is None:
            raise SystemExit("no run recorded for this set")
        print(report(record))
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from src.agent.graph import build_graph, sqlite_checkpointer
    from src.agent.state import RequestOptions
    from src.config import get_settings
    from src.retrieval.service import RetrievalService

    evalset = EvalSet.read(args.set)
    settings = get_settings()
    out = run_path(evalset.sha256, args.tag)

    existing = None if args.fresh else load_record(out)
    if existing is not None:
        record = existing
        print(f"resuming {out}: {len(record.items)} items already recorded")
    else:
        record = RunRecord(
            arm=args.arm or "v3",
            arm_config=dict(ARMS.get(args.arm, {})),
            set_path=str(args.set),
            set_sha256=evalset.sha256,
            corpus_sha256=load_eval_corpus_checksum(),
            generator_model=settings.model_name(),
            prompt_version=RequestOptions().prompt_version,
            started_at=datetime.now(UTC).isoformat(),
        )

    wanted = {i.strip() for i in args.only.split(",") if i.strip()}
    todo = [
        i
        for i in evalset.items
        if (not wanted or i.item_id in wanted) and i.item_id not in record.items
    ]
    print(f"{len(todo)} items to run on {record.generator_model} (free tier, {FREE_TIER_RPM} RPM)")

    service = RetrievalService.load()
    # A separate thread database: eval threads must not mix with interactive ones.
    async with sqlite_checkpointer(
        RUN_DIR / f"threads{'_' + args.tag if args.tag else ''}.sqlite"
    ) as saver:
        graph = build_graph(service, checkpointer=saver)
        for n, item in enumerate(todo, start=1):
            started = time.monotonic()
            options = RequestOptions(
                section_filter=section_for(item.question) if args.arm == "section_filter" else None
            )
            run = await run_one(graph, item, options)
            if is_quota_error(run):
                # One retry after a full minute: the free tier's window is per minute.
                print(f"  {item.item_id}: quota error, waiting 60s and retrying once")
                await asyncio.sleep(60)
                run = await run_one(graph, item, options)
            record.items[item.item_id] = run
            save_record(out, record)
            calls = int(run.usage.get("llm_calls", 0) or 0)
            print(
                f"  [{n}/{len(todo)}] {item.item_id:8} {run.status:17} "
                f"{calls:2d} calls  {run.latency_s:6.1f}s"
            )
            # Pace so the calls just made fit the per-minute budget; latency excludes this.
            budget_s = 60.0 * calls / FREE_TIER_RPM
            elapsed = time.monotonic() - started
            if budget_s > elapsed:
                await asyncio.sleep(budget_s - elapsed)

    print(f"\nwrote {out}")
    print(f"status counts: {record.counts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
