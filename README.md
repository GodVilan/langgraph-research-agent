# arXiv Agent v3

A graph-orchestrated research agent over a 150-paper arXiv machine-learning corpus.

This is a rebuild of [arXiv-Agent v2.1](docs/AUDIT.md), which used a hand-rolled ReAct
loop. v3 keeps v2.1's retrieval unchanged and replaces the orchestration, guardrails,
observability, evaluation, and serving layers.

**Status: Phase 2 of 5.** The graph compiles, checkpoints, and terminates correctly under
the full test suite, and retrieval runs against the real index. There are **no answer
quality metrics yet** — the evaluation harness is Phase 4, and until it exists this README
makes no claim about how well the agent answers anything. See
[What is not built yet](#what-is-not-built-yet).

Every figure in [Repository facts](#repository-facts) is emitted by `make readme-stats`,
which measures the repo rather than trusting prose. Numbers stated outside that block are
either dated observations or linked to the command that reproduces them.

---

## Why a rebuild

v2.1 worked, but the audit that opened this project found problems that were structural
rather than incidental. The three that drove the decision:

**It had no reproducible baseline.** v2.1's README reported `MRR@5 = 0.990` and
`Context Precision = 1.000` from "100 benchmark QA pairs". That dataset was deleted in
v2.1's HEAD commit, and every one of its 100 `paper_id`s refers to a `2604.*` corpus with
**zero overlap** — by id and by title — with the `2605.*` corpus the project ships. The
corpus was replaced and the benchmark was orphaned. Those numbers cannot be regenerated,
so v3 does not carry them forward and does not claim improvement over them. Full evidence
in [AUDIT §5](docs/AUDIT.md).

**Its bounds were prompts, not structure.** Roughly 40% of the 630-line orchestrator was
loop-guard machinery — tracking repeated queries, counting duplicate top results, and
telling the model "⚠ You are looping" — because a free-tool-choice ReAct loop has no
structural bound. Worst case was ~43 LLM calls for one question with no cost ceiling
([AUDIT §4.7](docs/AUDIT.md)).

**Its guards failed open.** The scope check returned "in scope" on any exception, and the
critic returned a clean `pass` from a bare `except`. A critic that never ran and a critic
that approved were indistinguishable, and the CLI printed "✅ passed" for both.

The honest justification for the rewrite is checkpointing. v2.1 persisted *completed turns*
across 911 lines of hand-rolled SQLite; a run that died at sub-question 3 of 4 lost
everything. LangGraph's `SqliteSaver` persists state after every node, which is what makes
mid-flight resume possible at all.

---

## Graph

```mermaid
graph TD;
	__start__([__start__]):::first
	validate_input(validate_input)
	plan(plan)
	retrieve(retrieve)
	generate(generate)
	critique(critique)
	finalize(finalize)
	__end__([__end__]):::last
	__start__ --> validate_input;
	validate_input -.->|refused| finalize;
	validate_input -.->|ok| plan;
	plan --> retrieve;
	retrieve -.->|sub-questions remain| retrieve;
	retrieve -.->|plan exhausted| generate;
	generate --> critique;
	critique -.->|retry, budget ok| retrieve;
	critique -.->|pass / error / exhausted| finalize;
	finalize --> __end__;
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Regenerate from the compiled topology (this is the source of truth, not the drawing):

```bash
make graph
```

Every node is a coroutine returning a partial state update. Three conditional edges route,
and all three consult the budget guard. `finalize` is reachable from `validate_input` so
refusals, truncations, and successes leave through one exit with one response shape.

Two bounds operate together, tested independently:

- **the counters** — `plan_cursor` against `len(plan)`, `refinement_count` against
  `max_refinements`, plus token / cost / wall-clock / tool-call ceilings — are the intended
  bound;
- **`recursion_limit=25`** is the backstop that raises `GraphRecursionError` loudly if a
  counter is ever wrong. The longest legal path is 17 steps.

Design rationale, including a per-field reducer justification, is in
[MIGRATION_MAP §2](docs/MIGRATION_MAP.md).

---

## Setup

Python 3.13. The full dependency set is verified to resolve on 3.13/arm64
([DECISIONS D-011](docs/DECISIONS.md)).

```bash
make install
cp .env.example .env    # then add GOOGLE_API_KEY
make index              # ~5 min on MPS, 15-20 min on CPU
```

`make index` is required, not optional. `data/indices/` is a gitignored build artifact
rebuilt from the committed, checksummed chunks ([DECISIONS D-005](docs/DECISIONS.md)) — and
v2.1's prebuilt index cannot be reused, because its pickled sidecar references the module
path `rag.processing.chunker`, which does not exist here (`ModuleNotFoundError: No module
named 'rag'`).

### Is the rebuild faithful to v2.1's index?

Behaviourally yes, bitwise no, and the difference is measured rather than assumed:

```bash
python scripts/compare_index.py data/indices/BGE_cs512.faiss <other>/BGE_cs512.faiss
```

Measured 2026-08-20 against v2.1's artifact: rebuilding under newer `torch` /
`transformers` produced a file of exactly the same size with a different SHA-256, and **0 of
5,401 vectors bit-identical**. The divergence is float32 rounding noise (max element
difference 3.576e-07, minimum cosine similarity 0.99999988), and **top-5 and top-10
rankings were identical on all 500 probe queries**. `make index-verify` re-runs this check
against a fresh CPU build and exits non-zero on any ranking change.

```bash
make verify-corpus      # check data/ against data/CORPUS.sha256
```

### Ask something

```bash
.venv/bin/python -m src.cli query "What approaches address catastrophic forgetting?"
```

```bash
.venv/bin/python -m src.cli query "Compare LoRA and prefix tuning" --stream --verbose
```

Threads resume by id:

```bash
.venv/bin/python -m src.cli query "What about the memory cost?" --thread <thread-id>
```

---

## Repository facts

<!-- STATS:START -->
<!-- Generated by `make readme-stats`. Do not edit by hand. -->

| | | Regenerate with |
|---|---|---|
| Tests | 212, all passing | `make test` |
| First-party Python | 34 files, 3,709 lines under `src/` | `make readme-stats` |
| Papers | 150 (arXiv cs.LG, all published 2026-05-28) | `make corpus-info` |
| Chunks | 5,401 at chunk size 512 | `make corpus-info` |
| Mean tokens per chunk | 380.0 (whitespace tokens) | `make corpus-info` |
| Embedding | `BAAI/bge-large-en`, 1024-dim | — |
| Index | FAISS `IndexFlatIP`, 22,122,541 bytes (21.1 MiB) | `make index` |
| Sparse | Okapi BM25 over lowercased whitespace tokens | — |
| Committed corpus | `chunks_512.json` 16 MiB, `metadata.json` 268 KiB | `make verify-corpus` |

*Measured 2026-08-20.*
<!-- STATS:END -->

The source PDFs are not carried in this repo; the chunk file has the text. The corpus
checksum is asserted at eval startup and recorded in every baseline — precisely the record
v2.1 lacked when its corpus was swapped (see [AUDIT §5](docs/AUDIT.md)).

---

## What changed from v2.1

Retrieval components are **frozen** — chunker, embeddings, vector store, dense retriever,
BM25, and the merge rule are ported unchanged. Changing any of them would make the Phase 4
comparison uninterpretable ([DECISIONS D-002](docs/DECISIONS.md)).

What changed is everything around them:

| | v2.1 | v3 |
|---|---|---|
| Orchestration | 630-line ReAct loop, model picks tools by emitting JSON | Typed `StateGraph`, 6 nodes, 3 conditional edges |
| Tool choice | model's free choice, guarded by prompts | deterministic rule: dense → BM25 when dense under-delivers → live arXiv on opt-in |
| Loop bound | `AGENT_MAX_STEPS`, plus ~90 lines of loop-guard prompting | explicit counters + `recursion_limit` backstop |
| Cost control | none | token / cost / wall-clock / tool-call ceilings, per request |
| Scope guard failure | fails **open** — proceeds | fails **closed** — refuses, logs the event |
| Critique failure | returns `pass` | returns `error`, routes to finalize, flagged in the answer |
| Structured output | regex-strip fences, `json.loads`, permissive default | `with_structured_output` + 2 bounded repair attempts |
| Sources | recovered by regex from formatted tool output; 3 of 6 tools produced none | projected from typed state |
| Request scope | mutable attributes on a shared agent object | frozen `RequestOptions` in checkpointed state |
| Thread persistence | 911 lines, 7 SQLite tables, completed turns only | `AsyncSqliteSaver`, state after every node |
| Progress | `step_callback` | `astream_events` |

---

## Guardrails

Retrieved paper text is untrusted input. `fetch_arxiv` makes that concrete: the agent
downloads a PDF chosen by a *model-authored* query, extracts its text, and feeds it back
into its own context. v2.1 concatenated that text straight into the prompt with no
delimiter and no detector ([AUDIT §4.11](docs/AUDIT.md)).

Three layers, deliberately independent:

| Layer | What it does | Depends on foreseeing the attack? |
|---|---|---|
| **Structural** | Passage text cannot close, forge, or nest its own `<passage>` wrapper; invisible characters are stripped | No |
| **Instructional** | The prompt names the delimiter, states that passage content is data, and tells the model to *report* an embedded instruction rather than obey it | No |
| **Detection** | Pattern matching; BLOCK hits quarantine the chunk out of context, every hit becomes a `GuardrailEvent` | Yes |

Content fetched at runtime (`arxiv`, `upload`) runs in strict mode, where warnings become
blocks. Corpus chunks come from a committed, checksummed artifact and do not.

### Measured, not claimed

```bash
make injection-report   # detector rates on the adversarial corpus
make screen-corpus      # the detector against all 5,401 real chunks — no API calls
make injection-live     # what happens when the detector misses (needs an API key)
```

| | |
|---|---|
| Detected, whole adversarial corpus (n=36) | **28/36 (78%)** |
| Detected, targeted subset (n=28) | 28/28 — see the caveat below |
| Quarantined under `strict` (n=28 targeted) | 28/28 |
| **Real-corpus false positives, 5,401 chunks** | **0 quarantined (0.00%)**, 60 warned (1.11%) |

**The 100% on the targeted subset is close to meaningless** and is reported only so the
qualifier can be attached: those cases and the rules were written by the same author in the
same sitting, so the number measures internal consistency, not robustness. 78% is the
honest figure, and it is that high only because eight confirmed gaps were kept in the corpus
rather than quietly removed.

The number that actually constrains the design is the last row, and it came from
[`make screen-corpus`](docs/SCREEN.md) — a regex pass over all 5,401 committed chunks. The
first run of that screen found **175 chunks (3.24%) would have been quarantined, every one
of them legitimate academic text.** The cause was systemic: this corpus is 150 papers about
LLMs and agents, so they quote system prompts and tool calls constantly, and a detector keyed
on prompt-like language cannot be precise against papers about prompting. Severity is now
assigned by what a rule keys on — structural artifacts block, prompt-like language warns —
which took corpus quarantine to 0.00% with adversarial detection unchanged.

Four of the rules exist because of unplanned probing rather than design: after the rules
were written, twelve fresh evasions were tried and **ten got through**. Four were cheaply
fixable and are now pinned as regression cases. The rest — encoded payloads, paraphrase,
non-English, hypothetical framing — are documented in `tests/adversarial_corpus.py` with the
reason each is not, and pattern matching is treated as the *weakest* of the three layers
accordingly.

When detection fails, the other two layers still apply. All eight undetected injections were
driven through the live graph: all eight reached the model, and in all eight the agent still
answered the real question from the clean passage. One case named the embedded instruction
and refused it explicitly. That is one model, one prompt version, one question, no repeats,
against a non-deterministic model — an anecdote, not a rate.

---

## Tests

```bash
make check     # ruff + mypy strict + tests
```

The count is in [Repository facts](#repository-facts) and comes from pytest, not from this
sentence. They run in under a second because the whole graph is driven against a fake retrieval
service — no embedding model, no index, no network. Coverage is concentrated on the claims
this rebuild makes rather than spread for a percentage:

- **reducers** — that `merge_retrieved` dedupes (the refinement loop re-retrieves) and
  `merge_usage` sums rather than overwrites;
- **routing** — every conditional edge, including budget breach overriding a retry;
- **termination** — the counters stop a healthy run, and the `recursion_limit` backstop
  fires when a counter is deliberately broken;
- **fail-closed guards** — a scope check that cannot run refuses; a critique that cannot
  parse reports `error`, never `pass`;
- **checkpointing** — state persists before every node, threads are isolated, resume does
  not duplicate messages, and every state model survives serialisation as itself.

One test asserts a *wrong* answer on purpose:
`test_classifier_precision_is_known_to_be_imperfect` pins the section classifier
mislabelling a methodology chunk as `abstract`, so the limitation lives in the suite rather
than in a comment.

---

## What is not built yet

Stated plainly, because a reader should not have to infer it:

| | Phase | Status |
|---|---|---|
| Langfuse tracing | 3 | Guardrail events land in state; nothing exports them yet. |
| OpenTelemetry spans, Prometheus metrics | 3 | Not built. `Usage` carries the fields; nothing exports them. |
| Eval dataset, metrics, baseline, CI regression gate | 4 | Not built. **No answer quality number exists for v3.** |
| v2.1-vs-v3 comparison | 4 | Not run. |
| FastAPI service, Docker, deployment, load figures | 5 | Not built. |

Deferred design choices and their reasons are in [BACKLOG.md](docs/BACKLOG.md).

### Numbers this README does not report

There are no answer relevance, faithfulness, Recall@k, MRR@k, latency, or cost-per-query
figures here, because none have been measured. The `~16 LLM calls per query` design target
in [BUDGET.md](docs/BUDGET.md) is a target written down so Phase 4 can falsify it, not a
result.

---

## Cost

The agent runs on `gemini-3.5-flash-lite` free tier. OpenAI is reserved for the Phase 4
evaluation judge under a **$5 lifetime ceiling**; $0.00 has been spent.

The originally pinned `gemini-2.5-flash-lite` was retired for new API keys and returns 404,
found on the first live run ([DECISIONS D-012](docs/DECISIONS.md)). Its rates were briefly
used as a placeholder and understated cost by ~3.6x; every `PRICING` entry now carries a
`verified` flag and a source, and an unverified entry warns on every use.

**Determinism is gone, and pinning the thinking budget does not bring it back.** Running one
fixed prompt five times produced five distinct outputs at every working setting — the model
reports that it ignores `temperature` ("uses fixed sampling defaults"), and
`thinking_budget=0` is rejected outright. Reproduce with:

```bash
python scripts/determinism_probe.py --runs 5
```

Thinking is pinned to `minimal` anyway, because thinking tokens bill at the output rate and
`low` costs roughly 6x `minimal` on the same prompt ([DECISIONS D-014](docs/DECISIONS.md)).
Phase 4 must design a variance estimate rather than assume repeatability.

Because billed cost on a free tier is always zero, a cost ceiling checked against it would
never be exercised. `Usage` therefore tracks both the billed figure and a *notional* figure
pricing the same tokens at paid rates, and the ceiling checks the notional one
([DECISIONS D-004](docs/DECISIONS.md)). Any published cost figure states which it is.

---

## Documentation

| | |
|---|---|
| [AUDIT.md](docs/AUDIT.md) | v2.1 module inventory, ReAct control-flow trace, 20 catalogued latent problems, and the benchmark integrity finding |
| [MIGRATION_MAP.md](docs/MIGRATION_MAP.md) | v2.1→v3 concept mapping, `AgentState` field by field with reducer justifications, graph topology, and nine places a direct port would be a mistake |
| [DECISIONS.md](docs/DECISIONS.md) | Architecture decisions with cost and reversal conditions |
| [BUDGET.md](docs/BUDGET.md) | Spend ceiling, verified unit prices, allocation, and mandatory controls |
| [BACKLOG.md](docs/BACKLOG.md) | What was deliberately deferred, and why |

---

## License

MIT. See [LICENSE](LICENSE).
