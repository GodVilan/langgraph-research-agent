# BACKLOG

Things deliberately deferred, with the reason. Nothing here is committed work — this is the
list of what was considered and set aside, so that "we didn't build it" is distinguishable
from "we didn't think of it".

Opened during Phase 0. See [`AUDIT.md`](./AUDIT.md) and [`MIGRATION_MAP.md`](./MIGRATION_MAP.md).
v2.1 is [GodVilan/arXiv-Agent](https://github.com/GodVilan/arXiv-Agent).

---

## Deferred from v2.1 (dropped in the migration)

| Item | Why deferred | Would revisit when |
|---|---|---|
| `LiteratureAgent` — topic → themes → multi-section review (548 lines) | A second orchestrator duplicating retrieve→synthesise. Building the graph twice before either is measured. | v3.0 has shipped and the eval can score long-form output |
| `exporter.py` — `.docx` / `.tex` export | Only serves the literature-review feature above | Same as above |
| `MemoryConsolidator` — LLM-extracted facts/preferences/entities per turn | An extra unbudgeted LLM call per turn whose extraction quality was never measured, writing into a schema v3 does not carry | There is a metric that says whether consolidated memory improves answers |
| `ResearchMemory` — JSON note store | No eviction, no schema, last-write-wins on key collision | Superseded by checkpointed state; revisit only if a real cross-thread memory requirement appears |
| `trace_bibliography` tool | Finds "references" by semantic similarity over 512-token chunks and asks the LLM to extract citations from whatever comes back. Output never evaluated. | A real reference parser (e.g. GROBID) is in the pipeline — otherwise it should not exist |
| `pdf_uploader.py` — user PDF upload | A public API accepting arbitrary PDFs is an abuse surface not worth opening in Phase 5 | Auth exists on the endpoint |
| `enrich_with_semantic_scholar` | Unused by the agent; UI-only | A citation-count feature is actually requested |
| Streamlit UI (`app.py`, 1,946 lines) | v3 ships an HTTP API; a UI is not one of the five capability gaps | After v3.0 ships, if a demo surface is wanted |

## Reserved for the post-mortem (Phase 8)

Findings that are worth more as one principle than as the incidents that produced them.
Recorded here so they survive to `docs/POSTMORTEM.md` rather than staying scattered.

| Principle | Evidence |
|---|---|
| **An instruction is a request; a mechanism is a guarantee.** Where correctness depends on a model complying, the compliance has to be enforced by something that is not the model. | Learned twice, at opposite ends of the system. **Phase 2, defence side:** telling the model that text inside `<passage>` delimiters is data and not instruction did not stop delimiter-escape payloads; structural neutralisation of the retrieved text did. **Phase 4, generation side:** a drafting prompt banning "synthesis", "integrate" and "how can" in explicit hard rules still produced eight research proposals out of eight; a post-generation regex check rejects them. The prompt is still worth writing — it improves the odds — but it is a filter, not a boundary. |

## Deferred design choices

| Item | Why deferred | Revisit at |
|---|---|---|
| `Send`-based parallel fan-out over sub-questions | Changes the latency/cost profile in the same commit as the baseline it would be measured against; also invalidates the `refinement_count` reducer choice | Phase 6, as a measured variant experiment (MIGRATION_MAP §5.4) |
| Non-pickle chunk store (JSONL/parquet sidecar) replacing `*_meta.pkl` | Changing the index artifact in Phase 1 invalidates the v2.1 comparison | A point where the index can be rebuilt and re-baselined together (MIGRATION_MAP §5.7) |
| Reranker (cross-encoder) over the merged dense+sparse candidates | Phase 1 explicitly forbids improving retrieval before the baseline exists | After Phase 4's baseline is committed |
| `interrupt()` / human-in-the-loop approval before live arXiv fetch | Checkpointing makes it cheap, but it is not one of the five stated capability gaps | If Phase 5's cost ceiling proves insufficient |

## Known measurement gaps

| Gap | Note |
|---|---|
| `classify_section` precision unvalidated | Ordered substring match over the first 600 chars; "in the abstract" inside a methodology chunk classifies as `abstract`. If section filtering ships (MIGRATION_MAP §5.8 option 1), this needs a labelled slice. |
| Filtered vector search recall ceiling | `VectorStore.search` retrieves 200 candidates then post-filters by `allowed_paper_ids` (AUDIT §4.16). The recall cost of that ceiling on scoped queries has never been measured. |
| v2.1 has no reproducible baseline | AUDIT §5. v3 does not start from a measured number; Phase 4 establishes the first one. |

## Found during Phase 1

| Item | Why deferred | Revisit at |
|---|---|---|
| Follow-up query rewriting / coreference resolution | `retrieve` embeds the current question verbatim, so a follow-up like "What is its rank hyperparameter?" is retrieved without the antecedent resolved and returns nothing useful. Observed live on a three-turn thread. Fixing it means a rewrite step that reads `messages` — a new LLM call per turn, and a change to what gets retrieved, which would move the Phase 4 baseline. | Phase 6, as a measured variant. The graph answers "the context does not contain this" rather than fabricating, so the failure is safe, just unhelpful. |
| Real rate card for `gemini-3.5-flash-lite` | The pinned 2.5 model was retired mid-phase (DECISIONS D-012). Placeholder rates are marked `verified=False` and warn on use. | Before any cost figure is published |
| Run-to-run variance under fixed sampling | `gemini-3.5-flash-lite` ignores `temperature`, so v2.1's greedy-decoding determinism no longer holds. Eval needs repeated runs or a variance estimate rather than a single pass. | Phase 4 |
| Per-thread / per-day cost accumulator | D-013 made the budget strictly per-request. Phase 5's daily cost ceiling needs a separate accumulator. | Phase 5 |
| No score threshold on dense retrieval | An out-of-corpus query ("quantum chromodynamics lattice gauge theory") still returns 5 confident-looking hits at cosine ~0.86, because `IndexFlatIP` always returns *something*. Adding a floor changes retrieval behaviour and would move the baseline. | Phase 4 measures it via the unanswerable subset first |

## Carried into Phase 4 planning (logged Phase 1 review)

| Item | Why it matters | Act at |
|---|---|---|
| ~~Push v2.1's working copy~~ | **Done 2026-08-21** — published as `8d3e67f`. `AUDIT.md` citations were re-verified against it; all 20 findings still hold, only line numbers moved. | — |
| **Pin the v2.1 commit in every baseline** | v2.1 is a moving target: `react_agent.py` grew 631 → 760 lines between the Phase 0 audit and `8d3e67f`. A baseline that does not record the commit is not reproducible. | Phase 4 |
| **v2.1 baseline must use `gemini-3.5-flash-lite`** | Q2's rationale was holding the generator constant so the delta is attributable to orchestration. The forced 2.5→3.5 switch (D-012) preserves that **only if the v2.1 re-run uses 3.5 too**. Written down now so the constraint does not quietly lapse. | Phase 4 spec |
| **`dense_only` comparison arm** | v3 calls BM25 automatically when dense under-delivers; v2.1 called it only when the model chose to. That is a policy difference (D-015), so the fallback's contribution should be isolated from the orchestration delta rather than bundled with it. | Phase 4 |
| **Corpus diversity is a construction constraint** | 150 cs.LG papers all published 2026-05-28 is one day of one category. Multi-hop questions needing genuinely distinct papers may be hard to build, and the 15 unanswerable items need topics far enough outside that slice to be unambiguous. Discovering this halfway through QA generation would waste the effort. | Before QA generation |
| **Unanswerable subset must target the no-score-floor failure** | An out-of-corpus query returns 5 hits at cosine ~0.86 because `IndexFlatIP` always returns something. The unanswerable items should be designed to surface exactly this, not merely to be absent from the corpus. | Phase 4 |
| **Reset `max_notional_cost_usd` from measured data** | $0.05 was chosen against placeholder rates that understated cost ~3.6x. It is a round number calibrated on wrong inputs, not a derived one. | Phase 4 |
| **Variance estimate, not a single pass** | Determinism is unrecoverable (D-014): 5 runs of one prompt gave 5 distinct outputs at every setting. Every eval metric needs repeats or a stated variance. | Phase 4 |

## Found during Phase 2

| Item | Why deferred | Revisit at |
|---|---|---|
| Model-based injection classifier | Pattern matching leaves 8 documented gaps (paraphrase, encoded payloads, non-English, hypothetical framing). A classifier would cover them, at the cost of an LLM call per retrieved chunk — 5 extra calls per retrieval pass, against a budget where the whole query currently costs 3. | Phase 3, once Langfuse can measure what the calls buy |
| Multilingual injection rules | The rules are English-only; a French override passes cleanly. Either multilingual patterns or the classifier above. | With the classifier |
| ~~False-positive rate on the real corpus~~ | **Done in Phase 2** — pulled forward on review. `make screen-corpus`; results and retune in [`SCREEN.md`](./SCREEN.md). It found 175 chunks (3.24%) would have been quarantined, all legitimate; now 0.00%. | — |
| Recall cost of quarantine | A BLOCK hit drops a whole chunk. Whether that measurably hurts answer quality is unknown. | Phase 4 |
| Repeated live-probe runs | `make injection-live` is n=1 per case against a non-deterministic model. A rate needs repeats. | Phase 4's variance work |
| ~~Separate `warn` from `block` in the guardrail metric~~ | **Done (Phase 3)** — `arxiv_agent_guardrail_triggers_total` carries a `severity` label. | — |
| Recall check on the surviving WARNs | WARNs do not withhold anything today, but under `strict` they quarantine. If the corpus is ever extended with untrusted content, those 60 chunks become 60 quarantines. | Whenever corpus trust tier changes |

## Carried into Phase 4 (added at Phase 2 review)

| Item | Why it matters | Act at |
|---|---|---|
| **Measure variance of the metric, not the string** | D-014 established 5/5 distinct *outputs*, but five different strings can grade identically under a rubric judge. The variance that matters is of the score. Design the estimate as N repeats of the same item scored by the judge, reporting the standard deviation of the metric. | Phase 4 |

## Found during Phase 3

| Item | Why deferred | Revisit at |
|---|---|---|
| Token-bucket rate limiter in the agent path | v2.1 had one; v3 does not. The Gemini free tier allows **15 requests/minute**, and a batch of eval-construction calls hit `RESOURCE_EXHAUSTED` mid-run. `evals/ratelimit.py` paces the *construction* tooling, deliberately not the agent: a limiter inside `src/agent/llm.py` would inject sleep into the path Phase 4 measures p50/p95 latency on, turning a latency metric into a measurement of the limiter. | Phase 5, where a public endpoint needs real throttling and latency is reported per-request rather than as a batch figure |
| Trace sampling | Every run is traced. Free at this volume; a deployed instance with real traffic needs a policy and does not have one. | Phase 5, once there is traffic to sample |
| Prometheus + Grafana in `infra/` | Metrics are exposed; nothing scrapes them. Adding a stack now would be a dashboard nobody watches, on top of six Langfuse containers. | Phase 5, alongside the deployed service |
| First-class per-node cost attribution | Cost is attributed per model call by the handler and rolled up in trace metadata, not as a per-node cost field. | If Phase 4 needs per-node cost to explain a regression |
| ~~Reconcile Langfuse's cost estimate with ours~~ | **Done, ahead of Phase 4.** The gap was not a rate-card difference: `$0.01528` of Langfuse's total is duplicate root traces, and our own blended rate was impossible because the test suite was writing synthetic traces into the same project. `make reconcile-cost`, D-021. | — |
| Purge the contaminated `development` traces and recapture | 187 of the 214 traces in the store are synthetic runs from the test suite, and 5 more are orphan duplicate roots. `make budget` now classifies rather than merges them, so they no longer corrupt the table — and they are the evidence `make reconcile-cost` reproduces D-021 from, which is why they have not been discarded. | Start of Phase 4: `make langfuse-reset`, then capture a clean baseline window for the eval runs |
