# BACKLOG

Things deliberately deferred, with the reason. Nothing here is committed work — this is the
list of what was considered and set aside, so that "we didn't build it" is distinguishable
from "we didn't think of it".

Opened during Phase 0. See [`AUDIT.md`](./AUDIT.md) and [`MIGRATION_MAP.md`](./MIGRATION_MAP.md).

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
| Streamlit UI (`app.py`, 1,925 lines) | v3 ships an HTTP API; a UI is not one of the five capability gaps | After v3.0 ships, if a demo surface is wanted |

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
