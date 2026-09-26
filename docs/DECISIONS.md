# DECISIONS

Architecture decisions with their reasoning. Each entry states what was decided, why, what
it costs, and what would reverse it. Newest last.

Related: [`AUDIT.md`](./AUDIT.md) (what v2.1 does), [`MIGRATION_MAP.md`](./MIGRATION_MAP.md)
(the plan), [`BUDGET.md`](./BUDGET.md) (spend tracking).

v2.1, the system being rebuilt, is [GodVilan/arXiv-Agent](https://github.com/GodVilan/arXiv-Agent).

---

## D-001 — Gemini Flash-Lite free tier runs the agent; OpenAI judges only

> **Superseded in part by [D-012](#d-012--the-pinned-generation-model-was-retired-mid-phase-pricing-now-carries-provenance):**
> `gemini-2.5-flash-lite` was retired for new API keys during Phase 1. The agent now runs
> on `gemini-3.5-flash-lite`. The rest of this decision stands unchanged.

**Decided:** every LLM call inside the graph — plan, generate, critique, scope classifier —
uses Gemini Flash-Lite on the free tier, routed through `init_chat_model`. OpenAI is
used **only** as the Phase 4 evaluation judge, and never as a generator, planner, critic,
guardrail classifier, or embedding model.

**Why:** the project has **$5 total** for OpenAI for its entire lifetime. AUDIT §4.7 puts
v2.1 at up to 43 LLM calls per query; the graph bounds this to roughly 6–16. Even at the
lower figure, one 100-pair eval run against an OpenAI generator would consume a meaningful
share of the budget, and Phase 4 needs dozens of runs across development, baselining, and
CI. Judging is the repeated cost worth paying for, because judge quality is what makes
every other number trustworthy.

**Cost:** free-tier content may be used to improve Google's products. Acceptable here
because the corpus is public arXiv text and no user data passes through it. Noted in the
README rather than buried.

**Reverses if:** the budget changes, or free-tier data-use terms become unacceptable.
Provider swap is a config change (`agent_model`), not a code change.

---

## D-002 — Retrieval components are frozen; the routing policy is not

**Decided:** `chunker.py`, `embeddings.py`, `vector_store.py`, `dense.py`, `bm25.py`, and
the merge rule in `router.py` are ported from v2.1 unchanged, and marked FROZEN in
`src/config.py`. What *did* change is the routing **policy**: v2.1 let the model pick
between `search_corpus`, `keyword_search`, and `fetch_arxiv` by emitting a JSON action;
`RetrievalService.retrieve` now decides by rule — dense always, BM25 when dense returns
fewer than `min(top_k, sparse_fallback_threshold)` hits, live arXiv only when both fall
short and the request opted in. The `min` matters: an absolute threshold fires the
"fallback" on every query whenever `top_k` is below it, which is what the first live run
did before it was caught.

**Why:** the whole point of Phase 4's comparison is to attribute the v2.1→v3 delta to
orchestration. If the embedding model, chunk size, index type, or BM25 tokenisation moved
in the same rebuild, the delta would be uninterpretable. The routing policy, by contrast,
*is* the orchestration — changing it is the experiment.

**Cost:** v3 inherits v2.1's retrieval limitations wholesale, including the 200-candidate
ceiling on filtered search (AUDIT §4.16). A reranker and any retrieval tuning are deferred
to `BACKLOG.md` until after the baseline exists.

**Reverses if:** never during Phases 1–4. After the baseline is committed, retrieval
changes become measurable improvements rather than uncontrolled variables.

---

## D-003 — `section_type` is classified at load time and ships disabled

**Decided:** `load_chunks(..., classify_on_load=True)` recomputes `section_type` from chunk
text on every load, and `RetrievalSettings.enable_section_filter` defaults to **`False`**.

**Why:** AUDIT §4.15 found three compounding problems in v2.1 — the filter had no call site
anywhere in the repository, the committed chunk cache predates the field so all 5,401
chunks deserialise as `"general"`, and the classifier's precision was never measured.
Classification is deterministic and text-only, so recomputing it at load repopulates the
field **without re-embedding anything**, leaving the FAISS index bit-identical and D-002
intact. Shipping the filter disabled means it is a measured feature or it is not a feature.

**Cost:** a small per-load CPU cost (~5,400 substring scans, negligible against loading a
22 MB index). The feature remains unavailable until Phase 4 scores it.

**Known limitation, carried openly:** the classifier is an ordered substring match over the
first 600 characters, so "Our proposed method, as stated in the abstract…" classifies as
`abstract`. `tests/test_retrieval.py::test_classifier_precision_is_known_to_be_imperfect`
asserts this wrong answer deliberately, so the limitation is recorded in the suite rather
than in a comment nobody reads.

**Reverses if:** Phase 4 shows section filtering helps, at which point the classifier needs
a labelled slice before the flag flips.

---

## D-004 — Budget ceilings are enforced against *notional* cost

> **Note 2026-09-24 (D-046).** The premise "on the free tier, billed cost is always `0.0`" was
> an assumption about the key, not a measurement: its project had billing enabled throughout and
> Google billed $7.60. Enforcing ceilings on notional cost was right for a different reason than
> the one given — a bill is known only after the fact — and `Usage` no longer computes a billed
> figure at all.

**Decided:** `Usage` carries both `cost_usd` (what we are actually billed) and
`notional_cost_usd` (the same tokens priced at paid-tier rates). `BudgetLimits.
max_notional_cost_usd` is checked against the notional figure.

**Why:** on the free tier, billed cost is always `0.0`. A cost ceiling checked against it
would be dead code — permanently satisfied, never exercised, and untested until the first
day it mattered. Pricing the same tokens at paid rates makes the ceiling a live, testable
guard now, and gives Phase 5's daily cost ceiling a real number to work with.

**Cost:** two cost fields instead of one, and a reporting obligation — any published cost
figure must say which of the two it is.

**Calibration note (added after D-012).** `max_notional_cost_usd = $0.05` was set while the
active model's rates were a placeholder carrying `gemini-2.5-flash-lite`'s numbers, which
understate `gemini-3.5-flash-lite` by roughly 3.6x. The ceiling was therefore calibrated
against figures about 4x too low. Corrected observed costs still sit an order of magnitude
below it ($0.00335 for a single-hop query against a $0.05 ceiling), so no change was made —
but the ceiling is a round number chosen against wrong inputs, not a derived one, and Phase
4 should reset it from measured per-query distributions. **Reset 2026-09-11 to $0.025** =
`max_llm_calls × max observed per-call notional` (16 × $0.00156) over the 69-query Phase 4
run; observed per-query max $0.0057. Derivation in docs/BUDGET.md.

**Prices** (re-verify against provider docs with a date before editing, never from memory):

| Model | Input /1M | Output /1M | Verified |
|---|---|---|---|
| `gemini-2.5-flash-lite` free tier | $0 | $0 | yes, provider docs 2026-08-19 |
| `gemini-2.5-flash-lite` paid | $0.10 | $0.40 | yes, provider docs 2026-08-19 |
| `gemini-3.5-flash-lite` free tier (**in use**) | $0 | $0 | yes, provider docs 2026-08-19 |
| `gemini-3.5-flash-lite` paid standard | $0.30 | $2.50 | yes, provider docs 2026-08-19 |
| `gemini-3.5-flash-lite` paid batch/flex | $0.15 | $1.25 | yes, provider docs 2026-08-19 |

Notional cost uses the **standard** paid rates, not batch: the agent serves interactive
requests, so batch pricing would understate what a paid deployment would actually pay.
Context caching is not available on the free tier for this model, and the output rate
includes thinking tokens — which is why the thinking budget is pinned (D-014).

Every `PRICING` entry carries `verified: bool` and a `source` string, and an unverified or
missing entry logs a warning on use. Reasoning/thinking tokens bill at the **output** rate
on both providers, and are tracked separately in `Usage.reasoning_tokens`.

---

## D-005 — The FAISS index is a trusted build artifact, not source

**Decided:** `data/indices/` is gitignored. `data/chunks_512.json` and
`data/metadata.json` are committed and checksummed in `data/CORPUS.sha256`. `make index`
rebuilds the index from the committed chunks. `VectorStore.load` unpickles the chunk
sidecar and is documented as a trust boundary.

**Why:** the index is 37 MB of derived data. Committing it puts the repo over GitHub's
50 MB warning for no benefit, since BGE-large runs locally and the rebuild is free and
repeatable. Keeping the *inputs* committed and checksummed is what prevents the failure
that orphaned v2.1's benchmark (AUDIT §5.2): the corpus changed, nothing recorded that it
had, and 100 QA pairs silently stopped referring to anything.

**Discovered during Phase 1, and the reason this matters more than expected:** the prebuilt
index copied over from v2.1 **cannot be loaded by v3 at all**. Its `_meta.pkl` pickles
`Chunk` instances under the module path `rag.processing.chunker`, which does not exist in
this repo:

```
ModuleNotFoundError: No module named 'rag'
```

So `make index` is not a convenience target — it is the only way to get a working index,
and shipping the copied artifact would have failed at first load.

**Cost:** a first-time contributor pays one index build (~5 min on MPS, 15–20 min on CPU)
before the CLI works. Documented in the README.

**Non-pickle chunk store** (JSONL/parquet sidecar) is in `BACKLOG.md`, deferred until the
index can be rebuilt and re-baselined together.

---

## D-006 — The index is behaviourally reproducible, not bit-reproducible

**Decided:** reproducibility is claimed at the level of **retrieval rankings**, not file
hashes, and the claim is backed by `scripts/compare_index.py` rather than asserted.

**Why:** the first rebuild answered this empirically. Rebuilding v2.1's index from the
committed chunks under this project's newer `torch` / `transformers` / `sentence-transformers`
produced a file of **exactly the same size (22,122,541 bytes) but a different SHA-256**.
Zero of 5,401 vectors were bit-identical.

Measured divergence (`python scripts/compare_index.py <new> <old>`):

| | |
|---|---|
| Bit-identical rows | 0 / 5,401 |
| Max absolute element difference | 3.576e-07 |
| Mean absolute element difference | 1.692e-08 |
| Cosine similarity | min 0.99999988, mean 1.00000000 |
| Identical top-5 ordering, 500 probe queries | 500 / 500 |
| Identical top-10 ordering, 500 probe queries | 500 / 500 |

The divergence is float32 rounding noise. **Rankings are unchanged at every depth
checked**, which is the property D-002's "frozen retrieval" claim actually depends on — so
that claim survives, but it is now a measurement rather than an assumption.

**Cost:** `make index-verify` cannot assert hash equality, so it compares rankings instead.
BGE inference on MPS and CPU also differ, so `--device cpu` remains the pinned path for
anything that needs to be reproduced exactly.

**What would falsify this:** a future dependency bump that moves rankings rather than
bits. `make index-verify` is the check that would catch it, and it exits non-zero on any
ranking difference.

---

## D-007 — `TypedDict` state with Pydantic values, not a Pydantic state model

**Decided:** `AgentState` is a `TypedDict`; every non-scalar value in it is a Pydantic v2
model.

**Why:** LangGraph nodes return *partial* updates. A Pydantic state model revalidates in
full on every merge, which turns a large `retrieved` list into a per-node revalidation cost
for no benefit. `add_messages` and custom reducers are declared via `Annotated[...]` on
`TypedDict` keys, which is the better-supported path. Validation still happens where
untrusted data actually enters — the API schemas and `with_structured_output`.

**Cost:** state keys are not validated at assignment. Mitigated by `initial_state()`
populating every key explicitly, asserted by
`tests/test_state.py::test_populates_every_key`.

---

## D-008 — Checkpoint serialisation must enumerate every state model

**Decided:** `checkpoint_serde()` builds a `JsonPlusSerializer` allowlist by reflecting
over every Pydantic model defined in `src/agent/state.py`, and the `AsyncSqliteSaver` is
constructed with it.

**Why:** this was a live bug found by a failing test, not a precaution. Without the
allowlist LangGraph silently deserialises every `RequestOptions`, `Usage`, `Critique`, and
`RetrievedChunk` back as a **plain `dict`**. Nothing fails at write time. The break only
surfaces on resume, when a node evaluates `state["request"].prompt_version` and gets
`AttributeError` on a dict — which is to say, it would have surfaced in Phase 5 against a
real deployed thread rather than here.

An earlier attempt using module-level tuples (`[("src.agent.state",)]`) silently did
nothing; the API requires `(module, ClassName)` pairs. Reflection is used rather than a
hand-written list so a newly added state model cannot fall off the allowlist —
`tests/test_checkpointing.py::test_allowlist_covers_every_model_defined_in_state` asserts
the two stay in sync.

---

## D-009 — `AsyncSqliteSaver`, not `SqliteSaver`

**Decided:** the checkpointer is `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`
(requires `aiosqlite`).

**Why:** not a preference. Every node is a coroutine, and the synchronous `SqliteSaver`
raises `NotImplementedError` on the async checkpoint API. The Phase 1 brief says
"`SqliteSaver` checkpointer"; this is the async member of the same family, and the
substantive requirement — SQLite-backed thread checkpoints replacing v2.1's manual
persistence — is unchanged.

---

## D-010 — `refinement_count` is incremented node-side, not accumulated

**Decided:** `refinement_count` uses last-write-wins with an explicit read-modify-write in
the `critique` node, rather than `operator.add`.

**Why:** it is a safety bound. Under `operator.add`, any double write — a node retry, a
replayed checkpoint — inflates the counter and terminates a healthy run early. A guard that
fires spuriously is as bad as one that never fires.

**Cost:** correct only while exactly one node writes the field and no fan-out touches it.
If `Send`-based fan-out is adopted (BACKLOG, Phase 6) this must be revisited. Recorded here
so the constraint travels with the decision.

---

## D-011 — Python 3.13 across all three environments

**Decided:** the project pins 3.13. Local venv, Dockerfile, and the CI matrix must all
match.

**Why:** requested, and verified rather than assumed. The full dependency set resolves on
3.13/arm64: `faiss-cpu 1.15.0`, `torch 2.13.0` (MPS available), `sentence-transformers
6.0.0`, `numpy 2.5.2`, `transformers 5.15.1`, `langgraph 1.2.11`, `langchain 1.3.15`. No
fallback to 3.11 is needed.

**Reverses if:** a wheel becomes unavailable on 3.13 — in which case all three environments
move to 3.11 together, never one at a time.

---

## D-012 — The pinned generation model was retired mid-phase; pricing now carries provenance

**Decided:** the agent runs on `gemini-3.5-flash-lite`. Every `PRICING` entry carries
`verified: bool` and a `source` string, and `Settings.pricing()` logs a warning when the
active model's rates are unverified or missing entirely.

**Why:** the first live run failed. `gemini-2.5-flash-lite` — the model the Phase 1 kickoff
pinned and supplied verified rates for — returns 404 for this API key:

> `This model models/gemini-2.5-flash-lite is no longer available to new users. Please
> update your code to use models/gemini-3.5-flash-lite`

It still appears in `models.list()`, so availability had to be established by calling each
candidate. `gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, and `gemini-flash-lite-latest`
all respond; `gemini-2.5-flash-lite` does not.

**Pricing, resolved.** Initially no verified rate card for the 3.x flash-lite line existed,
so none was invented: the entry carried 2.5's rates with `verified=False`, which kept the
notional ceiling live — falling back to `UNKNOWN_MODEL_PRICING` would have zeroed it and
silently disabled the guard, the exact failure D-004 exists to prevent — while making it
impossible to publish the number by accident. Rates were verified on 2026-08-19 and the
entry is now `verified=True` at $0.30 / $2.50 standard. The placeholder had understated
cost by ~3.6x; see the calibration note on D-004.

**The `verified: bool` mechanism stays permanently.** Every future rate entry carries it,
and an unverified or missing entry warns on every use. It cost nothing and it caught a 3.6x
error before any number reached a document.

**Two behavioural consequences worth recording:**

1. `gemini-3.5-flash-lite` **ignores `temperature`** ("uses fixed sampling defaults; the
   sampling parameter(s) temperature will be ignored"). v2.1 ran everything at
   `temperature=0.0`, so the determinism assumption inherited from it no longer holds.
   Phase 4 must account for run-to-run variance rather than assuming greedy decoding.
2. This is the second time in this project that an unpinned external dependency silently
   invalidated an assumption — the first being v2.1's corpus swap orphaning its benchmark
   (AUDIT §5.2). It is the argument for Amendment 4's reproducibility guards, now with a
   concrete precedent.

---

## D-013 — Per-request budget ceilings need an explicit reset

**Decided:** `Usage` carries a `reset` flag, set only by `initial_state`. `merge_usage`
treats a delta with `reset=True` as *replace* rather than *add*, and clears the flag on the
way out.

**Why:** found by a live three-turn thread, not by reading the code. `usage` is
checkpointed and `merge_usage` sums, so the second turn on a thread inherited the first
turn's spend: `llm_calls` went 3 → 6 → 9 across three turns. The ceilings documented as
*per-request* were therefore being enforced *per-thread*, and with `max_llm_calls=16` any
conversation would hit the cap after roughly five turns and truncate every answer after
that — a guard firing on healthy traffic, which is the failure mode D-010 objects to in a
different field.

After the fix the same three turns report `llm_calls=3, tool_calls=1` independently, while
the thread still accumulates its transcript (6 messages) and restores `RequestOptions` as a
real model.

**Cost:** cumulative per-thread spend is no longer tracked in state. If Phase 5 wants a
per-thread or per-day total — and its daily cost ceiling will — that is a separate
accumulator, deliberately not conflated with the per-request guard.

---

## D-014 — Thinking budget is pinned to `minimal`; determinism is not recoverable

> **Note 2026-09-24 (D-046) — the pin worked, and thinking was never the lever.** The bill:
> 89% of Gemini cost was uncached input, 9% output (thinking included), 2% cached input.
> Pinning `minimal` kept thinking off the bill; the cost lever is context size.
>
> **Rescoped 2026-09-24 (D-035, D-042).** "Not recoverable" was true for the knobs tried —
> temperature, reasoning effort, thinking budget. `seed=0, top_k=1` were not tried, and they
> make the output byte-identical for both the scope classifier and the generator.

**Decided:** `agent_reasoning_effort` defaults to `"minimal"` and is passed explicitly on
every call. It is never left at the model default.

**Why pinned:** thinking tokens bill at the **output** rate ($2.50/M), which makes thinking
the dominant cost term rather than context size. A provider-side change to the default
would silently multiply the bill with no code change and no signal.

**Why `minimal` specifically** — measured, not assumed. `scripts/determinism_probe.py
--runs 5`, one fixed prompt, 2026-08-20:

| Setting | Distinct outputs | Mean thinking tokens | Mean output tokens |
|---|---|---:|---:|
| default (unset) | 5/5 | 0 | 84 |
| `reasoning_effort=minimal` | 5/5 | 0 | 79 |
| `reasoning_effort=low` | 5/5 | 426 | 508 |
| `thinking_budget=128` | 5/5 | 388 | 469 |
| `thinking_budget=0` | — | rejected: `400 INVALID_ARGUMENT` | |

Three findings, one of which inverts the expected story:

1. **Determinism is not recoverable this way.** Every working configuration produced 5
   distinct outputs from 5 runs. The obvious cause is ruled out, so Phase 4 must design a
   variance estimate rather than assume repeatability. That was the point of running this
   before Phase 4 rather than during it.
2. **Thinking cannot be switched off.** `thinking_budget=0` is rejected outright.
3. **Raising the budget costs more, not less — and `thinking_budget` is not a cap.** Asking
   for 128 produced ~388 thinking tokens. Moving from `minimal` to `low` took output from
   ~79 to ~508 tokens, roughly **6x the output cost** for the same prompt. The cheapest
   configuration is also the current default; pinning it locks in that position rather than
   trading anything away.

**Cost:** none today — `minimal` matches the default's token profile. If a Phase 4 metric
shows answer quality suffering at `minimal`, the trade is then a measured one.

---

## D-015 — The BM25 fallback bug was introduced by the port, not inherited from v2.1

**Decided:** the fallback-threshold and merge-truncation fixes ship as-is. **No
bug-compatibility flag is needed, and D-002 is intact.**

**Why — checked against v2.1 source directly, because the answer determined whether the
Phase 4 comparison was still interpretable:**

1. **v2.1 has no automatic sparse fallback of any kind.** The only call site of
   `bm25.retrieve` in the whole repository is inside `_keyword_search`
   (`rag/agent/tools.py:43`), which is a *tool the LLM chooses*. There is no threshold, no
   fallback rule, and no `sparse_fallback_threshold` equivalent anywhere:

   ```bash
   grep -rn "fallback\|threshold" --include="*.py" rag/   # nothing retrieval-related
   grep -rn "bm25\.retrieve" --include="*.py" rag/        # one hit: tools.py:43
   ```

2. **v2.1 already truncates to `top_k`.** `SourceRouter.search` ends
   `return merged[:top_k]` (`rag/sources/source_router.py:65`). That line is ported
   verbatim into v3's `SearchRouter` and still truncates.

Both defects lived in `RetrievalService`, which is the **new policy layer** with no v2.1
counterpart. No ported component was touched, so no frozen behaviour changed and there is
nothing to be bug-compatible with.

**What *is* a deliberate behaviour change, and stays one:** v3 calls BM25 automatically
when dense under-delivers, whereas v2.1 called it only when the model chose to. That is a
routing-policy change, which D-002 explicitly does not freeze — it is the substance of the
rebuild and the thing Phase 4 is meant to measure.

**Consequence for Phase 4:** because the fallback is a policy difference rather than a
component difference, the comparison should carry a `dense_only` arm to isolate what the
fallback contributes, separately from the orchestration delta. Logged in `BACKLOG.md`.

**Reverses if:** a later reading of v2.1 contradicts the two greps above. Both are stated
here so the finding is checkable rather than trusted.

---

## D-016 — Injection defence is three independent layers; detection is the weakest

**Decided:** retrieved text is defended by three layers that do not depend on each other:

1. **Structural** — `neutralise()` replaces every delimiter-shaped token and strips
   invisible characters. Applied unconditionally at screening *and* again at render. A
   passage cannot close, forge, or nest the `<passage>` block that wraps it.
2. **Instructional** — `generate.v2.md` names the delimiter, states that passage content is
   data, and tells the model to *report* an embedded instruction rather than obey it.
3. **Detection** — `scan()` pattern-matches known injection shapes. BLOCK-level hits
   quarantine the chunk out of the context entirely; every hit becomes a `GuardrailEvent`.

Screening happens in `retrieve`, at the point untrusted text enters state, so a detection is
recorded even when a budget ceiling later stops the run before `generate` executes.

**Trust tiers.** Corpus chunks come from a committed, checksummed artifact. `arxiv` and
`upload` chunks arrive at runtime, selected by a *model-authored* query — the path AUDIT
§4.11 singles out. Runtime chunks run in strict mode, where WARN is promoted to BLOCK.

**Why detection is explicitly the weakest layer.** Pattern matching cannot enumerate the
space of injections, so it is defence in depth and a measurement instrument, not the
primary control. `make injection-report`, on a corpus of n=33 adversarial and n=8 benign
chunks:

| | |
|---|---|
| Detected, targeted subset (n=25) | 25/25 (100%) |
| Detected, whole corpus (n=33) | **25/33 (76%)** |
| False positives on benign text (n=8) | 0/8 (0%) |

**The 100% on the targeted subset should be read as close to meaningless.** Those cases and
the rules were written by the same author in the same sitting; the number measures internal
consistency. The honest figure is 76%, and it is only that high because eight cases were
deliberately kept in the corpus after being confirmed as gaps.

**Four rules exist because of unplanned probing, not design.** After the rules were written,
twelve fresh evasions were tried against them; **ten got through**. Four had cheap
principled fixes, now shipped and pinned as regression cases: whitespace collapsing (a
phrase split across newlines evaded everything), a letter-spacing shape rule
(`I g n o r e`), a conditional-AI-reader rule (`If you are an AI summarising this…`), and a
markup-comment rule. The remaining gaps are recorded in `tests/adversarial_corpus.py` with
the reason each is not cheaply fixable — encoded payloads need a decoder, paraphrase needs a
classifier, and the rules are English-only.

**What happens when detection fails — measured, `make injection-live`.** All eight
undetected injections were driven through the live graph with a stub retriever and a real
model. All eight reached the model; in all eight the agent still answered the real question
from the clean passage, and in one it named the embedded instruction explicitly and refused
to act on it:

> *(Note: The passage containing replay buffers also includes text attempting to direct a
> change in guidance regarding replay being obsolete, which is treated as an embedded
> instruction per system rules rather than actual evidence of obsolescence.)*

**Read that as an anecdote, not a rate.** One model, one prompt version, one question, no
repeats, against a non-deterministic model (D-014). It shows the instructional layer is
doing something when detection fails. It does not establish how often.

**Cost:** quarantine drops a whole chunk on a BLOCK hit, so a false positive costs recall.
The benign corpus exists to keep that honest and currently reports 0/8 — including a paper
that *describes* prompt injection, which a naive keyword detector would quarantine. A chunk
that quotes a literal injection verbatim will still be quarantined; that trade is
deliberate.

**Reverses if:** the false-positive rate rises on real corpus text, or Phase 4 shows
quarantine measurably hurting recall. Both are measurable; neither is yet measured on the
real 5,401-chunk corpus, only on the 8-chunk benign set.

---

## D-017 — Severity is assigned from corpus evidence, not intuition

**Decided:** injection rules that key on **structural artifacts** are `BLOCK`; rules that key
on **prompt-like language** are `WARN`. `WARN` still blocks under `strict`, which applies to
runtime-fetched content. Full evidence in [`SCREEN.md`](./SCREEN.md).

**Why:** the first full-corpus screen found that **175 of 5,401 chunks (3.24%) would have
been quarantined, and every one was legitimate academic text.** Not mostly — inspection of
the top excerpts for all nine firing rules turned up no true positive.

The cause is systemic rather than a tuning miss. This corpus is 150 papers about LLMs and
agents; they quote system prompts, tool calls, and instruction text constantly, because that
is their subject. A detector keyed on prompt-like language cannot be high-precision on a
corpus of papers about prompting. `reveal_prompt` alone matched the bare word `token` across
39 chunks — `"output tokens"`, `"a 1024 output-token cap"`.

After the retune: **0 of 5,401 quarantined (0.00%)**, 60 warn (1.11%), with adversarial
detection unchanged at 25/25 targeted and 25/25 quarantined under `strict`.

**Cost:** a genuine injection sitting in the committed corpus would only warn, not
quarantine. Accepted because the corpus is a checksummed artifact we control, the threat
model is runtime-fetched content, and structural neutralisation applies unconditionally to
both.

**Reverses if:** the corpus is ever extended with content we do not control, at which point
corpus chunks stop being a trusted tier and should move to `strict`.

---

## D-018 — Whitespace normalisation is a detector-wide change, and needs an anchor audit

**Decided:** `re.MULTILINE` is not set on any rule. Anchored rules use `SENTENCE_START`, an
explicit boundary construct that survives whitespace collapsing.

**Why:** `scan()` collapses all whitespace before matching — the right fix for a phrase split
across newlines. But `system_role_claim` was anchored `^` under `re.MULTILINE`, and with no
newlines left, `^` degenerates to "position 0 of the chunk". The rule had gone silent on any
payload placed mid-chunk.

It was green in every test. It passed only because its corpus payload happened to start at
character 0 — and **neither the 33-case adversarial corpus nor the 12 fresh evasion probes
could see the regression, because every anchored payload had been written at position 0.**
Position-at-zero was a corpus artifact, not a property of real documents.

**What this generalises to:** a change to the normalisation step is a change to every rule,
not a local fix. Any future change there requires re-auditing every positional anchor, and
any rule with a positional anchor needs both a start-of-chunk and a mid-chunk case in the
corpus. The three mid-chunk variants added for this are permanent.

**Cost:** `SENTENCE_START` is a wider match than `^`, so anchored rules fire more often —
which is why `system_role_claim` was also downgraded to `WARN` under D-017.

---

## D-019 — The letter-spacing rule keys on the reconstruction, not the shape

**Decided:** a letter-spaced run is flagged only when de-spacing it yields an injection
keyword.

**Why:** as a bare shape rule, `(?:\b\w\s){6,}\w\b` fired on 33 corpus chunks, all
legitimate — `"Let x y z u v w t denote"`, `"A R T I C L E   I N F O"`, `"0 1 0 1 1 0 0 1"`.
Maths variable enumerations and letter-spaced headings are routine in PDF extraction, not
edge cases, and the rule was `BLOCK` with no `strict` gate.

Testing the reconstruction separates the two exactly: `"I g n o r e  a l l"` → `ignoreall`
fires; `"Let x y z u v w t"` → `xyzuvwt` does not. **A rule keyed on appearance blocks
papers; a rule keyed on what the text becomes blocks attacks.** Corpus hits 33 → 0, with
both spaced attack variants still caught.

**Two implementation subtleties, recorded because both were found by the rule failing:**
the reconstruction is matched against bare keywords rather than the main `RULES`, because
de-spacing destroys the word boundaries every rule depends on; and the scan runs on the
*collapsed* text, because a spaced phrase is separated by double spaces between words, which
otherwise splits it into fragments too short to reconstruct.

**Cost:** an attack that letter-spaces a phrase the keyword list does not cover passes. The
keyword list is a smaller surface than the rule set, and that is the trade for not
quarantining maths.

---

## D-020 — Never fold a provider's cost estimate into our own billed figure

**Decided:** `make budget` aggregates *only* traces carrying our own `usage` metadata. A
trace without it is counted as unattributed and excluded from the cost columns, never
back-filled from Langfuse's `total_cost`.

**Why:** the first run of the generated spend table reported **`$0.01528` billed on a free
tier**, with billed *above* notional — which is impossible, since notional prices the same
tokens at strictly higher paid rates. The cause was a fallback that used Langfuse's
`total_cost` when our metadata was absent. That figure is Langfuse's own estimate at its own
rate card, and folding it into a column labelled "billed" turned two different measurements
into one wrong number.

Corrected, the same 70 traces report `$0.00000` billed and `$0.01248` notional, with 13
traces declared unattributable because they predate the Phase 3 instrumentation.

**The dashboard still shows Langfuse's `$0.02776`, and that is fine.** It answers a different
question — what a generic rate card says these tokens are worth — and both figures appear in
`docs/OBSERVABILITY.md` with the difference explained. What is not allowed is one number that
silently means either.

> **Amended by D-021.** "They answer different questions" was too comfortable an
> explanation, and it let a real error sit unexamined for a phase. The `$0.01528` that this
> decision correctly refused to call "billed" is not a rate-card difference at all — it is
> five duplicate root traces of runs already counted. And the notional figure was being
> divided by a token count drawn from a different set of traces entirely. Both figures are
> now reconciled exactly; see D-021.

**Cost:** the spend table under-reports whenever instrumentation is missing. That is the
right direction to be wrong in, and the unattributed count makes the gap visible rather than
absorbing it into a plausible-looking total.

---

## D-021 — Cost and tokens must come from the same population of traces

**Decided:** a trace contributes to the token columns of the spend table only if it also
contributes to the cost columns. Traces are classified — priced / tokens-but-no-cost /
no-usage-metadata — and the classes are reported separately, never summed together.
`make budget` computes the blended per-1M rate and **refuses to write the table** when it
falls outside the configured rate card. `make reconcile-cost` checks our figure against
Langfuse's trace by trace and exits non-zero on any divergence.

**Why:** review caught that `$0.01248` notional over the `72.05K` tokens the dashboard
reported is a blended **`$0.173` per 1M** — below the `$0.30` input-only rate. A weighted
average of `$0.30` and `$2.50` cannot land under `$0.30`, so the number was not merely
suspicious, it was impossible.

Three hypotheses were offered: a stale placeholder rate, unattributed traces holding a
disproportionate share of tokens, or Langfuse's total inflated by the double-trace bug. The
diagnostic — print the tokens *our own* instrumentation counted and the rate it implies —
eliminated the first immediately and showed the real cause was none of them cleanly:

- **The calculator was correct.** For all six priced traces, the stored notional, a
  recompute from their own tokens at `$0.30`/`$2.50`, and Langfuse's independent figure all
  equal `$0.01248`. Three derivations, one number.
- **187 of 214 traces were synthetic.** `run_query` opens a Langfuse span unconditionally,
  and `ObservabilitySettings` is a *different* settings object from the `Settings` that
  `tests/conftest.py` isolated. Every test that ran the graph shipped a trace built from
  `tests/fakes.py` usage — round token counts, zero cost. The token columns summed all 214;
  the cost column summed the 6 that were priced.
- **`$0.01528` of the dashboard total is double-counted**, sitting on five orphan
  `LangGraph` roots from before the double-trace fix, whose `query` twins hold the metadata.

So the reviewer's second and third hypotheses were both real and both secondary; the
dominant term was test traffic in the production trace store. Restated on the priced
population, the blend is **`$0.4056` per 1M** — inside the rate card, where a mostly-input
workload belongs.

**The generalisation is worth more than the fix.** D-004 says never merge billed and
notional into one number. This is the same rule one level up: **never divide two numbers
that describe different populations.** A ratio silently asserts that its numerator and
denominator range over the same things, and nothing in the type system checks that. The
blended-rate bounds check is cheap precisely because it needs no ground truth — it only
needs the arithmetic to be possible.

**Root cause fixed, not just the symptom:** `tests/conftest.py` now disables observability
for the entire suite. Verified by running the 22 graph tests and confirming the trace count
in Langfuse was unchanged at 214.

**Cost:** the spend table now rests on 6 priced traces rather than an apparent 193, which is
a much thinner base and honestly labelled as such. Phase 4's eval runs will widen it. The
contaminated traces are deliberately *not* purged yet — they are the evidence for this
decision, and `make reconcile-cost` reproduces the finding from them.

**Reverse if:** a future Langfuse version attributes cost to the trace we consider canonical
rather than to a duplicate root, at which point the duplicate-root detector becomes dead
code — delete it rather than leave it asserting something no longer true.

---

## D-022 — A working local environment is not evidence of a working declared one

**Decided:** CI installs the project from a clean environment against `pyproject.toml` alone
and imports every module, on every push. Adding a dependency is not done until it is
declared.

**Why:** `langchain`, `aiosqlite`, `langfuse`, and the OpenTelemetry packages were all
installed in the local venv and all missing from `pyproject.toml`. Everything ran, every
test passed, and a fresh clone would have failed at import. This repo's central claim is
that every number in it is regenerable by a command in it — a claim that is void if the
commands only run on the one machine where the venv accumulated the right packages by
accident.

The failure is structural, not careless: `pip install X` mutates the environment the tests
run in, so the environment silently diverges from the manifest and every local signal keeps
saying green. No amount of care detects it, because there is nothing to notice. Only an
install from scratch can.

**Cost:** a slower CI job, and a real dependency-resolution failure now blocks a push rather
than surfacing at deploy. That is the trade being bought deliberately.

**Reverse if:** never. This one is close to free.

---

## D-023 — CI reports rather than blocks, and the README says so

**Decided:** `.github/workflows/ci.yml` runs on every push to every branch and gates
nothing. No branch protection, no required status check. The README states this in plain
words under "CI reports; it does not block".

**Why:** this repository is developed by committing to `main` directly — pull requests are
never opened (§2 of the working agreement). A required status check can only block a merge
that goes through a pull request, so adopting branch protection would mean adopting pull
requests, changing how the project is worked on in order to enforce a check.

That trade was not worth making mid-Phase-4, but the *silence* about it was the real
problem. A workflow file and a green run together imply enforcement. A reader who sees CI
in a repository reasonably assumes a red build stops a merge, and nothing here would have
corrected them. The rule this project runs on is that a reader must not have to discover a
limitation by reading the source, and an unstated non-enforcement is exactly that.

**This is the same defect twice already.** The first push of the workflow triggered zero
runs, because `push: [main] + pull_request` cannot fire on a branch in a repository that has
no pull requests. Once it did run, it revealed `test_docs` had been invoking
`.venv/bin/python` — absent on any runner — and `pytest.skip`ping its own failure, so the
README test-count guard had never executed in CI at all. A gate that cannot fire, a check
that skips its own failure, and a workflow nothing requires are three versions of one
mistake: *running is not blocking*.

**A fourth instance, from Phase 4.** The cross-citation rule in `evals/absence.py` eliminates
**0 of 979** candidate unanswerable items. That number is equally consistent with two
opposite readings — the corpus is clean, or the check cannot fire — and nothing about the
number itself separates them. It is the same shape as the three above: *not observing a
failure is not evidence the detector works.*

Resolved the way the others should have been, by making it fire on demand:
`verify_attribute_absent_in` takes a corpus argument, and
`tests/test_evals_absence.py::TestTheCrossCitationRuleCanActuallyFire` injects a chunk
attributing a benchmark to its anchor paper and asserts the elimination — through both the
method-name path and the weaker arXiv-id path — plus a negative case proving an unrelated
mention does *not* trip it. The zero is now a measurement rather than an absence of evidence.
(The real reason it is zero: all 150 papers were published on the same afternoon, so no paper
can cite another's results.)

**A fifth instance, and it extends the family.** `ConstructionReport.diagnosis()` fired
correctly and reported uninterpretably. Keyed on ratios — it required
`single_paper >= kept` before calling the drafter healthy — it labelled a pilot of 4 kept,
1 `single_paper`, 0 `neither_nor_joint` as *"Mixed cull reasons; no single dominant
cause."* That is the cleanest result the check can produce, reported as though it were
inconclusive.

The first four instances are checks that could not fire, skipped their own failure, or
gated nothing. This one detected correctly and then lost the signal on the way to being
read. **The family therefore covers reporting, not just detection**, and that is the version
most likely to recur in Phase 4, where every finding reaches a human through a report. A
metric computed correctly and summarised into "mixed" is indistinguishable from one that was
never computed.

Fixed by keying the diagnosis on *which* reasons are present rather than on their shares:
zero `neither_nor_joint` and zero `banned_phrasing` is healthy at any ratio. Both real runs
— the 7-of-8 failure and the 4-of-5 pilot — are pinned as tests, so the reporting layer has
regression cases the same way the detectors do.

**A sixth instance, and the worst of them.** The construction report printed
`banned_phrasing: 39`. Those 39 were lexical-overlap rejections: a helper hardcoded the
reason, so every cull site that used it reported under the wrong name.

The damage is not that a label was wrong. It is that **a check which had never fired once
appeared to be firing 39 times** — and the whole point of the live probe is to answer
"has this ever fired on real output?". A mislabel does not merely misinform; it *retires the
question*. Nobody probes a detector reporting 39 hits. This is the second instance in the
reporting layer and the first where the report actively suppressed the investigation that
would have caught it.

Fixed structurally, not by correcting the argument. The helper is **deleted** rather than
repaired: one callable from three sites with a default reason will eventually be called with
the wrong one. Every cull now constructs its record at the site that made the decision,
where the reason is not in question. `by_reason` raises rather than tallying any reason that
is not a member of the enum it groups by, and a `TOPIC_NOT_ABSENT` reason now exists because
"the term turned out to be present" had been borrowing `BANNED_PHRASING` as well — the same
bug twice in one function.

**A seventh instance, and the first where one function held three variants at once.**
`prompt_decision` in the verification CLI:

1. `default=ACCEPT` — an empty line, or a stray line from a paste, silently became "accept".
2. `answer.strip().lower()[:1]` — free text was truncated to its first character, so
   *"reject this"* was recorded as `r` and *"absolutely not"* as `a`. A verifier typing a
   sentence got a decision they never made.
3. No echo of the parsed decision, so neither of the above was visible while it happened.

Underneath all three, a fourth: multi-line notes were consumed one line per subsequent
prompt, shifting every answer after a paste one slot early — decisions filed as notes, notes
read as decisions, and the overflow falling through to the shell after the process exited.

A session recorded **22 accepts and 2 rejects against an intent of roughly 17 and 7**, which
turns a 28% disagreement rate into 8% — on the one number the exercise exists to produce.
The whole session was discarded; a partial re-run against a corrupted record is not
recoverable.

What makes this the family's worst case is not the count. Every earlier instance was a check
that failed to *observe* something. This one **manufactured observations that never
happened** and reported them with confidence. A detector that cannot fire under-reports; an
input handler that invents decisions produces a number that is precisely wrong and looks
precisely right.

Fixed: no default, strict membership against exactly `{a, e, r, s, q}` with re-prompting,
the parsed decision echoed back before advancing, notes read multi-line to an explicit
terminator, and the input buffer drained before every decision so nothing typed in answer to
one question can be read as the answer to another.

**An eighth instance, and it is the family closing a loop.** The two integration tests were
*failing* when Langfuse was down, because `get_client` constructs lazily and a stopped stack
looked healthy until the first request. Making them skip was the right local fix — an
unreachable backend is a missing precondition, not a failure of the code under test.

But it converted a noisy failure into a silent absence. Those two tests are the **only**
automated coverage of the trace path, in the layer where all four Phase 3 bugs were found by
running the agent rather than by the suite. `449 passed, 2 skipped` reads as green, and the
2 were the only thing testing the thing that has broken most.

So skipping beat failing and was still not enough: a test that never runs is a check that
never fires. CI now stands Langfuse up as a service and **fails if the integration suite did
not actually execute** — a skip in that job is an error. The local skip stays, because a
developer without Docker running should not be blocked; the enforcement belongs where the
claim is made, not where the code is written.

**A ninth instance, and a distinct class: the check ran on the wrong property.**

Every earlier instance was a check that *could not fire*, *skipped its own failure*, *gated
nothing*, *was applied to one stratum only*, or *reported uninterpretably*. Two failures found
in human verification were none of those. They fired, passed, and were wrong, because what
they measured was not what the item needed.

**Containment measured digits; multi-hop answers are qualitative.** The figure `2` appears in
3,772 of 5,401 chunks — 70%. `mh-006` asserted a *"lightweight 2D U-Net"* and a *"Graph
Convolutional Network"* against four gold chunks that were **entirely bibliography**, and
passed, because its only figure was `2`. Four of five rejected items had reference lists,
acknowledgements, coordinate plots or an XML prompt template as gold.

**The mutual reinforcement is more instructive than either defect alone.** Two components
independently chose the same wrong property, and each then concealed the other's error:

* the **selector** ranked candidate chunks by how many of the answer's *figures* they
  contained;
* the **check** validated gold chunks by whether they contained the answer's *figures*.

Neither is absurd in isolation. Together they form a closed loop that actively seeks out the
worst possible gold chunk. A reference list is among the most figure-dense text in a paper —
bracketed numerals, years, page numbers — so the selector *preferred* bibliographies, and the
check then confirmed them, because the very property that made them attractive to the selector
was the only property the check inspected. The measurement that exposed it: **the figure `2`
appears in 3,772 of 5,401 chunks — 70%.** A check requiring a gold chunk to contain `2` is not
a weak check; it is close to no check, and the selector was optimising to satisfy it.

The general lesson: when a selector and its validator share a scoring property, the validator
cannot audit the selector. They agree by construction, and their agreement looks like
confirmation. Any two components on the same axis need a third property to check against —
here, structural eligibility (a bibliography cannot support an architecture claim regardless
of what it contains) and per-paper contribution, neither of which is expressible in the
figure-density terms both original components used.

A third instance of the same defect surfaced while fixing these two: `unsupported_chunks`
measured figures while selection had moved to claims, so a chunk chosen for containing `CLAP`
was reported as contributing nothing and flagged five sound items. Same file, same wrong
property, found only because the artifact gate ran after the selector changed.

**Necessity measured answerability; the item needed truth.** `mh-009` stated that a paper used
a Franka arm in MuJoCo locomotion simulation; the paper used one in real-world experiments
only. Necessity returned "no single paper answers it, and the papers together do" for a claim
neither paper makes. Answerability and truth are different properties. Claim-level attribution
does not close it either, because every *entity* in that answer appears somewhere in the
papers — the falsehood is in the relation between them, which is why grounding is a model call
and not a regex.

**What this class costs to find.** Nothing automated found either one. Both required a human
reading gold chunks against answers, and `gold_chunks_exist: all present` passed every one —
that check reports id resolution, not containment, and its name invited the wrong reading.

**The general rule this family points at:** every detector needs a case that makes it fire,
written at the same time as the detector — *and* a case that makes its report readable. A
clean run proves nothing on its own; neither does a correct number nobody can act on; and a
number attributed to the wrong check is worse than no number, because it answers a question
that was never asked and buries the one that was.

**Cost:** an unreviewed red commit can reach `main`. Accepted because the author is the sole
committer and runs `make check` locally, but it is a real weakness and is named as one.

**Reverse when:** Phase 4's regression gate lands. A gate nobody is required to pass is a
much weaker claim than a gate that blocks, and the eval gate is the first check where the
difference genuinely matters — a silent regression is exactly what it exists to stop.
Revisit then, with pull requests and branch protection on both jobs.

**Correction (D-056, 2026-09-25).** The integration step described above — CI standing Langfuse
up and failing if the suite did not run — **never executed on GitHub.** It was added in
`eaa5e4b` (run #9), and every run from #9 failed at the unit-test step of the same job, so every
later step was skipped. The eighth instance's fix was itself an instance of this decision until
the workflow was split into independent jobs.

---

## D-024 — A test of a reimplementation proves the reimplementation works

**Decided:** the item check chain exists once, as ``evals.verify_items.classify_candidate``.
Every builder, the artifact gate, and the defect matrix delegate to it. A test asserts that
each caller does, and asserts *delegation* rather than the presence of reason names in
source — the earlier version looked for strings and broke the moment the checks were
correctly extracted into one function.

**Why:** the chain was written four times and diverged four ways.

* `draft_multi_hop` ran five checks.
* `build_factual` ran two, then four — never `banned_phrasing` or `false_premises`.
* `verify_dataset` ran a superset, and each side reported itself complete.
* `tests/test_check_matrix.py` ran **its own copy**, written in the test file.

The consequence is the part worth remembering: **the matrix passed green while the builder it
spoke for was missing two checks.** It could not have failed. A test of a copy establishes
that the copy works, and says nothing about the code in production — which is the same defect
as a source-inspecting test, one level subtler, because it looks like a behavioural test.

Three defects reached a written artifact through that gap and were found by the artifact gate
rather than by the matrix.

**Cost:** callers pass six arguments to one function instead of inlining the checks they care
about, and two of those arguments (`banned`, `leaked`) must be computed by the caller because
they need the drafter's own gold passage. That awkwardness is the price of there being one
chain.

**Reverse if:** never. The alternative is a convention that four implementations already
failed to follow.

---

## D-025 — A solved problem that fails to propagate is its own defect class

**Decided:** string matching happens in exactly one place, `evals/matching.py`, with two
named modes. Every call site delegates; none re-derives a boundary. `make audit-entities`
prints the whole surface the premise check asserts so borderline cases are reviewed as a
list rather than discovered one rejection at a time.

**Why:** `absence._mentions` learned word boundaries in Phase 4 after `Gram` matched inside
"n-gram" and falsely eliminated two candidate items. The lesson was written down. It did not
travel. Months later `false_premises` was still matching substrings, so **`cifar-10` matched
inside `cifar-100`** — a question asking what a paper reports on CIFAR-10, of a corpus that
only uses CIFAR-100, was not flagged as a false premise. It had been passing all 93 items.

Auditing every other matching site found a third: `_mentions_any`, the paraphrase screen
gating *all thirteen* unanswerable items, had a left boundary and no right one at all.

This is not the D-023 family. Those are checks that could not fire, skipped their own
failure, gated nothing, or reported uninterpretably. **This one fired correctly, in one
place, and the correctness stayed there.** Two correct implementations in two places is a
coincidence; three sites with three different boundary rules is what a convention produces
when it is written in prose rather than in one function.

**The audit also showed the sites were not simply right and wrong.** They answer different
questions, and the difference had never been stated:

| | preceding | trailing | because |
|---|---|---|---|
| `mentions_term` — "is this concept discussed?" | no alphanumeric | no **digit** | "minibatches" *is* batch size; `cifar-100` is not `cifar-10` |
| `mentions_name` — "is this exact thing named?" | no alphanumeric **or hyphen** | no alphanumeric | a question naming `Gram` is not satisfied by "n-gram" |

Each errs in the direction that is safe for its own question. In an absence check,
over-matching rejects an item — recoverable, visible as a shortfall. Under-matching
certifies a term absent that the corpus discusses, producing a **backwards item** whose
expected answer is a refusal while the corpus holds the answer. So term mode deliberately
permits hyphenated compounds; name mode forbids them, because there over-matching is what
lets a false premise through.

**Cost:** one more module, and two mode names a caller must choose between. That choice is
the point — it was previously made implicitly by whoever wrote each regex.

**Reverse if:** never. The alternative is a convention, and a convention is what failed.

---

## D-026 — A check written in response to a rejection is not a fix until it is called

**Decided:** the shared chain takes the clause-scoped inputs (`source_paper_ids`,
`gold_by_paper`), the drafter passes the drafted answer to the necessity check, and
`tests/test_check_matrix.py` asserts the *reason* each defect class produces. A check that
nothing calls fails the matrix rather than sitting in the module looking like coverage.

**Why:** the previous round's five multi-hop rejections produced two checks, each documented
in its own docstring as the fix for a named item. Neither ran.

- **`papers_contributing_nothing` was called from nowhere.** It was written to catch `mh-005`
  and `mh-011`, whose gold came from one of their two papers. It was imported by no builder,
  no verifier, and not by the chain. Dead code, described in prose as a guarantee.
- **The grounding check ran with an empty answer.** `MultiHopCheck.grounded`,
  `GROUNDING_SYSTEM` and `is_admissible` were added after necessity certified a hallucinated
  Franka/MuJoCo relation. `draft_multi_hop` then called
  `check_single_paper_sufficiency(question, [a, b])` — no answer — so `answer.strip()` was
  always empty and `grounded` always defaulted to `True`. It fired only in its own fixtures.

Both were reported to a human as closed. The next verification round found the same defect
rate on the set they were supposed to have protected, which is what a fix that does not run
predicts.

**Why it is not simply D-023's tenth and eleventh instances:** those are checks that ran and
measured the wrong thing. These never executed on real input at all, and the reason they
looked fine is that each had a *fixture* proving it worked. A passing fixture plus an
unreferenced function reads exactly like a passing fixture plus a wired-up one.

**Cost:** the chain now takes eight arguments, two of them only meaningful for multi-hop
items, and every caller must supply them. The alternative — a builder deciding which checks
apply to it — is what D-024 was written about.

**Third instance (2026-09-22): a submit that reported success without happening.** During the
account outage, two batch submissions (`section_filter` q1, `v21` q1) received a 401, wrote no
receipt file, and left the driver log reading "SUBMITTED". Nothing downstream would have
noticed: both arms would simply have been absent from the results, with a log line saying they
were queued. Same shape as the two checks above — the report and the reality diverged, and the
report was the optimistic one. The distinction worth keeping is that the 401 was external and
the false "SUBMITTED" was ours; only the second is a defect.

The fix is structural, not a log message: `write_receipt` refuses an info dict with no
`batch_id`, writes the receipt, then **re-reads it and compares the id**, raising if either
step fails. A submit that cannot produce a readable receipt cannot report success, because the
receipt is the handle a later `collect` needs — no receipt means the work is unaddressable even
if the provider queued it. `tests/test_judge.py::TestASubmitCannotReportSuccessWithoutHappening`
simulates a rejected submit and asserts it raises and leaves no receipt behind.

**Reverse if:** never.

---

## D-027 — A structural exclusion needs the same evidence as a metric

**Decided:** a chunk is a reference list when at least half its `[n]` markers are followed by
whitespace and a capital — the shape of an author name beginning an entry. Marker *counts* and
URL density are not used. The criterion's separation is measured and recorded: 318 chunks at
share 0.0, 218 at 1.0, 36 in between.

**Why:** `is_ineligible_gold` was introduced to stop bibliographies being selected as gold,
after four of five rejected multi-hop items had reference lists or an XML template as gold.
It worked, and it barred **670 chunks — 11.6% of the corpus — of which the majority were
ordinary prose, results tables and a license table.** Three of the false positives were named
by a human as the correct gold for items whose selector had been forbidden to see them:

| chunk | what it holds | why it was barred |
|---|---|---|
| `29628_0021` | "reduce the dimensionality from 1024 to 100" — the only chunk in its paper stating 1024 | 35 mid-sentence citations |
| `30232_0100` | "Language models used as starting checkpoints for maze training" | 4 markers in a Citation column |
| `29601_0047` | "Claude Haiku 4.5 costs 19.746 per 1,000 evaluations" | 8 pricing URLs |

So the fix for "gold selection picks bibliographies" *caused* "gold selection cannot pick the
right chunk", and the second failure was invisible because a chunk that is never a candidate
never appears in any report. The corrected rule excludes 290 chunks (5.4%).

**The URL rule is deleted rather than retuned, and measurement is the reason.** It survived one
round longer than the bracket rule on the grounds that no observed item had been harmed by it;
`sp-009` then lost the only chunk stating its answer. Marker density cannot separate the
classes either — a real bibliography runs 1.23 URL markers per 100 words and that results
paragraph runs 1.43 — so there was no threshold to move. A rule that cannot be made to
discriminate is removed, not tuned.

**The general form.** An exclusion is a check whose *rejections are never reviewed*, because
what it rejects leaves the pipeline before anything reports on it. That makes it the easiest
place in a system to be confidently wrong, and it needs a stated criterion, a measured
separation, and a fixture on each side — the same evidence a metric needs.

**Cost:** a bibliography keyed `[CJS12]` rather than `[8]` is no longer excluded. It would
still have to support one of the answer's claims to be selected.

**Reverse if:** a reviewer finds a bibliography chosen as gold. Then the criterion is wrong,
not the threshold.

---

## D-028 — What counts as evidence is a corpus measurement, not a shape

**Decided:** a figure is a claim when it appears in at most 91 of the corpus's 5,401 chunks.
The ceiling is derived from a stated coincidence budget — a pruned gold set holds at most three
chunks, and the chance that any one of them contains the figure by accident must stay under
5%: `1 - (1 - p)**3 < 0.05` gives `df <= 91`. `GOLD_SET_CEILING` is a premise, and
`make verify-dataset` fails an item whose gold exceeds it rather than letting the budget widen
silently.

**Why:** "a decimal, or three or more digits" was a guess at rarity and wrong by an order of
magnitude. It admitted `100` (581 chunks, 10.8%) as evidence on the same footing as `183,098`
(0 chunks). Three multi-hop items were grounded on figures of that kind: `0.1` (240 chunks)
matched Algorithm 1 pseudocode containing neither quantity the answer stated; `2.5` came from
the model name "Qwen 2.5 7B"; `3.2` came from "Llama 3.2" and matched a *section number* in the
other paper. This is the figure `2` appearing in 70% of chunks (D-023, ninth), one notch less
obvious and therefore live for a round longer.

**Rarity also subsumes the name problem without a stop list.** Version digits were excluded
only for single-token names like `Qwen3-4B-Instruct-2507`; the spaced form no pattern caught.
Both "Llama 3.2" and "Section 3.2" fail a rarity test for the same reason.

**This loosens claim extraction, which is the dangerous direction**, so it is paired with three
tightenings rather than shipped alone: parameter scales (`405B`, `1B`) become claims of their
own kind, attribution is scoped to the clause naming each paper instead of pooled across the
answer, and a clause left with no groundable value is a rejection.

**Cost, stated plainly.** Nine of 46 figures in the set stop being claims, and `0.001` is a
plausible learning rate rather than a commonplace. Four factual answers are a bare common
figure and nothing else — "0.001", "0.07", "1,000", "0.6/0.2/0.2" — which the first version
rejected as "nothing checkable at all". That is the short-answer failure a second time, aimed
at vacuous multi-hop support and landing on the most precise answers in the factual stratum.
Containment on a common figure is now required and reported as **weak**, not discarded.

**Reverse if:** the corpus changes. The threshold is a function of it and must be recomputed,
not carried forward.

**Addendum (2026-09-10) — the guard, and what it found on its first run.**
`tests/test_no_orphan_checks.py` asserts that every public function in
`evals/verify_items.py` has a call site in `evals/` or `scripts/` — tests excluded, because
a call from a test is exactly the false comfort this removes — and that `draft_multi_hop`
passes the drafted answer to the necessity check. The first run failed on two more orphans:
`claims_as_lists` (never used) and **`gold_chunk_supports`, the single-chunk containment
check, which six tests in `test_evals_gold_chunks.py` proved "can reject" while every builder
and the gate called `gold_chunks_support_jointly`.** The class of tests titled *the gold-chunk
check must be able to reject* was asserting it of dead code. Both deleted; the six tests now
run against the function that runs.

---

## D-029 — One rule, three surfaces: the paraphrase requirement fails the classifier, the retriever and the generator

**Decided:** nothing is changed in the drafter, the input guardrail, the retriever or the
generator before Phase 4's metrics are reported. The interaction is recorded here, measured on
each surface, and reserved for the post-mortem as its own class: **a failure that no component
produces alone, that each component's own validation could not see, and that only real
paraphrased input exposes — on three components at once.**

**The rule.** Eval construction (Phase 4, amendment 2) requires the drafter to paraphrase away
distinctive ML vocabulary — model names, dataset names, multi-word phrases — so that retrieval
cannot succeed on string matching. That is correct: without it Recall@k measures lexical
overlap, not retrieval.

**The three surfaces it failed on.** Every component downstream of the question keys on the
vocabulary the rule removes, and each was validated in isolation on input that still had it.

| surface | validated on | what the paraphrased question did to it | evidence |
|---|---|---|---|
| **Input scope classifier** (Phase 2) | 36 adversarial probes + 8 hand-written benign questions, 0 false refusals | refused it as off-topic before retrieval ran — *"What is the age of the female patient described in the clinical case example?"* | 5 of 43 factual items (11.6%, n=43); **17.9% of the 28 the LLM classifier judged** (15 were keyword-fast-pathed); `make run-report` |
| **Retriever** (v2.1, frozen, D-002) | 500 probes, top-5/10 rankings identical to v2.1 | did not return the gold chunk at k=10; often not the gold paper | 5 of the 7 hallucinated refusals in the human-scored 25, plus 3 of the 5 guardrail cases probed with the guard bypassed; `make bypass-probe` |
| **Generator** (Phase 1) | live runs, critique loop | had the fact in context, *quoted it*, and refused: *"The text discusses context lengths (such as 8192 for Llama 3.2 …) but does not provide information regarding text window sizes for smaller language versions."* Gold answer: 8192. | **1 item, sp-036**, found by hand-scoring; no automated check saw it |

**Why the split was measured, not argued.** It was possible that the refused items are simply
under-anchored — a question with no paper named and no ML term is arguably not a corpus
question — and that reading makes the headline smaller, so it could not be decided after
seeing the number. The five guardrail-blocked items were retrieved with the guard bypassed at
k=5 and k=10: two find the gold chunk at rank 2 (pure guardrail failures, floor 2/43 = 4.7%),
three fail retrieval (under-anchored). Then the seven human-labelled hallucinated refusals
were bucketed **by the rubric's own grounding criterion — was the gold fact in what the agent
retrieved** — not by gold-chunk-retrieved. That distinction is the whole finding on the third
surface: by the stricter criterion sp-036 is a retrieval failure and the only generation
failure in the set disappears. Coincidental figures (sp-047's gold `1,000` appears as `n =
1000` in two *other* papers' bootstraps) are reported with their source papers so they cannot
masquerade as grounding. `evals/runs/bypass_probe_hallucinated_refusals.json`.

**What this is not.** It is not a reason to loosen the paraphrase rule, the guardrail, or the
generator prompt before the metrics run; any change now tunes the eval to the system. It is a
Phase 5 input on all three surfaces: the scope check needs a benign set drawn from *the eval's
own questions* rather than written by hand; the drafter needs an anchor rule — paraphrase the
terms, keep a referent; and the generator's refusal has to be checked against its own context
before it is emitted, since it refused with the answer in its quotation.

**v2.1 refuses the same questions (2026-09-19).** Run on the same set and generator, v2.1's
own scope check short-circuited `sp-003`, `sp-014` and `sp-026` before retrieval — three of
the five v3 blocked. The surface is the paraphrased question, not one system's classifier.
And on retrieval the two systems miss the gold chunk on the same 30 of 43 factual items:
Recall@5 0.267 (v3, spread 0.000 over three runs) vs 0.233 (v2.1) is the paraphrase rule's
retrieval surface measured across the whole set, belonging to neither system.

**Both denominators, always.** "5 refusals" reads differently over 69 items than over the 44
the classifier actually judged (11.4%, n=44; 7.2%, n=69). On factual items: 17.9% of the 28
judged.

**Reversal condition:** if the judge-validated scores on the 25 show these items scored as
anything other than `hallucinated_refusal`, the rubric is wrong, not this record.

---

## D-030 — The judge decides Q1 blind, in a separate call; and the open-weights arm does not deliver the reproducibility it was chosen for

**Decided (a):** each judge arm scores an item in two calls. Stage `q1` carries the rubric's
Q1 section and the question plus agent answer — nothing else, asserted by
`tests/test_judge.py` the way the human's view is asserted: the payload contains no stratum,
no gold answer, no "expected behaviour", no chunk text. Stage `q3` runs only where the scoring
CLI asked Q3 — the judge said `answer` and the item expects one — and carries the full rubric
and the complete scorer's view with the judge's own Q1 verdict restated. The label is computed
by `rubric.outcome_for` in both cases; the judge never names one.

**Why:** a single call shows the stratum alongside the answer, and the rubric's Q2 table maps
expected x observed straight onto a label. A judge that reads "expected: answer" and sees a
refusal can emit `hallucinated_refusal` off a lookup without judging what the agent did — and
that pair, 17 of the 25 human labels, is the thing the experiment exists to measure. It would
score high while measuring nothing. The human scored Q1 blind; giving the judge an easier
task and comparing the two would measure the task difference, not the judge.

**Measured, not assumed (2026-09-18, `luna-low`).** On the cell the split was built to
protect — the seven items the human labelled `hallucinated_refusal` — the judge agreed on
**7 of 7**, having decided the behaviour from the answer text alone before the stratum fixed
the label. The two disagreements it did have (`ua-008`, `ut-003`) were the *other* direction:
explained refusals it read as answers under the rubric's own hedge sentence, a rubric
ambiguity rather than a lookup. The extra call earned itself.

**Cost:** two stages, sequenced — q3 cannot be submitted until q1 is collected — and a
second batch round-trip per arm. Not 2x the tokens: q1 is small and q3 runs on at most the 10
answerable items. `make judge-estimate`: 35 requests per arm, ~47K input tokens, ~$0.045 for
both paid arms together.

**Decided (b): the reproducibility argument for choosing `gpt-oss-120b` did not survive
contact with the API, and is struck — not softened.** Amendment 4 chose it on two grounds:
$0, and permanent reproducibility through a pinned weights revision (open weights over
deprecable API snapshots). Neither holds. What the Hugging Face router actually exposes
(`GET /v1/models/openai/gpt-oss-120b`, 2026-09-18): the model id, eleven serving providers
each with its own price and stack, and **no revision, commit or checkpoint field anywhere in
the response**. Which provider serves a call is a routing decision; the served model id is
recorded per response and asserted to match the request (`assert_served_model`), which
prevents silent substitution but does not identify the weights revision or the provider's
quantisation. **The arm runs anyway**, because the comparative question — does a cheaper
judge agree with the human as well as the reasoning one — survives intact and is the one
worth answering. The original wording ("free, pinned revision") is struck wherever it stood;
what stands is the limitation below.

**The unpinnable-judge limitation, stated in the form the artifact carries it.** Every other
input to a Phase 4 number is pinned: corpus `e1be96d1`, eval set `de699d68`, rubric sha
(stamped on every score sheet), prompt version, and the full per-item scoring record
(question, gold, agent answer, retrieved chunk text, every judge answer). **The judge's
weights revision is the single link that is not**, because no provider exposes one — not the
HF router for `gpt-oss-120b`, and not OpenAI for `gpt-5.6-luna`, which has no dated snapshot
alias in this account. Contrast with the Phase 0 finding about v2.1: those numbers were
unreproducible *and unlabelled* — a deleted dataset, zero paper overlap, no record of what
was measured. This one is reproducible on every dimension except one, and the artifact names
the dimension. That contrast is the point; without it the limitation reads as the same
disease.

Two further facts from the same probe, stated plainly:

* **No provider on the router lists this model as free.** Every one of the eleven carries a
  per-token price (from $0.037/$0.17 to $0.35/$0.75 per 1M). "Free" rests on Hugging Face's
  monthly inference credit, an account allowance, not on the model. Billed to HF, not to the
  $5 OpenAI cap; at the cheapest listed rate the 35 requests are ~$0.005.
* **Quota exhaustion fails loudly.** The router returns an HTTP error rather than routing to
  a different model, and `assert_served_model` would reject a response naming any other model
  regardless. The arm could not be run on 2026-09-18: the account's token returned
  `403 … does not have sufficient permissions to call Inference Providers` — a scope the
  token owner has to grant. That is the failure mode working as intended.

**The guard fired on real input (2026-09-18).** With the correctly scoped token the arm ran
seven stage-q1 items, then every call returned `402: You have depleted your monthly included
credits`. An HTTP error, no reroute, nothing scored by a model that was not asked for; the
seven verdicts were kept and the run resumes from the eighth. That is the fail-loudly
assertion proving itself in production use rather than in a fixture — the third guard in
this record to do so, after the orphan guard (`test_no_orphan_checks.py`, two orphans on its
first run) and the D-021 bound check (`make budget` refusing a table whose blended rate sat
outside the rate card). The arm is **paused at 7 of 25, reported as a partial, and no
agreement rate is computed from it**: the seven are all `unanswerable_attribute`, so the
`hallucinated_refusal` cell — the discriminating one — has 0 of 7 coverage and the arm has
not yet tested what it exists to test.

**Two facts from the successful responses.** Default routing selected **cerebras** at
$0.35/$0.75 per 1M — the most expensive input rate of the eleven providers, against a
cheapest of $0.037/$0.17 — so the router's default landed on the wrong side of a ~10x
spread. And the response carries a `system_fingerprint` (`fp_b546658c8e93d2e57ef2`), which is
a serving-stack fingerprint and not a weights revision; recorded as such. If the arm is
finished, the provider is pinned first (`openai/gpt-oss-120b:deepinfra`) and the spend record
names the serving stack, since pinning the provider changes it.

**The self-preference control, run before any further spend (2026-09-18).**
`gemini-3.5-flash-lite` — the generator itself — scored its own 25 answers under the same
two-stage prompt, $0. Result: **agreed with the human on 23 of 25**, `hallucinated_refusal`
**7 of 7**, `correct_refusal` 8 of 10 — and **agreed with `luna-low` on 25 of 25 with each other**, both arms
disagreeing with the human identically on `ua-008` and `ut-003`. Three consequences, stated
as measurements:

1. **Agreement in the 90s does not demonstrate independence on this sample.** A judge with
   maximal reason to favour the generator scored exactly as the third-vendor reasoning judge
   did. The 25 items are mostly easy to score once the rubric is applied; the experiment
   cannot separate an independent judge from a self-preferring one on them.
2. **No self-preference effect is visible at Q1.** The generator labelled all seven of its
   own hallucinated refusals as refusals. If same-model bias exists here it is smaller than
   one item in 25, which is the resolution of the sample.
3. **The two disagreements belong to the rubric, not to any model.** Two unrelated models
   read the hedge sentence the same way against the human. That settles where the amendment
   goes: the text, once, with a responsiveness criterion — after all arms land.

Reported as a **control**, not an arm; it is not counted toward the independence question,
because it cannot bear on it.

**`luna-medium` landed the same day: 23 of 25, `hallucinated_refusal` 7 of 7, the same two
misses.** All three complete sheets — the control, `luna-low`, `luna-medium` — agree with
each other on **25 of 25**. Reasoning effort at `medium` changed no label at ~1.6x the
output tokens of `low`; Amendment 4's low-vs-medium question is answered on this sample,
and the answer is that the sample cannot distinguish them. OpenAI spend for the whole
experiment so far: ≈$0.0057 billed at batch rates.

**And on the paid arms:** `gpt-5.6-luna` has **no dated snapshot alias** in the account's model
list (`gpt-5.5` has `gpt-5.5-2026-04-23`; the 5.6 variants do not), so "pin snapshot IDs" cannot
be satisfied for it either. The served id is recorded per batch and asserted per response.

**A D-023 recurrence inside a D-023 fix (2026-09-18).** The rewrite of
`test_necessity_fixtures.py` for N repeats — itself a fix for a D-023-family defect, a test
asserting one draw of a nondeterministic instrument — shipped with a `pytest.skip` that
swallowed "Event loop is closed" as "could not reach the model": a test skipping its own
failure, inside the fix for a test that could not fail honestly. Removed; the except is
narrowed to provider unreachability. One line, because the recurrence is the point.

**Reverse (a) if:** never — the asymmetry is structural. **Revisit (b) if:** the router or the
provider starts exposing a weights revision, or the weights are served from a pinned local
checkpoint, at which point the original rationale holds and this caveat is deleted.


---

## D-031 — Verify absence of what the question asks about, not of the seed term

**Decided:** an `unanswerable_topic` item's absence is verified against the *question's*
topic, not the term the item was drafted from. The transferable rule: **verify absence of what
the question asks about, not of the seed term.** Applied to the next set, not this one — the
frozen set is not edited, and `ut-003` stays in it with this record attached.

**Why:** `ut-003` was built on the verified-absent term *AlphaFold*. Its question asks *"What
protein structure prediction results are reported in this corpus?"* The corpus contains
OmegaFold predicting protein 3D structures as a feature extractor for drug–target interaction
(`2605.29926_0004`). The absence screen verified the seed term — correctly: AlphaFold appears
nowhere — and the question generalised above it to a topic that is partly present. The agent
answered: *"protein 3D structures are predicted using OmegaFold as part of a feature
extraction process … the retrieved passages do not report specific performance metrics."*
**The agent was right and the item was wrong.** The human scorer labelled it `correct_refusal`
on the second half of that sentence; the judge labelled it `hallucinated_answer` on the first
half; both were scoring an item whose expected behaviour was mis-specified.

**Same family as Hazard 1, inverted.** The false-absence hazard is a question paraphrased
*below* the verified term — "easy to hard" for "curriculum learning" — so the corpus has it
under a surface form the screen missed. This is a question generalised *above* the verified
term, so the corpus has the topic under a different name. In both, the screen verified a
string and the question asked about a concept, and the gap between them is where the item
broke.

**Consequence for reporting:** `unanswerable_topic` is now **effectively n=1 clean**
(`ut-004`, click-through rate). It was already a case study rather than a rate at n=2; every
place that framing appears now says n=1 clean of 2, and the stratum's result is reported per
item by name.

**It is the second time hand-scoring caught what automated checking passed.** `sp-036` (the
generator refusing with the answer quoted in its own refusal) and `ut-003` (an item whose
expected behaviour was wrong) each survived the full check chain, the artifact gate, the
defect matrix, and the freeze — eight rounds of automated verification — and each was found
by a human reading the agent's words against the corpus. That is not an argument against the
checks; it is the measured size of what they cannot see.

**Reverse if:** never; the rule is a strict strengthening of the absence screen.

---

## D-032 — The judge experiment closed: the question was answered as unanswerable on this sample, and `gpt-5.6-luna` at `low` ships because cost and pinnability decide

**Decided:** the judge is `gpt-5.6-luna`, `reasoning_effort=low`, via the OpenAI Batch API,
two-stage prompt (D-030a). **It did not win.** Agreement is undetermined among the arms —
three complete sheets agree with each other on 25 of 25 — so cost and pinnability decide,
and `low` is the cheapest of three indistinguishable options. `oss120b` and `gemma-4-31b` were
not finished: a fourth and fifth arm would confirm rather than inform.

**What the experiment was designed to measure, and what it measured instead.** Amendment 4
asked whether a free open judge is competitive with a paid one — judge independence from the
generator. What it measured is that **25 items scorable by a fixed rubric do not distinguish
any judge from any other, including one with maximal incentive to favour the system under
test**: the generator scoring its own answers (the control) produced the same 25 labels as
`luna-low` and `luna-medium`. That is a finding about the instrument, not a failure of the
experiment. The sample's resolution is one item in 25, and no judge differed from another by
even that.

**Two numbers, stated prominently.**

1. **Reasoning effort `medium` spent ~1.6x the output tokens of `low` and moved zero labels**
   (q1: 1,051 vs 1,000 tokens; q3: 267 vs 177; 25 of 25 labels identical with each other). A plausible
   assumption — more reasoning, better judging — measured and refuted at this sample size.
   Most people never test it.
2. **Total experiment spend: ≈$0.0057 billed against the $5 ceiling** (42,538 input / 2,495
   output tokens across four batches, plus two 2-request rescoring batches). Stated beside
   BUDGET.md's allocation table, whose line for this work had reserved $0.05.

**The rubric amendment, and what it cost.** All three judge sheets missed the same two items
(`ua-008`, `ut-003`) the same way — explained refusals read as answers under v1's hedge
sentence. Rubric v2 replaces the gloss with a criterion: *a fact stated in service of
explaining why the requested fact is absent is part of the refusal; a fact offered as the
answer is an answer; the test is responsiveness to the question asked.* Every sheet is
sha-stamped; the v1 sheets are archived beside the v2 ones; the two items were re-judged by
every arm under v2 (the human's labels were unchanged — v2 codifies the reading the scorer
applied). **A post-amendment agreement figure is never printed without the amendment in the
same line**, and `make judge-agreement` enforces that: a perfect cell containing a rescored
item carries "(after rubric v2 amendment; rescored […])" on the line itself. Before/after per
arm is in the report and in EVALS.md. A rubric that visibly changed and cost something is a
stronger record than one that was always right.

**Cost:** the shipping judge has no dated snapshot alias (D-030b); the served id is asserted
per response. **Reverse if:** a larger or harder human-scored slice separates the arms —
then this decision is remade on that evidence, and the control is run again first.

---

## D-033 — A tool in scope for the repo is not in scope for a directory the brief declares read-only

**Decided:** the v2.1 baseline harness refuses to run unless the clone is at the pinned commit
(`8d3e67f`) with a clean tree, checked on every run, with the check recorded in the run's
`arm_config`. `evals/baselines/` is excluded from ruff and mypy by configuration, not by
convention.

**Why:** the first `ruff --fix` after the clone existed rewrote **38 files** of published
`main` — import order, formatting, fixable lint — because the clone sat under `evals/`, which
ruff was configured to fix. Caught by a file count in the same command's output and reverted
before anything ran. Had it not been caught: a reformatted v2.1 still runs and still produces
numbers, and nothing in the output would have shown that the baseline compared v3 against a
modified system. The first constraint in the brief — v2.1 is read-only — would have been
violated silently by a tool doing exactly what it was configured to do.

**Not a D-023 instance.** Those are checks that ran and did not block, or measured the wrong
property. This is a *correct* tool with a *correct* configuration applied to a directory that
was never in its scope, with a consequence no output reports. The class is: scope declared
in prose (a brief, a README) is not scope enforced by the tools that act on the tree. Every
tool that rewrites files needs the read-only directory in its exclude list, and every consumer
of that directory needs to assert it is unmodified — because the exclusion can be forgotten
and the assertion cannot be.

**The guard fired on demand:** one comment appended to the clone's `rag/config.py` made the
harness exit with `v2.1 clone has local modifications; refusing to run a baseline against a
modified system: M rag/config.py`. Reverted; tree clean.

**Cost:** a run of the baseline cannot start with any local experiment in the clone, even a
harmless one. That is the point. **Reverse if:** never.

---

## D-034 — Five external dependencies changed under this project; the pattern is the finding

**Decided:** recorded as one pattern, not five incidents, and reserved for the post-mortem.
Nothing is changed in response — the point is that each was absorbed, and *why* each was
absorbable is the transferable part.

**The five, in order of occurrence.**

| # | What changed | How it surfaced | What it cost |
|---|---|---|---|
| 1 | **v2.1's benchmark corpus was swapped.** Its README reported `MRR@5 = 0.990` from 100 QA pairs whose `paper_id`s are all `2604.*`; the corpus that ships is `2605.*` — zero overlap, and the dataset was deleted in v2.1's HEAD commit | Phase 0 audit, reading the numbers against the data | v3 carries no v2.1 numbers forward and claims no improvement over them (AUDIT §5) |
| 2 | **`gemini-2.5-flash-lite` was retired for new API keys** mid-project, returning 404 | the first live run (D-012) | a model swap, and a ~3.6x pricing correction; every `PRICING` entry now carries `verified` and a source |
| 3 | **Cerebras pruned its free catalog** — a model the plan named was no longer served free | availability check before use | the judge experiment's arm list was re-costed |
| 4 | **Hugging Face's included inference credit was exhausted**, `402`, mid-arm at 7 of 25 | the fail-loudly guard, on real input (D-030b) | that arm paused as a partial; no rate computed from it |
| 5 | **The OpenAI account was deactivated** (`401 account_deactivated`) with ≈$0.022 total spend and no policy issue identified | every call after 2026-09-19 00:40Z | **four days blocked** on a third-party action; judging resumed 2026-09-22 |

**The pattern:** in a project whose whole value is reproducible numbers, the provider surface
is the least reproducible part of it — five distinct changes in one build, none of them caused
by anything in the repo, none foreseeable from the code. "Pin the version" does not cover it:
two of the five were the *provider* withdrawing something, one was the *account*, one was
credit, one was a dataset someone else deleted.

**What made the fifth survivable, specifically.** Worth naming because it is the part that
generalises:

* **Batch ids and submission times were archived** to `evals/runs/batch_<arm>_<stage>.json` at
  submit time, so four days later the work was addressable rather than lost. Of the three
  stranded batches, two were `completed` and one `expired` at 67 of 69 — and **all three output
  files were still retrievable**; the collector now reads an `expired` batch's partial output
  instead of refusing it, which recovered 67 paid verdicts.
* **The runs themselves were provider-independent.** Everything the judge scores — questions,
  gold, agent answers, full retrieved chunk text — is persisted by the Gemini-side run, so all
  the judge-free work (three repeat runs, `dense_only`, `section_filter`, the embedding arm,
  the v2.1 baseline) completed *during* the outage.
* **The judge had a measured substitute ready.** The `flash-lite-self` control had already
  scored the same 25 items and agreed with the human on 25 of 25 after the rubric v2 amendment
  — so a Gemini fallback was available and its agreement was known rather than assumed. It was
  not needed; having measured it is what made that a decision rather than a gamble.

**Reverse if:** never — it is a record, not a policy.

**Not in this record: the invisible failed submit.** Two submissions issued after 00:40Z
(`section_filter` q1, `v21` q1) 401'd, wrote no batch file, and left the driver log reading
"SUBMITTED". That is not an external-dependency failure — it is a step reporting success
without having happened, which is **D-026's class**, and it is recorded there.

---

## D-026, fourth instance — never infer an effect from a status code

**What happened.** Re-pushing eval scores to Langfuse required clearing the previous push
first, because `create_score` is not idempotent. The delete endpoint answers
**`202 Score deletion queued successfully`**. The code counted `202` as success, reported
"deleted 125 existing score(s)", and pushed immediately — producing **239 scores where 115
were intended**, with the dashboard averaging two generations of scores together. Nothing had
been deleted; Langfuse's worker drains that queue at roughly one score every two minutes.

**Why this instance is the instructive one.** It was committed **one turn after** the third
instance was recorded — the batch submit that 401'd, wrote no receipt, and left the log
reading "SUBMITTED". The lesson was written down, in this file, by the same author, about the
same class of mistake, and it did not prevent the next one. The reason is mechanical rather
than careless: **a 2xx reads like success at the call site.** `if resp.status_code in (200,
202, 204): removed += 1` looks like careful error handling. It is an inference about an effect
drawn from an acknowledgement of a request.

**The rule, stated so it can be applied rather than remembered: never infer an effect from a
status code — assert the effect.** A delete is done when the thing is gone. A submit is done
when the receipt is readable. A score is pushed when the server returns it.

**The audit this triggered**, over every place in the codebase that reads a status code or
calls `raise_for_status`:

| site | verdict |
|---|---|
| `judge.assert_model_available` (200 on `GET /models`) | a **read**, not an effect; the effect is separately asserted per response by `assert_served_model` |
| `judge.submit` file upload / batch create | the effect is asserted downstream: `write_receipt` re-reads the receipt, and `collect` reads the batch back by id |
| `judge.collect` / `judge_spend` / `push_scores` list calls | reads |
| `judge.run` chat completions | the response **body** is the effect and is parsed |
| `arxiv_client` fetch | a read |
| **`push_scores` → `lf.record_score`** | **the same defect, still live.** `create_score` not raising means the SDK accepted it, not that Langfuse ingested it — ingestion is asynchronous too. Fixed: the push now reads the scores back from the server and reports `verified N of M`, exiting non-zero when they disagree. |

That last row is the point of doing the audit: the class was present in a second place, in the
same file, and inferring from "no exception raised" is the same mistake as inferring from
`202`.

**Cost:** every push and delete now pays a read-back and up to 120s of polling.
**Reverse if:** never.

---

## D-021, applied to a metric — `retrieval_recall5` on the dashboard was not `Recall@5` in the docs

**What happened.** `scripts/push_scores.py` pushed a `retrieval_recall5` score for every item
except the two refusal strata — **56 items**. `evals/metrics.py` computes Recall@5 over items
whose gold chunks are *answer support*, which also excludes `ambiguous` — **46 items**, and the
published factual figure is over **43**. So the dashboard would have displayed a number named
"recall" over a different population from the one `docs/EVALS.md` calls Recall@5 (0.267), with
nothing on either surface saying they were different.

This is **D-021's rule one level up**: that decision was about never dividing two numbers that
describe different populations, in a cost table. This is the same error in a metric — two
populations under one name, on two surfaces, where a reader would reasonably compare them.
Ambiguous gold chunks are *competing referents*, not answer support; recall over them asks a
different question.

**Fix:** one predicate, `evals.metrics.has_answer_gold(stratum)`, used by the metrics report
and the score pusher. It cannot diverge again without failing both.

**How it was caught, which is the part worth keeping.** By checking a screenshot against the
documentation before committing it — the same artifact-verification discipline that produced
the D-021 reconciliation and the judge-format check in `docs/PHASE4.md` §4. **Third defect
found by that practice.** In each case the number was already in hand and looked fine; the
check was performed because publishing it was the next step.

---

## D-035 — The scope classifier ships with pinned sampling, and its decision is returned in every response

**The problem, measured.** Across seven v3 runs on identical input — r1, r2, r3, the traced
run, and three comparison arms whose changes all sit *after* the guardrail — the input scope
guardrail blocked **5, 3, 4, 4, 6, 3 and 5** of the same 43 verified factual questions. Three
questions were blocked in all seven; three (`sp-016`, `sp-021`, `sp-044`) flipped between
answered and refused with nothing changed (`make guardrail-variance`, no API calls). On a
public endpoint the same question can be answered or refused by chance.

**The brief offered three options. The measurement decided between them.** `make
guardrail-probe` calls the classifier alone over the 28 factual questions the keyword fast
path does not decide, 5 draws each, under two configurations (280 calls, Gemini free tier,
$0 billed):

| | refused per draw | verdict flips | items whose output varied |
|---|---|---|---|
| production (unpinned) | 5, 4, 5, 3, 5 | 3 of 28 | **28 of 28** |
| pinned: `seed=0, top_k=1` | 5, 5, 5, 5, 5 | **0 of 28** | **0 of 28 — byte-identical on 140 of 140 calls** |

**Decided — all three, not one of them:**

1. **Pin** the scope classifier (`get_chat_model(pinned=True)`, `PIN_SCOPE_CLASSIFIER=true`).
   Scoped to the classifier only: the generator, planner and critic stay unpinned, because
   changing them moves the Phase 4 baseline and that is a measured Phase 6 question, not a
   serving tweak (BACKLOG).
2. **Return the decision** in every response: `guardrail.stage`, `reason`, `deterministic`,
   and a `note` for the classifier stage (`src/guardrails/decision.py`). The classifier stage
   is still reported `deterministic: false` — byte-identical in a 140-call probe is a
   measurement, not a guarantee the provider makes, and a model update can move it.
3. **Document** the seven-draw range and the probe in the README's Known limitations.

**This corrects D-014 in part.** D-014 concluded "determinism is not recoverable" after trying
temperature (ignored), reasoning effort and thinking budget — five distinct outputs from five
runs at every setting. It never tried `seed` or `top_k`. For this call, they recover it. The
conclusion held for the parameters tested and was stated more broadly than they supported.

**What it costs.** The served system is no longer exactly the measured one: Phase 4's metrics
were produced unpinned, and `PIN_SCOPE_CLASSIFIER=false` reproduces that configuration. The
pinned classifier refuses a fixed 5 of 43 (`sp-003`, `sp-014`, `sp-016`, `sp-026`, `sp-044`),
inside the unpinned range of 3–6 — consistent rather than better, and still refusing the
paraphrased questions D-029 is about.

**Reverses if:** a repeat of `make guardrail-probe` shows the pinned configuration varying —
at which point the note's claim is withdrawn and the stage is legible only.

---

## D-036 — Live arXiv fetch is off on the public endpoint

**Decided:** `POST /query` has no `use_arxiv` field (unknown fields are rejected) and the
service always runs with `use_arxiv=False`. The CLI keeps the option.

**Why:** a live fetch makes the server download and parse PDFs on an anonymous caller's
behalf — outbound traffic and CPU the caller chooses — and every fetched paper is embedded
into a process-wide session index (`src/retrieval/session_index.py`) that later requests
search. That sharing was a documented cache on a single-user CLI; on a public endpoint it is
one caller writing into every other caller's retrieval. BACKLOG already declined PDF upload
until auth exists for the same reason.

**Reverses if:** the endpoint gains authentication, and the session index becomes per-request.

---

## D-037 — The daily cost ceiling reserves before it runs, and refuses to start on a ledger that forgets

**Decided:** a global daily ceiling on *notional* cost (`API__DAILY_NOTIONAL_CEILING_USD`,
$0.50), kept in a ledger separate from graph state (`src/api/ledger.py`). Each request
atomically **reserves** the per-request ceiling ($0.025) before the graph runs and **settles**
to its actual cost after. A deployed container **refuses to start** with `API__LEDGER=memory`,
with `sqlite` whose path is on the container's own root filesystem, or with a daily ceiling
below one reservation.

**Why reserve rather than check.** A check-then-run ceiling lets N concurrent requests each see
room for one. Reservation makes committed-plus-reserved spend a hard bound however many race:
50 concurrent reservations against room for 4 admit exactly 4 (`tests/test_api.py`).

**Why the startup refusal.** An in-memory total resets on restart, and on a host that sleeps
when idle every wake is a restart — a ceiling that never binds, as the brief put it. A README
line asking for a persistent ledger would be an instruction; refusing to boot is a mechanism
(§8.9). Verified on the real image: both refusals fire; a volume-backed ledger carried
$0.003628 across a container restart.

**Why notional.** D-004's reason, unchanged: billed cost is $0 on the free tier.

**Fails closed.** An unreachable ledger refuses the query with 503 rather than serving it
uncounted. The Upstash backend asserts each command's result, not the HTTP status — a 200
carrying a per-command error is not a write (D-026, fourth instance).

**Cost:** one ledger round-trip before and after each query; for `upstash`, a network call
each way. A crashed request's reservation stays counted until midnight — over-counting, the
safe direction.

---

## D-038 — Hugging Face Docker Spaces now require a paid plan: the sixth external change

**Found before deploying, from the provider's own docs (2026-09-23):** "Gradio and Docker
Spaces run on compute and require a paid plan to create: PRO for personal accounts." PRO is
$9/month. The account this project deploys from is not PRO and has no payment method on file
(`whoami`: `isPro: False`, `canPay: False`). The brief's "Hugging Face Spaces, free CPU tier"
no longer exists for a new Docker Space on a free account; the CPU Basic hardware is still
free, but only under PRO.

**This is D-034's pattern, a sixth time** — the provider surface changed under the plan. What
made the earlier five survivable applies here too: the image is host-agnostic (one
Dockerfile, env-configured, two durable ledger backends), so the choice of host is a
configuration and billing decision, not a code change.

**Resolved 2026-09-24 (Srikanth, Phase 5 review): Hugging Face PRO ($9/month), CPU Basic,
Upstash Redis free tier for the daily ledger, Langfuse Cloud Hobby for traces.** Reasons:

* the deploy tooling already targets it — `make deploy-space`, `make space-secrets`, the
  context checks and the dry run. Another host means new tooling, new guards, new failure modes;
* **the cost is fixed and known.** A pay-per-use host behind a public endpoint has an
  open-ended bill, bounded only by this project's own ceilings holding;
* Fly.io costs more and needs a volume.

**Fallback, not built:** Google Cloud Run, request-based billing, `max-instances=1`, 2 GiB,
Upstash for the ledger. Its free tier covers demo traffic, but it needs a billing account with
no hard cap. Nothing is built for it unless Srikanth says so.

---

## D-039 — Every request is traced; the sampling knob exists, is derived, and is verified in effect

**Measured (`make trace-units`, the 69-trace Phase 4 window):** 43 Langfuse units per query
(median; trace + observations). At the most queries the $0.50/day ceiling admits — 146/day at
the $0.00342 mean notional cost — tracing everything is ~167k units/month against Langfuse
Cloud Hobby's 50k. A head-sampling rate of **0.30** makes that worst case fit.

**Decided:** ship `LANGFUSE_SAMPLE_RATE=1.0`, and state the arithmetic rather than pre-emptively
discard 70% of traces from an endpoint that will see a few queries a day. Hobby covers about
1,160 traced queries a month (~38/day). If sustained traffic approaches that, set the derived
0.30. What Langfuse does when a Hobby project exceeds its allowance is **not verified** — the
pricing page does not say.

**The guard that made the knob trustworthy.** Langfuse applies `sample_rate` only when it
creates the global OpenTelemetry provider itself; if one is already registered it reuses it and
drops the rate without a warning. A configured policy that is silently not in effect is the
D-023 family. The service checks the *effective* sampler at startup and refuses to start if a
configured rate is not in effect. A sampled-out request returns `trace_id: ""`, never a dead id.

---

## D-040 — The load check bypasses the per-IP bucket, and only that

**Decided:** when `LOADCHECK_TOKEN` is set on the instance, a request carrying a matching
`X-Loadcheck-Token` (constant-time compare) skips the per-IP bucket. The daily ceiling and
the concurrency gate still apply. Unset, no bypass exists.

**Why:** every load-check request comes from one address. Without the bypass, 10 concurrent
users against a 3-request burst measures the limiter, not the service. The two limits it does
not bypass are the ones that protect the key. A run *without* the token is reported alongside,
as evidence that the per-IP limit fires on the deployed instance.

---

## D-041 — Phase 5, what running found that the tests did not

The §8.1 pattern again — seven instances, each found by building or running the real thing rather than by a test written in advance:

| Found by | Defect | Fix |
|---|---|---|
| reading the CLI while wiring SSE | `--stream` ran the graph **twice** per question — once to stream, once for the answer — paying for every streamed query double | one run path, `run_query(on_event=…)`; a test asserts one set of model calls per streamed question |
| first real query in the container | the model pacer started **empty with a bucket of one**, so every call after the first waited 5 s with a single user: 16.6 s for a 3-call query | bucket of 3, starts full; 4.3 s from idle; the 60 s window still admits ≤ 15 calls |
| first container start on a volume | the volume mounted **root-owned**; the uid-1000 service could not open its own database | `/data` created and owned by uid 1000 in the image |
| the sqlite-on-a-volume test | `mount_point()` skipped non-existent path components and judged a not-yet-created ledger file by the wrong mount | walk the path lexically |
| extending the seeded-key guard | `host_is_local` was a **substring** test: `https://localhost.attacker.example` counted as local | parse the hostname |
| writing the sampling test | Langfuse **silently drops** `sample_rate` when another OTel provider exists | startup check on the effective sampler (D-039) |
| `/ready` in the container | the chunk count read a non-existent attribute and reported 0 | read the FAISS index's `ntotal`; now 5,401 |

And the one found by *not* running: the daily ceiling could be configured below one
reservation, admitting nothing while `/ready` said ready. It is now a startup refusal.

---

## D-042 — `seed=0, top_k=1` makes the generator deterministic too; measured, not shipped

**Measured (`make generator-determinism`, Gemini free tier, $0, probed 2026-09-24):** N=5
calls per cell, `gemini-3.5-flash-lite` at `reasoning_effort=minimal`.

| Prompt | Unpinned (production) | Pinned `seed=0, top_k=1` |
|---|---|---|
| D-014's own fixed prompt, verbatim | 5 distinct of 5 — reproduces D-014 | **1 distinct of 5 — byte-identical** |
| the real `generate` call (v2 prompt, fixed 5-chunk context from r1's `sp-001`) | 3 distinct of 5 | **1 distinct of 5 — byte-identical** |

The pinned generator's output equals one of the unpinned draws (`9719d9c9`): pinning selects
one of the answers sampling can produce, it does not invent a new one.

**Decided: not shipped in v3.0.** `top_k=1` on the generator is greedy decoding. It would
change answer quality, and so every Phase 4 outcome metric, not only the classifier's refusal
count. It goes to BACKLOG as a measured Phase 6 arm: three pinned runs against the three
unpinned, same judge. What changes now is wording: Phase 4's variance "lives in *unpinned*
generation and judging" (EVALS.md, PHASE4.md), and D-014 carries a rescoping note.

**Not a contract.** Five identical outputs on one day, one model version, one key. The
classifier result (140 of 140, D-035) is stated the same way, with its probe date.

---

## D-043 — The published OpenAI spend was stale and undercounted; the account is now the check

**Found 2026-09-24, while pricing G-1.** Every document quoted **$0.0942 over 465 requests**
(CLAUDE.md, README, BUDGET, the Phase 5 brief). The provider's own records say **$0.1196 over
673 requests** (`make judge-spend`, 21 batches, none unreceipted). Two separate errors:

| | Requests | Spent | Why it was missing |
|---|---:|---:|---|
| traced run's judging (q1 + q3) | 91 | $0.0172 | judged *after* $0.0942 was measured; the typed figure never moved — C-1 again |
| three batches with no receipt | 117 | $0.0082 | the expired `dense_only` q1 (67 paid of 69) and the two rubric-v1 validation q1 batches (25 + 25): each receipt was **overwritten** by a later submit for the same arm and stage, so `judge-spend`, which summed receipts, could not see them |

The second is the worse one: a tool built so the spend figure would be emitted rather than
typed was emitting from an incomplete source, and nothing compared it to the provider's.

**Fixed three ways.** `write_receipt` archives a receipt for a different batch instead of
overwriting it (tested); `judge-spend` lists the account's batches and **refuses to write a
total while any judge batch is unreceipted**; the three lost receipts are reconstructed from
the account's batch list, marked as reconstructed. The spend line in README and BUDGET, and a
per-line breakdown in BUDGET, are rendered from the artifact, and `tests/test_docs.py` fails
a stale copy or a spend figure typed outside the rendered blocks — which is how the allocation
table's "$0.00 spent" on full runs that had cost $0.0654 was caught.

**Cost of the error:** none in money — $0.1196 is 2.4% of the $5 ceiling. The cost is to the
claim "every number is regenerable": this one was regenerable from an incomplete source.

---

## D-044 — The shipped (pinned) configuration is re-measured, passes the gate, and becomes the CI baseline

D-035 shipped a pinned scope classifier while every Phase 4 number had been measured unpinned
— D-002's problem again: a changed component invalidates the comparison until it is
re-measured. So it was re-measured (Phase 5 review, G-1).

**Pre-registered before judging** (CLAUDE.md, 2026-09-24): the gate passes and Recall@5 stays
0.267, because every item the pinned classifier blocks was already a refusal or a gold miss —
`sp-016` was a hallucinated refusal in r1–r3 and misses gold at k=5.

**Run:** v3, pinned default, frozen set `de699d68`, Gemini free tier ($0): the same 5 blocked
as the probe, 0 errors. **Judged:** `gpt-5.6-luna` @ `low`, Batch, two-stage, 69 q1 + 19 q3.
Cost cap $0.07 for this run (the estimator said $0.0640, conservative by design; measured
comparable runs $0.0154–$0.0173). **Actual: $0.0154** (q1 $0.0053, q3 $0.0102) — no anomaly.

| gated metric | Phase 4 baseline (unpinned, r1–r3) | pinned run | |
|---|---|---:|---|
| factual Recall@5 (n=43; v2.1 0.233) | 0.267 ±0.000 | **0.267** | as pre-registered |
| factual MRR (n=43; v2.1 0.196) | 0.175 ±0.000 | 0.175 | |
| hallucinated refusals (n=46 answerable) | 25.3 ±2 (26, 26, 24) | **27** | inside tolerance, **above all three unpinned runs** |
| correct answers (n=46) | 15.7 ±1 (15, 16, 16) | 15 | |
| attribute correct refusals (n=11) | 11 ±0 | 11 | |

**Gate: passed.** Stated with its caveat: 27 hallucinated refusals is one more than any
unpinned run — the direction pinning predicts (5 blocks every call against an unpinned mean of
4.3), and one pinned run cannot separate that from noise.

**Decided:** `evals/baseline_metrics_pinned.json` is committed **beside** the Phase 4 baseline,
which stays untouched. Values are the pinned run's own; tolerances are the Phase 4 three-run
spread, **carried** — one run cannot measure its own spread (the rule was written into
`evals/gate.py` before the result existed). CI now gates the shipped configuration's
committed run artifact (replayed, not re-run — D-050): the
must-pass step and the injected-regression step both run against the pinned run and baseline,
and the injected regression fails it (replayed locally before committing).

**Cost:** the tolerance on the shipped baseline is borrowed, not measured. Two more pinned
runs would measure it; that is the Phase 6 arm in BACKLOG, not a v3.0 blocker.

### D-044, amended 2026-09-24 — three pinned draws, and the answer to "was 27 noise?"

**The correction (Phase 5 review).** Re-baselining on one pinned draw *loosened* the gate: the
centre became the one draw already worse than all three unpinned runs, so hallucinated
refusals would have passed up to 29 (was ~27.3) and correct answers down to 14 (was ~14.7).
Until the fix landed, CI gated every metric on the **stricter** of the Phase 4 and pinned
baselines (`evals/gate.py --baseline A --baseline B`).

**Two more pinned full runs**, judged the same way (Batch, `low`, two-stage). Cap $0.06;
projected $0.0346 from the estimator's actuals line; **actual $0.0320** (r2 $0.0150, r3
$0.0170). The classifier blocked the same 5 items in all three pinned runs.

| n=46 answerable / n=43 factual | unpinned (r1, r2, r3) | pinned (p1, p2, p3) |
|---|---|---|
| hallucinated refusals | 26, 26, 24 → **25.3 ±2** | 27, 27, 25 → **26.3 ±2** |
| correct answers | 15, 16, 16 → 15.7 ±1 | 15, 15, 15 → 15.0 ±0 |
| Recall@5 (v2.1 0.233) / MRR (v2.1 0.196) | 0.267 / 0.175, spread 0 | 0.267 / 0.175, spread 0 |
| attribute correct refusals (n=11) | 11 ±0 | 11 ±0 |

**The answer, stated either way as asked: 27 was partly noise, and a rise is not shown.** The
third pinned draw gave 25. Across three draws pinning moves the mean by +1.0 — the direction
the mechanism predicts (5 blocks every call against an unpinned mean of 4.3) — but +1.0 is
inside the spread of either configuration (2), and the ranges overlap (24–26 vs 25–27). By this
project's reporting rule, a difference no larger than the spread has not been shown to be a
difference. Three draws, no confidence interval.

**The baseline, re-centred by code.** `evals/baseline_metrics_pinned.json` is now the mean of the
three pinned draws with their own max-min spread, written by `evals/gate.py --derive-baseline`
— no hand edits. Doing so exposed that the Phase 4 baseline's stated rule had never been
implemented in committed code; the one derivation function now regenerates **both** committed
baselines exactly from the runs they name, and `tests/test_gate.py` fails if either drifts. CI
gates the shipped configuration's committed artifact (replayed, not re-run — D-050) against the re-centred baseline; the injected regression still
fails it. One consequence stated plainly: pinned correct answers had spread 0 across three
draws, so the shipped gate allowed no drop in correct answers at all — **superseded by the
note below**.

### D-044, note 2026-09-24 — an outcome tolerance is floored by the unpinned spread

**The correction (Phase 5 review).** A zero tolerance on pinned correct answers is a flaky gate.
Generation and judging are still unpinned — D-042 measured generator pinning and did not ship
it — and unpinned correct answers ranged 15–16. Three equal draws (15, 15, 15) from a process
known to vary are a small-sample artifact, not evidence that the variance is gone.

**The rule, in `evals/gate.py` (`derive_baseline`), applied by regeneration, not by hand:**

* **retrieval metrics** (Recall@k, MRR): tolerance = their own observed spread. Retrieval is
  deterministic by construction, and `make index-verify` enforces identical rankings;
* **outcome metrics** (correct answers, hallucinated refusals, anything downstream of
  generation or judging): tolerance = **max(own spread, the unpinned spread for the same
  metric)**, the unpinned spread read from the reference baseline (`evals/baseline_metrics.json`).
  Each metric records which basis it used (`tolerance_basis`).

**Result:** pinned correct answers 15.0 **±1.0** (floored — own spread 0); hallucinated
refusals 26.3 ±2.0 (own spread = floor); Recall@5 0.267 ±0 and MRR 0.175 ±0 (retrieval, own).
The Phase 4 baseline regenerates unchanged (derived against itself, the floor is its own
spread), both CI steps replay correctly, and `tests/test_gate.py` asserts each branch.

**What the floor does not fix — and deliberately so.** Attribute correct refusals (n=11) is an
outcome metric whose unpinned spread was also 0 (11, 11, 11), so under this rule it stays at
±0.

### D-044, note 2 (2026-09-24) — the ±0 gate on attribute refusals is intentional

**Decided (Phase 5 review):** keep attribute correct refusals at ±0.

**What the zero means.** It is a **ceiling metric**: 11 of 11 (1.000) in all six judged runs,
pinned and unpinned. Zero spread at the ceiling is a ceiling artifact, not evidence of
stability — the stratum cannot vary upward, and at 1.000 it is above the 0.95 too-easy line
(CLAUDE.md §3), so it cannot tell configurations apart. The README states that beside the
figure, paired with the hallucinated-refusal count as the refusal pair always is.

**Why the strict gate anyway.** A missed refusal on an unanswerable item — the agent answering
a question the corpus cannot support — is the regression most worth catching. A one-item drop
fails CI; that is accepted, **including an occasional false alarm** from unpinned generation or
judging, as the price of never letting that regression through silently. The asymmetry is the
decision: a false alarm costs a rerun, a missed regression costs a fabricated answer served as
fact.

**Reverses if:** false alarms on this metric recur often enough to be ignored — at which point
a gate that is routinely overridden is worse than a looser one, and the stratum needs hardening
(more, harder attribute items) rather than a wider tolerance.

**Correction (D-056, 2026-09-25).** "CI gates the shipped configuration", "both CI steps replay
correctly" and the interim stricter-of-both gate above describe what the workflow was *written*
to do. **None of it executed on GitHub:** the gate steps followed the unit-test step in one job,
and every run from #9 (before the pinned baseline existed) to the fix failed at that step. Every
gate result recorded here — pass, fail, injected regression — was a local replay, which is what
`tests/test_gate.py` and `make gate` verify. The gate first runs in CI, in its own job, with the
D-056 fix.

---

## D-045 — The Phase 4 baseline's stated derivation had no producing code

**Found 2026-09-24** while re-centring the pinned baseline. `evals/baseline_metrics.json` says
of itself *"value = mean of three complete runs; tolerance = their max-min spread"* — and no
committed code implemented that sentence. The file was produced outside the repo, and every
review since Phase 4 closed read the derivation string as if it were a derivation.

**It is the C-1 class** — a number that could not be regenerated by a command in the repo —
in its least visible form: not a typed figure in prose but a machine-read artifact carrying
its own provenance text, which is exactly what makes it look regenerable. The regression gate,
the part of the project that exists to catch drift, rested on it.

**Fixed:** one derivation function (`evals/gate.py derive_baseline`) now regenerates the
Phase 4 baseline exactly from r1–r3, and the pinned baseline from its three runs;
`tests/test_gate.py` fails if either committed baseline differs from what the function
produces from the runs it names. Baselines are written by `--derive-baseline`, never by hand.
Reserved for the post-mortem (BACKLOG).

---

## D-046 — "Billed" was $0 by assumption, not measurement; Gemini was billed $7.60

**Found 2026-09-24 by Srikanth, from Google Cloud Billing** (SKU export, 2026-08-19 to
2026-09-24; not committed — the figures are hand-entered in `docs/billing/gemini.json` with
source and date). The Gemini key's project had billing enabled throughout. **Every "Gemini billed
$0" in this repo was wrong**: README, BUDGET, OBSERVABILITY, SERVING, PHASE4, the API's
`billed_cost_usd: 0.0`, the `arxiv_agent_cost_usd_total{kind="billed"}` series, the trace
metadata, every run artifact's `cost_usd: 0.0`, and the self-judge receipt's "free tier — $0".

| SKU | tokens | billed |
|---|---:|---:|
| 3.5 flash-lite input (uncached) | 22,573,359 | $6.77 |
| 3.5 flash-lite input (cached) | 4,703,996 | $0.14 |
| 3.5 flash-lite output (incl. thinking) | 274,770 | $0.69 |
| 2.5 flash-lite input + output (the first live run, D-012) | 13,904 | $0.002 |
| **total** | | **$7.60** |

All rates match the verified card ($0.30 / $0.03 cached / $2.50; 2.5 at $0.10 / $0.40). That
is **45× the OpenAI judge spend** ($7.60 / $0.1670, `make judge-spend`) the project tracked to four decimal places under a $5 ceiling,
on the provider it had no ceiling for.

**How it happened.** `ModelPricing` carried "billed" rates of $0 from the belief that the key
was on the free tier; `Usage.cost_usd` multiplied tokens by them; everything downstream reported
the product as a measurement. Nothing ever compared it with the provider's record. It is the
C-1 class again — a number nobody could regenerate from a source — and the same shape as D-043
(OpenAI spend summed from an incomplete source), one level down: this figure had no source at
all, only a default.

**Decided — the rule:** **a billed figure comes from the provider's own record, or it is shown
as "unverified" — never 0 by default.** In code: `ModelPricing` has no billed rates;
`Usage.cost_usd` is `None` unless a provider record supplied it, and the reducer keeps
unverified unverified; the API returns `billed_cost_usd: null` with a `billed_cost_basis`; no
billed metric series is written; traces carry `cost_usd_billed: null` and no `cost_details`.
The provider's figure is rendered into README and BUDGET from `docs/billing/gemini.json`, and
`tests/test_docs.py` fails a doc that asserts Gemini was billed $0.

**Cached input (the second defect).** Google billed 4.7M prompt tokens as cached at $0.03/1M.
LangChain reports them (`input_token_details.cache_read`, inside `input_tokens`); `Usage`
ignored them and priced every prompt token at $0.30 — an overstatement, the safe direction for
a ceiling, but wrong. `Usage.cached_input_tokens` now records them and notional prices them at
the cached rate. The blended-rate sanity checks (`make budget`, `make reconcile-cost`) use the
*recorded* cached share for their floor — using the cached rate for every token would have
made the D-021 blend look possible and disabled the check that caught it.

**Reconciliation, on tokens (`make gemini-reconcile`).** Billed prompt tokens 27,277,355;
recorded by any instrumentation 6,530,193 (eval runs 6.22M, CLI 0.20M, tests 0.07M, the
Gemini judge arm 0.02M, probes 0.02M). **The gap — 20,747,162 prompt tokens, 76%, and 42% of
output — is uninstrumented usage, named, not spread.** Only full-text work is large enough to
fill it: one multi-hop construction candidate is ~99k prompt tokens (eight calls carrying whole
papers), one run of the necessity-fixture suite ~504k, the v2.1 baseline run at least 771k —
all three discarded or never exposed their usage. **Srikanth's hypothesis that full-text QA
drafting dominates is consistent with the records and not confirmed by them:** the gap equals
~210 candidates' worth, the 4.7M cached tokens point to long identical prefixes sent repeatedly,
and construction and the necessity tests make the same kind of call — nothing recorded can say
which of the two dominates.

**Cost of the error:** $7.60, and the claim that every number here is regenerable. **Reverses
if:** never — this is a rule about where numbers come from.

**Also found while reconciling:** Langfuse Cloud answers `GET /api/public/traces` with 410 for
organisations created on or after 2026-09-16; `make smoke-live`'s trace check and
`span_loss_probe.py` use it and must move to `/api/public/v2/observations` before deploy.

### D-046, note 2026-09-24 — the gap was callers discarding usage; the fix is structural

The 76% gap had one cause, not several: **callers discarded the usage the wrapper handed
them.** `call_structured` and `call_text` always returned a `Usage`; eval-set construction
(`drafted, _ = await call_structured(...)`), the necessity checks and the probes threw it away,
and three scripts built their own models and bypassed the wrapper entirely. Recording that
depends on every caller choosing to keep the number is an instruction, not a mechanism (§8.9).

**The mechanism.** `usage_from_message` — which every response through `call_text` and
`call_structured` passes through — now appends one row per call to a local usage log
(`Settings.usage_log`, `.usage/llm_usage.jsonl`, gitignored): timestamp, activity (the running
program, or an explicit `USAGE_ACTIVITY`), model, input, cached, output, thinking, notional. A
caller can drop the `Usage` object; it cannot drop the row. The three bypassing scripts now build
models through `get_chat_model` (which gained `overrides` for D-014's probe settings), and the
judge's synchronous arms — which call Gemini's OpenAI-compatible HTTP endpoint around LangChain —
write a row per call. `tests/test_usage_log.py` asserts a wrapper call leaves a correct row, that
a caller discarding its `Usage` still leaves one, and **fails if any module other than
`src/agent/llm.py` builds a chat model, calls Gemini over HTTP without `record_usage`, or calls
`.ainvoke` on a model without `usage_from_message`** (mutation-checked: an injected bypass fails
it). The suite writes its rows to a temporary file, never the real log (§8.10).

**What it does not do.** It records calls from now on; it cannot recover the 20.7M tokens
already spent unrecorded. A log that cannot be written is reported as an error rather than
failing the query — a served request is not taken down by its own bookkeeping.

---

## D-047 — Deployed: what the live host changed, and what running against it found

**Deployed 2026-09-24** to `godvillain/Scholium` (HF PRO, CPU Basic, public), commit `c40fc900`
of the Space repo. Build 201 s; a variable-change restart took ~30 s to `/ready`. Upstash ledger,
Langfuse Cloud traces, the deploy key a free-tier key in its own no-billing project (15 RPM /
500 RPD, read from AI Studio by Srikanth).

**Verified on the host, by effect:**

| Check | Result |
|---|---|
| `make smoke-live` | every check passes — README example cites a returned source, `/ready` 5,401 chunks and the Phase 4 index checksum, limiter keys on the real client, forged `X-Forwarded-For` ignored, trace **complete** in Langfuse Cloud (32 observations, root carries the answer) via the v2 API, effective sample rate 1.0 in `/ready` and in the run log |
| ledger durability | $0.003606 today before a restart, $0.003606 after — read back from Upstash by the new container |
| seeded-key refusal | the Space's own built image (`registry.hf.space/godvillain-scholium`), run with the seeded pair and a Cloud host: "refusing to start", startup failed |
| span loss on stop | Space paused the instant a response landed: trace **complete** (33 observations, root answer) — the pause is a graceful stop and the shutdown flush runs. A natural idle sleep cannot be triggered on demand; it is inferred from pause, not observed |

**Found by running it — none visible locally:**

1. **The trusted proxy hop count is 1, and 0 was actively wrong.** At the default of 0 the limiter
   keyed on the socket peer, which on Spaces is a *rotating* proxy node: 6 distinct keys in 8
   plain requests from one client. The per-IP limit would have been shared by strangers on the
   same node and escapable by landing on another. With `API__TRUSTED_PROXY_HOPS=1` (a Space
   variable) the key is stable, equals the client's address, and ignores forged headers.
2. **The generator's citation format drifted on the free-tier key.** Recorded eval answers (all
   on the paid key) cited full ids; on the free-tier key 4 of 4 sampled answers used
   abbreviated ids (`[30179_0006]`) and one used a corrupted id (`[32605.30179_0006]`). Whether
   the tier or a week of model drift is the cause cannot be separated from here. It exposed a
   pre-existing parser defect: **134 of 645 completed eval answers (21%) put several ids in one
   bracket and `finalize` recognised none of them** — so citation-ordered sources and the
   invented-citation warning had not worked for a fifth of answers. One parser now accepts all
   forms, resolves an abbreviation only when unique, and reports corrupted ids as unresolved.
   No gated metric reads it (retrieval uses retrieved ids; the judge reads text).
3. **`smoke-live` could not assert on one draw.** Unpinned generation (D-042) makes the README
   example's citation form a draw. (A typed "about 2 of ~10 draws" stood here; it was never
   regenerable and is replaced by a measurement: in the load check against the deployed
   instance, 11 of 32 served responses cited no returned source — 5 guardrail refusals, 6
   generator refusals read by hand, none an answer without a citation (n=32, live Space,
   `make load-report`). That covers the 43 factual questions, not the README example's own
   draw-to-draw rate, which stays unmeasured.) The check now runs up to 3 draws, prints each draw's citations, passes if any cites
   a returned source, and warns with the count when not all did — a relaxation of the brief's
   single-draw wording, flagged rather than made silently.
4. **Local Langfuse could not test the v2 API.** The v2 observations endpoint needs self-hosted
   Langfuse v4, whose migrations need a newer ClickHouse than the pinned 24.3 (25.8 also fails).
   The migration was tested on a throwaway v4 stack (ClickHouse 26.9, its own volumes, the
   Phase 4 trace store untouched), then against Cloud. `infra/docker-compose.langfuse.yml` is
   unchanged and still v3.
5. **`make serve` did not run uvicorn like the container** (no `--no-proxy-headers`), so a local
   forged header rewrote the client address — caught by `smoke-live`'s forgery check. Fixed.

---

## D-048 — The first deployed load check found a reservation leak, and was itself a runaway

**2026-09-24, first G-3 run against the Space, aborted by hand.** Three failures, two of them
mine in the measuring tool and one in the service:

1. **The service leaked daily-ceiling reservations.** A request reserved $0.025, then waited at
   the concurrency gate *in the request's own coroutine*. A client that disconnected while queued
   cancelled that coroutine between reserve and settle, and the reservation stayed counted until
   midnight. The run's 83 client-side connection errors left enough stranded reservations that
   the $0.50 ceiling answered `429 daily_cost_ceiling` **after one served query**. On a public
   endpoint that is a denial-of-service: open requests, drop them, and the day's ceiling is gone
   for everyone. **Fixed and redeployed** (`caefad03`): admission (reserve + gate) runs as its own
   task awaited through `asyncio.shield`, so no client can cancel it half-way; a client that has
   left gets its slot and reservation handed back by a done-callback. Tested by reproducing the
   disconnect while queued; the pre-fix sequence, run against the same scenario, leaks $0.025.
2. **The load tool hot-looped.** On an instant rejection it retried at once: ~74,000 requests in
   minutes, **70,510 of them answered by Hugging Face's own platform rate limiter** (HTML 429s),
   not by this service. It now honours `Retry-After`, backs off on non-JSON (platform) 429s,
   **aborts on `daily_cost_ceiling`** (which cannot clear before 00:00 UTC), and stops at 600
   requests sent whatever happens — each tested.
3. **The cold-start figure timed the wrong container.** After `restart_space` the old container
   kept answering `/ready` for seconds; the tool took that 200 as the new container. `/ready` now
   reports a per-process `boot_id`, and cold start is timed until a *new* boot id answers.

**What the aborted run does not support:** any latency or throughput figure. Its artifact was
never written (the tool was killed); its per-request log is not a load-check result. **Not
published.**

**Consequence today:** the leaked reservations stay counted until 00:00 UTC (an over-count, the
safe direction; the ledger was not hand-edited). With ~$0.19 of the day left, a load check to the
G-3 spec (≥30 served, 10 users each holding a $0.025 reservation) cannot run until the reset.

---

## D-049 — Deploy mapping: tag `deploy-2026-09-24` (git `d6b0369`) is Space commit `caefad03`

**Recorded 2026-09-24, verified by `make verify-deploy SPACE=godvillain/Scholium REV=d6b0369`.**
Every file in the Space's tree at `caefad03` was compared with `d6b0369`:

* **60 identical, 0 mismatched.** 57 repo files byte-for-byte; `Dockerfile` equals
  `infra/Dockerfile` (the deploy copies it to the Space root); the two gitignored FAISS files
  match `data/INDEX.sha256` as committed at `d6b0369`; the Space card (`README.md` in the Space)
  equals what `scripts/deploy_space.py` at `d6b0369` renders.
* **1 not compared:** `.gitattributes`, Hugging Face's own LFS rules, not a repo file.
* **Not deployed, by design, so not mismatches:** 222 tracked files — `evals/` 132, `tests/` 41,
  `scripts/` 26, `docs/` 16, the repo `README.md` (the Space serves the generated card under the
  same name), `Makefile`, `.env.example`, `.gitignore`, `.github/workflows/ci.yml`,
  `infra/docker-compose.langfuse.yml`, `data/LICENSES.json`, `data/traces_d021.json`.

A later deploy is mapped the same way: tag the commit, deploy, run `make verify-deploy` with that
commit, and record the pair here.

**The deployed commit, gated (2026-09-25).** A pinned full run of the frozen 69 from `d6b0369`
— a `git archive` export with this repo's venv and the index verified against
`data/INDEX.sha256@d6b0369`, the deploy project's Gemini key, tracing off, run locally rather
than through the public endpoint — completed 69 of 69 with 0 errors and blocked the same 5 items
as the three baseline draws (`evals/runs/v3_de699d68_deployed.json`, `code_rev` stamped). Judged
by `gpt-5.6-luna` @ `low` under the $0.03 cap, submitted in stages so q3 went out only after
q1's actual cost plus a q3 projection from measured runs fit ($0.0051 + $0.0106 = $0.0157
actual). `make gate RUN=evals/runs/v3_de699d68_deployed.json
SHEET=evals/runs/scores_luna-low_de699d68_all_deployed.json` **passes** against
`evals/baseline_metrics_pinned.json`: every gated metric inside the three-draw spread. One draw,
not added to the baseline (no re-baselining was asked for). A first gate invocation on a sheet
still awaiting q3 printed a false regression and is not this result (D-053).

---

## D-050 — CI replays committed run artifacts through the gate; it does not evaluate the pushed code

**The fact, stated where it could be misread (README "CI reports; it does not block", EVALS,
PHASE4, D-044):** the regression gate in `.github/workflows/ci.yml` reads a committed run artifact
and its judge sheet — outputs recorded when the configuration was last run and judged — and
checks them against the committed baseline, plus an injected regression that must fail. **No
model is called in CI.** A change to prompts, nodes, retrieval or the guardrail that alters
answers passes CI unchanged until the set is re-run (`make run-set`), judged, and the new
artifact gated. Wording that implied CI evaluates the current code ("a live regression gate",
"CI gates the shipped configuration") was corrected on 2026-09-24.

**Why the Phase 4 spec's "eval on a fast subset" is not in CI:**

* **No key in CI, deliberately.** CI has no Gemini or OpenAI credentials; every push to every
  branch runs the workflow, and a key there would spend on any push from anyone with write
  access.
* **Free-tier quota.** The Gemini keys are free tier at 15 RPM / 500 RPD, shared with the live
  Space. Even a 10-item subset is ~35 model calls per push, paced at 15 RPM — minutes per run —
  and a few pushes a day would eat the Space's daily quota.
* **Judging is Batch.** The outcome metrics need the judge, which runs through the OpenAI Batch
  API (mandatory, half-price): minutes to hours of latency per run, and real money per push.
  Retrieval-only metrics could skip the judge, but they are the metrics that cannot move without
  a retrieval change (spread 0.000).

**Reverses if:** a dedicated paid Gemini key with its own budget cap exists for CI, stored as a
secret available only to protected branches; the subset is small enough to pace inside a CI job;
and either the judge is replaced for CI by a cheap synchronous arm validated against the human
sheet, or CI gates retrieval metrics only and says so. Until then, re-running the set is a
manual step recorded with its artifact.

### D-047, amended 2026-09-24 — the citation drift is time-bound, not key-bound

Tested rather than assumed (Phase 5 review): the README request five times on the **old** key
(project `langgraph-research-agent`) against the deploy key's draws on the Space.

| key | when (UTC) | tier then | citing answers | abbreviated ids |
|---|---|---|---:|---:|
| old | 2026-09-24 15:11–16:45 (three pinned runs) | paid (billing attached) | 156 | **0** |
| old | 2026-09-24 ~21:45 (README request ×5) | free (billing detached) | 5 | **4** |
| deploy | 2026-09-24 21:05–21:45 (Space, smoke draws) | free | every draw | **all** |

**It is not the key**: both keys abbreviate now. The change happened on the old key between
16:45 and ~20:50 UTC on one day, and coincides with that key's move to the free tier. A model or
serving update in the same window **cannot be excluded**, so this is recorded as a time-bound
change coinciding with the tier switch — not as "caused by the free tier". No doc claims more.

**Correction (D-056, 2026-09-25).** This entry is right about *what* the gate compares — committed
artifacts, not the pushed code — and wrong by implication about *whether* it ran: until the
D-056 fix the gate had never executed in CI at all. Before that fix, "CI replays committed
artifacts through the gate" described the workflow file, not anything that happened.

---

## D-051 — Keep all 150 papers; credit every one; publish a takedown route

**Decided (Srikanth, 2026-09-24):** the corpus stays as it is — all 150 papers, nothing
re-chunked, re-indexed or removed. The license audit (`make corpus-licenses`, `data/LICENSES.json`,
arXiv OAI-PMH) found 85 CC BY 4.0, 6 CC BY-NC-SA 4.0, 3 CC BY-NC-ND 4.0, 2 CC BY-SA 4.0, and **54
under arXiv's non-exclusive distribution license**, which grants distribution rights to arXiv, not
to third parties. This repository commits all 150 papers' full text in `data/chunks_512.json`, the
container bakes it in, and the service quotes excerpts.

**What was done instead of removal:** `CORPUS_ATTRIBUTION.md` credits every paper — arXiv id
(linked), title, all authors, license (linked) — rendered from the audit and the metadata, with a
test that fails a stale copy; the README License section states that MIT covers the code only,
lays out the open question for the 54 papers as not verified, and gives a takedown route (a
GitHub issue on this repository).

**Why keeping is acceptable here, as a judgment rather than a legal conclusion:** a
non-commercial research demo over public preprints, every excerpt attributed to its paper by id
and title, with a published removal path. Why the question stays open: whether redistributing the
full text of arXiv-licensed papers in a public repository is permitted is **not verified**, and the
README says so.

**Reverses if:** a takedown request arrives for a paper (remove its chunks, rebuild the index,
re-run the frozen set against the reduced corpus and re-baseline, and record the corpus checksum
change); or an authoritative reading establishes that the arXiv license does not permit this use
(remove all 54 the same way). Either reversal invalidates every retrieval number measured on the
150-paper corpus, which is why it is a recorded decision rather than a quiet edit.

## D-052 — The first G-3 rerun measured a sleeping laptop; the load tool now checks its own clock

**2026-09-25, first G-3 rerun after the 00:00 UTC reset. Invalid, not published.** The run was
launched at 00:26 UTC from a MacBook on battery with its lid closed; the session was executing
only during macOS dark wakes. The power log (`pmset -g log`) shows sleep entered at 00:26:12 UTC —
seconds after the tool issued the Space restart — with 2-second dark wakes at 00:44:03 and
00:59:52 and a lid-open wake at 02:43:34. The Space's run log shows the other side: new container
ready at 00:26:51, the cold-start query answered at 00:44:07, the load's first ten requests
answered (503 busy) at 00:44:27, and served answers from 00:44:35 onward, delivered to a client
that was asleep.

**What the tool reported, and why every figure was wrong:** "134 requests over 600 s", "cold start
2.3 s to ready", served p50/p95, and 10 client `ReadError`s and 10 `ReadTimeout`s. Every timing was
`time.monotonic`, which stops while the machine sleeps, so wall time (00:44 → 02:49 UTC, about two
hours) collapsed to ten minutes; the cold start's "2.3 s" was the poll after a 17-minute sleep;
the errors were connections that died across sleep. Nothing in the artifact showed any of this.

**Kept, quarantined:** `evals/runs/loadcheck_deployed-client-slept.json(l)`, carrying an
`invalid` field with the evidence above; `make load-report LABEL=deployed-client-slept` prints
the reason and refuses to print latency, throughput or rates. Cost: about 142 model calls on the
deploy key (Langfuse Cloud generation count since 00:26 UTC) and $0.139 of the day's notional
ceiling.

**The fix is a mechanism, not a note to keep the lid open.** `ClientClock` compares wall time
with monotonic time; more than 5 s of divergence means the client slept. The run stops, writes an
`invalid` line into the per-request stream (so a killed run carries the evidence too), records
the drift on every request, and the cold start returns an error instead of a number. The report
refuses an invalid artifact. `make load-check` also runs under `caffeinate -i`, which holds off
idle sleep but cannot stop a closed lid on battery — so the check is the guarantee, and
`caffeinate` only makes a valid run likelier. Tests: a fake clock that jumps two hours stops the
run, marks the stream, and blanks the report; the real clock on an awake machine does not
trip it.

**Post-mortem class:** a measuring tool that assumes the machine running it is continuously
awake. The same family as D-048's tool defects: the instrument failed, and without a check its
failure read as a measurement. Recorded beside "external-facing tool without backoff".

**Reverses if:** load checks move to a host that cannot sleep (a CI runner or a VM), at which
point the clock check stays as a cheap assertion rather than being removed.

## D-053 — The gate scored a half-judged sheet as a regression; unfinished input is now refused

**2026-09-25, item 4 (the pinned run of the deployed commit).** Judging is two-stage (D-030a):
`collect` for q1 assembles a score sheet from the items q1 settles alone and leaves the rest
awaiting q3. My poll loop waited for that sheet *file* to exist, so it returned the moment q1
landed, and I ran the gate on a sheet scoring **49 of 69** items. The gate printed:

    correct_answers 0.000 vs baseline 15.000 (tolerance ±1.000) — regression

Every answered item was still awaiting q3, so none could be `correct_answer`. The gate had no
completeness check: it computed outcome counts over whatever the sheet held. **Not a result, and
not reported as the item 4 outcome** — the deployed code was never measured by that invocation.

**Fixed, in the gate, not the loop:** `gated_values` raises `IncompleteSheetError` unless the
sheet scores every item of the set, so both the gate and `--derive-baseline` refuse; the CLI
exits 2 and names the unscored items. Tests: a sheet with 20 items removed exits 2 with "covers
49 of 69 items"; `gated_values` refuses a sheet missing one item.

**The same defect one level up, found by the fix's own test.** A first version also refused a
sheet whose `run_path` differed from `--run`. CI's injected-regression step pairs a doctored copy
of a run with the original sheet, so it would have been refused with exit 2 — and CI read *any*
non-zero exit as "the gate fired". The injected regression would have "passed" on a refusal, the
D-023 pattern exactly. The run-path check is dropped; CI now requires **exit 1 and
`factual_recall@5: 0.000` in the output**, so a refusal, a crash or a wrong failure all fail CI.
Replayed locally: the injected regression exits 1 naming recall; the committed pinned run passes.
The replay also showed an unreadable sheet crashing the gate with a traceback — exit 1, the
regression code. It now exits 2 ("refusing: cannot read the run or sheet"), tested.

**Post-mortem class:** a detector that accepts input its author assumed would be complete, and a
test harness that accepts any failure as the failure it wanted. D-023 and D-026 again.

**Not extended:** `make metrics-report` and `metrics-spread` still compute outcomes over whatever
a sheet holds; they print n per stratum, so a partial sheet shows, but they do not refuse
(BACKLOG).

## D-054 — Where a malformed Upstash result failed, the redeploy that fixes it, and why the Phase 5 measurements carry over

**Which call raised the unhandled 500 (deployed `d6b0369`).** The **reservation, before the
query**: `UpstashLedger.reserve` did `float(str(total_raw))` on `INCRBYFLOAT`'s answer, and
`_reserve` in `src/api/app.py` catches only `LedgerUnavailableError`, so a `ValueError` became an
unhandled 500. That path **fails closed**: nothing runs, and the $0.025 reservation — already
applied on the server when the answer came back unparseable — stays counted, an over-count.
`/ready` failed the same way (`spent` parsed `GET` the same way). The fixed build returns `503
cost_ledger_unavailable` for both.

**Settlement could not raise that 500 — because it never read its answer at all.** `settle` sent
`INCRBYFLOAT <actual − reserved>` and discarded the result, so a 200 with a malformed body was
taken as a completed write. Losing a settlement is safe when the delta is negative (the usual
case: a query gives back most of its $0.025), and **fails open when it is positive**: the
per-request budget is checked after each model call (`src/guardrails/budget.py`), so a run can
overshoot its reservation by its last call, and a lost positive settlement under-counts the
daily ceiling by that overshoot. Never observed — the largest per-query notional measured is
$0.0057 against the $0.025 reservation (docs/BUDGET.md) — but reachable. Now `settle` parses the
new total and raises `LedgerUnavailableError`, which `_settle` logs; the settlement is still
lost, but not silently. D-026 again: a write assumed from a status code. Tests: a malformed
settlement raises, a well-formed one passes, and the reservation-path cases (unreachable, 429,
command error in a 200, malformed result, timeout) refuse the query before any model call.

**Redeploy, 2026-09-25.** `make deploy-space` from the working tree pushed Space commit
`7745886e`; the rebuilt container came up as boot `a050a88ad358` with the same index sha
(`ad6c35cf…`). `make verify-deploy REV=WORKTREE` (a mode added for exactly this: a deploy made
before the commit that carries it exists): **60 identical, 0 mismatched**, `.gitattributes`
listed. `make smoke-live` against the redeployed Space: every check passed (README curl cites a
source on the first draw; trace complete, 32 observations). **The tag `deploy-2026-09-25` is not
created**: it must point at a commit holding this fix, and commits are Srikanth's. After he
commits: `git tag deploy-2026-09-25 <commit>` and `make verify-deploy SPACE=godvillain/Scholium
REV=deploy-2026-09-25`, expected 60 / 0; the pair to record is `<commit>` == Space `7745886e`.

**Mapping recorded (2026-09-25): tag `deploy-2026-09-25` = git `957b711` = Space `7745886e`.**
Srikanth committed and tagged; `make verify-deploy SPACE=godvillain/Scholium
REV=deploy-2026-09-25`: identical 60, mismatched 0, not compared 1 (`.gitattributes`, Hugging
Face's own). This was the second deploy run from uncommitted files — the first, `caefad03`, was
mapped to `d6b0369` only after the fact (D-049) — so `make deploy-space` now refuses to do it
again (D-055).

**Why the G-3 load check and the item 4 gate result carry over.** The deployed files differ from
`d6b0369` in exactly one: `make verify-deploy REV=d6b0369` against the new Space reports **59
identical, 1 mismatched — `src/api/ledger.py`**, and `git diff d6b0369` over the deploy
allow-list (`.dockerignore`, `infra/Dockerfile`, `pyproject.toml`, `src/`, the corpus and index
checksums, and `scripts/deploy_space.py`, which renders the Space card) shows the same single
file. Its tests and this record are not deployed.
* *The gate result (item 4)* never touched the ledger: the pinned run executes the graph through
  `evals.run_set`, not the API. It carries over by construction.
* *The load check* ran through the ledger, but only on well-formed Upstash answers — the Space's
  run log shows 33 × 200 and 137 × 503 `busy`, no 500 and no `cost_ledger_unavailable`. On a
  numeric answer the new code does the same arithmetic (`_number(x)` is `float(str(x))`) and
  sends the same single pipeline per call; the one addition is parsing settlement's reply,
  which costs no request. Nothing the load check measured passes through a changed line.

**Reverses if:** a later deploy changes any other allow-listed file, in which case the load check
and the gate are re-run against it rather than carried.

## D-021, third instance — the Gemini reconciliation counted a run from outside the bill

**2026-09-25, found while drafting the post-mortem.** `make gemini-reconcile` globbed every
`v3_de699d68*.json` run artifact as "measured" usage. The day after the billing period closed,
that included `v3_de699d68_deployed.json` — a run on 2026-09-25, on the deploy key, in a
no-billing project — so 658,690 of its input tokens were counted against a bill they are not on,
and the published gap moved from 76% to 74%. The command also rewrote the committed artifact
(`evals/runs/gemini_reconcile.json`); the committed version was restored from git.

Same rule as D-021: a ratio asserts its two sides describe one population. Here the population is
fixed by the bill (`docs/billing/gemini.json` `period`), so `run_artifacts()` now drops any run
started after `period.to`. Tests: every artifact is counted exactly when it started inside the
period, and the committed reconciliation lists exactly the runs the script now selects. The usage
log is not a measured source in this script, so the 236 deploy-key rows appended to it are not
counted either.

## D-055 — Deploy only a tagged commit; the override is explicit and recorded

**Decided (Srikanth, 2026-09-25):** Srikanth commits and tags; the deploy runs from the tag.
Two deploys had run from uncommitted working trees — `caefad03`, mapped to `d6b0369` only after
the fact (D-049), and `7745886e`, verified against the working tree and mapped to `957b711` once
the commit existed (D-054). Both mappings held, but only because someone checked afterwards.

**The mechanism.** `scripts/deploy_space.py` computes what would ship before uploading:
`HEAD`, the `deploy-*` tags on it, and `git status --porcelain --untracked-files=all` over every
deployed path — the `.dockerignore` allow-list plus `infra/Dockerfile`, `.dockerignore` and the
script itself, whose text renders the Space card. The upload refuses if any deployed path is
modified or untracked, or if `HEAD` has no `deploy-*` tag. Paths git ignores (the FAISS index)
are outside git's vouching; their checksums are committed and verified by `check_context`, as
before. A dirty file that is not deployed (docs, tests, evals) does not block.

`--allow-dirty` (`make deploy-space ALLOW_DIRTY=1`) overrides the refusal, prints a warning,
names the Space commit "deploy arXiv Agent v3 from UNCOMMITTED (--allow-dirty) on <sha>", and
the deploy record carries `allow_dirty: true` with the dirty paths. Every upload appends one line
to `infra/deploy_log.jsonl` (time, Space, Space commit, git `HEAD`, tags, override, dirty paths)
and reads it back (D-026). A clean deploy's Space commit message names its tag and short sha.
The log starts with the two earlier deploys, backfilled and marked so (`backfilled`, with the
after-the-fact mapping), recorded as what they were: uncommitted deploys.

**Tests** (`tests/test_deploy_guards.py::TestDeployOnlyFromATaggedCommit`, in a throwaway git
repo): a clean tagged tree is accepted; a modified deployed file, an untracked deployed file, a
changed Dockerfile or card renderer, and a clean but untagged `HEAD` are each refused; a dirty
file outside the deployed set does not block; `main()` refuses before any upload, and the
override passes the guard with a warning while still stopping at the publish confirmation; the
deploy record is written and read back. On the real repo the dry run reported
`an upload would be refused: deployed files differ from HEAD: scripts/deploy_space.py` — this
change itself, uncommitted.

**Reverses if:** never silently. A deploy that must go out before a commit uses the override,
and the record says so.

## D-021, fourth instance — "gold paper in the top 5" counted v2.1's whole retrieval

**2026-09-25, found while applying Srikanth's POSTMORTEM review** (no winner on a 1–2 item
difference, which needed per-item counts). PHASE4, EVALS, README and the post-mortem draft all
published *"gold paper in the top 5: v3 21–22 of 43, v2.1 28 of 43"*. The metric
(`gold_paper_hit` in `evals/metrics.py`) counted a hit anywhere in what the run retrieved, and
`make metrics-compare` printed it as `gold-paper-in-top5`. v3 retrieves 5 chunks, so for v3 the
two are the same; v2.1's loop unions several searches, a median of 18 chunks from 6 papers. **In
v2.1's first 5 retrieved chunks the gold paper appears for 22 of 43 — level with v3.** The 28 is
real, but it is the wide-net mechanism PHASE4 §2 already describes, not a better first page, and
"v2.1 finds the gold paper more often" is true only at 3–4× the depth.

Two populations under one label again: the same shape as D-021 (6 priced traces over 214) and
its metric instance (`retrieval_recall5` over 56 items against 43). **Fixed:** the metric now
records both depths (`gold_paper_hit_at5` beside `gold_paper_hit`); `make metrics-compare` prints
"in first 5" and "in all retrieved (max k)"; `make metrics-versus` gives per-item counts and the
median retrieved per run. The table rows in PHASE4 and EVALS are split into the two depths, and
README's line states the depth. No gated metric used it; MRR and Recall@5 are unchanged.

Found only because a review asked for item counts instead of means. The per-item view also shows
Recall@5 0.267 vs 0.233 is two items (`sp-023`, `sp-033`), both v3's.

## D-056 — CI was red from run #9 to this fix, and the regression gate never once ran in it

**Found by Srikanth, 2026-09-25, from the Actions tab.** The last green run was #8 (`ee95155`,
2026-08-25). Every run from #9 (`eaa5e4b`, the Phase 4 close-out, 2026-09-24) through
`c7c7e63` failed — six consecutive, plus the duplicate runs pushes produced. Nobody noticed,
because every "green" reported in this project over that month was a local run.

**What failed** (`gh run view … --log-failed`, read-only):

| From run | Test | Why it passed locally |
|---|---|---|
| #9 | `tests/test_docs.py::…test_every_25_of_25_line_names_the_amendment_or_is_cross_arm` — `FileNotFoundError: …/CLAUDE.md` | `CLAUDE.md` is gitignored; it exists only on this machine |
| #10 | `tests/test_deploy_guards.py::…refuses_the_compose_file` and `…refuses_the_provisioning_block` — `ContextError: data/indices/… missing` | the tests assembled the real deploy context, which needs the gitignored FAISS index |

Everything else passed, nothing was skipped.

**What that switched off — worse than the failures.** The unit-test step came first in its job,
and the integration suite, the dataset re-verification and both regression-gate steps were later
steps of the same job, so each was skipped on every run. Run #8, the last green, predates all
four: they were added in `eaa5e4b`, the commit whose run first failed. **So none of them has ever
executed on GitHub.** Every "the gate passed", "the gate fired on an injected regression", "CI
replays committed artifacts" and "a skip is an error" in this repository described the workflow
file and local replays. The gate *logic* is verified — `tests/test_gate.py`, `make gate`, and now
`make ci-local` — but it had never run in CI.

It is the D-023 family's largest instance: not a detector that cannot fire, but **four detectors
switched off by an unrelated failure upstream of them**, with a local environment that made the
upstream failure invisible — the D-022 lesson ("a working local environment is not evidence of a
working declared one") one level up, applied to *files* rather than dependencies.

**Fixed:**
* **The tests read only what git holds.** The docs test checks tracked files only (`tracked()`:
  `git ls-files`, or presence in an export), and a test asserts the filtered list still holds
  README, EVALS and DECISIONS so the filter cannot quietly empty it. The deploy-context tests build
  a fake repository in `tmp_path` — the real `.dockerignore`, Dockerfile and `src/`, stand-in data
  files with manifests that checksum them — and gained a checksum-mismatch and a missing-file case.
* **Mutation-checked.** Disabling each of `check_context`'s four refusals in turn (compose/env
  filename, provisioning block, checksum, missing admitted file) now fails a test each time. The
  check found the compose-file test had never tested the filename rule: it wrote the real compose
  file, whose `LANGFUSE_INIT_` block tripped the text scan with a message that also contains
  "compose", so the filename check could be deleted and it still passed. It now writes a compose
  file with no provisioning block, and a `.env` case was added.
* **The workflow is four independent jobs** — `check` (lint, type, unit tests), `integration`,
  `gate` (dataset re-verification and both gate steps), `clean-install` — with no `needs:`, and
  each gate-job step runs `if: !cancelled()`, so one failure cannot hide the others.
  `tests/test_ci_local.py` fails if a job gains a `needs:` or a gate step leaves the gate job.
* **`make ci-local`** runs the workflow's own `run:` blocks, parsed from `ci.yml`, in a `git
  archive` export of HEAD committed as a one-commit repository (what `actions/checkout` gives the
  runner), in a minimal environment with no `.env` and no shell keys, after asserting the code
  under test is the export's and not this checkout's editable install. `WORKTREE=1` exports the
  working tree's tracked changes via `git stash create` (writes nothing) and lists the untracked
  files it leaves out. What it cannot run — the Docker integration suite, the from-scratch clean
  install — it names as NOT RUN. Its first run found one more gap in itself: a bare archive has no
  `.git`, so `test_space_secrets` (which asks git whether `.env.deploy` is ignored) failed there
  and would have passed on GitHub; hence the one-commit repository.
* **`pyyaml` declared** in the dev extras: `tests/test_deploy_guards.py` had imported it all
  along and only had it transitively through LangChain (D-022's class).

**Result:** `make ci-local` green — Lint, Type check, Tests (691 passed, 42 deselected), dataset
re-verification, gate baseline passes, injected regression fails on recall. **Not verified until
Srikanth pushes:** the GitHub run itself, and in particular the integration job, which has never
run anywhere but locally and needs Docker on the runner; it is independent of the gate now, so if
it fails it fails alone.

**Still not covered by CI, stated:** 42 tests are deselected there — the 7 `network` /
`integration` ones and 35 marked `slow`, which load the embedding model or the FAISS index, absent
on a runner (`pytest --collect-only -q -m "slow or network or integration"`). They run only
under `make test` locally.

**Reverses if:** never. The rule it adds: a green that was not produced from what git holds is not
evidence about CI; run `make ci-local` before reporting CI state, and read the Actions tab.


## D-057 — A local image cache masked a missing dependency: MinIO's public images are gone

**Found 2026-09-26 by the integration job's first-ever run** (run #15, `a17b522`, the first run
in which it executed at all — D-056): `pull access denied for minio/minio`. MinIO stopped
publishing images to Docker Hub and quay.io and went source-only on 2025-10-15 — before this
project began. `make langfuse-up` worked here only because `minio/minio:latest` had been cached
about a year earlier (`docker images`: created 12 months ago). **A fresh clone of this repository
could never start the Langfuse stack**, and nothing local could show it.

**The D-022 family, one layer down.** D-022: a dependency present in the developer's virtualenv
and absent from the manifest. D-056: files present on the developer's disk and absent from git.
Here: an image present in the developer's Docker cache and absent from every registry. Each time
the local environment held something the declared one did not, and each time only a clean
environment could tell them apart.

**Checking the other images turned up drift, not removal.** Every other image still resolves,
but three floating tags no longer meant what was tested here: `langfuse/langfuse:3` and
`langfuse-worker:3` resolve to builds newer than the cached **v3.225.4** the Phase 4 trace window
was captured on and the integration suite ran against, and `postgres:16-alpine` and
`redis:7-alpine` resolve to different digests than the cached 16.15 and 7.4.11. Only
`clickhouse-server:24.3` still matched. A fresh clone would have run a different Langfuse.

**Decided:** every image in `infra/docker-compose.langfuse.yml` is pinned `tag@digest` to the
build tested here (each digest confirmed still pullable, all multi-arch). MinIO moves to the image
upstream Langfuse moved to, `cgr.dev/chainguard/minio` (langfuse/langfuse#10585), pinned at
`sha256:bd014394…` (RELEASE.2026-09-22T19-25-18Z). Upstream's compose at v3.225.4 uses it
*untagged*, so there is no upstream tag to copy; command and healthcheck follow upstream's (the one
difference from ours: an explicit `--address ":9000"`).

**Proved without the cache.** `minio/minio:latest` was saved first (`docker save`, 57 MB, to
`~/Documents/Projects/langfuse-backups/` — it can no longer be downloaded, so removing it
unbacked would have been irreversible), then untagged; the Chainguard image was removed too, so
Compose had to pull it from `cgr.dev` by digest. A fresh stack under a separate project name
(`arxiv-agent-langfuse-fresh`, fresh volumes, web on 127.0.0.1:3100) came up healthy, reported
Langfuse `3.225.4`, MinIO running as uid 65532 and owning its bucket, and the trace integration
suite passed against it (2 passed). The existing stack and its four volumes were not touched; the
test stack and its volumes were removed afterwards.

**The non-root problem is real for the existing volume.** Chainguard's MinIO runs as uid 65532;
the old image ran as root, and the existing volume (`arxiv-agent-langfuse_langfuse-minio`) is
root-owned, mode 755, top to bottom — read through a read-only mount. A fresh volume is created
with `/data` world-writable and works. **The existing volume cannot be used as it is**, so running
`make langfuse-up` against the existing stack with this compose file would recreate the MinIO
container on a volume it cannot write. **Do not run it until the migration below.** The running
containers are unaffected. Migration plan (not done): BACKLOG, "Migrate the local Langfuse
MinIO volume".

**Two more holes in the integration step, found while proving it.**
1. CI ran `pytest -m integration`, which also selects the multi-hop necessity fixtures (marked
   `integration` *and* `network`). They call Gemini; CI has no key by design, so there they
   **skip** — and the step's check (`^[0-9]+ (skipped|no tests ran)`) could not see a partial
   skip, because "2 passed, 5 skipped" starts with "2 passed". The job would have gone green
   with five of seven selected tests skipped.
2. The same step piped `pytest … || true` and then only required "N passed" somewhere, so
   "1 failed, 1 passed" would have read as green.

Now: CI selects `integration and not network` (the two trace tests) and fails on a non-zero
pytest exit, on "skipped" or "no tests ran" anywhere in the summary, or on a summary not starting
"N passed". Checked on the real run (green) and on four doctored summaries: partial skip, a
failure with exit 1, a failure with a false exit 0, no tests (all red). Neither hole had fired —
the step had never run — but either would have hidden a failure on its first run.

**A cost of finding it.** Running CI's old selection locally against the fresh stack picked up
the local `.env` and made **27 Gemini calls** (624,852 input tokens, $0.0858 notional — the
usage log's `pytest` rows, 2026-09-26 05:20–05:26 UTC) on the local key, which sits in a
no-billing free-tier project. I should have checked what `-m integration` selects before running
it.

**Reverses if:** never for the pins. A digest is changed deliberately, with the reason in this
record; `make langfuse-up` on a fresh clone is the test.

**Same round, two follow-ups.**
* **`make ci-local` now reads `ci.yml` from the export**, not the checkout: the commands run are
  the commit's own. Read from the checkout, an edited-but-uncommitted workflow would have been
  run against a commit that does not contain it (tested: the path is relative and resolved
  against the export).
* **The test that starts a span exporter leaked it.** `tests/test_api_observability_faults.py`
  runs Langfuse's real OTLP exporter against local stubs (a 429 server and a socket that never
  answers — nothing leaves the machine). Its teardown started `client.shutdown()` in a
  fire-and-forget daemon thread and returned; the fixtures then closed the stubs, and the
  exporter kept retrying against dead ports into later tests — **six threads still alive two
  seconds after the file finished**, logging `Connection refused … retrying`, all of it hidden by
  pytest's output capture (neither CI's log nor a local run showed it). Two SDK facts made a
  synchronous shutdown not enough on its own: Langfuse 4.14's `shutdown()` does not shut down a
  tracer provider it was handed (the span processor belongs to the provider), and it leaves its
  prompt-cache consumer running. The test now shuts down the client, its provider and the
  prompt-cache task manager before the stubs close, and asserts every thread the client started
  has stopped: 0 left, the file ~0.3 s slower. Reaching the prompt-cache manager uses private SDK
  attributes; if they move, the test errors rather than passing.
