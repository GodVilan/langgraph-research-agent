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
4 should reset it from measured per-query distributions.

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

**Cost:** the spend table under-reports whenever instrumentation is missing. That is the
right direction to be wrong in, and the unattributed count makes the gap visible rather than
absorbing it into a plausible-looking total.
