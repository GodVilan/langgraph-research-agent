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
