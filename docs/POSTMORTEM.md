# Post-mortem — arXiv Agent v3

**Draft for review, 2026-09-25.** Covers Phases 0–5: the audit of v2.1, the LangGraph rebuild,
guardrails, observability, evaluation with a regression gate, and the deployed service. Every
figure below names the command that regenerates it or the decision record that holds its
evidence (`docs/DECISIONS.md`, `D-nnn`). Where a figure has no command, that is stated.

This is written to be useful to whoever works on this next, including me. It is not a summary of
what went well.

---

## 1. Why the rewrite

v2.1 ([GodVilan/arXiv-Agent](https://github.com/GodVilan/arXiv-Agent), audited at `8d3e67f`)
worked as a demo. The audit (`docs/AUDIT.md`) found problems that were structural rather than
incidental:

* **No reproducible baseline.** Its README reported `MRR@5 = 0.990` and
  `Context Precision = 1.000` from "100 benchmark QA pairs". That file was deleted in v2.1's HEAD
  commit; recovered from history, all 100 `paper_id`s are `2604.*`, and the corpus that ships is
  `2605.*` — zero overlap by id and by title (AUDIT §5.2, reproducible with the script there). The
  only eval script in the repo scored three hardcoded questions and hardcoded 5/5/5 for one of
  them (AUDIT §5.1). v3 carries none of those numbers forward.
* **Bounds were prompts.** About 40% of the 760-line orchestrator was loop-guard machinery
  ("⚠ You are looping"), because a free-tool-choice ReAct loop has no structural bound. Worst case
  was about 43 model calls for one question, with no cost ceiling (AUDIT §4.7).
* **Guards failed open.** The scope check returned "in scope" on any exception; the critic
  returned `pass` from a bare `except`. A critic that never ran and one that approved were
  indistinguishable (AUDIT §4).
* **Persistence was of completed turns only**, in 911 lines of hand-rolled SQLite. A run that died
  at sub-question 3 of 4 lost everything (MIGRATION_MAP §1.4).

The honest justification, stated at Phase 0 and still true: **checkpointing and structure, not
retrieval quality**. Retrieval was frozen on purpose (D-002) so the comparison would isolate
orchestration. The measurement in §4 bears that out — v3 does not retrieve better.

---

## 2. What broke in the migration

Each of these was found by running the ported system, not by reading it, and most were not caught
by the tests written for the phase.

| What broke | How it surfaced | Record |
|---|---|---|
| Checkpointed Pydantic models came back as plain `dict` on resume. Nothing failed at write time; the first node to read `state["request"].prompt_version` would have raised on a real deployed thread. An allowlist written as module tuples silently did nothing. | a failing checkpoint test | D-008 |
| Synchronous `SqliteSaver` raises `NotImplementedError` under async nodes. | first graph run | D-009 |
| Per-request budget ceilings were enforced **per thread**: `usage` is checkpointed and summed, so `llm_calls` went 3 → 6 → 9 across three turns, and any conversation would truncate after about five. | a live three-turn thread | D-013 |
| The BM25 "fallback" fired on **every** query, and the merge did not truncate. Introduced by the port's new policy layer, not inherited — v2.1 had no automatic fallback at all, which is why no bug-compatibility flag was needed. | a live run | D-015 |
| The pinned model, `gemini-2.5-flash-lite`, returned 404 for new keys mid-phase; its successor ignores `temperature`, so v2.1's greedy-decoding determinism assumption no longer held. | the first live run | D-012, D-014 |
| The placeholder rate card for the new model understated cost about 3.6×. | the `verified: bool` flag on every rate entry | D-012 |
| Phase 5, while serving: the CLI's `--stream` ran the graph **twice** per question; the model pacer started empty with a bucket of one (16.6 s for a 3-call query); the volume mounted root-owned; `host_is_local` was a substring test (`localhost.attacker.example` counted as local); Langfuse silently dropped `sample_rate`; `/ready` reported 0 chunks. | building and running the container | D-041 |

What did *not* break: the ported index. It is not bit-identical to v2.1's (float32 noise), but
top-5 and top-10 rankings are identical on 500 probe queries (`make index-verify`, D-006).

---

## 3. What traces revealed that v2.1 hid

v2.1's only instrumentation was logging: at `8d3e67f`, 20 files under `rag/` log, and none imports
Langfuse, OpenTelemetry or a metrics client (`grep -rl "langfuse\|opentelemetry\|prometheus"
--include='*.py'` over the clone finds nothing). A slow or expensive question could not be
attributed to a stage without re-running it by hand. v3 emits one trace per query — root, each node,
each model call, retrieval and embedding spans; 32 observations for the README example on the
deployed instance (`make smoke-live`, trace-complete check).

What that made visible, in order of how much it changed the project:

1. **The spend table was arithmetic on two populations.** It divided cost from 6 priced traces by
   tokens from 214, and published a blended rate *below* the input-only price, which no token mix
   can produce. 187 of the 214 were synthetic runs the test suite had been writing into the same
   project, and 5 were duplicate roots (D-021; `make reconcile-d021` reproduces it from a frozen
   fixture). After the fix, three surfaces agree on the Phase 4 window — Langfuse's total, the
   stored notional, and a recomputation from tokens all give $0.236004, 0 duplicate roots, blended
   $0.3433 per 1M (`make reconcile-cost`).
2. **What a query actually costs.** Median 4 model calls per query, maximum 4, $0.0033 notional
   median and $0.0057 maximum over the 69 frozen items (`make run-report`) — against v2.1's worst
   case of about 43. The per-request reservation of $0.025 (16 × the maximum observed per-call
   cost) and the daily ceiling were derived from these, not guessed (docs/BUDGET.md, D-037).
3. **What tracing costs.** 43 Langfuse units per traced query, so the Hobby plan's 50k units cover
   about 38 traced queries a day (`make trace-units`, D-039).
4. **What a crash loses.** SIGTERM leaves a complete trace (32 observations); SIGKILL leaves 15 of
   32 with no answer, because the root span and the late nodes were still buffered
   (docs/OBSERVABILITY.md, `scripts/span_loss_probe.py`). Existence of a trace was reported for
   both; only completeness distinguished them.
5. **Which guardrail fired, and whether a critique ran.** Every trace carries the guardrail events
   and the critique verdict, so "critique errored" and "critique passed" — the pair v2.1 made
   indistinguishable — are separate records.

Two limits on this section. First, several of the largest findings came from **running**, not
from traces: the per-thread budget, the BM25 fallback, the double-run CLI, and all of §7's
failure classes. Second, traces measure what they are attached to. They said nothing about the
76% of Gemini usage in §5, because the code paths that spent it were not traced.

---

## 4. v2.1 versus v3 — the numbers, including where v3 is worse

Same frozen 69 items (sha `de699d68`), same 150-paper corpus, same generator
(`gemini-3.5-flash-lite`), v2.1 a clone of published `main` pinned at `8d3e67f` and asserted clean
before the run, retriever frozen identical (D-002, D-033). Judge: `gpt-5.6-luna` at `low`,
two-stage, validated against 25 human-scored answers (D-030, D-032). Local batch, single user —
not serving figures. v3 is three unpinned runs; v2.1 is one. `make metrics-compare`,
`make metrics-spread`.

**Retrieval, n=43 factual items** (v3's retrieval spread across three runs is 0.000):

| | v3 | v2.1 | Better |
|---|---:|---:|---|
| Recall@5, chunk level | 0.267 | 0.233 | v3 |
| MRR | 0.175 | 0.196 | **v2.1** |
| gold paper in the top 5 | 21–22 of 43 | 28 of 43 | **v2.1** |
| items where neither retrieves the gold chunk | 30 of 43, the same 30 | | neither |

**Outcomes, n=46 answerable, n=10 ambiguous** (v3 shown as r1 / r2 / r3, spread in brackets):

| | v3 | v2.1 | Better |
|---|---:|---:|---|
| correct answers | 15 / 16 / 16 (1) | 6 | v3, outside the spread |
| wrong answers | 5 / 4 / 6 (2) | 11 | v3, outside the spread |
| hallucinated refusals — refused an answerable item | 26 / 26 / 24 (2) | 29 | v3 by 3, just outside the spread; **both are bad** |
| ambiguous items where it asked which paper | 4 / 3 / 3 (1) | 0 | v3 |
| attribute refusals, n=11 | 11 / 11 / 11 (0) | 11 | tie — a ceiling; this stratum cannot separate systems |

The mechanism behind every row: v2.1's loop issues several searches and unions them — a median of
18 chunks from 6 papers per item, against v3's 5 from 3 (docs/PHASE4.md §2). The wider net finds
the gold paper more often and ranks worse, and it hands the generator more wrong papers: 7 of
v2.1's 11 wrong answers cite only non-anchor papers. v3, given less, refuses more and answers
wrongly less. **It is a precision-for-recall trade, not an improvement**, and neither system is
good at this set.

**Where v3 is worse, stated plainly:**

* It ranks worse (MRR 0.175 vs 0.196) and finds the gold paper less often (21–22 vs 28 of 43).
* **It refuses 26 of 46 answerable questions.** That is the largest single failure in the
  artifact. D-029 traces it to one construction rule failing on three surfaces (§7).
* Its input guardrail refuses in-scope questions v2.1 answered. Pinned, as it ships, the scope
  classifier refuses the same 5 of 43 verified factual questions on every call; unpinned it
  refused 3 to 6 per draw, mean 4.3 over 7 draws on identical input (`make guardrail-probe`,
  `make guardrail-variance`, D-035). Pinning fixed the stricter end in place: `sp-016`, refused in
  2 of 7 unpinned draws, is now refused every time, and a user who hits one of the five no longer
  has a chance on retry.
  v2.1's own scope check marked 3 of the 69 out of scope in its one run (`make metrics-compare`):
  fewer than v3's pinned 5.

**Not claimed:** anything about multi-hop (n=3, a per-item case study, EVALS.md) or unanswerable
topics (n=2, effectively 1 clean, D-031); any confidence interval (three draws cannot support one);
any latency comparison (the two systems were never timed under the same conditions).

**The shipped configuration** (classifier pinned) was re-measured over three more runs and gates
on its own baseline: hallucinated refusals 27, 27, 25 against the unpinned 26, 26, 24 — a +1.0
difference inside either spread, not shown to be a difference (D-044,
`evals/baseline_metrics_pinned.json`). A pinned run of the deployed commit passes that gate
(D-049 addendum).

**Serving, deployed instance, one client machine, not live traffic** (`make load-report`,
`make load-report LABEL=single_user`): single user, warm, n=10, p50 7.1 s, p95 14.3 s; ten
concurrent users, n=32 served of 169, p50 48.4 s, p95 88.3 s, 3.2 served queries a minute —
the ceiling is the model's free-tier quota, not the service; cold start 39.6 s to ready.

---

## 5. The spend story

Two providers, watched very differently.

**OpenAI — the judge — $0.1827 of a $5 lifetime ceiling**, 1,028 requests across 29 batches
(`make judge-spend`). Batch API only, a hard cap and a soft alert confirmed before the first call,
a receipt per batch, a per-run cap stated before every submission, the estimator's figure shown
beside the last three measured actuals, and an account-side cross-check that refuses to publish a
total while any batch is unreceipted. Even so, the published figure was wrong for a phase:
$0.0942 over 465 requests, when the account said $0.1196 over 673 — stale, and undercounted
because three receipts had been overwritten, hiding 117 paid requests (D-043).

**Gemini — the agent, assumed free — $7.60**, from Google Cloud Billing's SKU export for
2026-08-19 to 2026-09-24, hand-entered with source and date in `docs/billing/gemini.json`
(D-046). About 42× the OpenAI spend, on the provider with no ceiling, no receipts and no
cross-check, because the key's project had billing enabled throughout and every "billed $0" in
the repo was a default multiplied by tokens. 89% of it was uncached input and 9% output
(`make readme-stats`, rendered from the billing record): the lever was context size, not the
pinned thinking budget.

**76% of the billed prompt tokens were never recorded by anything in the repo.** Google billed
27,277,355 prompt tokens; every source of instrumentation together saw 6,530,193; the gap is
20,747,162, about 210 multi-hop construction candidates' worth (`make gemini-reconcile`). Eval-set
construction, the necessity fixtures, v2.1's baseline run and the probes all discarded the usage
the wrapper handed them. Recording is now structural — every call through the wrapper appends a
row to the usage log whether or not the caller keeps the usage, and a test fails any code path
that builds a model or calls the Gemini API around it (`tests/test_usage_log.py`). That fixes the
future, not the $7.60.

The asymmetry is the finding. Every control guarded the provider that was being watched; the one
assumed free had none. The rule that came out of it: **a billed figure comes from the provider's
own record, or it is shown as "unverified" — never 0 by default** (D-046). Both keys now sit in
separate no-billing projects.

---

## 6. Provider churn, and what each change cost

Seven external changes under one build. None was caused by anything in the repo, and pinning a
version would have prevented none of them.

| # | What changed | Cost | Record |
|---|---|---|---|
| 1 | v2.1's corpus swapped and its benchmark deleted | no inherited baseline; the 69-item eval set was built from scratch | AUDIT §5 |
| 2 | `gemini-2.5-flash-lite` retired for new keys | a mid-phase model switch, loss of determinism, and a 3.6× pricing error caught by `verified: bool` | D-012 |
| 3 | Cerebras pruned its free catalog | a judge arm re-costed | D-034 |
| 4 | Hugging Face included credit exhausted, `402` at 7 of 25 | the open-weights judge arm left partial; no rate computed from it | D-030b |
| 5 | OpenAI account deactivated for four days, at about $0.022 spent, no policy issue found | judging blocked; recovered because batch ids were archived at submit time and every judge input was persisted provider-independently | D-034 |
| 6 | Hugging Face Docker Spaces moved behind PRO | $9/month and a host decision at the ship gate | D-038 |
| 7 | Langfuse Cloud answers `410` on the legacy traces API for orgs created from 2026-09-16 | the read-back rewritten to the v2 observations API; the local stack (v3) can no longer test it, and v4 needs ClickHouse ≥ 26 | D-046 (note), D-047 |

What made these survivable generalises: the work was provider-independent (runs persist
everything a judge needs), handles were archived at the moment of submission, fallbacks were
measured before they were needed, and every failure was loud — no judge was ever silently
substituted (D-030b).

---

## 7. Recurring failure classes

Individual defects were fixed as found. These are the shapes that kept coming back. Each has more
than one instance, and the later instances happened after the earlier ones were recorded.

### 7.1 A step that reports success without having happened — the D-026 family

* A check written to catch a rejected item, imported by nothing (D-026).
* A grounding check that always ran with an empty answer, so it always passed (D-026).
* A batch submit that `401`'d, wrote no receipt, and logged "SUBMITTED" — two arms would have been
  silently absent (D-026, D-034).
* `202 deletion queued` counted as deleted: 239 scores where 115 were intended (D-026, fourth).
* Score ingestion assumed from "no exception" (found by the audit that followed the fourth).
* A ledger settlement that never read Upstash's reply, so a malformed `200` was taken as a
  completed write — and for a run that overshot its reservation, a lost settlement under-counts
  the daily ceiling (D-054).
* A regression gate that computed outcomes over a half-judged sheet (49 of 69) and reported
  correct answers 0 against 15 as a regression (D-053).

The fix that generalises is the one from D-026's fourth instance: **assert the effect, not the
status**. Read back what was written; count what was scored; refuse input that is not complete.

### 7.2 Running is not blocking — the D-023 family

Detectors that existed and could not fire: a CI workflow that never triggered because the project
never opens pull requests; a docs test calling `.venv/bin/python`, absent on every runner, and
skipping its own failure; a `pytest.skip` swallowing "Event loop is closed" as "could not reach
the model" inside the very fix written for this class; and CI's injected-regression step, which
read any non-zero exit as the gate firing, so a *refusal* would have passed it (D-023, D-053).
**Every detector needs a case that makes it fire, and the harness must check it fired for the
right reason.**

### 7.3 Numbers that cannot be regenerated — the C-1 class

A number without a producing command survives every review if it looks authoritative:

* README's "Phase 3 of 5" and "$0.00 spent", outliving Phase 4's close.
* The OpenAI spend, $0.0942 (D-043).
* The Phase 4 regression baseline, whose JSON said "mean of three runs; tolerance = max-min
  spread" while no committed code produced it. It survived every review *because* it described
  its own derivation (D-045).
* Every "Gemini billed $0", which had no source at all — only a default (D-046).
* D-047's "about 2 of ~10 draws" citation estimate, replaced by a measurement (D-047 note).

Now: generated blocks in README and BUDGET, rendered from artifacts by `make readme-stats`, with
tests that fail a stale or hand-typed copy.

### 7.4 Two populations in one ratio — the D-021 family

The spend table's blend (6 priced traces over 214), the dashboard's `retrieval_recall5` over 56
items against the documents' 43 (D-021 on a metric), and — while drafting this document — the
Gemini reconciliation counting a run from outside the billing period, which moved the published
gap from 76% to 74% (D-021, third instance). Where a bound is structurally knowable, assert it;
where a population is defined by an external record, filter to it.

### 7.5 External-facing tools that trust their environment

* **No backoff.** The first deployed load check retried every instant rejection at once and sent
  about 74,000 requests in minutes; 70,510 were answered by Hugging Face's platform rate limiter
  (D-048). The service's limits were correct; the client written to measure them had none.
* **Assumes the machine never sleeps.** The first rerun ran from a laptop that slept for two hours;
  the tool timed on `time.monotonic`, which stops in sleep, and reported a 600 s run and a 2.3 s
  cold start that were neither (D-052).

Any tool that calls someone else's infrastructure now needs `Retry-After`, a backoff, a hard stop
on terminal refusals, a request cap, and a check on its own clock — each with a test that makes it
fire.

### 7.6 One rule, three surfaces — D-029

The eval construction rule "paraphrase away the paper's vocabulary so retrieval cannot win by
string matching" was right on its own. Three components key on that vocabulary, and each had
passed its own validation: the scope classifier refused 5 of 43 verified questions before
retrieval; the retriever missed the gold chunk at k=10; and the generator refused `sp-036` with
the answer quoted inside its own refusal. Each component was validated on input that still had
the vocabulary. The interaction had no test until the shared input ran through all three.

### 7.7 An instruction is a request; a mechanism is a guarantee

Learned at both ends: telling the model that delimited text is data did not stop delimiter-escape
payloads (structural neutralisation did, Phase 2); a drafting prompt banning "synthesis" still
produced 8 research proposals out of 8 (a post-generation regex rejects them, Phase 4). The same
principle drove the ledger that refuses to start when non-durable (D-037) and the usage log that
cannot be bypassed (D-046).

---

## 8. What I would do differently

1. **Check the provider's billing record on day one, and weekly.** A single comparison of our
   "billed" figure with Google's would have caught D-046 in August instead of September. Cost:
   minutes.
2. **Make usage recording structural before the first model call.** The 76% gap exists because
   recording was each caller's job.
3. **Emit every number from the start.** The generated-block mechanism arrived in Phase 5; every
   C-1 instance predates it.
4. **Build the eval set before the orchestration it measures, and review it by hand early.** 69 of
   a planned 100 items survived, and the shortfall is almost entirely tooling measuring the wrong
   property — multi-hop went from 20 to 3 (EVALS.md). The corpus was not the constraint for any
   stratum except unanswerable topics.
5. **Run shared input through every component before trusting any component's validation** — the
   D-029 lesson, which cost the project its largest failure.
6. **Test pinned sampling in Phase 1.** D-014 concluded determinism was unrecoverable after trying
   temperature and thinking budget; `seed=0, top_k=1` recovers it for both the classifier and the
   generator (D-035, D-042). Knowing that before Phase 4 would have changed the variance design.
7. **Run measurement tools from a host that cannot sleep and has backoff by default.** D-048 and
   D-052 were both the client, not the service.
8. **Write the gate's input contract with the gate**, not after it misfired (D-053).

---

## 9. What this system still cannot do

* **Answer most answerable questions in its own eval set.** It refuses 26 of 46 in Phase 4
  (`make metrics-spread`), and the scope guardrail blocks 5 of 43 in-scope questions on every call.
* **Support any claim about multi-hop retrieval or orchestration benefit on multi-hop.** n=3.
* **Serve more than a few queries a minute.** 3.2 served per minute under ten concurrent users;
  p50 48.4 s under that load (`make load-report`). The ceiling is the model's free-tier quota.
* **Resolve follow-up questions.** A follow-up is retrieved without its antecedent (BACKLOG).
* **Scale past one process.** Rate limits, the concurrency gate and the model pacer are per
  process; only the daily ceiling is shared.
* **Authenticate anyone.** Thread ids are bearer capabilities.
* **Detect injection it was not written for.** 28 of 36 adversarial cases detected (78%); the 8
  known gaps — encoded, non-English, persuasion without imperatives — pass detection and rely on
  structural neutralisation (`make injection-report`).
* **Pin its judge.** No provider exposes a weights revision for the judge model; it is the one
  input the eval artifact cannot pin (D-030).
* **Know whether it may redistribute 54 of its 150 papers.** They are under arXiv's non-exclusive
  distribution license; whether a public repository may carry their full text is not verified
  (D-051).
* **Answer outside one day of cs.LG.** 150 papers, one category, one publication date; live arXiv
  fetch is off on the public endpoint (D-036).
* **Report billed cost from its own instrumentation.** Billed figures come only from the
  provider's record, entered by hand (D-046).

---

*Sources: `docs/DECISIONS.md` (D-001–D-054), `docs/BACKLOG.md` ("Reserved for the post-mortem"),
`docs/AUDIT.md`, `docs/MIGRATION_MAP.md`, `docs/PHASE4.md`, `docs/EVALS.md`, `docs/BUDGET.md`,
`docs/OBSERVABILITY.md`, `docs/SERVING.md`.*
