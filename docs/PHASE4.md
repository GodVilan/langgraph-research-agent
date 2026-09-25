# Phase 4 close-out — evaluation, metrics, and the regression gate

What was built, what it measured, and what the measuring cost. Written as the raw material for
the Phase 8 post-mortem, so the findings are stated once and in full rather than summarised.

Every number here is regenerable: `make verify-dataset`, `make metrics-compare`,
`make metrics-spread`, `make judge-spend`, `make gate`, `make run-report`.

---

## 1. The artifact

**A frozen 69-item eval set**, `evals/datasets/phase4.json`, sha `de699d68`, over the
150-paper corpus `e1be96d1`. 43 `single_paper_factual` (29 strong-gold / 14 weak),
3 `multi_hop`, 2 `unanswerable_topic` (1 clean — D-031), 11 `unanswerable_attribute`,
10 `ambiguous`. Target was 100; **why it is 69 and not 100 is construction defect, not corpus
scarcity**, itemised in `docs/EVALS.md`. The freeze criterion was a *reporting* condition, not
a nothing-found condition: zero gate disagreements, a reviewed hand-authored defect matrix, and
every shortfall attributed.

**One scoring rubric** (`evals/rubric.py` → `docs/RUBRIC.md`), applied by the human scorer and
every judge arm from the same text, amended once mid-experiment (v2, the responsiveness
criterion) with the amendment named on every figure it touches.

**A shipping judge**: `gpt-5.6-luna` at `low`, Batch API, two-stage blind-Q1 prompt. It did
not win — three sheets agree with each other on 25 of 25 — so cost and pinnability decided
(D-032).

**Total OpenAI spend through Phase 4: $0.1196 of the $5.00 lifetime ceiling**, 673 requests
across 21 batches (`make judge-spend`, which now checks the account's own batch list). This
document originally said $0.0942 over 465 requests; that figure was stale — the traced run's
judging landed after it was measured — and undercounted — three receipts had been overwritten,
hiding 117 paid requests. Corrected 2026-09-24, DECISIONS D-043. The agent side was
recorded here as "$0.00 billed (Gemini free tier)". It was not: the key's project was billed
throughout, $7.60 over the whole project by the provider's record (D-046). $0.0033 notional per
query median.

---

## 2. v2.1 versus v3 — a precision/recall trade, not an improvement

**The mechanism first, because it explains every difference in one sentence.** v2.1's ReAct
loop issues several searches per question and unions the results — a median of **18 chunks from
6 distinct papers** per item, against v3's **5 chunks from 3**. A wider net finds the gold
*paper* more often and ranks worse, and it puts more wrong papers in front of the generator:
**7 of v2.1's 11 wrong answers cite only non-anchor papers.** It answers confidently from the
wrong paper where v3, given a narrower context, refuses.

Same 69 items, same corpus, same generator (`gemini-3.5-flash-lite`), v2.1 a clone of published
`main` pinned at `8d3e67f` with a clean tree asserted before the run, **retriever frozen
identical between the two (D-002)**.

| n=43 factual | v3 (r1–r3, spread 0.000) | v2.1 |
|---|---:|---:|
| Recall@5, chunk level | **0.267** | 0.233 |
| MRR | 0.175 | **0.196** |
| gold paper in the top 5 | 21–22 of 43 | **28 of 43** |
| items where neither retrieves the gold chunk | 30 of 43 — the same 30 | |

| n=46 answerable | v3 (r1 / r2 / r3) | v2.1 |
|---|---:|---:|
| correct answers | 15 / 16 / 16 (spread 1) | **6** |
| wrong answers | 5 / 4 / 6 (spread 2) | **11** |
| hallucinated refusals | 26 / 26 / 24 (spread 2) | 29 |
| ambiguous clarified (n=10) | 4 / 3 / 3 | **0** — 9 of 10 silently disambiguated |

**This is a trade, not an improvement.** v3 buys answer precision with recall breadth: fewer
wrong answers (5 vs 11) and more correct ones (15 vs 6), both outside the three-run spread, at
the cost of ranking and gold-paper hit rate. Neither system is good at this set.

**Both failure modes are visible and quantified.** v2.1 hallucinates from wrong papers (7 of 11
wrong answers) and never asks which paper is meant (0 of 10 ambiguous items). v3 **refuses 26
of 46 answerable items** — the largest single failure in this artifact, explained by D-029.

**v3's justification is orchestration, checkpointing, guardrails and observability. The
measurement shows no retrieval gain, and v3 must never be presented as retrieving better.**

---

## 3. The variance decomposes

Three complete runs of the shipping system, each judged in full:

* **retrieval spread 0.000** — Recall@5 and MRR identical across three runs
* **outcome spreads nonzero** — correct answers ±1, wrong ±2, hallucinated refusals ±2,
  ambiguous clarified ±1, attribute refusal accuracy ±0

**The variance lives entirely in unpinned generation and judging; retrieval contributes none of it.** (Rescoped 2026-09-24: generation here is unpinned. `seed=0, top_k=1` makes the generator's output byte-identical on a 5-draw probe — D-042 — so the generation share is a property of the sampling configuration, not of the model.)
The reporting rule that follows: outcome metrics carry their spread inline, retrieval metrics do
not, and a difference no larger than the spread has not been shown to be a difference. Three
draws, no confidence interval — n=3 cannot support one.

The gate's tolerances *are* this spread (`evals/baseline_metrics.json`), so the gate fires on
real movement and not on noise. It has been shown to fire three ways: on a real regression
(`section_filter`, four metrics named), on an injected retrieval collapse (test plus two CI
steps, one of which errors if the injection ever passes), and it passes the clean run. CI replays
committed run artifacts through the gate; it does not run the agent on pushed code (D-050).

---

## 4. Method note — verifying a flattering result before publishing it

The v2.1 outcome comparison favours the system I built, which is the condition under which a
number deserves the most suspicion, not the least. v2.1's answers carry HTML citation spans
(`<span class="citation" data-paper-id="…">`) that v3's do not, and the rubric says formatting
is not scored — but a judge could still be penalising the format while appearing to score
content.

So before publishing the 15-versus-6 figure I read the judge's reasons on v2.1's wrong answers
against the gold. They are substantive: on `sp-022` the gold paper is `2605.29713` and v2.1
answered from `2605.30189`; on `sp-043` gold is `2605.29908` and it answered from `2605.30213`.
The errors are wrong-paper answers, not formatting penalties. Counting them gave the 7-of-11
figure that became the mechanism in §2 — the check for an artifact produced the explanation.

**Stated as a practice, not an incident: a result that flatters the system under test gets
checked for measurement artifact before it is published, and the check is reported whether or
not it finds anything.** The same discipline produced the D-021 cost reconciliation and the
`bypass-probe` decomposition, both of which were run *because* the convenient reading was
available.

---

## 5. The central methodological finding — running found what reasoning did not

Across four phases, **every significant defect was found by running the system against real
data, not by reasoning about it, and in several cases not by the tests written to catch it**.
Phase 1's live runs caught three bugs the suite passed — per-request budgets enforced
per-thread, the BM25 "fallback" firing on every query, checkpoint models deserialising as
`dict`. Phase 2's twelve fresh adversarial probes, written *after* the rules, evaded ten of
them; four became new rules. Phase 3's full-corpus screen found **175 chunks (3.24%) would be
wrongly quarantined** where a hand-written benign set of eight had reported zero false
positives. Phase 4's hand-scoring of 25 items found defects that **eight rounds of automated
checking, an artifact gate, and a hand-authored defect matrix had all passed** — `sp-036`, where
the generator retrieved the fact, quoted it, and refused anyway; and `ut-003`, where the item
itself was wrong and the agent was right. And the judge experiment's own control invalidated
its discriminating power: the generator scoring its own answers agreed with the human exactly as
the third-vendor reasoning judge did, so **25 rubric-scorable items cannot distinguish any
judge from any other** — a finding about the instrument that no amount of reasoning about judge
selection would have produced.

The corollary, which is the transferable part: a check that has never fired is not evidence,
and the fastest way to learn whether a check can fire is to run the thing it guards against.
Ten instances of "running is not blocking" (D-023, D-026) came from that principle applied to
this project's own tooling.

---

## 6. What is not done

* ~~Screenshots not recaptured~~ — **done 2026-09-23.** Both images are the Phase 4 window:
  69 traces, `$0.236004`, and a **populated Scores panel** (115 scores — `outcome_correct` on
  69 items, `retrieval_recall5` on 46). The synthetic-volume caveat is removed from
  `README.md` and `docs/OBSERVABILITY.md`, because the count is now real usage rather than
  test traffic; the contamination it described is still documented as the diagnosis that
  produced D-021, reproducible from the frozen fixture via `make reconcile-d021`.
* ~~No Phase 4 run is traced~~ — **a traced run of all 69 exists**
  (`evals/runs/v3_de699d68_traced.json`), judged 69/69 by the shipping judge, with per-item
  `trace_id` recorded. `make budget` and `make reconcile-cost` are regenerated from it: 69
  priced traces, **0 unpriced, 0 duplicate roots**, blended `$0.3433` per 1M, Langfuse's own
  total agreeing with our instrumentation to the cent. The spend table is trace-generated as
  `docs/BUDGET.md` promises.

  Two defects were found by checking those screenshots against the docs before committing
  them — the artifact-verification discipline's third and fourth (see §4): `retrieval_recall5`
  had been pushed over 56 items against the docs' 46, and the re-push counted Langfuse's
  `202 deletion queued` as deletion completed. Both recorded in `docs/DECISIONS.md`; the
  second prompted an audit of every status-code site in the codebase, which found the same
  inference live in a second place.
* `oss120b` and `gemma-4-31b` judge arms were **skipped, not run** (D-032): three sheets
  already agreed 25 of 25, so a fourth and fifth would confirm rather than inform.
* Multi-hop is **n=3, a case study reported per item by name**, and cannot support Recall@k,
  MRR@k, or an orchestration delta. `unanswerable_topic` is n=2, effectively n=1 clean.
