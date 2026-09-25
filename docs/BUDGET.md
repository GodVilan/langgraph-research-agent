# BUDGET

**Hard ceiling: $5.00 total OpenAI spend for the lifetime of this project.** Gemini had no
ceiling, because it was assumed free — and it was the larger bill (below, D-046).

OpenAI is the evaluation judge only — never the agent's generator, planner, critic, or
guardrail classifier, and never the embedding model. The agent graph runs on Gemini
Flash-Lite. It was believed to run on the free tier; the key's project had billing enabled
throughout ([`DECISIONS.md`](./DECISIONS.md) D-046). Rationale for the split: D-001.

<!-- JUDGESPEND:START -->
<!-- Rendered from evals/runs/judge_spend.json by `make readme-stats`; the figure is summed from the batches by `make judge-spend`. -->
**OpenAI judge spend: $0.1827 of the $5.00 lifetime ceiling** — 1028 Batch requests, 1,487,447 input / 56,605 output tokens at batch rates, measured 2026-09-25.
<!-- JUDGESPEND:END -->

<!-- GEMINI:START -->
<!-- Rendered from docs/billing/gemini.json (hand-entered from the provider's record) and evals/runs/gemini_reconcile.json by `make readme-stats`. -->
**Gemini API spend: $7.60** — provider-sourced (Google Cloud Billing — Gemini API (service AEFD-7695-64FA), SKU export, 2026-08-19 to 2026-09-24), hand-entered 2026-09-24 into `docs/billing/gemini.json`. 27,277,355 prompt tokens (4,703,996 of them cached) and 274,770 output tokens on `gemini-3.5-flash-lite`; **89% of the cost was uncached input and 9% output**. Only 24% of those prompt tokens were seen by any instrumentation in this repo (`make gemini-reconcile`, DECISIONS D-046).
<!-- GEMINI:END -->

**A billed figure comes from the provider's own record, or it is shown as unverified — never
as $0 by default** (D-046). The Gemini line above is the provider's record, hand-entered with
its source and date; nothing in this repo computes a billed Gemini figure.

**Which parts of this file are generated.** The OpenAI spend line, the Gemini line (from the
hand-entered provider record `docs/billing/gemini.json` and `make gemini-reconcile`), the
per-line spend table, and the table under
[Agent-side spend, from traces](#agent-side-spend-from-traces) are rendered by commands
(`make judge-spend`, `make gemini-reconcile`, then `make readme-stats`; `make budget`) and must
not be edited by hand —
`tests/test_docs.py` fails a stale copy. Everything else here (unit prices, allocation,
controls, the update protocol) is hand-written and dated where it states a fact.

---

## Unit prices

Provider documentation, verified 2026-08-19 except where marked. **Do not edit these from
memory** — re-verify against provider docs and update the date.

| Model | Input /1M | Output /1M | Notes |
|---|---|---|---|
| `gemini-3.5-flash-lite` paid standard (**in use, billed**) | $0.30 | $2.50 | Output rate **includes thinking tokens**. Cached input **$0.03** — all three rates matched by the Cloud Billing SKUs, 2026-09-24 |
| `gemini-3.5-flash-lite` paid batch/flex | $0.15 | $1.25 | |
| `gemini-2.5-flash-lite` (paid) | $0.10 | $0.40 | **Retired for new API keys** — see below |
| `gpt-5.6-luna` | $0.20 | $1.20 | Cached input $0.02; cache writes 1.25× uncached |

> **The agent runs on `gemini-3.5-flash-lite`.** `gemini-2.5-flash-lite` returns 404 for
> this API key ("no longer available to new users"), found on the first live run
> ([DECISIONS D-012](./DECISIONS.md)). Its rates were briefly used as a `verified=False`
> placeholder and **understated cost by roughly 3.6x**; the figures below are the corrected
> ones. This line used to add that billed cost was "$0 on the free tier regardless". It was
> not: the key was billed at exactly these paid rates (D-046).
>
> Notional cost uses **standard**, not batch: the agent serves interactive requests, so
> batch pricing would understate what a paid deployment would pay. Batch pricing does apply
> to the OpenAI judge, where it is mandatory.

Both providers bill thinking/reasoning tokens at the **output** rate, which is why
`reasoning_effort` is pinned explicitly rather than left at its default. This file used to call
reasoning tokens "the dominant cost variable, not input size". The bill says the opposite: **89%
of Gemini cost was uncached input, 9% output** (thinking included). The pin worked; context size
is the lever (D-046).

OpenAI has **no free tier**. Tier 1 is the entry point: 500 RPM, 500K TPM, 5M batch queue
limit. Credit must be loaded before any judge call. A full 100-item run is roughly 510K
enqueued tokens — comfortably inside the Tier 1 batch queue, so no throttling is expected.

---

## Spend outside the OpenAI ceiling

A budget document that tracks one provider while another accrues silently is the wrong shape,
however small the other is. The `gpt-oss-120b` judge arm is served through the Hugging Face
inference router and billed to **Hugging Face account credit**, not to the $5 OpenAI ceiling.
No router provider lists the model as free — prices on 2026-09-18 ranged from $0.037/$0.17 to
$0.35/$0.75 per 1M tokens (in/out) across eleven providers, a ~10x spread — so the provider
actually served matters and is recorded per run in `evals/runs/spend_oss120b.json`.

| Line | Requests | Provider served | Tokens | Billed to |
|---|---:|---|---|---|
| Judge arm `oss120b`, 25-item validation, two stages — **paused at 7 of 25 q1 verdicts** (2026-09-18) | 7 + 2 probes | **cerebras**, chosen by default routing | ≈2.8K in / 0.9K out (one measured call: 400 in, 384 cached / 132 out, 87 reasoning) | HF monthly included credit, now depleted (HTTP 402); no purchase made |

**The default routing chose the wrong side of a 10× spread.** Eleven providers serve this
model at prices from $0.037/$0.17 to $0.35/$0.75 per 1M (in/out). The router, unpinned,
served every successful call from **cerebras at $0.35/$0.75** — the most expensive input rate
listed. A document whose purpose is being right about money records that, not only the
spend: finishing the remaining 28 requests costs ≈$0.026 at the provider the router picked
and ≈$0.004 at the cheapest, and the choice is made by pinning `openai/gpt-oss-120b:deepinfra`
explicitly, which changes the serving stack and is named in the spend record when it does.

## Planned allocation

Assumes ~4,000 input and ~1,100 output tokens per judged item (300 visible + ~800 reasoning
at `medium`), Batch API throughout.

| Line | Items judged | Allocation |
|---|---:|---:|
| Judge validation (25 items × 2 configs) | 50 | $0.05 |
| Phase 4 dev — 25-item subset runs (×10) | 250 | $0.27 |
| Phase 4 dev — 100-item full runs (×6) | 600 | $0.64 |
| Baselines: v2.1, v3, embedding comparison arm | 300 | $0.32 |
| CI subset runs (×40) | 1,000 | $1.06 |
| Phase 6 variant experiment, if reached | 200 | $0.21 |
| **Planned total** |  | **$2.55** |
| **Unallocated reserve** |  | **$2.45** |

The allocation above is the plan and is not updated as money is spent. What was actually
spent, per line, is generated from the batch receipts (the typed "Spent" column this table
used to carry still read $0.00 for runs that had cost most of the total):

<!-- SPENDTABLE:START -->
<!-- Rendered from evals/runs/judge_spend.json by `make readme-stats`. -->

| Line | Batches | Requests | Spent (batch rates) |
|---|---:|---:|---:|
| Judge validation — the 25-item sample, three judge configurations | 6 | 60 | $0.0061 |
| Full runs of the shipped configuration (r1, r2, r3, traced, pinned) | 16 | 715 | $0.1286 |
| Comparison arms (dense_only, section_filter, v2.1) | 7 | 253 | $0.0481 |
| **Total** | 29 | 1028 | **$0.1827** |
<!-- SPENDTABLE:END -->

Judge validation also showed that `medium` spent ~1.6× `low`'s output tokens and moved zero
labels (D-032).

---

## Mandatory cost controls

These are requirements, not suggestions. Each one has a specific failure it prevents.

| Control | Prevents | Status |
|---|---|---|
| **Batch API for every judge call** | Standard rates are exactly 2× batch. Forgetting doubles the project to $5.10 and consumes the entire budget. The eval is asynchronous by nature; there is no reason to pay standard rates. | Phase 4 |
| **`reasoning_effort` pinned in config, never defaulted** | Luna's default is `medium`. Validate at `low` first — for rubric scoring against a fixed schema it may suffice and saves roughly $1 across the project. Escalate only if judge–human agreement is poor at `low`, and record the comparison. `high` and above are out of budget. | Phase 4 |
| **Log `reasoning_tokens` per call to Langfuse** | This table is generated from trace data, not hand-maintained. | **Done (Phase 3)** — `Usage.reasoning_tokens` is populated from `output_token_details.reasoning`, attached to every trace, and exported as `arxiv_agent_tokens_total{kind="thinking"}`. |
| **Pin the snapshot ID, not the alias** | An alias silently moving to a new snapshot mid-project makes every prior baseline incomparable — the v2.1 failure mode through a different door. | Phase 4 |
| **`run_eval.py` prints estimated cost before executing, and requires `--confirm-cost` above a threshold** | Discovering a $3 run after it has happened. | Phase 4 |
| **OpenAI dashboard hard cap $5, soft alert $2** | Everything above failing at once. | Manual, before the first judge call |
| **Cost and tokens summed over the same traces, checked by a blended-rate bound** | A ratio whose numerator and denominator range over different populations. The spend table divided cost from 6 priced traces by tokens from 214, and reported a rate below the input-only price. `make budget` now refuses to write a table whose blended rate falls outside the rate card. | **Done** — [D-021](./DECISIONS.md), `make reconcile-cost` |
| **A global daily ceiling on the public endpoint, in a ledger that survives restarts** | A public endpoint with a real key behind it spending without limit; and a ceiling that resets every time a sleeping host wakes, which never binds. Each request reserves the $0.025 per-request ceiling before it runs and settles after, so concurrent requests cannot jointly pass it; a deployed container refuses to start on a non-durable ledger. | **Done (Phase 5)** — $0.50/day notional, [D-037](./DECISIONS.md), [SERVING.md](./SERVING.md) §3 |
| **Model calls paced below the free-tier quota in the served container** | A burst of queries turning into provider 429s and silent client-side retries. 12 calls/min, burst 3 — at most 15 in any minute. Off in the eval path so its latency carries no limiter sleep. | **Done (Phase 5)** — `AGENT_REQUESTS_PER_MINUTE`, [D-041](./DECISIONS.md) |
| **The test suite never writes to the trace store `make budget` reads** | Synthetic runs with fake token counts and no cost polluting a published spend figure. 187 of 214 traces were test traffic. | **Done** — `tests/conftest.py` disables observability suite-wide |

---

## Agent-side spend (Gemini)

The graph meters every call, because a ceiling needs a figure at request time and a bill
arrives after the fact (D-004). `Usage` carries:

- `notional_cost_usd` — the tokens priced at paid standard rates, the cached part of the
  prompt at $0.03/1M. **This is what every ceiling checks.**
- `cached_input_tokens` — the part of the prompt the provider served from its cache (added
  2026-09-24; before that, cached tokens were priced at the full input rate).
- `cost_usd` — billed. **`None` (unverified) unless a provider record supplied it.** It used to
  be computed as $0 from a free-tier assumption on a key that was being billed (D-046).

Any cost figure published anywhere must say which of the two it is.

Current per-request ceilings (`src/config.py::BudgetLimits`):

| Ceiling | Default |
|---|---|
| `max_input_tokens` | 120,000 |
| `max_output_tokens` | 12,000 |
| `max_notional_cost_usd` | $0.025 (derived, see below) |
| `max_wall_clock_s` | 120 |
| `max_tool_calls` | 12 |
| `max_llm_calls` | 16 |

Breaching any one returns a partial answer with `truncated=True` and an explicit reason —
never a silent stop. Asserted by `tests/test_termination.py::TestBudgetTermination`.

**LLM calls per query.** The design target was ~16 worst case against v2.1's ~43
(AUDIT §4.7). Observed on single local CLI runs, single-user, n=5 queries — an
illustrative sample, not a measurement, and no substitute for Phase 4:

| Query shape | LLM calls | Tool calls | Input/output tokens | Notional cost |
|---|---:|---:|---|---:|
| Single-hop ("What is LoRA?") | 3 | 1 | 7,295 / 465 | $0.00335 |
| Multi-hop (3 sub-questions) | 3 | 3 | 12,969 / 516 | $0.00518 |
| Out-of-scope refusal | 1 | 0 | 134 / 31 | $0.00012 |

Notional cost is at **verified** paid-standard rates ($0.30 / $2.50). These runs were also
*billed* at those rates — the key was on a billing-enabled project (D-046). The v3 worst-case
call count remains a target for Phase 4 to falsify, not a result.

**The ceiling is a pathological-case bound, not a description of observed behaviour.** Over
69 queries (`make run-report`): median 4 LLM calls per query against the 16 the budget
assumed, per-call notional median $0.00100 / p95 $0.00131 / max $0.00156, per-query max
$0.0057. The ceiling is `max_llm_calls × max observed per-call cost` = 16 × $0.00156 =
$0.0249 → **$0.025** — every permitted call priced at the largest call ever seen, which no
real query does. It therefore binds only on a run that is pathological in *size*, the one
dimension the call ceiling cannot bound, and never on an ordinary run (4.4× headroom over the
observed per-query max). That is its job; it is not a forecast. The previous $0.05 was a
round number calibrated against placeholder rates that understated cost ~3.6×.

**Thinking was expected to be the dominant cost lever, so it is pinned — and the pin
worked.** The bill shows output (thinking included) at 9% of cost and uncached input at 89%:
context size, not thinking, is what costs money (D-046). `reasoning_effort` is set
explicitly to `minimal` rather than left at the model default, because thinking tokens bill
at the output rate. Measured over 5 runs on one prompt (`scripts/determinism_probe.py`,
2026-08-20): `minimal` produces 0 thinking tokens and ~79 output tokens; `low` produces
~426 thinking and ~508 output — roughly **6x the output cost** for the same prompt. Thinking
cannot be disabled entirely (`thinking_budget=0` is rejected). Details in
[DECISIONS D-014](./DECISIONS.md).

**Per-request, not per-thread.** `usage` is checkpointed, so the ceilings were briefly
being enforced across a whole thread — three turns accumulated `llm_calls` 3 → 6 → 9 before
[D-013](./DECISIONS.md) fixed it. Phase 5's daily ceiling has its own accumulator (D-037).

---

## Agent-side spend, from traces

> **The trace store was reset at the start of Phase 4** (`make langfuse-reset`), so this
> table counts only runs since then and reads `$0.00` until the eval runs land. The
> pre-reset window — 214 traces, of which 6 were real agent runs totalling `$0.01248`
> notional — is committed as a checksummed fixture at `data/traces_d021.json` and stays
> reproducible with `make reconcile-d021`. Deleting the store without first preserving the
> evidence for [D-021](./DECISIONS.md) would have repeated the v2.1 failure in
> [AUDIT §5](./AUDIT.md): a documented number whose dataset no longer exists.
>
> Rows marked _(test traffic)_ are traces written by `tests/test_trace_integration.py`.
> They are real traces and are shown, but they are not agent spend and are excluded from
> the total.

<!-- SPEND:START -->
<!-- Generated by `make budget` from Langfuse traces. Do not edit by hand. -->

**Agent-side spend, last 30 days** (generated 2026-09-24):

| Environment | Priced traces | Input | Output | Thinking | Notional USD |
|---|---:|---:|---:|---:|---:|
| `development` | 276 | 2,678,570 | 54,238 | 0 | $0.93917 |
| `integration-test` _(test traffic)_ | 8 | 7,820 | 1,248 | 0 | $0.00547 |
| `span-loss-probe` | 2 | 15,172 | 934 | 0 | $0.00689 |
| **total** | **278** | **2,693,742** | **55,172** | | **$0.94605** |

_Blended $0.3442 per 1M tokens, between the $0.3 input and $2.5 output rates as a mostly-input workload should be (cached input, at $0.03, is the only thing allowed below it). `make reconcile-cost` checks this._

_Excluded from every column above, not estimated: 0 traces carrying tokens but no cost (synthetic runs from the test suite, which priced nothing), and 2 traces with no usage metadata (pre-instrumentation runs and the duplicate roots of the double-trace bug). 278 of 288 traces in the window are real, priced agent runs. Counting the excluded traces' tokens against the priced traces' cost is what produced an impossible blended rate before (DECISIONS D-021)._

Notional prices the tokens at paid standard rates (DECISIONS D-004). **There is no billed
column.** It used to read $0 from a free-tier assumption while the key's project was billed;
billing comes only from the provider's own record — see the Gemini line above, from
docs/billing/gemini.json (DECISIONS D-046). **This table is the agent side only.** OpenAI
judge spend is a different provider on a different ceiling and is never summed with the figures
above; regenerate it with `make judge-spend`, which reads usage from the batch objects
themselves.
<!-- SPEND:END -->

Regenerate with `make budget`. Reads Langfuse directly, so this table cannot drift from what
was actually spent — the same reason `make readme-stats` generates the test count.

Generated is not the same as correct, though, and this table proved it: it was machine-built
from real trace data and still published an impossible blended rate, because it summed
tokens and cost over different sets of traces. A generated number inherits the authority of
the pipeline that made it, so the pipeline needs its own check. `make reconcile-cost` is
that check — it verifies our figure against Langfuse's trace by trace and exits non-zero on
any divergence. Run it before quoting a cost per query ([D-021](./DECISIONS.md)).

---

## Update protocol

This file is updated **every phase**. Once Langfuse is wired (Phase 3), the spent column is
generated from trace data rather than typed by hand. If a phase looks likely to exceed its
allocation, that gets raised **before** the phase starts, not partway through.
