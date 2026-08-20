# BUDGET

**Hard ceiling: $5.00 total OpenAI spend for the lifetime of this project.**

OpenAI is the evaluation judge only — never the agent's generator, planner, critic, or
guardrail classifier, and never the embedding model. The agent graph runs on Gemini
Flash-Lite free tier throughout. Rationale in [`DECISIONS.md`](./DECISIONS.md) D-001.

**Status at the end of Phase 1: $0.00 spent. No OpenAI call has been made.** The judge is
not built until Phase 4, and no credit needs to be loaded before then. Agent-side Gemini
usage is free-tier and also $0.00 billed.

---

## Unit prices

Provider documentation, verified 2026-08-19 except where marked. **Do not edit these from
memory** — re-verify against provider docs and update the date.

| Model | Input /1M | Output /1M | Notes |
|---|---|---|---|
| `gemini-3.5-flash-lite` free (**in use**) | $0 | $0 | No context caching on the free tier |
| `gemini-3.5-flash-lite` paid standard | $0.30 | $2.50 | Output rate **includes thinking tokens** |
| `gemini-3.5-flash-lite` paid batch/flex | $0.15 | $1.25 | |
| `gemini-2.5-flash-lite` (paid) | $0.10 | $0.40 | **Retired for new API keys** — see below |
| `gpt-5.6-luna` | $0.20 | $1.20 | Cached input $0.02; cache writes 1.25× uncached |

> **The agent runs on `gemini-3.5-flash-lite`.** `gemini-2.5-flash-lite` returns 404 for
> this API key ("no longer available to new users"), found on the first live run
> ([DECISIONS D-012](./DECISIONS.md)). Its rates were briefly used as a `verified=False`
> placeholder and **understated cost by roughly 3.6x**; the figures below are the corrected
> ones. Agent-side *billed* cost is $0 on the free tier regardless — this affects reporting
> and the notional ceiling, not spend.
>
> Notional cost uses **standard**, not batch: the agent serves interactive requests, so
> batch pricing would understate what a paid deployment would pay. Batch pricing does apply
> to the OpenAI judge, where it is mandatory.

Both providers bill thinking/reasoning tokens at the **output** rate. Reasoning tokens are
the dominant cost variable, not input size, which is why `reasoning_effort` is pinned
explicitly rather than left at its default.

OpenAI has **no free tier**. Tier 1 is the entry point: 500 RPM, 500K TPM, 5M batch queue
limit. Credit must be loaded before any judge call. A full 100-item run is roughly 510K
enqueued tokens — comfortably inside the Tier 1 batch queue, so no throttling is expected.

---

## Planned allocation

Assumes ~4,000 input and ~1,100 output tokens per judged item (300 visible + ~800 reasoning
at `medium`), Batch API throughout.

| Line | Items judged | Allocation | Spent |
|---|---:|---:|---:|
| Judge validation (25 items × 2 configs) | 50 | $0.05 | $0.00 |
| Phase 4 dev — 25-item subset runs (×10) | 250 | $0.27 | $0.00 |
| Phase 4 dev — 100-item full runs (×6) | 600 | $0.64 | $0.00 |
| Baselines: v2.1, v3, embedding comparison arm | 300 | $0.32 | $0.00 |
| CI subset runs (×40) | 1,000 | $1.06 | $0.00 |
| Phase 6 variant experiment, if reached | 200 | $0.21 | $0.00 |
| **Planned total** | | **$2.55** | **$0.00** |
| **Unallocated reserve** | | **$2.45** | |

---

## Mandatory cost controls

These are requirements, not suggestions. Each one has a specific failure it prevents.

| Control | Prevents | Status |
|---|---|---|
| **Batch API for every judge call** | Standard rates are exactly 2× batch. Forgetting doubles the project to $5.10 and consumes the entire budget. The eval is asynchronous by nature; there is no reason to pay standard rates. | Phase 4 |
| **`reasoning_effort` pinned in config, never defaulted** | Luna's default is `medium`. Validate at `low` first — for rubric scoring against a fixed schema it may suffice and saves roughly $1 across the project. Escalate only if judge–human agreement is poor at `low`, and record the comparison. `high` and above are out of budget. | Phase 4 |
| **Log `reasoning_tokens` per call to Langfuse** | This table is generated from trace data, not hand-maintained. `Usage.reasoning_tokens` already exists and is populated from `output_token_details.reasoning`. | Field ready (Phase 1); wiring in Phase 3 |
| **Pin the snapshot ID, not the alias** | An alias silently moving to a new snapshot mid-project makes every prior baseline incomparable — the v2.1 failure mode through a different door. | Phase 4 |
| **`run_eval.py` prints estimated cost before executing, and requires `--confirm-cost` above a threshold** | Discovering a $3 run after it has happened. | Phase 4 |
| **OpenAI dashboard hard cap $5, soft alert $2** | Everything above failing at once. | Manual, before the first judge call |

---

## Agent-side spend (Gemini, free tier)

Billed spend is $0.00 by construction. The graph nevertheless meters every call, because a
ceiling that is permanently satisfied is a ceiling that has never been tested (D-004).
`Usage` carries both figures:

- `cost_usd` — actually billed. `0.0` on the free tier.
- `notional_cost_usd` — the same tokens priced at paid rates. **This is what the budget
  ceiling checks**, so the guard is live and exercised today.

Any cost figure published anywhere must say which of the two it is.

Current per-request ceilings (`src/config.py::BudgetLimits`):

| Ceiling | Default |
|---|---|
| `max_input_tokens` | 120,000 |
| `max_output_tokens` | 12,000 |
| `max_notional_cost_usd` | $0.05 |
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

Notional cost is at **verified** paid-standard rates ($0.30 / $2.50). Billed cost is $0.00
on the free tier. The v3 worst-case call count remains a target for Phase 4 to falsify, not
a result.

**The $0.05 ceiling is a round number, not a derived one.** It was chosen while the active
model's rates were a placeholder understating cost ~3.6x, so it was calibrated against
figures about 4x too low. Corrected costs still sit an order of magnitude under it, so it
was left alone — but Phase 4 should reset it from a measured per-query distribution rather
than inherit it.

**Thinking is the dominant cost lever, and it is pinned.** `reasoning_effort` is set
explicitly to `minimal` rather than left at the model default, because thinking tokens bill
at the output rate. Measured over 5 runs on one prompt (`scripts/determinism_probe.py`,
2026-08-20): `minimal` produces 0 thinking tokens and ~79 output tokens; `low` produces
~426 thinking and ~508 output — roughly **6x the output cost** for the same prompt. Thinking
cannot be disabled entirely (`thinking_budget=0` is rejected). Details in
[DECISIONS D-014](./DECISIONS.md).

**Per-request, not per-thread.** `usage` is checkpointed, so the ceilings were briefly
being enforced across a whole thread — three turns accumulated `llm_calls` 3 → 6 → 9 before
[D-013](./DECISIONS.md) fixed it. Phase 5's daily cost ceiling needs its own accumulator.

---

## Update protocol

This file is updated **every phase**. Once Langfuse is wired (Phase 3), the spent column is
generated from trace data rather than typed by hand. If a phase looks likely to exceed its
allocation, that gets raised **before** the phase starts, not partway through.
