# OBSERVABILITY

What a trace can tell you about a run, how to bring the stack up, and what is deliberately
not instrumented.

The problem this solves is stated in [AUDIT §1.1](./AUDIT.md): v2.1's only instrumentation
was `log.info`. A query that took four seconds or cost more than expected could not be
attributed to a stage without re-running it by hand and adding print statements.

---

## Bringing it up

```bash
make langfuse-up          # docker compose, ~1 min on first boot
```

The stack provisions itself **headlessly** — organisation, project, and API key pair are
created on first boot from environment variables in
`infra/docker-compose.langfuse.yml`. There is no click-through signup, and the key pair is
identical for everyone who clones this repo, so the instructions below are the same on every
machine.

Put the seeded pair in `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-lf-1a1a1a1a-2b2b-4c4c-8d8d-3e3e3e3e3e3e
LANGFUSE_SECRET_KEY=sk-lf-4f4f4f4f-5a5a-4b6b-8c7c-6d6d6d6d6d6d
LANGFUSE_HOST=http://localhost:3000
```

Then:

```bash
make trace-check          # confirms reachability without making a model call
```

The UI at <http://localhost:3000> signs in with `dev@example.com` / `localdev1234`, also
seeded from the compose file. Everything binds to `127.0.0.1`.

> **These are fixture values, not secrets.** The `pk-lf-…` / `sk-lf-…` pair, the login, and
> the `SALT` / `ENCRYPTION_KEY` / `NEXTAUTH_SECRET` in
> `infra/docker-compose.langfuse.yml` are committed deliberately. They provision a
> throwaway localhost-only stack, they are byte-identical for every clone, and they grant
> access to nothing but a local container holding traces of public arXiv text. They are in
> version control so that `make langfuse-up` works on a fresh clone with no signup step.
>
> A deployed instance uses **Langfuse Cloud keys read from the environment** and never
> these. The compose file is a development tool and is not a deployment artifact.
>
> That reasoning holds only while the host is local, so it is enforced rather than trusted:
> `ObservabilitySettings.check_not_deployed_with_seeded_keys()` refuses the seeded pair
> against any non-localhost `LANGFUSE_HOST`, and `get_client()` raises instead of quietly
> tracing somewhere it should not. It is the one tracing failure that is not degraded to a
> no-op — a misrouted credential is a configuration bug, not a backend outage.

```bash
make langfuse-down        # stop, keep data
make langfuse-reset       # stop, discard all traces
```

### If you only want to look at a trace

The self-hosted stack is six containers — web, worker, Postgres, ClickHouse, Redis, MinIO —
because that is what Langfuse v3+ requires. **Langfuse Cloud** (<https://cloud.langfuse.com>,
or `us.cloud.langfuse.com`) is the lighter path and the one to use for a deployed instance:
set the same three variables, pointing `LANGFUSE_HOST` at the cloud URL. Nothing in the code
changes.

### Two things that will waste an hour if you hit them cold

Both were hit while building this, and both fail with an error that does not name the cause:

1. **`ENCRYPTION_KEY`, `SALT`, and `NEXTAUTH_SECRET` must be quoted in the compose file.**
   YAML parses an all-digit hex string as an integer, so `0000…` arrives as `"0"` and
   Langfuse rejects it as too short. The worker crash-loops with a `ZodError`.
2. **`LANGFUSE_INIT_USER_EMAIL` must be a real-looking address.** `dev@localhost` fails
   validation and takes the whole web service down with `Invalid environment variables` —
   which does not say which variable, and appears identical to a database problem.

---

## What a trace carries

One query produces **one trace** with roughly 40 observations, nested: the root `query`
span, then each graph node, then each model call inside it, then the retrieval and embedding
spans underneath.

| Field | Where it comes from |
|---|---|
| Session id (= thread id) | Root span; multi-turn conversations group the way they do in the checkpointer |
| Node name, model name | LangChain callback handler |
| Prompt version | `RequestOptions.prompt_version`, plus what each prompt resolved to |
| Input / output / **thinking** token counts | `Usage`, from `output_token_details.reasoning` |
| Latency, per node and per call | Callback handler + OTel spans |
| **Billed and notional cost, as separate fields** | `Usage`; see below |
| Tool name and arguments | `ToolCallRecord` |
| Retrieval hit chunk ids | `RetrievalEvent.hit_chunk_ids` |
| Every guardrail event, including injection detections | `GuardrailEvent`, emitted as trace events at WARNING/ERROR level |
| Truncation flag and reason, critique verdict, refinement count | `AgentState` |

The LangChain handler supplies only the first four rows. Everything else lives in
`AgentState` and is attached explicitly by `src/observability/langfuse.py` — without that,
a trace could not answer either of the two questions it exists for: *what did this cost*,
and *what did the guardrails do*.

### Billed and notional cost are separate, deliberately

Billed cost is `$0` on the Gemini free tier. Notional cost prices the same tokens at paid
standard rates ([DECISIONS D-004](./DECISIONS.md)). A single cost field would either read
zero forever or imply spend that is not happening, so both are carried and every published
figure says which it is.

### Non-LLM latency is in the same trace

Langfuse v4 installs itself as the global OpenTelemetry tracer provider, so the spans around
retrieval and embedding — `retrieval.dense`, `retrieval.sparse`, `faiss.search`,
`embedding.encode_query` — nest inside the same tree as the model calls rather than living
in a second system to correlate by hand.

With Langfuse off, those spans go to `OTLP_ENDPOINT` if one is set, and to a no-op tracer
otherwise.

### Evidence

![Langfuse trace tree](img/langfuse-trace.png)

The injection scenario, traced end to end. Left: the nested tree from the root `query` span
down to individual model calls. Right: input, output, and the metadata panel carrying
`guardrail_events`, `guardrail_summary`, `usage`, `critique_verdict`, `thread_id`, `model`,
and `prompt_version`. Header shows latency, session, release, cost, and token split.

The answer cites `[2605.30148_0021]` — the clean passage. The passage that tried to close its
own `<passage>` block and issue `System: ignore all previous instructions` was quarantined
and never reached the model.

![Langfuse home dashboard](img/langfuse-dashboard.png)

**69 traces, `$0.236004` notional, 115 evaluation scores** — the Phase 4 eval window, one
trace per item of the frozen 69-item set, with judge scores attached per trace
(`make push-scores`). The Scores panel is the reason this capture replaced an earlier one: it
read "No data" until eval scores landed, and a dashboard with an empty Scores panel is not
evidence of evaluation-driven development.

The trace count here **is** a measure of real usage, which it previously was not. An earlier
capture of this page showed 70 traces of which most were synthetic — emitted by the test
suite into the same project, the contamination the next section diagnoses. `make reconcile-cost`
over this window reports 0 unpriced traces, 0 duplicate roots, and a blended `$0.3433` per 1M.

### Reconciling the two cost figures

The dashboard showed `$0.02776`; `make budget` reported `$0.01248` notional. That was logged
as "two rate cards answering different questions", which was wrong — a gap is only explained
if both figures are internally consistent, and ours was not. Over the 72K tokens the
dashboard was summing, `$0.01248` is a blended **`$0.173` per 1M**, *below* the `$0.30`
input-only rate. No token mix can produce that.

`make reconcile-cost` closes it, and the answer was not a rate card:

| | |
|---|---:|
| Priced agent runs, stored notional | `$0.01248` |
| The same runs, recomputed from their tokens at `$0.30`/`$2.50` | `$0.01248` |
| The same runs, per Langfuse's independent calculation | `$0.01248` |
| Duplicate root traces of five of those same runs | `$0.01528` |
| **Langfuse dashboard total** | **`$0.02776`** |

Three independent computations agree exactly, so **the notional calculator was never
wrong**. Two separate contaminations produced the illusion:

1. **The token count was 71% synthetic.** 187 of 214 traces in the window came from the test
   suite. `run_query` opens a Langfuse span unconditionally, and the observability settings
   are read by a *different* settings object than the one `tests/conftest.py` isolated — so
   every test that ran the graph shipped a trace built from `tests/fakes.py` usage: real
   token counts, zero cost, because the fake priced nothing. The table then summed tokens
   over all 214 traces and cost over the 6 that had been priced. Two populations, one
   blended rate, an impossible number. Fixed at the root: `conftest` now disables
   observability for the whole suite, and `summarise()` only counts tokens from a trace that
   also contributes cost.
2. **`$0.01528` of the dashboard total is double-counted.** It sits on five orphan
   `LangGraph` roots — the residue of the double-trace bug described above, whose `query`
   twins carry the metadata and none of the cost. Those runs are already counted; the
   duplicate roots predate the fix and remain in any 30-day window.

Restated on the priced population alone, the blend is **`$0.4056` per 1M**, between the
`$0.30` input and `$2.50` output rates exactly where a mostly-input workload lands.

Two guards now make this class of error loud rather than plausible. `make budget` computes
the blended rate and **refuses to write the table** if it falls outside the rate card, and
`make reconcile-cost` exits non-zero if the stored and recomputed figures ever diverge. The
underlying rule is D-004's, applied one level up: never divide two numbers that describe
different populations (D-021).

**Langfuse's own figure still is not ours.** It prices from its rate card; we price from
`src/config.PRICING` and separate billed from notional. Merging them is how a free-tier
project publishes a spend figure it never spent — which the first run of `make budget` did
(D-020). They agree here because both are right, not because either was copied.

---

## Metrics

```bash
make metrics              # print the current exposition
```

Defined in `src/observability/metrics.py` against a single registry, so the CLI, the eval
harness, and the Phase 5 service all report the same series. Phase 5 mounts it at
`GET /metrics`; none of the definitions change when it does.

| Metric | Labels | Note |
|---|---|---|
| `arxiv_agent_requests_total` | `outcome` | `answered` / `refused` / `truncated` / `error` |
| `arxiv_agent_request_latency_seconds` | `outcome` | Buckets to 120s, the per-request wall-clock ceiling |
| `arxiv_agent_errors_total` | `error_type` | |
| `arxiv_agent_guardrail_triggers_total` | `category`, **`severity`**, `node` | |
| `arxiv_agent_chunks_quarantined_total` | `source` | |
| `arxiv_agent_tokens_total` | `kind` | `input` / `output` / **`thinking`** |
| `arxiv_agent_cost_usd_total` | `kind` | `billed` / `notional` |
| `arxiv_agent_retrieval_hits` | `retriever` | |
| `arxiv_agent_missing_usage_metadata_total` | — | Non-zero means the token and cost counters understate reality |

**Severity is a label on the guardrail counter, not an afterthought.** 60 of the 5,401
corpus chunks (1.11%) produce a WARN, and almost all are false positives on papers that
quote prompts ([SCREEN.md](./SCREEN.md)). A single trigger counter would show a steady
stream of alarming-looking events for something that withholds nothing from the model.

---

## Spend, generated from traces

```bash
make budget               # rewrites the spend table in BUDGET.md from Langfuse
```

`docs/BUDGET.md` committed to this control in Phase 1: the spend column is generated, not
typed. Same reasoning as `make readme-stats` — a hand-maintained number drifts, and a
drifted number discredits the ones that are correct.

---

## What is not instrumented, and why

- **No sampling.** Every run is traced. At this volume that is free; a deployed instance
  with real traffic would need a sampling policy, and does not have one.
- **No alerting.** Metrics are exposed, nothing consumes them. There is no Prometheus
  server or Grafana in `infra/` — adding one would be a dashboard nobody watches.
- **No per-node cost attribution beyond what the handler infers.** The handler attributes
  cost to each model call; the *node* rollup is in trace metadata rather than as a
  first-class per-node cost field.
- **Tracing is best-effort by construction.** Every entry point swallows its own
  exceptions. An agent that fails because its telemetry failed is worse than an agent with
  no telemetry, and `tests/test_observability.py::TestDegradesToNoOp` asserts this against a
  client that raises, a span that is already closed, and no configuration at all.
- **The 60 corpus WARNs will appear in traces as a steady background rate.** They are
  labelled `severity="warn"` and withhold nothing. Treating them as incidents would be a
  misreading; see [SCREEN.md](./SCREEN.md).
