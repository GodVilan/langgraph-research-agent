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

70 traces: volume by name, cost by model, and both over time.

**Langfuse's cost figure is not ours.** The dashboard shows `$0.02776` from Langfuse's own
rate card for `gemini-3.5-flash-lite`. `make budget`, reading our instrumentation, reports
`$0.00000` billed and `$0.01248` notional. Both are defensible; they answer different
questions. Merging them is how a free-tier project ends up publishing a spend figure it never
spent — which is precisely what the first run of `make budget` did before it was fixed
(D-020).

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
