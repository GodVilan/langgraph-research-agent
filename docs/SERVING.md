# Serving — the HTTP API, its limits, and how it is deployed

Phase 5. The graph, unchanged, behind FastAPI in one container. This document is the
reference for what the service does, what it refuses, and why each limit is where it is.
Decisions are D-035…D-044 in [DECISIONS.md](DECISIONS.md).

---

## 1. Endpoints

| | |
|---|---|
| `POST /query` | Ask a question. **SSE by default**; `"stream": false` returns one JSON body. Starts a new thread. |
| `POST /threads/{thread_id}/query` | A new turn on an existing, checkpointed thread. 404 if the thread does not exist on this instance. |
| `GET /threads/{thread_id}` | The thread's transcript and last outcome. |
| `GET /health` | Liveness: the process is up. Always 200 while it is. |
| `GET /ready` | Readiness: index and model loaded, ledger reachable. 503 with a reason otherwise. Also reports today's committed notional spend against the ceiling. |
| `GET /metrics` | Prometheus exposition of the existing registry (`src/observability/metrics.py`) — the same series the CLI and eval harness write. No new definitions. |
| `GET /docs` | OpenAPI. |

Request body (`src/api/schemas.py`, `extra` fields rejected):

```json
{"question": "What is LoRA?", "stream": false, "top_k": 5, "paper_ids": ["2605.30179"]}
```

`question` 1–2,000 chars; `top_k` 1–10; `paper_ids` up to 20 arXiv ids restricting retrieval
to those papers. There is no `use_arxiv`: live arXiv fetch is off on the public endpoint
(D-036).

### The response

```json
{
  "request_id": "f0210eb0761244de",
  "thread_id": "009276c9442348bbb8f28c77a751ec7f",
  "trace_id": "",
  "answer": "…",
  "sources": [{"paper_id": "2605.30179", "title": "…", "score": 0.7972}],
  "guardrail_blocked": false,
  "guardrail": {"decision": "passed", "stage": "keyword_fastpath",
                "reason": "scope keyword fastpath", "deterministic": true, "note": null},
  "guardrail_events": [],
  "truncated": false,
  "truncation_reason": null,
  "usage": {"llm_calls": 3, "tool_calls": 1, "input_tokens": 7652, "output_tokens": 533,
            "cached_input_tokens": 0, "billed_cost_usd": null,
            "billed_cost_basis": "unverified: …", "notional_cost_usd": 0.003628},
  "latency_ms": 16620.3
}
```

* **`guardrail`** says which stage of the input guardrail decided and whether that stage is
  deterministic. `keyword_fastpath`, `input_validation` and `injection_check` are; the model
  classifier (`scope_classifier`) is not, and its decisions carry a `note` saying that asking
  again can change the answer (§5, D-035).
* **`guardrail_blocked`** is true only when the *input* guardrail refused. An answer saying
  the corpus does not cover the question is not a block.
* **`truncated`** means a per-request ceiling stopped the run; `answer` is the partial answer
  and ends with a note naming the ceiling. Never a silent stop.
* **`usage`** carries notional cost — the tokens at paid standard rates, cached input at
  $0.03/1M — which every ceiling checks, and `billed_cost_usd: null` with a
  `billed_cost_basis`: billing is known only from the provider's record, never assumed to be
  $0 (D-046; the key behind every run to date was on a billing-enabled project).
* **`trace_id`** is the Langfuse trace this request wrote, or `""` when tracing is off or the
  request was sampled out (§7).

### SSE

```
event: node
data: {"node": "validate_input"}

event: token
data: {"text": "Based on the retrieved "}

event: final
data: {…the JSON response above…}
```

`node` events fire as each graph node completes; `token` events stream the answer draft
from `generate` only (the classifier, planner and critic also call the model, but their
output is JSON, not answer text). If the critic sends the run back for another round, a
second draft streams too — **the `final` event's `answer` is the authoritative one**,
including any truncation note. A failure after streaming has started arrives as
`event: error` with the status it would have had. `: keepalive` comment lines every 15 s
keep proxies from closing a connection that is waiting on the model pacer.

One run per request either way: the CLI's old `--stream` ran the graph a second time after
streaming it, paying for every question twice. `run_query(on_event=…)` is now the single path.

---

## 2. The order of checks

Cheapest and least trusting first (`src/api/app.py`):

| # | Check | Refusal | Survives restart? |
|---|---|---|---|
| 1 | index and model loaded | 503 `not_loading` / `not_failed` | — |
| 2 | per-IP token bucket | 429 `rate_limited` + `Retry-After` | no — a smoothing limit, not a budget |
| 3 | daily notional cost ceiling (reserve) | 429 `daily_cost_ceiling`, `Retry-After` = seconds to 00:00 UTC; 503 `cost_ledger_unavailable` if the ledger cannot be reached | **yes — it must** |
| 4 | concurrency gate | 503 `busy` after `queue_timeout_s` | — |
| — | the graph's own per-request ceilings | 200 with `truncated: true` | — |

Every rejection increments `arxiv_agent_requests_total{outcome="rejected"}` and
`arxiv_agent_guardrail_triggers_total{node="api", category=…}` — existing series, new label
values. A thread already running a turn returns 409 `thread_busy`.

### Defaults (`ApiLimits` in `src/config.py`, env `API__*`)

| Setting | Default | Why this value |
|---|---|---|
| `per_ip_per_minute` / `per_ip_burst` | 4 / 3 | The whole service sustains ~4 queries a minute (below); one client should not take all of it. |
| `max_concurrent_queries` | 2 | The model quota, not CPU, bounds throughput. More in-flight queries only queue inside the provider client's retry loop, where nothing reports it. |
| `queue_timeout_s` | 20 | How long an admitted request may wait for a slot before 503. |
| `daily_notional_ceiling_usd` | 0.50 | ~150 median queries ($0.0033, `make run-report`), or 20 at the $0.025 per-request bound. |
| `AGENT_REQUESTS_PER_MINUTE` (container) | 10, burst 3 | Any 60 s window admits at most 3 + 10 = 13 calls, under the documented 15/min quota with margin (`tests/test_api.py` reads the Dockerfile). The first version was 12 + 3 = exactly 15 — no margin. To be set to the key's actual quota from AI Studio. |

**Throughput is the quota, stated plainly.** A median query makes 3–4 model calls
(`make run-report`), so 10 calls a minute is about 3 queries a minute for the whole instance.
Measured in the local container at the earlier 12/min setting (2026-09-23): a single query from idle took 4.3 s; two more sent back to
back took 11.9 s and 15.1 s, because the pacer's burst had drained. That is the free tier,
not the service; a paid key with the pacer raised would lift it.

---

## 3. The daily cost ceiling

D-013 made every budget in the graph per-request, deliberately. The daily ceiling is the
separate accumulator it said Phase 5 would need (`src/api/ledger.py`).

**Reserve, then settle.** Each request atomically reserves the per-request ceiling
(`BUDGET__MAX_NOTIONAL_COST_USD`, $0.025) before running and settles to its actual notional
cost afterwards. Committed-plus-reserved spend therefore never passes the ceiling however
many requests race: 50 concurrent reservations against room for 4 admit exactly 4
(`tests/test_api.py`). The bound that remains is inside one request — the graph checks its
budget between nodes, so a request can exceed its own $0.025 by the last call it made before
the check. A failed run settles to what its last checkpoint shows it spent; if nothing can be
read, the full reservation stays counted. Both errors are in the safe direction.

**It must survive a restart.** An in-memory total resets on every restart, and a host that
sleeps when idle restarts every time it wakes — a ceiling that never binds. So:

| `API__LEDGER` | Durable when | Use |
|---|---|---|
| `memory` | never | tests and `make serve` only |
| `sqlite` | its file is on a **mounted volume** | Fly.io volume, a local `docker run -v`, a Spaces storage bucket |
| `upstash` | always (Redis over HTTPS, httpx only) | hosts with no persistent disk |

This is enforced, not documented: the image sets `API__DEPLOYED=true`, and a deployed
container **refuses to start** on `memory`, or on `sqlite` whose path resolves to the
container's own root filesystem (`check_ledger_is_durable`). Verified against the real image
— both refusals fire, and a volume-backed start keeps the day's spend across a container
restart ($0.003628 before, $0.003628 after, then growing). It also refuses a daily ceiling
below one reservation, which would 429 every request while reporting healthy.

**Fails closed.** A ledger that cannot be reached refuses the query (503) and marks
`/ready` unready. A query that cannot be counted is not served — the same rule as the scope
guard (AUDIT §4.2).

---

## 4. Client addresses and the load-check token

`X-Forwarded-For` is a list each proxy appends to; everything left of what *our* proxies
added was written by the client. `API__TRUSTED_PROXY_HOPS=n` takes the n-th entry from the
right; `0` (default) ignores the header and uses the socket peer. Trusting the leftmost entry
would let any caller pick their own rate-limit key. uvicorn runs with `--no-proxy-headers` so
it never rewrites the peer itself. **The correct hop count for a given host is set after
observing that host's headers — not verified until the deploy step records it.**

`LOADCHECK_TOKEN`, when set, lets a request carrying `X-Loadcheck-Token` skip the per-IP
bucket **only**. The daily ceiling and the concurrency gate still apply, because those protect
the key. Unset, no bypass exists. It exists because every load-check request comes from one
address and would otherwise measure the limiter (D-040). Compared in constant time.

Request logs are JSON lines carrying a truncated SHA-256 of the client address, never the
address itself.

---

## 5. The guardrail's nondeterminism is a serving problem

Across **seven v3 runs on identical input** (the four full runs plus three comparison arms,
which change retrieval only — after the guardrail), the input scope guardrail blocked
**5, 3, 4, 4, 6, 3 and 5** of the same 43 verified factual questions. Three questions were
blocked in all seven; three flipped between answered and refused with nothing changed
(`make guardrail-variance`). On a public endpoint the same question can be answered or refused
depending on nothing the caller controls.

What ships (D-035, D-044): **the classifier pinned (`seed=0, top_k=1`), its decision returned
in every response, and the trade documented.** Pinned, the classifier gave byte-identical
output on 140 of 140 probe calls where unpinned varied on all 28 items (`make
guardrail-probe`, 2026-09-24 — not a provider guarantee). It refuses the same 5 of 43 in-scope
questions on every call, against an unpinned mean of 4.3 (range 3–6, n=7): consistency bought
by fixing the stricter end in place — `sp-016`, refused in 2 of 7 unpinned draws, is now
refused every time. The pinned configuration was re-judged end to end and passes the
regression gate — three pinned draws: Recall@5 0.267 as pre-registered, hallucinated refusals
27, 27, 25 against unpinned 26, 26, 24, a +1.0 shift inside either spread and so not shown
to be a difference (D-044) — and its three-draw baseline is the gate's (verified locally; the
gate never executed in CI before D-056). Every response still
names the deciding stage and reports a classifier decision as `deterministic: false`.

---

## 6. Threads

A thread id is a uuid4 minted by the service and acts as a bearer capability: anyone holding
it can read and continue that thread. Only the 32-hex-character shape is accepted. Two limits:

* **Threads live on the instance's disk** (`CHECKPOINT_DB`). Without a volume they are lost
  on restart — on a sleeping free host, after every sleep.
* **A follow-up is not rewritten with earlier turns.** `retrieve` embeds the new question
  verbatim, so "what is its rank hyperparameter?" retrieves without its antecedent
  (BACKLOG, Phase 1). The transcript persists; the context does not carry into retrieval.

Each new turn gets its own per-request budget (D-013) — asserted through the API.

---

## 7. Tracing and sampling

The deployed instance traces to **Langfuse Cloud**, keys from the environment. The seeded
`pk-lf-…`/`sk-lf-…` pair is a localhost fixture and is refused four ways: the client refuses
either half against a non-loopback host (by parsed hostname — the Phase 3 substring check
called `localhost.example.com` local); the compose file must publish only loopback ports while
it carries the seeds, and cannot use host networking (`tests/test_deploy_guards.py`); the
`.dockerignore` allow-list never admits `infra/`; and the Space deploy refuses a context
containing a compose file or the `LANGFUSE_INIT_*` provisioning block.

Sampling: `LANGFUSE_SAMPLE_RATE` (head sampling, default 1.0). The policy and its arithmetic
are in [OBSERVABILITY.md](OBSERVABILITY.md#sampling). A sampled-out request returns
`trace_id: ""`, never an id pointing at a trace that was not exported. Langfuse silently drops
`sample_rate` if another OpenTelemetry provider was registered first, so the service checks
the *effective* sampler at startup and refuses to start if a configured rate is not in effect.

---

## 8. The container

`infra/Dockerfile`, three stages: dependencies (CPU-only torch), the embedding model pinned
to commit `abe7d9d8` with safetensors only (the repo's duplicate 1.3 GB `pytorch_model.bin` is
not downloaded), and the runtime with the prebuilt FAISS index **checksum-verified against
`data/INDEX.sha256`** — the build fails if the index differs from the one Phase 4 measured.
Nothing is built or downloaded at start (`HF_HUB_OFFLINE=1`). One uvicorn worker: each worker
would load its own copy of the model and keep its own limiters.

**BGE-large ships, 1.3 GB and all**, because Phase 4 measured the alternative: bge-small
retrieves gold on 5 fewer of 43 factual items and on none that large misses
([EVALS.md](EVALS.md)).

**Image size: 3.10 GB of layers** (`docker history`, summed). The venv is 1.54 GB — CPU torch
0.67 GB (`torch 2.14.0+cpu`; no `nvidia-*`, `cuda` or `triton` packages in the final stage),
then scipy, transformers, sympy, PyMuPDF and sklearn, which arrive with sentence-transformers
and the ported retrieval code — the model 1.34 GB, the Debian base 0.16 GB, corpus and index
0.05 GB. An earlier report said 4.39 GB: that is Docker Desktop's containerd image store,
which counts compressed blobs alongside the unpacked layers. No CUDA libraries were there to
remove. PyMuPDF (64 MB) serves only live arXiv fetch and index building, neither of which the
API runs; it stays because it is a declared dependency of the ported retrieval package.

```bash
make docker-build
make docker-run          # sqlite ledger and threads on a named volume, :7860, tracing off
```

Local measurements (arm64 laptop, Docker Desktop, not the deploy host, 2026-09-23): ready 9 s
after start with the model already in the image; 826 MiB resident after load.

**One worker, asserted.** The per-IP buckets, the concurrency gate and the model pacer are
in-process, so a second worker would silently double every limit. A deployed container takes
an exclusive `flock` at startup and a second worker cannot start.

**Observability is off the request path.** The server never flushes spans per request; the
batch exporter ships them in the background and the server flushes once, bounded, at
shutdown. Before this, `run_query`'s per-request flush added 3.2 s to one query against a
Langfuse that accepted the connection and never answered; now a 429-ing or silent backend
leaves `/query` answering in well under a second with fakes
(`tests/test_api_observability_faults.py`, real Langfuse client and OTLP exporter).

**Cold start on a sleeping host** is container start plus model load. It is paid by the
first request after every sleep and is not hidden: `/health` answers immediately and
`/ready` says `loading` until the model is in memory. The deployed host's figure is measured
separately from any load, after a restart, and published in the README's latency table
(`make load-report`).

### Deploying: a tagged commit, nothing else

Srikanth commits and tags `deploy-YYYY-MM-DD`; the deploy runs from that tag:

```bash
make deploy-space SPACE=godvillain/Scholium DRY=1
```

```bash
make deploy-space SPACE=godvillain/Scholium
```

```bash
make verify-deploy SPACE=godvillain/Scholium REV=deploy-YYYY-MM-DD
```

The upload **refuses** unless every deployed path — the `.dockerignore` allow-list, `infra/Dockerfile`,
`.dockerignore` itself and `scripts/deploy_space.py`, which renders the Space card — is clean in
git (untracked files count) and `HEAD` carries a `deploy-*` tag. The dry run prints the same
verdict without refusing. `ALLOW_DIRTY=1` overrides it: it warns, names the Space commit
"UNCOMMITTED (--allow-dirty)", and the deploy log records the dirty paths. Every upload appends
one line to `infra/deploy_log.jsonl` (time, Space commit, git `HEAD`, its tags, override, dirty
paths), read back after writing. The gitignored FAISS index is not git's to vouch for; its
checksums are committed and verified before upload. Why: two deploys ran from uncommitted files
and were mapped to git only afterwards (D-049, D-054, D-055).

---

## 9. Verifying a deployment — `make smoke-live`

Asserts effects, never status codes (D-026): the README's curl example is extracted from the
README and run verbatim, and must return an answer citing a chunk id; `/ready` must report
5,401 chunks and the FAISS checksum in `data/INDEX.sha256`; a forged `X-Forwarded-For` must not
change the `X-Client-Key` response header (the limiter's key, a truncated hash), and that key
must be this machine's public address — the hop-count check that cannot run locally; the
response's trace must be readable back from Langfuse Cloud; and the sampler's *effective* rate
(`/ready` `trace_sample_rate`, and the startup log line) must equal the configured one.

## 10. What is not verified

* Behaviour under real traffic. The deployed URL is verified by `make smoke-live`, the proxy hop
  count was observed on the host (1, D-047), and cold start and throughput come from **one load
  check, one client machine against one instance** (README, Serving; `make load-report`) — not
  from live callers, whose arrival pattern nothing here has seen.
* Anything above one worker or one instance: the per-IP buckets and the concurrency gate are
  per process by design.
* Nothing scrapes `/metrics`. It is exposed; no Prometheus or Grafana runs (BACKLOG).
