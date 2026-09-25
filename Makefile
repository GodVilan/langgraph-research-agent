PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install lint fmt type test test-all test-fast test-integration test-necessity graph index index-verify compare-index verify-corpus corpus-info corpus-diversity audit-entities verify-evals select-attributes prune-gold gold-report rubric judge-sample run-set run-report bypass-probe score judge-print judge-estimate judge-agreement metrics-report metrics-compare metrics-spread judge-spend push-scores gate run-v21 integrity readme-stats injection-report injection-live screen-corpus langfuse-up langfuse-down langfuse-reset budget reconcile-cost reconcile-d021 metrics guardrail-variance guardrail-probe generator-determinism gemini-reconcile corpus-licenses trace-units smoke-live verify-deploy serve docker-build docker-run deploy-space space-secrets load-check latency-single load-report check clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install the project with dev extras
	python3 -m venv .venv 2>/dev/null || true
	$(PIP) install -q -e ".[dev]"

lint:  ## ruff check + format check
	$(PY) -m ruff check src tests scripts evals
	$(PY) -m ruff format --check src tests scripts evals

fmt:  ## ruff format
	$(PY) -m ruff format src tests scripts evals
	$(PY) -m ruff check --fix src tests scripts evals

type:  ## mypy strict
	$(PY) -m mypy src evals

test:  ## the suite minus live-model and live-Langfuse tests (those are test-all)
	$(PY) -m pytest -q -m "not network and not integration"

test-all:  ## everything, including tests that call Gemini and Langfuse; slow under backoff
	$(PY) -m pytest -q

test-fast:  ## skip tests that load the embedding model or hit the network
	$(PY) -m pytest -q -m "not slow and not network and not integration"

test-necessity:  ## the multi_hop_necessity fixtures (needs GOOGLE_API_KEY, 27 calls: 3 cases x 3 repeats)
	$(PY) -m pytest -q tests/test_necessity_fixtures.py -m "integration and network"

test-integration:  ## one real trace against a live local Langfuse (needs make langfuse-up)
	$(PY) -m pytest -q -m integration

check: lint type test-fast  ## what CI runs on every push

graph:  ## regenerate docs/img/graph.mmd from the compiled topology
	$(PY) -m src.cli graph

index:  ## build the FAISS index from the committed chunks
	$(PY) scripts/build_index.py --print-hashes

index-verify:  ## rebuild on CPU into a temp dir and compare rankings with the current index
	@rm -rf .index-verify && mkdir -p .index-verify
	$(PY) scripts/build_index.py --device cpu --out .index-verify --force
	$(PY) scripts/compare_index.py data/indices/BGE_cs512.faiss .index-verify/BGE_cs512.faiss

compare-index:  ## compare two FAISS indexes: INDEX_A=... INDEX_B=...
	$(PY) scripts/compare_index.py $(INDEX_A) $(INDEX_B)

verify-corpus:  ## check data/ against CORPUS.sha256
	$(PY) -m src.cli verify-corpus

corpus-info:  ## print reproducible corpus statistics
	$(PY) -m src.cli corpus-info

corpus-diversity:  ## can the corpus honestly fill the four eval strata?
	$(PY) scripts/corpus_diversity.py

select-attributes:  ## deterministic, stratified pick of the 11 absent-attribute anchors
	$(PY) -m evals.select_attributes

draft-evals:  ## draft the whole eval set (~180 model calls, ~15 min at the free-tier 15 RPM)
	$(PY) -m evals.build_set

audit-entities:  ## every entity the premise check asserts, across the whole set
	$(PY) scripts/audit_entities.py

verify-dataset:  ## re-verify the frozen artifact independently of the builder (freeze gate)
	$(PY) -m evals.verify_dataset evals/datasets/phase4.json

verify-evals:  ## verify eval items by hand (FILE=evals/datasets/phase4.json)
	$(PY) -m evals.verify_cli $(FILE)

prune-gold:  ## prune gold to the minimal jointly-supporting set (DROP=id,id to remove items)
	$(PY) -m evals.prune_gold --drop "$(DROP)"

gold-report:  ## print gold chunks in full with every claim located (AGAINST=rev to diff gold)
	$(PY) scripts/gold_report.py --minimal $(if $(AGAINST),--against $(AGAINST),)

rubric:  ## write docs/RUBRIC.md from evals/rubric.py — the one scoring rubric
	$(PY) -m evals.rubric

judge-sample:  ## the seeded 25-item judge-validation draw from the frozen set
	$(PY) -m evals.judge_sample

run-set:  ## run v3 over the frozen set on Gemini free tier; resumable; ~75 min at 15 RPM
	$(PY) -m evals.run_set

run-report:  ## every number the run report states, from evals/runs/v3_<sha8>.json
	$(PY) -m evals.run_set --report

bypass-probe:  ## retrieval vs generation vs guardrail for refused items (SCORES=path to a score sheet)
	$(PY) scripts/bypass_probe.py $(if $(SCORES),--from-scores $(SCORES),)

score:  ## score the 25 sampled agent answers by hand under docs/RUBRIC.md (human-first)
	$(PY) -m evals.score_cli

judge-print:  ## one judge request body, verbatim (ARM=luna-low ITEM=sp-036)
	$(PY) -m evals.judge print-one --arm $(or $(ARM),luna-low) --item $(or $(ITEM),sp-036)

judge-estimate:  ## request count and estimated cost per judge arm; submits nothing
	$(PY) -m evals.judge estimate

judge-agreement:  ## per-arm agreement with the human sheet, refusal labels never pooled
	$(PY) -m evals.judge agreement

metrics-report:  ## Phase 4 metrics per stratum with n (SHEET=path to a score sheet for outcomes)
	$(PY) -m evals.metrics $(if $(SHEET),--sheet $(SHEET),)

metrics-compare:  ## judge-free retrieval across runs and arms, with the repeat-run spread
	$(PY) -m evals.metrics --compare r1,r2,r3,dense_only,section_filter,embedding_small,v21

metrics-spread:  ## the three-draw outcome spread, and each arm measured against it
	$(PY) -m evals.metrics --spread dense_only,section_filter,v21

judge-spend:  ## total OpenAI judge spend, summed from the batches themselves
	$(PY) scripts/judge_spend.py --write

push-scores:  ## attach judge scores to the traces of the judged run (RUN=... SHEET=... [REPLACE=1])
	$(PY) scripts/push_scores.py --run $(RUN) --sheet $(SHEET) $(if $(REPLACE),--replace,)

gate:  ## the regression gate (RUN=path [SHEET=path] [BASELINE=path; default: the shipped, pinned one])
	$(PY) -m evals.gate --run $(RUN) $(if $(SHEET),--sheet $(SHEET),) --baseline $(or $(BASELINE),evals/baseline_metrics_pinned.json)

run-v21:  ## v2.1 (published main, pinned) over the frozen set on the same generator; 6h timebox
	$(PY) -m evals.run_baseline_v21

integrity:  ## the forbidden-phrase grep over our own prose
	@# data/, evals/runs/ and evals/baselines/ hold verbatim paper text or published v2.1 code, which says "at scale" and "high-throughput"
	@# in the papers' own words; excluding them is not a loophole, it is the boundary of "our prose".
	@! grep -rn --include='*.md' --include='*.py' --include='*.yml' --include='*.toml' \
	  -E 'production-grade|production-ready|enterprise-scale|at scale|high-throughput' \
	  --exclude-dir=.venv --exclude-dir=data --exclude-dir=runs --exclude-dir=baselines --exclude=CLAUDE.md . \
	  || (echo "forbidden phrase in our prose (CLAUDE.md §3)"; exit 1)
	@echo "integrity: no forbidden phrases in our prose"

readme-stats:  ## regenerate the README statistics block from the repo
	$(PY) scripts/readme_stats.py

injection-report:  ## detection and false-positive rates for the injection guardrail
	$(PY) scripts/injection_report.py

injection-live:  ## drive undetected injections through the live graph (needs GOOGLE_API_KEY)
	$(PY) scripts/injection_live_probe.py

screen-corpus:  ## run the injection detector over all committed chunks (no API calls)
	$(PY) scripts/screen_corpus.py

langfuse-up:  ## start the self-hosted Langfuse stack
	docker compose -f infra/docker-compose.langfuse.yml up -d
	@echo "Langfuse starting at http://localhost:3000 (first boot takes a minute)"

langfuse-down:  ## stop Langfuse, keeping its data
	docker compose -f infra/docker-compose.langfuse.yml down

langfuse-reset:  ## stop Langfuse and discard all trace data
	docker compose -f infra/docker-compose.langfuse.yml down -v

budget:  ## regenerate the spend table in docs/BUDGET.md from Langfuse traces
	$(PY) scripts/budget_from_traces.py

reconcile-cost:  ## check our notional cost against Langfuse's own figure, trace by trace
	$(PY) scripts/reconcile_cost.py

reconcile-d021:  ## reproduce the D-021 finding from the committed trace fixture
	$(PY) scripts/reconcile_cost.py --from-fixture

metrics:  ## print the current Prometheus exposition
	$(PY) -m src.cli metrics

guardrail-variance:  ## the scope guardrail across the seven v3 runs on identical input (no API calls)
	$(PY) scripts/guardrail_variance.py

guardrail-probe:  ## does pinning the classifier's sampling remove the flips? (--report reads the stored probe)
	$(PY) scripts/guardrail_variance.py --report

generator-determinism:  ## does seed=0,top_k=1 pin the generator? (REPORT=1 reprints the stored probe)
	$(PY) scripts/generator_determinism.py $(if $(REPORT),--report,)

gemini-reconcile:  ## Gemini bill vs every recorded token count, on tokens; names the gap (REPORT=1 reprints)
	$(PY) scripts/gemini_reconcile.py $(if $(REPORT),--report,)

corpus-licenses:  ## each paper's arXiv license, counted; changes nothing (REPORT=1 reprints, ATTRIBUTION=1 writes CORPUS_ATTRIBUTION.md)
	$(PY) scripts/corpus_licenses.py $(if $(REPORT),--report,) $(if $(ATTRIBUTION),--attribution,)

trace-units:  ## Langfuse units per traced query, and the sampling rate the daily ceiling implies
	$(PY) scripts/trace_units.py

# ── Phase 5: serving ───────────────────────────────────────────────────────────
serve:  ## run the API locally on :8000 (memory ledger; not a deployed configuration)
	$(PY) -m uvicorn src.api.main:app --port 8000 --no-proxy-headers

docker-build:  ## build the serving image (index and pinned model baked in)
	docker build -f infra/Dockerfile -t arxiv-agent-v3 .

docker-run:  ## run the image locally with a volume-backed ledger and threads on :7860
	@# Only GOOGLE_API_KEY crosses into the container — not the whole .env, which also holds
	@# the judge's OpenAI key that the agent never uses. Tracing is off locally.
	set -a && . ./.env && set +a && docker run --rm -p 127.0.0.1:7860:7860 -e GOOGLE_API_KEY \
	  -e API__LEDGER=sqlite -e API__LEDGER_PATH=/data/ledger.sqlite -e CHECKPOINT_DB=/data/threads.sqlite \
	  -v arxiv-agent-data:/data arxiv-agent-v3

deploy-space:  ## deploy to a Hugging Face Docker Space: SPACE=owner/name [DRY=1]
	$(PY) scripts/deploy_space.py --space $(SPACE) $(if $(DRY),--dry-run,--i-confirmed-public)

space-secrets:  ## (owner runs this) copy the Space's secrets from a gitignored deploy env file: SPACE=owner/name [ENV_FILE=.env.deploy]
	$(PY) scripts/space_secrets.py --space $(SPACE) --env-file $(or $(ENV_FILE),.env.deploy)

smoke-live:  ## assert a running instance's effects: URL=... [SPACE=owner/name] [ENV_FILE=.env.deploy] [SAMPLE_RATE=1.0] [NO_TRACE=1]
	$(PY) scripts/smoke_live.py --url $(URL) $(if $(SPACE),--space $(SPACE),) --env-file $(or $(ENV_FILE),.env) --sample-rate $(or $(SAMPLE_RATE),1.0) $(if $(NO_TRACE),--no-trace,)

verify-deploy:  ## does the live Space serve exactly git REV's files? SPACE=owner/name REV=sha
	$(PY) scripts/verify_deploy.py --space $(SPACE) --rev $(REV)

load-check:  ## G-3 load check: URL=... [SPACE=owner/name] [C=10] [SERVED=30] [MAX_MIN=15] [LABEL=deployed] [NO_TOKEN=1]
	@# caffeinate -i holds off idle sleep on macOS (not a closed lid on battery); the tool's own
	@# clock check marks the run INVALID if the machine sleeps anyway (D-052).
	$(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i) $(PY) scripts/load_check.py --url $(URL) $(if $(SPACE),--space $(SPACE),) --concurrency $(or $(C),10) --min-served $(or $(SERVED),30) --max-minutes $(or $(MAX_MIN),15) --label $(or $(LABEL),deployed) $(if $(NO_TOKEN),--no-token,) --key-exclusive

latency-single:  ## single user, warm: N=10 sequential requests SPACING=30 s apart -> evals/runs/latency_single_user.json
	$(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -i) $(PY) scripts/load_check.py --url $(URL) --single-user $(or $(N),10) --spacing $(or $(SPACING),30)

load-report:  ## every load-check number, from evals/runs/loadcheck_<LABEL>.json
	$(PY) scripts/load_check.py --report --label $(or $(LABEL),deployed)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .index-verify
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
