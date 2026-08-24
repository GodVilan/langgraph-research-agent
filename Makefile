PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install lint fmt type test test-fast test-integration graph index index-verify compare-index verify-corpus corpus-info corpus-diversity verify-evals select-attributes readme-stats injection-report injection-live screen-corpus langfuse-up langfuse-down langfuse-reset budget reconcile-cost reconcile-d021 metrics check clean

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

test:  ## full test suite
	$(PY) -m pytest -q

test-fast:  ## skip tests that load the embedding model or hit the network
	$(PY) -m pytest -q -m "not slow and not network and not integration"

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

verify-evals:  ## verify eval items by hand (FILE=evals/datasets/draft.json)
	$(PY) -m evals.verify_cli $(FILE)

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

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .index-verify
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
