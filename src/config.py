"""Central configuration.

All settings are Pydantic v2 models loaded from the environment. Nothing here reads a
secret at import time except through ``Settings``, so tests can construct a ``Settings``
instance directly without touching the filesystem.

Retrieval constants carried over from v2.1 are marked FROZEN. Changing one invalidates
the v2.1-vs-v3 comparison the evaluation exists to make; see docs/DECISIONS.md D-002.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "indices"
CHECKPOINT_DIR = REPO_ROOT / ".checkpoints"

# ── FROZEN retrieval constants (ported verbatim from v2.1 rag/config.py) ───────
EMBEDDING_MODEL = "BAAI/bge-large-en"
EMBEDDING_DIM = 1024
DEFAULT_CHUNK = 512
CHUNK_OVERLAP = 64
INDEX_NAME = f"BGE_cs{DEFAULT_CHUNK}"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class ModelPricing(BaseModel):
    """Paid-tier USD per 1M tokens, used for the *notional* cost every ceiling checks (D-004).

    There is deliberately no "billed" rate here. Until 2026-09-24 this model carried billed
    rates of $0 from an assumption that the key was on the free tier; the key's project had
    billing enabled throughout and Google billed $7.60 (DECISIONS D-046). A billed figure now
    comes only from the provider's own record (docs/billing/), never from a rate in code.

    Reasoning/thinking tokens bill at the output rate. ``notional_cached_input_usd`` prices the
    part of the prompt the provider served from its implicit cache; ``None`` means the cached
    rate is not verified, and cached tokens are then priced at the full input rate — an
    overstatement, the safe direction for a ceiling.
    """

    notional_input_usd: float
    notional_output_usd: float
    notional_cached_input_usd: float | None = None
    verified: bool = False
    source: str = ""


# Do not edit from memory — re-verify against provider docs and update `source`.
PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash-lite": ModelPricing(
        notional_input_usd=0.10,
        notional_output_usd=0.40,
        verified=True,
        source="provider docs, 2026-08-19; matched by the Cloud Billing SKU rates, 2026-09-24",
    ),
    # In use. Paid *standard* rates, not batch — the agent serves interactive requests.
    # Batch/Flex are $0.15 / $1.25 for reference. The output rate includes thinking tokens,
    # which is why the thinking budget is pinned (D-014). Cached input: $0.03/1M, the rate
    # Google Cloud Billing actually charged on SKU 3D01-132D-D29C (docs/billing/gemini.json).
    "gemini-3.5-flash-lite": ModelPricing(
        notional_input_usd=0.30,
        notional_output_usd=2.50,
        notional_cached_input_usd=0.03,
        verified=True,
        source="Google pricing page 2026-08-19 (paid standard); all three rates matched by the "
        "Cloud Billing SKU breakdown, 2026-09-24",
    ),
    # Not in use. Rates unchecked; the 3.5 standard rates are a closer placeholder than
    # 2.5's, but the entry stays unverified so switching to it warns loudly.
    "gemini-3.1-flash-lite": ModelPricing(
        notional_input_usd=0.30,
        notional_output_usd=2.50,
        verified=False,
        source="placeholder: 3.5-flash-lite standard rates, unverified for 3.1",
    ),
}
# A model with no entry prices at zero, which would silently disable the notional cost
# ceiling — the exact failure D-004 exists to prevent. `Settings.pricing()` logs a warning
# when this is hit so it cannot pass unnoticed.
UNKNOWN_MODEL_PRICING = ModelPricing(
    notional_input_usd=0.0,
    notional_output_usd=0.0,
    verified=False,
    source="unknown model — cost ceiling is inert",
)


class BudgetLimits(BaseModel):
    """Per-request ceilings. Exceeding any one returns a partial answer with an explicit
    truncation flag — never a silent stop (Phase 2 requirement, enforced from Phase 1).

    The cost ceiling is checked against *notional* cost so that it is a live, testable
    guard on the free tier rather than dead code that only wakes up if we start paying.
    """

    max_input_tokens: int = 120_000
    max_output_tokens: int = 12_000
    # A pathological-case bound, not observed behaviour: max_llm_calls (16) x the largest
    # per-call notional cost seen over the 69-item Phase 4 run ($0.00156, `make run-report`)
    # = $0.0249 — every permitted call at the largest size ever seen, which no real query
    # does. It binds on call *size*, the one thing the call ceiling cannot bound, and never
    # on an ordinary run (observed per-query max $0.0057, 4.4x headroom). Previous $0.05 was
    # calibrated against placeholder rates understating cost ~3.6x (DECISIONS D-004 note).
    max_notional_cost_usd: float = 0.025
    max_wall_clock_s: float = 120.0
    max_tool_calls: int = 12
    max_llm_calls: int = 16


class GraphLimits(BaseModel):
    """Structural bounds on the graph itself."""

    max_refinements: int = 2
    max_sub_questions: int = 4
    # Longest legal path is 17 steps (MIGRATION_MAP §4). 25 leaves headroom while still
    # raising GraphRecursionError loudly if a counter is ever wrong.
    recursion_limit: int = 25
    # Bounded repair attempts for structured-output parse failures.
    max_structured_output_attempts: int = 2


class RetrievalSettings(BaseModel):
    """FROZEN values are ported from v2.1. The routing policy is not frozen — replacing
    the LLM's free tool choice with a deterministic rule is the point of the rebuild.
    """

    top_k: int = 5  # FROZEN: v2.1 DEFAULT_TOP_K
    # Deterministic route: always run dense; add BM25 only when dense under-delivers.
    # This replaces v2.1's LLM-chosen search_corpus/keyword_search (MIGRATION_MAP §5.1).
    # The effective trigger is min(top_k, this), so lowering top_k does not silently turn
    # the fallback into an always-on second retriever.
    sparse_fallback_threshold: int = 5
    # Section filtering ships disabled. AUDIT §4.15 found it dead in v2.1 and the
    # classifier's precision is unvalidated; it stays off until Phase 4 measures it.
    enable_section_filter: bool = False
    classify_sections_on_load: bool = True
    # The embedding model, index name and query prefix default to the FROZEN constants above
    # and are settable only so the Phase 4 embedding-comparison arm can point the same graph
    # at a second index (`RETRIEVAL__EMBEDDING_MODEL=…`). The shipping values are the frozen
    # ones; `tests/test_retrieval_defaults.py` asserts that.
    embedding_model: str = EMBEDDING_MODEL
    index_name: str = INDEX_NAME
    query_prefix: str = BGE_QUERY_PREFIX


class ApiLimits(BaseModel):
    """Serving-side ceilings for the public endpoint (Phase 5).

    These sit *outside* the graph. ``BudgetLimits`` bounds one request; nothing in it can
    bound a day, a client, or how many requests run at once, and D-013 deliberately made the
    per-request budget forget everything between requests. Each limit here protects the API
    key behind the endpoint from a different shape of abuse: many requests from one client
    (per-IP), many clients at once (concurrency), and many requests over a day (the cost
    ceiling, which is the only one that must survive a restart — docs/SERVING.md).
    """

    # Per-IP token bucket: `per_ip_per_minute` refill, `per_ip_burst` capacity. In memory
    # and per process: a smoothing limit, not a budget, so losing it on restart costs a
    # burst, not money. The daily ceiling below is the one that has to persist.
    per_ip_per_minute: float = 4.0
    per_ip_burst: int = 3
    # Queries admitted to the graph at once. Gemini's free tier allows 15 requests/minute
    # and the median query makes 4 model calls (`make run-report`), so the model quota —
    # not CPU — is the ceiling on throughput. More in-flight queries would only queue
    # inside the model client's retry loop, where nothing reports it.
    max_concurrent_queries: int = 2
    # How long an admitted-but-waiting request may queue for a slot before 503.
    queue_timeout_s: float = 20.0
    # Global daily ceiling on *notional* cost — the tokens priced at paid standard rates. Not
    # on billed cost: billing is known only from the provider's record, after the fact
    # (D-046), and a ceiling needs a figure at request time. $0.50/day is ~150 median
    # queries ($0.0033 each, `make run-report`) and 20 pathological ones at the $0.025
    # per-request bound.
    daily_notional_ceiling_usd: float = 0.50
    # Where the daily ledger lives. `memory` is refused in a deployed container; `sqlite`
    # is refused there unless the path is on a mounted volume (src/api/ledger.py).
    ledger: Literal["memory", "sqlite", "upstash"] = "memory"
    ledger_path: Path = CHECKPOINT_DIR / "ledger.sqlite"
    # Entries appended by proxies in front of the service. 0 = trust only the socket peer;
    # X-Forwarded-For is client-controlled and would let anyone pick their own rate-limit
    # key. Set to the number of proxies that *append* to the header (docs/SERVING.md).
    trusted_proxy_hops: int = 0
    # Set by the container image. Turns ephemeral-state guards from warnings into refusals.
    deployed: bool = False
    # Live arXiv fetch stays off on the public endpoint: it triggers outbound fetches and
    # PDF parsing on an anonymous caller's behalf, and fetched papers enter a process-wide
    # session index (src/retrieval/session_index.py). The CLI keeps it (DECISIONS D-036).
    allow_arxiv: bool = False
    max_top_k: int = 10
    # Held exclusively for the process lifetime in a deployed container (single worker).
    worker_lock_path: Path = Path("/tmp/arxiv-agent-v3.worker.lock")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    # ── Secrets ────────────────────────────────────────────────────────────────
    google_api_key: SecretStr = SecretStr("")
    hf_token: SecretStr = SecretStr("")
    # The daily cost ledger on a host with no persistent disk (API__LEDGER=upstash).
    upstash_redis_rest_url: str = ""
    upstash_redis_rest_token: SecretStr = SecretStr("")
    # Lets the load check bypass the per-IP limiter only — never the daily ceiling or the
    # concurrency cap, which protect the key. Unset = no bypass exists (docs/SERVING.md).
    loadcheck_token: SecretStr = SecretStr("")

    # ── Models ─────────────────────────────────────────────────────────────────
    # Routed through init_chat_model so a provider swap is config, not code.
    agent_model: str = "google_genai:gemini-3.5-flash-lite"
    # Ignored by gemini-3.5-flash-lite, which uses fixed sampling defaults. Kept because it
    # still applies to other providers reachable through init_chat_model.
    agent_temperature: float = 0.0
    # Pinned, never left at the model default. Thinking tokens bill at the *output* rate
    # ($2.50/M), so this is the dominant cost lever, and a provider-side change to the
    # default would silently multiply the bill. Measured on 2026-08-20 with
    # `scripts/determinism_probe.py --runs 5` (DECISIONS D-014): "minimal" produces 0
    # thinking tokens and ~79 output tokens, while "low" produces ~426 thinking and ~508
    # output — roughly 6x the output cost for the same prompt. Thinking cannot be disabled
    # outright; thinking_budget=0 is rejected with INVALID_ARGUMENT.
    agent_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "minimal"
    # Process-wide pacing of model calls, requests per minute; 0 = off. Off by default so
    # the eval path — where p50/p95 are measured — carries no limiter sleep (BACKLOG, Phase
    # 3). The container sets it to the free-tier quota, so a burst queues here, visibly,
    # rather than inside the provider client's silent retry loop.
    agent_requests_per_minute: float = 0.0
    # The input scope classifier runs with seed=0, top_k=1 (DECISIONS D-035): measured
    # byte-identical over 140 calls where the unpinned model flipped verdicts on 3 of 28
    # questions. Phase 4's metrics were measured with this False; set False to reproduce them.
    pin_scope_classifier: bool = True

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: Path = DATA_DIR
    index_dir: Path = INDEX_DIR
    checkpoint_db: Path = CHECKPOINT_DIR / "threads.sqlite"
    # Every model response appends one row here (src/agent/llm.py record_usage) — the record
    # the D-046 reconciliation did not have. Gitignored; the test suite redirects it.
    usage_log: Path = REPO_ROOT / ".usage" / "llm_usage.jsonl"

    # ── Device ─────────────────────────────────────────────────────────────────
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    # ── Nested ─────────────────────────────────────────────────────────────────
    budget: BudgetLimits = Field(default_factory=BudgetLimits)
    graph: GraphLimits = Field(default_factory=GraphLimits)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    api: ApiLimits = Field(default_factory=ApiLimits)

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / f"chunks_{DEFAULT_CHUNK}.json"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "metadata.json"

    @property
    def corpus_checksum_path(self) -> Path:
        return self.data_dir / "CORPUS.sha256"

    def model_name(self) -> str:
        """Bare model id without the provider prefix, for pricing lookup."""
        return self.agent_model.split(":", 1)[-1]

    def pricing(self) -> ModelPricing:
        name = self.model_name()
        entry = PRICING.get(name)
        if entry is None:
            log.warning(
                "No pricing entry for %r — notional cost will compute as $0 and the cost "
                "ceiling is inert for this model. Add it to PRICING in src/config.py.",
                name,
            )
            return UNKNOWN_MODEL_PRICING
        if not entry.verified:
            log.warning(
                "Pricing for %r is unverified (%s). The ceiling still applies, but do not "
                "publish a cost figure from it.",
                name,
                entry.source,
            )
        return entry


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device. Imports torch lazily so config stays cheap to import."""
    if preference != "auto":
        return preference
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
