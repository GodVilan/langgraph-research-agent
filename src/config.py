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
    """USD per 1M tokens. Reasoning/thinking tokens bill at the output rate.

    ``input_usd``/``output_usd`` are what we are actually billed on the configured tier.
    ``notional_*`` are the paid-tier rates, used to compute a shadow cost so the budget
    ceiling is exercised and testable even while the agent runs on a free tier.
    See docs/DECISIONS.md D-004.

    ``verified`` records whether these rates were checked against provider documentation.
    Unverified rates still drive the ceiling — a ceiling computed from a placeholder is far
    better than one silently computed from zero — but they must never be published as a
    cost figure without checking first.
    """

    input_usd: float
    output_usd: float
    notional_input_usd: float
    notional_output_usd: float
    verified: bool = False
    source: str = ""


# Do not edit from memory — re-verify against provider docs and update `source`.
# Free tier: no charge. Free-tier content may be used to improve Google's products;
# acceptable here because the corpus is public arXiv text (README, DECISIONS D-003).
PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash-lite": ModelPricing(
        input_usd=0.0,
        output_usd=0.0,
        notional_input_usd=0.10,
        notional_output_usd=0.40,
        verified=True,
        source="provider docs, 2026-08-19",
    ),
    # In use. Notional rates are paid *standard*, not batch — the agent serves interactive
    # requests, so batch pricing would understate what a paid deployment would pay.
    # Batch/Flex are $0.15 / $1.25 for reference. Context caching is not available on the
    # free tier for this model, and the output rate includes thinking tokens, which is why
    # the thinking budget is pinned rather than left at the model default (D-014).
    "gemini-3.5-flash-lite": ModelPricing(
        input_usd=0.0,
        output_usd=0.0,
        notional_input_usd=0.30,
        notional_output_usd=2.50,
        verified=True,
        source="Google official pricing page, 2026-08-19 (paid standard tier)",
    ),
    # Not in use. Rates unchecked; the 3.5 standard rates are a closer placeholder than
    # 2.5's, but the entry stays unverified so switching to it warns loudly.
    "gemini-3.1-flash-lite": ModelPricing(
        input_usd=0.0,
        output_usd=0.0,
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
    input_usd=0.0,
    output_usd=0.0,
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
    max_notional_cost_usd: float = 0.05
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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

    # ── Secrets ────────────────────────────────────────────────────────────────
    google_api_key: SecretStr = SecretStr("")
    hf_token: SecretStr = SecretStr("")

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

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: Path = DATA_DIR
    index_dir: Path = INDEX_DIR
    checkpoint_db: Path = CHECKPOINT_DIR / "threads.sqlite"

    # ── Device ─────────────────────────────────────────────────────────────────
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    # ── Nested ─────────────────────────────────────────────────────────────────
    budget: BudgetLimits = Field(default_factory=BudgetLimits)
    graph: GraphLimits = Field(default_factory=GraphLimits)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)

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
