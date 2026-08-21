"""Prometheus metrics.

Defined here rather than in the API layer so the CLI, the eval harness, and the Phase 5
FastAPI service all report against the same registry. Phase 5 mounts ``exposition()`` at
``GET /metrics``; nothing else about these definitions changes when it does.

Two deliberate choices:

* **Guardrail triggers are labelled by severity as well as category.** 60 of the 5,401
  corpus chunks (1.11%) produce a WARN, and almost all are false positives on papers that
  quote prompts (docs/SCREEN.md). A single trigger counter would show a steady stream of
  alarming-looking events for something that withholds nothing. `warn` and `block` mean
  different things and are counted separately.
* **Cost is two counters, not one.** Billed cost is $0 on the free tier; notional cost
  prices the same tokens at paid rates (D-004). Collapsing them would make the dashboard
  read zero forever, or imply spend that is not happening.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()

# Wall-clock buckets sized for this agent: a refusal returns in well under a second, a
# multi-hop query with a refinement round takes tens of seconds, and the per-request
# ceiling is 120s (BudgetLimits.max_wall_clock_s).
LATENCY_BUCKETS = (0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0)

requests_total = Counter(
    "arxiv_agent_requests_total",
    "Queries received, by terminal outcome.",
    labelnames=("outcome",),  # answered | refused | truncated | error
    registry=REGISTRY,
)

request_latency_seconds = Histogram(
    "arxiv_agent_request_latency_seconds",
    "End-to-end latency of one query.",
    labelnames=("outcome",),
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

errors_total = Counter(
    "arxiv_agent_errors_total",
    "Failures by type, so a spike can be attributed without reading logs.",
    labelnames=("error_type",),
    registry=REGISTRY,
)

node_latency_seconds = Histogram(
    "arxiv_agent_node_latency_seconds",
    "Latency of one graph node.",
    labelnames=("node",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

guardrail_triggers_total = Counter(
    "arxiv_agent_guardrail_triggers_total",
    "Guardrail events. Severity is a label because a warn withholds nothing and a block "
    "removes a passage from the model's context.",
    labelnames=("category", "severity", "node"),
    registry=REGISTRY,
)

chunks_quarantined_total = Counter(
    "arxiv_agent_chunks_quarantined_total",
    "Retrieved passages withheld from the model by the injection guardrail.",
    labelnames=("source",),
    registry=REGISTRY,
)

tokens_total = Counter(
    "arxiv_agent_tokens_total",
    "Tokens consumed. Thinking tokens are billed at the output rate and counted separately "
    "because they are the dominant cost lever (DECISIONS D-014).",
    labelnames=("kind",),  # input | output | thinking
    registry=REGISTRY,
)

cost_usd_total = Counter(
    "arxiv_agent_cost_usd_total",
    "Accumulated cost. 'billed' is what the provider charges (0 on the free tier); "
    "'notional' prices the same tokens at paid rates (DECISIONS D-004).",
    labelnames=("kind",),  # billed | notional
    registry=REGISTRY,
)

llm_calls_total = Counter(
    "arxiv_agent_llm_calls_total",
    "Model calls, by node.",
    labelnames=("node", "model"),
    registry=REGISTRY,
)

retrieval_hits = Histogram(
    "arxiv_agent_retrieval_hits",
    "Passages returned by one retrieval pass, by retriever.",
    labelnames=("retriever",),
    buckets=(0, 1, 2, 3, 5, 8, 10, 20),
    registry=REGISTRY,
)

refinements_total = Counter(
    "arxiv_agent_refinements_total",
    "Critique-driven refinement rounds.",
    registry=REGISTRY,
)

missing_usage_metadata_total = Counter(
    "arxiv_agent_missing_usage_metadata_total",
    "Model responses that reported no token usage. Non-zero means the token and cost "
    "counters understate reality.",
    registry=REGISTRY,
)

build_info = Gauge(
    "arxiv_agent_build_info",
    "Static build labels; always 1.",
    labelnames=("version", "model", "prompt_version"),
    registry=REGISTRY,
)


def exposition() -> tuple[bytes, str]:
    """Render the registry. Phase 5 returns this from ``GET /metrics``."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def record_run(state: dict[str, Any], elapsed_s: float) -> None:
    """Record one completed run from its terminal state.

    Called once per query rather than sprinkled through the nodes, so the counters cannot
    disagree with the state that produced them.
    """
    from src.agent.state import GuardrailEvent, RetrievalEvent, Usage

    refused = bool(state.get("refused"))
    truncated = bool(state.get("truncated"))
    outcome = "refused" if refused else ("truncated" if truncated else "answered")

    requests_total.labels(outcome=outcome).inc()
    request_latency_seconds.labels(outcome=outcome).observe(elapsed_s)

    usage = state.get("usage")
    if isinstance(usage, Usage):
        tokens_total.labels(kind="input").inc(usage.input_tokens)
        tokens_total.labels(kind="output").inc(usage.output_tokens)
        tokens_total.labels(kind="thinking").inc(usage.reasoning_tokens)
        cost_usd_total.labels(kind="billed").inc(usage.cost_usd)
        cost_usd_total.labels(kind="notional").inc(usage.notional_cost_usd)
        if usage.missing_usage_metadata:
            missing_usage_metadata_total.inc(usage.missing_usage_metadata)

    for event in state.get("guardrail_events") or []:
        if isinstance(event, GuardrailEvent):
            category = event.kind.removeprefix("injection_")
            guardrail_triggers_total.labels(
                category=category, severity=event.severity, node=event.node or "unknown"
            ).inc()
            if event.kind == "injection_quarantine":
                chunks_quarantined_total.labels(source="retrieved").inc()

    for event in state.get("retrieval_events") or []:
        if isinstance(event, RetrievalEvent):
            retrieval_hits.labels(retriever=event.retriever).observe(event.n_hits)

    refinements = state.get("refinement_count")
    if isinstance(refinements, int) and refinements > 0:
        refinements_total.inc(refinements)


def record_error(error_type: str, elapsed_s: float = 0.0) -> None:
    errors_total.labels(error_type=error_type).inc()
    requests_total.labels(outcome="error").inc()
    if elapsed_s:
        request_latency_seconds.labels(outcome="error").observe(elapsed_s)
