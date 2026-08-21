"""Command-line entry point."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from src.config import get_settings

app = typer.Typer(add_completion=False, help="arXiv Agent v3 — graph-orchestrated research agent.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The research question.")],
    thread: Annotated[str | None, typer.Option(help="Resume an existing thread id.")] = None,
    arxiv: Annotated[
        bool, typer.Option(help="Allow live arXiv fetch when the corpus falls short.")
    ] = False,
    top_k: Annotated[int, typer.Option(help="Passages per retrieval pass.")] = 5,
    stream: Annotated[bool, typer.Option(help="Stream node-level progress.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Answer one question end to end."""
    _setup_logging(verbose)
    asyncio.run(_query(question, thread, arxiv, top_k, stream))


async def _query(question: str, thread: str | None, arxiv: bool, top_k: int, stream: bool) -> None:
    from src.agent.graph import build_graph, sqlite_checkpointer
    from src.agent.runner import new_thread_id, run_query, stream_events
    from src.agent.state import RequestOptions
    from src.retrieval.service import RetrievalService

    typer.secho("Loading corpus and index…", fg=typer.colors.BRIGHT_BLACK)
    service = RetrievalService.load()
    options = RequestOptions(use_arxiv=arxiv, top_k=top_k)
    thread_id = thread or new_thread_id()

    async with sqlite_checkpointer() as saver:
        graph = build_graph(service, checkpointer=saver)

        if stream:
            async for event in stream_events(graph, question, thread_id, options):
                if event["event"] == "on_chain_start":
                    typer.secho(f"  -> {event['node']}", fg=typer.colors.BRIGHT_BLACK)
                elif event["event"] == "token":
                    typer.echo(event["text"], nl=False)
            typer.echo()

        state = await run_query(graph, question, thread_id, options)

    typer.echo()
    typer.secho(state.get("answer") or "(no answer)", fg=typer.colors.WHITE)

    sources = state.get("sources") or []
    if sources:
        typer.echo()
        typer.secho("Sources", fg=typer.colors.CYAN, bold=True)
        for i, src in enumerate(sources[:8], 1):
            typer.echo(f"  [{i}] {src.title[:78]}  ({src.score:.3f}, {len(src.chunk_ids)} chunks)")

    usage = state["usage"]
    crit = state.get("critique")
    typer.echo()
    typer.secho(
        f"thread={state['thread_id']}  "
        f"llm_calls={usage.llm_calls}  tool_calls={usage.tool_calls}  "
        f"tokens={usage.input_tokens}in/{usage.output_tokens}out  "
        f"notional=${usage.notional_cost_usd:.5f}  "
        f"critique={crit.verdict if crit else 'n/a'}  "
        f"refinements={state.get('refinement_count', 0)}",
        fg=typer.colors.BRIGHT_BLACK,
    )
    if usage.missing_usage_metadata:
        typer.secho(
            f"  note: {usage.missing_usage_metadata} response(s) reported no token usage; "
            f"the figures above understate actual spend.",
            fg=typer.colors.YELLOW,
        )
    if state.get("truncated"):
        typer.secho(f"  TRUNCATED: {state.get('truncation_reason')}", fg=typer.colors.RED)
    for guardrail in state.get("guardrail_events") or []:
        if guardrail.severity != "info":
            typer.secho(
                f"  guardrail[{guardrail.severity}] {guardrail.kind}: {guardrail.detail}",
                fg=typer.colors.YELLOW,
            )


@app.command()
def graph(
    output: Annotated[Path, typer.Option(help="Write the Mermaid diagram here.")] = Path(
        "docs/img/graph.mmd"
    ),
) -> None:
    """Regenerate the graph diagram from the compiled topology."""
    from src.agent.graph import draw_mermaid

    mermaid = draw_mermaid()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(mermaid, encoding="utf-8")
    typer.echo(mermaid)
    typer.secho(f"\nWrote {output}", fg=typer.colors.GREEN)


@app.command("verify-corpus")
def verify_corpus() -> None:
    """Check the committed corpus against data/CORPUS.sha256.

    Every baseline JSON and the QA dataset's ``generated_by`` block record this checksum,
    so a silent corpus drift cannot orphan a benchmark the way it did in v2.1 (AUDIT §5.2).
    """
    settings = get_settings()
    checksum_file = settings.corpus_checksum_path
    if not checksum_file.exists():
        typer.secho(f"Missing {checksum_file}", fg=typer.colors.RED)
        raise typer.Exit(1)

    failures = 0
    for line in checksum_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        expected, name = line.split(None, 1)
        path = Path(name.strip())
        if not path.exists():
            typer.secho(f"  MISSING  {path}", fg=typer.colors.RED)
            failures += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected:
            typer.secho(f"  OK       {path}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"  MISMATCH {path}\n           expected {expected}\n           actual   {digest}",
                fg=typer.colors.RED,
            )
            failures += 1

    if failures:
        typer.secho(f"\n{failures} file(s) failed verification.", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("\nCorpus matches CORPUS.sha256.", fg=typer.colors.GREEN)


@app.command("corpus-info")
def corpus_info() -> None:
    """Print corpus statistics, all of which are reproducible from committed files."""
    from src.retrieval.chunker import load_chunks

    settings = get_settings()
    with open(settings.metadata_path) as f:
        papers = json.load(f)
    chunks = load_chunks(settings.chunks_path, classify_on_load=True)

    from collections import Counter

    sections = Counter(c.section_type for c in chunks)
    typer.echo(f"papers           : {len(papers)}")
    typer.echo(f"chunks           : {len(chunks)}")
    typer.echo(f"mean tokens/chunk: {sum(c.token_count for c in chunks) / len(chunks):.1f}")
    typer.echo("section_type (recomputed at load; see AUDIT §4.15):")
    for name, count in sections.most_common():
        typer.echo(f"  {name:<14} {count:>6}  ({count / len(chunks):.1%})")


@app.command()
def metrics() -> None:
    """Print the current Prometheus exposition.

    Phase 5 serves this from `GET /metrics`; this command exists so the metric definitions
    can be inspected and tested without a running service.
    """
    from src.observability.metrics import exposition

    payload, _ = exposition()
    typer.echo(payload.decode("utf-8"))


@app.command("trace-check")
def trace_check() -> None:
    """Report whether observability is configured, without making a model call."""
    from src.observability.config import get_observability_settings
    from src.observability.langfuse import get_client

    obs = get_observability_settings()
    typer.echo(f"langfuse host        : {obs.langfuse_host}")
    typer.echo(f"langfuse environment : {obs.langfuse_environment}")
    typer.echo(f"keys configured      : {obs.langfuse_enabled}")

    if not obs.langfuse_enabled:
        typer.secho(
            "\nLangfuse is disabled. Tracing is a no-op; the agent runs normally.\n"
            "  make langfuse-up   then set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(0)

    client = get_client()
    if client is None:
        typer.secho("Keys are set but the client would not start.", fg=typer.colors.RED)
        raise typer.Exit(1)
    try:
        reachable = bool(client.auth_check())
    except Exception as exc:  # a reachability failure is the answer, not a crash
        typer.secho(f"\nNot reachable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.secho(f"\nreachable            : {reachable}", fg=typer.colors.GREEN)


def main() -> None:
    sys.exit(app())


if __name__ == "__main__":
    main()
