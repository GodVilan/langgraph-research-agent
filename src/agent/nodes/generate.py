"""Answer synthesis from retrieved context.

Context is rendered as delimited blocks tagged with their ``chunk_id``. Passage text is
neutralised before it gets here (``src/guardrails/screening.py``), and ``render_context``
neutralises again as a belt-and-braces measure — the structural guarantee that a passage
cannot close its own wrapper is the layer that does not depend on recognising an attack, so
it is applied at both the entry point and the render point.

The prompt itself (``generate.v2.md``) supplies the instruction layer: it names the
delimiter, states that content inside is data, and tells the model to report an embedded
instruction rather than obey it.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from src.agent.llm import call_text
from src.agent.prompts import load_prompt
from src.agent.state import AgentState, RetrievedChunk
from src.guardrails.injection import neutralise

log = logging.getLogger(__name__)

NODE = "generate"

MAX_CONTEXT_CHARS = 24_000
NO_CONTEXT_ANSWER = (
    "I could not find passages in the corpus that address this question. "
    "The corpus covers a fixed set of arXiv machine-learning papers, and this topic does "
    "not appear to be represented in it."
)


def render_context(chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Render retrieved chunks as delimited, id-tagged blocks.

    Titles are neutralised too: a paper title is attacker-controllable on a live arXiv
    fetch, and it is rendered inside the opening tag where a forged quote would be most
    effective.
    """
    blocks: list[str] = []
    total = 0
    for chunk in chunks:
        title = neutralise(chunk.title).replace('"', "'")
        block = (
            f'<passage chunk_id="{chunk.chunk_id}" paper="{title}" '
            f'section="{chunk.section_type}">\n{neutralise(chunk.text).strip()}\n</passage>'
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


async def generate(state: AgentState) -> dict[str, object]:
    chunks = state.get("retrieved") or []
    question = state["question"]

    if not chunks:
        log.info("No retrieved context; emitting the corpus-gap answer")
        return {
            "draft_answer": NO_CONTEXT_ANSWER,
            "messages": [AIMessage(content=NO_CONTEXT_ANSWER)],
        }

    context = render_context(chunks)
    critique = state.get("critique")
    gap_note = ""
    if critique is not None and critique.gaps:
        gap_note = (
            "\n\nA previous draft was judged incomplete for these reasons; address them:\n"
            + "\n".join(f"- {g}" for g in critique.gaps[:4])
        )

    answer, usage = await call_text(
        system=load_prompt("generate", state["request"].prompt_version),
        user=f"Question: {question}\n\nRetrieved passages:\n{context}{gap_note}",
    )

    return {
        "draft_answer": answer,
        "messages": [AIMessage(content=answer)],
        "usage": usage,
    }
