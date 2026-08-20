"""Apply the injection policy to a batch of retrieved chunks.

Sits between retrieval and state: chunks are screened at the point they enter
``AgentState``, not at render time, so a detection is recorded even when the run is later
truncated by a budget ceiling and ``generate`` never executes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agent.state import GuardrailEvent, RetrievedChunk
from src.guardrails.injection import (
    Severity,
    is_untrusted_source,
    neutralise,
    scan,
    worst_severity,
)

log = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    kept: list[RetrievedChunk]
    quarantined: list[RetrievedChunk]
    events: list[GuardrailEvent]

    @property
    def n_flagged(self) -> int:
        return len({e.chunk_id for e in self.events if e.chunk_id})


def screen_chunks(chunks: list[RetrievedChunk], node: str = "retrieve") -> ScreeningResult:
    """Neutralise every chunk, quarantine the ones that trip a BLOCK rule.

    Neutralisation is unconditional — it costs nothing and does not depend on the detector
    recognising anything. Quarantine is the detector's call, and drops the chunk from the
    context entirely rather than passing it through with a warning label the model may
    ignore.
    """
    kept: list[RetrievedChunk] = []
    quarantined: list[RetrievedChunk] = []
    events: list[GuardrailEvent] = []

    for chunk in chunks:
        strict = is_untrusted_source(chunk.source, chunk.retriever)
        detections = scan(chunk.text, strict=strict)
        severity = worst_severity(detections)

        cleaned = neutralise(chunk.text)
        if cleaned != chunk.text:
            events.append(
                GuardrailEvent(
                    kind="injection_neutralised",
                    severity="info",
                    node=node,
                    chunk_id=chunk.chunk_id,
                    detail="delimiter or invisible characters removed from passage text",
                )
            )

        for detection in detections:
            events.append(
                GuardrailEvent(
                    kind=f"injection_{detection.category.value}",
                    severity="block" if detection.severity is Severity.BLOCK else "warn",
                    node=node,
                    chunk_id=chunk.chunk_id,
                    detail=(
                        f"{detection.pattern}"
                        f"{' [strict: runtime-fetched source]' if strict else ''} — "
                        f"{detection.excerpt[:160]}"
                    ),
                )
            )

        safe = chunk.model_copy(update={"text": cleaned})
        if severity is Severity.BLOCK:
            log.warning(
                "Quarantined chunk %s (%s): %s",
                chunk.chunk_id,
                chunk.source,
                ", ".join(d.describe()[:60] for d in detections[:2]),
            )
            quarantined.append(safe)
        else:
            kept.append(safe)

    if quarantined:
        events.append(
            GuardrailEvent(
                kind="injection_quarantine",
                severity="block",
                node=node,
                detail=(
                    f"{len(quarantined)} of {len(chunks)} retrieved passage(s) withheld from "
                    f"the model: {', '.join(c.chunk_id for c in quarantined[:5])}"
                ),
            )
        )

    return ScreeningResult(kept=kept, quarantined=quarantined, events=events)
