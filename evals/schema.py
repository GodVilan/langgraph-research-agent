"""Eval item schema, versioned and checksummed.

One item is one question with its gold chunks, its stratum, and the provenance of how it
was made. The schema is deliberately strict: an item that cannot say where it came from or
what it is testing is not admissible, because the whole point of this dataset is to be the
thing v2.1 did not have — a benchmark that can still be re-derived after the corpus moves
(AUDIT §5).

**The two unanswerable sub-strata are separate types and must never be merged.** They test
different capabilities:

* ``unanswerable_topic`` — the corpus does not discuss the subject at all. Tests scope
  refusal. Supply is hard-limited: only four topics survive full-corpus screening
  (``make corpus-diversity``).
* ``unanswerable_attribute`` — the subject is a real paper in the corpus, but the specific
  fact asked for is one it never reports. Tests attribute refusal, and is the sharper probe
  of the two: retrieval lands on the anchor paper's own chunks at high similarity, so a
  system with no score floor has every reason to answer confidently and wrongly.

Reporting them as one "refusal accuracy on n=15" would average two different skills and
hide which one failed. ``Stratum.reporting_group`` keeps them apart in the report.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1


class Stratum(StrEnum):
    SINGLE_PAPER = "single_paper_factual"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE_TOPIC = "unanswerable_topic"
    UNANSWERABLE_ATTRIBUTE = "unanswerable_attribute"
    AMBIGUOUS = "ambiguous"

    @property
    def expects_refusal(self) -> bool:
        return self in {Stratum.UNANSWERABLE_TOPIC, Stratum.UNANSWERABLE_ATTRIBUTE}

    @property
    def reporting_group(self) -> str:
        """Metrics are reported per group. The two refusal strata stay distinct."""
        return str(self.value)


class AbsenceShape(StrEnum):
    """What *kind* of fact is missing, for ``unanswerable_attribute`` items.

    Eleven items all shaped "what did X report on benchmark Y" is one item measured eleven
    times. Spreading the shape is what makes the sub-stratum a measurement rather than a
    repetition, so the shape is recorded per item and the balance is asserted at freeze.
    """

    BENCHMARK = "benchmark"
    DATASET = "dataset"
    ABLATION = "ablation"
    BASELINE = "baseline"
    HYPERPARAMETER = "hyperparameter"
    COMPUTE = "compute"


class Provenance(BaseModel):
    """How this item came to exist. Required — an unattributable item is not admissible."""

    model_config = ConfigDict(extra="forbid")

    generator_model: str = Field(description="Model that drafted it, or 'hand' if written")
    generator_snapshot: str = Field(default="", description="Pinned snapshot/revision id")
    prompt_version: str = Field(default="", description="Generation prompt version")
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source_paper_ids: list[str] = Field(default_factory=list)


class AnchorClass(StrEnum):
    """How reliably the cross-citation check can recognise an item's anchor paper.

    Only 45 of 150 papers carry a method name distinctive enough to match on; for the rest
    the check rests on the arXiv id alone and is correspondingly weaker. Recording the class
    per item keeps that difference identifiable instead of averaged into one number.
    """

    STRONG = "strong"  # distinctive method name plus arXiv id
    WEAK = "weak"  # arXiv id only


class MachineCheck(BaseModel):
    """One automated verdict, recorded so a disagreement points at a specific check."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str = ""


class Verification(BaseModel):
    """What has actually been checked. Every flag defaults to *unchecked*.

    The human label and the machine verdicts are stored side by side and never merged. The
    verification CLI takes the human decision **before** showing any machine verdict, so
    ``human_accepted`` is an independent judgement rather than a ratification: a verifier
    shown "necessity: PASS" first produces an agreement rate that measures agreement with
    the checker, not the generator-vs-human disagreement this stratum's validity rests on.
    """

    model_config = ConfigDict(extra="forbid")

    # None until a human has ruled. Distinct from ``human_verified`` (accepted) so that
    # "rejected" and "not yet seen" cannot collapse into the same falsy value.
    human_accepted: bool | None = None
    machine_checks: list[MachineCheck] = Field(default_factory=list)
    disclosure_order: str = "human_first"

    # Full-corpus, never the anchor paper alone. Papers cite each other's numbers in related
    # work, so "absent from paper X" does not imply "unanswerable from the corpus" — the
    # same false-absence trap that reduced the absent-topic supply from 28 to 4.
    absence_verified_against_corpus: bool = False
    absence_chunks_scanned: int = 0

    # Multi-hop only. Shared vocabulary establishes topical relatedness, not multi-hop
    # necessity; an item is only multi-hop if no single source paper answers it alone.
    single_paper_sufficiency_checked: bool = False
    answerable_by_one_paper: bool = False

    lexical_overlap_with_gold: float = -1.0
    human_verified: bool = False
    human_notes: str = ""

    @property
    def machines_all_passed(self) -> bool | None:
        return all(c.passed for c in self.machine_checks) if self.machine_checks else None

    @property
    def agrees(self) -> bool | None:
        """Whether the human and the automated pipeline reached the same verdict."""
        machine = self.machines_all_passed
        if self.human_accepted is None or machine is None:
            return None
        return self.human_accepted == machine

    def disagreeing_checks(self) -> list[MachineCheck]:
        """The specific checks a disagreement is attributable to.

        A blanket "3 disagreements" says nothing actionable. Naming the check that differed
        is what turns a disagreement into either a fixed check or a documented limit.
        """
        if self.agrees is not False or self.human_accepted is None:
            return []
        # Human accepted, some check failed -> those checks are over-strict (or wrong).
        # Human rejected, all checks passed -> every check missed something.
        if self.human_accepted:
            return [c for c in self.machine_checks if not c.passed]
        return list(self.machine_checks)


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    stratum: Stratum
    question: str
    # Empty for both unanswerable strata: the expected behaviour is a refusal.
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_answer: str = ""
    absence_shape: AbsenceShape | None = None
    anchor_paper_id: str = ""
    anchor_class: AnchorClass | None = None
    absent_term: str = ""
    provenance: Provenance
    verification: Verification = Field(default_factory=Verification)

    @model_validator(mode="after")
    def _check_stratum_invariants(self) -> Self:
        if self.stratum.expects_refusal:
            if self.gold_chunk_ids:
                raise ValueError(
                    f"{self.item_id}: an unanswerable item cannot have gold chunks — if a "
                    f"chunk answers it, it is not unanswerable"
                )
            if not self.absent_term:
                raise ValueError(f"{self.item_id}: unanswerable items must record absent_term")
        elif not self.gold_chunk_ids:
            raise ValueError(f"{self.item_id}: an answerable item needs at least one gold chunk")

        if self.stratum is Stratum.UNANSWERABLE_ATTRIBUTE:
            if not self.anchor_paper_id:
                raise ValueError(
                    f"{self.item_id}: an absent-attribute item is anchored to a real paper"
                )
            if self.absence_shape is None:
                raise ValueError(
                    f"{self.item_id}: absence_shape is required so the sub-stratum cannot "
                    f"collapse into eleven copies of one question shape"
                )
            if self.anchor_class is None:
                raise ValueError(
                    f"{self.item_id}: anchor_class is required — the cross-citation check "
                    f"is weaker for papers with no distinctive method name, and that has to "
                    f"stay visible per item rather than averaged away"
                )
        elif self.stratum is Stratum.UNANSWERABLE_TOPIC and self.anchor_paper_id:
            raise ValueError(
                f"{self.item_id}: an absent-topic item has no anchor paper — anchoring it "
                f"would make it an absent-attribute item and blur the two sub-strata"
            )

        if self.stratum is Stratum.MULTI_HOP and len(self.provenance.source_paper_ids) < 2:
            raise ValueError(f"{self.item_id}: a multi-hop item cites at least two papers")
        return self


class EvalSet(BaseModel):
    """A frozen, checksummed collection of items."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    name: str
    corpus_sha256: str = Field(description="data/CORPUS.sha256 at generation time")
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    items: list[EvalItem] = Field(default_factory=list)
    sha256: str = ""

    def payload_checksum(self) -> str:
        rows = [i.model_dump(mode="json") for i in self.items]
        return hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in Stratum}
        for item in self.items:
            counts[item.stratum.value] += 1
        return counts

    def absence_shape_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            if item.absence_shape is not None:
                counts[item.absence_shape.value] = counts.get(item.absence_shape.value, 0) + 1
        return counts

    def agreement_by_stratum(self) -> dict[str, dict[str, int]]:
        """Human-vs-machine agreement, per stratum, never pooled.

        Pooled agreement would hide exactly what matters: multi-hop and
        unanswerable-attribute rest entirely on automated checks, so their agreement rate is
        evidence about those checks. Single-paper factual agreement is close to free and
        would dilute the figure if averaged in.
        """
        report: dict[str, dict[str, int]] = {}
        for item in self.items:
            row = report.setdefault(item.stratum.value, {"ruled": 0, "agree": 0, "disagree": 0})
            agrees = item.verification.agrees
            if agrees is None:
                continue
            row["ruled"] += 1
            row["agree" if agrees else "disagree"] += 1
        return report

    def disagreements(self) -> list[tuple[str, str, list[str]]]:
        """(item_id, stratum, names of the checks the disagreement is attributable to)."""
        return [
            (
                item.item_id,
                item.stratum.value,
                [c.name for c in item.verification.disagreeing_checks()],
            )
            for item in self.items
            if item.verification.agrees is False
        ]

    def anchor_class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            if item.anchor_class is not None:
                counts[item.anchor_class.value] = counts.get(item.anchor_class.value, 0) + 1
        return counts

    def write(self, path: Path) -> None:
        self.sha256 = self.payload_checksum()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> EvalSet:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        evalset = cls.model_validate(data)
        actual = evalset.payload_checksum()
        if evalset.sha256 and actual != evalset.sha256:
            raise ValueError(
                f"{path} failed its checksum: expected {evalset.sha256[:16]}, got "
                f"{actual[:16]}. The set has been edited since it was frozen."
            )
        return evalset
