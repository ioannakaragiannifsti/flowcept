"""Domain-agnostic decision provenance data models."""

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any
from uuid import uuid4

from flowcept.commons.flowcept_dataclasses.retrieval_provenance import (
    EvidenceUse,
    Retrieval,
)


@dataclass
class Candidate:
    """An explicit alternative considered during a decision."""

    candidate_id: str
    status: str
    content: Any = None
    content_ref: str | None = None
    origin_type: str = "explicit_generation"
    rank: int | None = None

    def __post_init__(self):
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if self.status not in {"proposed", "selected", "rejected", "filtered"}:
            raise ValueError(f"Unsupported candidate status: {self.status}")
        if self.content is not None and self.content_ref is not None:
            raise ValueError("Use either content or content_ref, not both")

    def to_dict(self) -> dict:
        """Serialize non-null candidate fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class Assessment:
    """An evaluation of one candidate by a model, agent, rule, or human."""

    candidate_id: str
    evaluator_id: str
    score_type: str
    score: float | None = None
    explanation: str | None = None
    criteria: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    assessment_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict:
        """Serialize non-null assessment fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class DecisionRecord:
    """A generic provenance record describing a decision among alternatives."""

    decision_type: str
    context: str | dict
    candidates: list[Candidate]
    selected_candidate_ids: list[str]
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    assessments: list[Assessment] = field(default_factory=list)
    input_entity_ids: list[str] = field(default_factory=list)
    output_entity_ids: list[str] = field(default_factory=list)
    retrievals: list[Retrieval] = field(default_factory=list)
    evidence_uses: list[EvidenceUse] = field(default_factory=list)
    timestamp: float = field(default_factory=time)
    schema_version: str = "0.1.0"

    def __post_init__(self):
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate IDs must be unique")

        unknown_selected = set(self.selected_candidate_ids) - set(candidate_ids)
        if unknown_selected:
            raise ValueError(f"Selected candidate IDs do not exist: {sorted(unknown_selected)}")

        assessed_ids = {assessment.candidate_id for assessment in self.assessments}
        unknown_assessed = assessed_ids - set(candidate_ids)
        if unknown_assessed:
            raise ValueError(f"Assessment candidate IDs do not exist: {sorted(unknown_assessed)}")

        retrieval_ids = [retrieval.retrieval_id for retrieval in self.retrievals]
        if len(retrieval_ids) != len(set(retrieval_ids)):
            raise ValueError("Retrieval IDs must be unique")

        # Evidence use is only meaningful against items that were actually retrieved, so a
        # reference to an unknown item means the grounding trail is broken, not merely odd.
        retrieved_ids = {item_id for retrieval in self.retrievals for item_id in retrieval.item_ids}
        unknown_evidence = {use.item_id for use in self.evidence_uses} - retrieved_ids
        if unknown_evidence:
            raise ValueError(f"Evidence use references unretrieved items: {sorted(unknown_evidence)}")

        unknown_evidence_candidates = {
            use.candidate_id for use in self.evidence_uses if use.candidate_id is not None
        } - set(candidate_ids)
        if unknown_evidence_candidates:
            raise ValueError(f"Evidence use candidate IDs do not exist: {sorted(unknown_evidence_candidates)}")

    def to_dict(self) -> dict:
        """Serialize the complete decision provenance record."""
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "context": self.context,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "assessments": [assessment.to_dict() for assessment in self.assessments],
            "selected_candidate_ids": self.selected_candidate_ids,
            "input_entity_ids": self.input_entity_ids,
            "output_entity_ids": self.output_entity_ids,
            "retrievals": [retrieval.to_dict() for retrieval in self.retrievals],
            "evidence_uses": [use.to_dict() for use in self.evidence_uses],
            "timestamp": self.timestamp,
        }

    def grounding_summary(self) -> dict:
        """Summarize what was retrieved against what the decision kept.

        The counts make the common question answerable without walking the record:
        how much evidence the tools returned, how much survived into the decision,
        and which items were dropped.
        """
        retrieved_ids = [item_id for retrieval in self.retrievals for item_id in retrieval.item_ids]
        kept = [use.item_id for use in self.evidence_uses if use.used]
        dropped = [use.item_id for use in self.evidence_uses if not use.used]
        return {
            "tools_used": sorted({retrieval.tool_name for retrieval in self.retrievals}),
            "retrieved_count": len(retrieved_ids),
            "kept_item_ids": kept,
            "dropped_item_ids": dropped,
            "unreported_item_ids": [
                item_id for item_id in retrieved_ids if item_id not in {use.item_id for use in self.evidence_uses}
            ],
        }
