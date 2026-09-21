"""Domain-agnostic retrieval provenance data models.

A retrieval is what an agent's tool brought back before it decided anything: the tool
that ran, the query it ran, and every item it returned. Those items are the "ground
truth" the decision is measured against, so they are recorded in full and separately
from whatever the model later chose to keep. :class:`EvidenceUse` records that keeping:
one entry per retrieved item saying whether the decision used it and why.
"""

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any
from uuid import uuid4

TOOL_TYPES = {
    "web_search",
    "database",
    "vector_store",
    "file_system",
    "api",
    "code_execution",
    "other",
}

EVIDENCE_ROLES = {"supporting", "contradicting", "irrelevant", "redundant", "unreliable"}

# Roles that describe evidence being set aside. An item cannot be both used for a decision
# and dismissed as irrelevant, redundant, or unreliable; "contradicting" is not here,
# because evidence that argues against the chosen option is still evidence that was used.
DISCARDING_ROLES = frozenset({"irrelevant", "redundant", "unreliable"})


@dataclass
class RetrievedItem:
    """One item a tool returned, recorded as retrieved regardless of later use."""

    item_id: str
    content: Any = None
    source: str | None = None
    rank: int | None = None
    score: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.item_id:
            raise ValueError("item_id is required")

    def to_dict(self) -> dict:
        """Serialize non-null retrieved-item fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class Retrieval:
    """One tool invocation: the tool, the query it ran, and everything it returned."""

    tool_name: str
    query: Any
    tool_type: str = "other"
    items: list[RetrievedItem] = field(default_factory=list)
    tool_args: dict = field(default_factory=dict)
    # Set when result-size limits dropped items or shortened their content. Recording it
    # keeps "everything the tool returned" honest: a capped result set says so explicitly
    # rather than looking like a complete one.
    truncation: dict = field(default_factory=dict)
    retrieval_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time)
    schema_version: str = "0.1.0"

    def __post_init__(self):
        if not self.tool_name:
            raise ValueError("tool_name is required")
        if self.tool_type not in TOOL_TYPES:
            raise ValueError(f"Unsupported tool_type: {self.tool_type}; expected one of {sorted(TOOL_TYPES)}")
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Retrieved item IDs must be unique within a retrieval")

    @property
    def item_ids(self) -> list[str]:
        """Identifiers of every item this retrieval returned."""
        return [item.item_id for item in self.items]

    def to_dict(self) -> dict:
        """Serialize the complete retrieval record, including every retrieved item."""
        return {
            "schema_version": self.schema_version,
            "retrieval_id": self.retrieval_id,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type,
            "query": self.query,
            "tool_args": self.tool_args,
            "retrieved": [item.to_dict() for item in self.items],
            "retrieved_count": len(self.items),
            "truncation": self.truncation,
            "timestamp": self.timestamp,
        }


@dataclass
class EvidenceUse:
    """Whether one retrieved item was kept for the decision, and why.

    The set of ``EvidenceUse`` entries is the audit trail between what a tool returned
    and what the model actually reasoned over, which is what makes dropped evidence
    visible instead of silent.
    """

    item_id: str
    used: bool
    retrieval_id: str | None = None
    role: str | None = None
    explanation: str | None = None
    candidate_id: str | None = None

    def __post_init__(self):
        if not self.item_id:
            raise ValueError("item_id is required")
        if self.role is not None and self.role not in EVIDENCE_ROLES:
            raise ValueError(f"Unsupported evidence role: {self.role}; expected one of {sorted(EVIDENCE_ROLES)}")

    def to_dict(self) -> dict:
        """Serialize non-null evidence-use fields."""
        return {key: value for key, value in asdict(self).items() if value is not None}
