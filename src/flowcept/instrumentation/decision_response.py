"""Validated LLM output contract, separate from runtime provenance metadata."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flowcept.commons.flowcept_dataclasses.retrieval_provenance import DISCARDING_ROLES, EVIDENCE_ROLES


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _CandidateResponse(_ResponseModel):
    candidate_id: str = Field(min_length=1)
    content: Any


class _AssessmentResponse(_ResponseModel):
    candidate_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    criteria: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=1)


class _EvidenceUseResponse(_ResponseModel):
    item_id: str = Field(min_length=1)
    used: bool
    role: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    # Which alternative this item bears on. Optional because a discarded item often bears
    # on none, but when present it turns the record into a per-alternative account: this
    # evidence is why that candidate was considered, and why it lost.
    candidate_id: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        """Keep roles inside the recorded vocabulary so they stay queryable."""
        if value not in EVIDENCE_ROLES:
            raise ValueError(f"Unsupported evidence role: {value}; expected one of {sorted(EVIDENCE_ROLES)}")
        return value


class DecisionResponse(_ResponseModel):
    """Model-authored fields; identifiers and attribution come from the runtime."""

    candidates: list[_CandidateResponse] = Field(min_length=1)
    assessments: list[_AssessmentResponse] = Field(min_length=1)
    selected_candidate_ids: list[str] = Field(min_length=1)

    @classmethod
    def response_format(cls) -> dict:
        """Build the provider's structured-output request from the validation schema."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "decision_response",
                "strict": True,
                "schema": cls.model_json_schema(),
            },
        }

    @classmethod
    def build_prompt(cls, decision_type: str, context: str | dict, retrieved: list[dict] | None = None) -> str:
        """Build domain-neutral instructions from the validated response contract."""
        return (
            "Perform the requested decision by explicitly generating alternatives, assessing every alternative, "
            "and selecting one or more candidates. Report only alternatives generated for this request, "
            "not a reconstruction of hidden internal reasoning. Scores are model-reported confidence in [0, 1], "
            "not calibrated probabilities. Give concise assessment explanations and criteria. "
            "Return only one JSON object conforming to the supplied schema, without Markdown fences. "
            "Candidate identifiers must be unique; assessments and selections must reference those identifiers.\n"
            f"Decision type: {json.dumps(decision_type)}\n"
            f"Context: {json.dumps(context)}\n"
            f"JSON schema: {json.dumps(DecisionResponse.model_json_schema())}"
        )

    @model_validator(mode="after")
    def validate_references(self):
        """Require unique candidates, valid selections, and an assessment for each candidate."""
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        known = set(candidate_ids)
        if len(known) != len(candidate_ids):
            raise ValueError("Candidate IDs must be unique")
        if len(set(self.selected_candidate_ids)) != len(self.selected_candidate_ids):
            raise ValueError("Selected candidate IDs must be unique")
        if not set(self.selected_candidate_ids) <= known:
            raise ValueError("Selected candidate IDs must reference known candidates")
        if {assessment.candidate_id for assessment in self.assessments} != known:
            raise ValueError("Assessments must cover all candidates and reference only known candidates")
        return self


class GroundedDecisionResponse(DecisionResponse):
    """A decision the model had to ground in named retrieved items.

    The extra ``evidence_uses`` field is what turns a retrieval into provenance: the
    model must say, item by item, whether it kept what the tool returned and why, so a
    dropped search hit is recorded rather than silently absent from the answer.
    """

    evidence_uses: list[_EvidenceUseResponse] = Field(min_length=1)

    @classmethod
    def build_prompt(cls, decision_type: str, context: str | dict, retrieved: list[dict] | None = None) -> str:
        """Extend the base instructions with the retrieved items the model must judge."""
        base = DecisionResponse.build_prompt.__func__(cls, decision_type, context)
        items = retrieved or []
        return "\n".join(
            [
                base,
                (
                    "The evidence below was returned by tools for this request. Judge every item: for each "
                    "item_id, report in evidence_uses whether you used it for your decision (used) and why "
                    f"(role, one of {sorted(EVIDENCE_ROLES)}, plus a one-sentence explanation). Use "
                    f"used=false for any item whose role is one of {sorted(DISCARDING_ROLES)}. Report every "
                    "item_id exactly once, including the ones you discarded, and do not invent item_ids. "
                    "Where an item bears on one particular alternative, name it in candidate_id, so the "
                    "record shows which evidence supported the alternative you chose and which supported "
                    "the ones you rejected. "
                    "Base your candidates and assessments only on the items you marked as used."
                ),
                f"Retrieved evidence: {json.dumps(items, default=str)}",
            ]
        )

    @model_validator(mode="after")
    def validate_evidence(self):
        """Require one coherent judgement per reported item and no duplicate items."""
        item_ids = [use.item_id for use in self.evidence_uses]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Evidence use item IDs must be unique")

        # "I used it" and "it was irrelevant" cannot both be true. Recording the pair would
        # make kept_item_ids overcount and the stored reason contradict the stored verdict,
        # so the response is rejected and the caller can retry instead.
        contradictions = [
            use.item_id for use in self.evidence_uses if use.used and use.role in DISCARDING_ROLES
        ]
        if contradictions:
            raise ValueError(
                f"Items marked used cannot have a discarding role {sorted(DISCARDING_ROLES)}: {contradictions}. "
                "Set used=false, or give a role explaining how the item supported the decision."
            )

        known = {candidate.candidate_id for candidate in self.candidates}
        unknown = sorted({use.candidate_id for use in self.evidence_uses if use.candidate_id} - known)
        if unknown:
            raise ValueError(f"Evidence use candidate IDs must reference candidates you returned: {unknown}")
        return self
