"""Validated LLM output contract, separate from runtime provenance metadata."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class DecisionResponse(_ResponseModel):
    """Model-authored fields; identifiers and attribution come from the runtime."""

    candidates: list[_CandidateResponse] = Field(min_length=1)
    assessments: list[_AssessmentResponse] = Field(min_length=1)
    selected_candidate_ids: list[str] = Field(min_length=1)

    @staticmethod
    def response_format() -> dict:
        """Build the provider's structured-output request from the validation schema."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "decision_response",
                "strict": True,
                "schema": DecisionResponse.model_json_schema(),
            },
        }

    @staticmethod
    def build_prompt(decision_type: str, context: str | dict) -> str:
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
