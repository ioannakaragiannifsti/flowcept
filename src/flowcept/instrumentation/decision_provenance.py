"""Capture generic decision provenance through Flowcept tasks."""

from uuid import uuid4

from flowcept.commons.flowcept_dataclasses.decision_provenance import (
    Assessment,
    Candidate,
    DecisionRecord,
)
from flowcept.commons.vocabulary import PROV_AGENT
from flowcept.instrumentation.task_capture import FlowceptTask


def record_decision(
    decision: DecisionRecord,
    agent_id: str | None = None,
    workflow_id: str | None = None,
    parent_task_id: str | None = None,
) -> str:
    """Record a completed decision and return its Flowcept task identifier."""
    task = FlowceptTask(
        activity_id="decision",
        subtype=PROV_AGENT.DECISION,
        agent_id=agent_id,
        workflow_id=workflow_id,
        parent_task_id=parent_task_id,
        used={
            "context": decision.context,
            "input_entity_ids": decision.input_entity_ids,
        },
        custom_metadata={
            "decision_id": decision.decision_id,
            "decision_type": decision.decision_type,
            "schema_version": decision.schema_version,
        },
        capture_telemetry=False,
    )

    task.end(
        generated={
            "decision": decision.to_dict(),
            "output_entity_ids": decision.output_entity_ids,
        }
    )
    return task.get_id()


class DecisionCapture:
    """Build and record a decision using a context manager."""

    def __init__(
        self,
        decision_type: str,
        context: str | dict,
        agent_id: str | None = None,
        workflow_id: str | None = None,
        parent_task_id: str | None = None,
        input_entity_ids: list[str] | None = None,
        output_entity_ids: list[str] | None = None,
        llm=None,
    ):
        self.decision_type = decision_type
        self.context = context
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.parent_task_id = parent_task_id
        self.input_entity_ids = input_entity_ids or []
        self.output_entity_ids = output_entity_ids or []
        self.candidates = []
        self.assessments = []
        self.selected_candidate_ids = []
        self.record = None
        self.task_id = None
        self.llm = llm
        self._automatic = llm is not None

    def __enter__(self):
        return self

    def invoke(self, request: str, **kwargs) -> DecisionRecord:
        """Invoke an attached, unwrapped LangChain model and record one validated decision.

        The request is a user message. The model must support the OpenAI-compatible
        JSON-schema response_format parameter, which capture supplies automatically.
        Other keyword arguments are forwarded to the model. Use a fresh capture per
        decision; tool-bound models are not supported.
        Requires the ``llm_agent`` extra and an active ``Flowcept`` context.
        """
        if self.llm is None:
            raise ValueError("Attach an llm to DecisionCapture before invoking it")
        self._check_automatic_capture()
        # Keep optional LLM dependencies out of manual capture and package imports.
        from flowcept.instrumentation.decision_response import DecisionResponse
        from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM

        self._automatic = True
        invocation_task_id = str(uuid4())
        wrapped = FlowceptLLM(
            self.llm,
            agent_id=self.agent_id,
            workflow_id=self.workflow_id,
            parent_task_id=self.parent_task_id,
            task_id=invocation_task_id,
        )
        response = wrapped.invoke(
            [
                {"role": "system", "content": DecisionResponse.build_prompt(self.decision_type, self.context)},
                {"role": "user", "content": request},
            ],
            response_format=DecisionResponse.response_format(),
            **kwargs,
        )
        return self.record_response(response, invocation_task_id=invocation_task_id)

    def _check_automatic_capture(self):
        if self.record is not None or self.candidates or self.assessments or self.selected_candidate_ids:
            raise ValueError("DecisionCapture already contains a decision; use a fresh capture")
        if not self.agent_id:
            raise ValueError("Automatic decision capture requires an agent_id for evaluator attribution")

    def record_response(self, response: str, *, invocation_task_id: str) -> DecisionRecord:
        """Validate decision JSON and record it as a child of its generating LLM task.

        Supports callers that already captured the LLM invocation themselves. Runtime
        identifiers and evaluator attribution are supplied by this capture, never the model.
        """
        self._check_automatic_capture()
        # Pydantic is provided by the optional LLM dependencies.
        from flowcept.instrumentation.decision_response import DecisionResponse

        self._automatic = True
        output = DecisionResponse.model_validate_json(response)
        for candidate in output.candidates:
            self.add_candidate(candidate.candidate_id, content=candidate.content)
        for assessment in output.assessments:
            self.assess(
                **assessment.model_dump(),
                evaluator_id=self.agent_id,
                score_type="model_reported_confidence",
            )
        self.select(*output.selected_candidate_ids)
        self._finish(parent_task_id=invocation_task_id)
        return self.record

    def add_candidate(
        self,
        candidate_id: str,
        content=None,
        content_ref: str | None = None,
        origin_type: str = "explicit_generation",
        rank: int | None = None,
    ) -> Candidate:
        """Add an explicit candidate alternative."""
        candidate = Candidate(
            candidate_id=candidate_id,
            content=content,
            content_ref=content_ref,
            status="proposed",
            origin_type=origin_type,
            rank=rank,
        )
        self.candidates.append(candidate)
        return candidate

    def assess(
        self,
        candidate_id: str,
        evaluator_id: str,
        score_type: str,
        score: float | None = None,
        explanation: str | None = None,
        criteria: list[str] | None = None,
        evidence_ids: list[str] | None = None,
    ) -> Assessment:
        """Add an assessment for a candidate."""
        assessment = Assessment(
            candidate_id=candidate_id,
            evaluator_id=evaluator_id,
            score_type=score_type,
            score=score,
            explanation=explanation,
            criteria=criteria or [],
            evidence_ids=evidence_ids or [],
        )
        self.assessments.append(assessment)
        return assessment

    def select(self, *candidate_ids: str):
        """Select one or more candidates."""
        known_ids = {candidate.candidate_id for candidate in self.candidates}
        unknown_ids = set(candidate_ids) - known_ids
        if unknown_ids:
            raise ValueError(f"Selected candidate IDs do not exist: {sorted(unknown_ids)}")

        self.selected_candidate_ids = list(candidate_ids)
        for candidate in self.candidates:
            candidate.status = "selected" if candidate.candidate_id in candidate_ids else "rejected"

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None or self._automatic:
            return False

        self._finish(parent_task_id=self.parent_task_id)
        return False

    def _finish(self, parent_task_id):
        self.record = DecisionRecord(
            decision_type=self.decision_type,
            context=self.context,
            candidates=self.candidates,
            assessments=self.assessments,
            selected_candidate_ids=self.selected_candidate_ids,
            input_entity_ids=self.input_entity_ids,
            output_entity_ids=self.output_entity_ids,
        )
        self.task_id = record_decision(
            decision=self.record,
            agent_id=self.agent_id,
            workflow_id=self.workflow_id,
            parent_task_id=parent_task_id,
        )
