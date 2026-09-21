"""Capture generic decision provenance through Flowcept tasks."""

import json
from uuid import uuid4

from flowcept.commons.flowcept_dataclasses.decision_provenance import (
    Assessment,
    Candidate,
    DecisionRecord,
)
from flowcept.commons.flowcept_dataclasses.retrieval_provenance import (
    EvidenceUse,
    Retrieval,
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
            # The retrievals this decision consumed, so the decision task points back at the
            # agent_tool tasks that produced its evidence.
            "retrieval_ids": [retrieval.retrieval_id for retrieval in decision.retrievals],
            "queries": [retrieval.query for retrieval in decision.retrievals],
        },
        custom_metadata={
            "decision_id": decision.decision_id,
            "decision_type": decision.decision_type,
            "schema_version": decision.schema_version,
            "grounding": decision.grounding_summary(),
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
        retrievals: list[Retrieval] | None = None,
        prompt_item_chars: int = 800,
        prompt_max_items: int = 0,
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
        # No retrievals argument means "whatever this agent's tools just retrieved", so a
        # MAS gets grounded decisions without threading retrievals through its call stack.
        # An explicit empty list still means "no evidence", which is how an agent that uses
        # no tools keeps the plain decision contract.
        from flowcept.instrumentation.tool_provenance import current_retrievals

        self.retrievals = list(retrievals) if retrievals is not None else current_retrievals()
        self.evidence_uses = []
        # These budget only the copy of the evidence sent to the model. The recorded
        # retrievals keep whatever the tools returned, in full: a context window is a hard
        # limit on what a model can read, not a reason to store less than was retrieved.
        self.prompt_item_chars = prompt_item_chars
        self.prompt_max_items = prompt_max_items
        self.record = None
        self.task_id = None
        self.llm = llm
        self._automatic = llm is not None

    def __enter__(self):
        return self

    def add_retrieval(self, retrieval: Retrieval) -> Retrieval:
        """Attach a tool retrieval whose items this decision must be grounded in.

        Attach it before invoking: ``invoke`` puts the retrieved items in the prompt and
        switches to the grounded contract, which makes reporting their use mandatory.
        """
        if self.record is not None:
            raise ValueError("DecisionCapture already recorded a decision; use a fresh capture")
        self.retrievals.append(retrieval)
        return retrieval

    def use_evidence(
        self,
        item_id: str,
        used: bool,
        role: str | None = None,
        explanation: str | None = None,
        candidate_id: str | None = None,
    ) -> EvidenceUse:
        """Record whether one retrieved item was kept for this decision, and why."""
        evidence_use = EvidenceUse(
            item_id=item_id,
            used=used,
            retrieval_id=self._retrieval_id_for(item_id),
            role=role,
            explanation=explanation,
            candidate_id=candidate_id,
        )
        self.evidence_uses.append(evidence_use)
        return evidence_use

    def _retrieval_id_for(self, item_id: str) -> str | None:
        """Find which retrieval returned an item, so evidence points back at its tool call."""
        for retrieval in self.retrievals:
            if item_id in retrieval.item_ids:
                return retrieval.retrieval_id
        return None

    def _retrieved_for_prompt(self) -> list[dict]:
        """Render retrieved items for the prompt, tagged with the tool that returned them.

        Content is shortened to ``prompt_item_chars`` so a large result set cannot overrun
        the model's context window, which providers truncate silently and which would leave
        the model reading a cut-off schema. Only this copy is shortened; the stored
        retrievals are untouched, and every ``item_id`` is still present, so the verdicts
        the model returns still cover the whole result set.
        """
        rendered = []
        for retrieval in self.retrievals:
            for item in retrieval.items:
                content = item.content
                if self.prompt_item_chars:
                    try:
                        text = content if isinstance(content, str) else json.dumps(content, default=str)
                    except Exception:  # noqa: BLE001 - an unrenderable payload still needs an entry
                        text = str(content)
                    if len(text) > self.prompt_item_chars:
                        content = text[: self.prompt_item_chars]
                        rendered.append(
                            {
                                "item_id": item.item_id,
                                "tool_name": retrieval.tool_name,
                                "query": retrieval.query,
                                "source": item.source,
                                "content": content,
                                # Told explicitly, so the model treats a partial payload as
                                # partial rather than as the whole document.
                                "content_is_truncated_preview": True,
                                "full_content_chars": len(text),
                            }
                        )
                        continue
                rendered.append(
                    {
                        "item_id": item.item_id,
                        "tool_name": retrieval.tool_name,
                        "query": retrieval.query,
                        "source": item.source,
                        "content": content,
                    }
                )

        if self.prompt_max_items and len(rendered) > self.prompt_max_items:
            # Dropping items entirely means the model cannot report on them, so the
            # decision would be incomplete; keep this off unless a caller opts in.
            rendered = rendered[: self.prompt_max_items]
        return rendered

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
        from flowcept.instrumentation.decision_response import DecisionResponse, GroundedDecisionResponse
        from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM

        # With retrievals attached, the model is held to the grounded contract, which adds
        # a mandatory per-item verdict on the evidence it was given.
        response_model = GroundedDecisionResponse if self.retrievals else DecisionResponse
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
                {
                    "role": "system",
                    "content": response_model.build_prompt(
                        self.decision_type, self.context, self._retrieved_for_prompt()
                    ),
                },
                {"role": "user", "content": request},
            ],
            response_format=response_model.response_format(),
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
        from flowcept.instrumentation.decision_response import DecisionResponse, GroundedDecisionResponse

        self._automatic = True
        response_model = GroundedDecisionResponse if self.retrievals else DecisionResponse
        output = response_model.model_validate_json(response)
        for evidence_use in getattr(output, "evidence_uses", []):
            # An item_id the model invented has no retrieval behind it, so it would make the
            # grounding trail untrue; DecisionRecord rejects it rather than storing it.
            self.use_evidence(**evidence_use.model_dump())
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
            retrievals=self.retrievals,
            evidence_uses=self.evidence_uses,
        )
        self.task_id = record_decision(
            decision=self.record,
            agent_id=self.agent_id,
            workflow_id=self.workflow_id,
            parent_task_id=parent_task_id,
        )
