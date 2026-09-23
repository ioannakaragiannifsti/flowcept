"""Real local-LLM multi-agent incident-response simulation captured by Flowcept.

Every agent must enumerate the alternatives it considered, assess each one, and
select exactly one. Each of those choices is captured as its own Flowcept
decision record, so the UI and the post-hoc analyzers can show not only what each
agent answered but which options it weighed and why it rejected the others.

Each agent delegates its model call to ``DecisionCapture.invoke``, which builds the
prompt from the decision context, enforces the ``DecisionResponse`` contract, and records
both the invocation and the decision. The narrative an agent passes downstream is derived
from the recorded decision rather than asked for separately, so a visible answer can never
contradict the decision it is supposed to describe.
"""

import argparse
import json

from langchain_openai import ChatOpenAI

from flowcept import DecisionCapture, Flowcept, FlowceptTask
from flowcept.commons.vocabulary import PROV_AGENT
from flowcept.configs import AGENT, AGENT_API_KEY

INCIDENT = (
    "At 09:10, customers began reporting that the online checkout accepts orders but "
    "does not send confirmations. The order API is healthy, the notification queue "
    "grew from 20 to 18,000 messages, and yesterday's release changed the email worker "
    "configuration. Payment processing must not be interrupted."
)

# The incident commander's alternatives are fixed so the final outcome label is comparable
# across runs, which is what the causal analyzer ablates against.
COMMANDER_CANDIDATES = ("execute", "modify", "reject")

# Enough headroom for candidates, assessments, and the narrative answer in one JSON object.
# Too low and the object is truncated mid-string, which reads as invalid JSON.
MAX_TOKENS = 1600


def _decision_context(agent_role: str, question: str, model_name: str, fixed: tuple[str, ...] | None) -> dict:
    """Build the decision context DecisionCapture turns into the model's system prompt.

    ``DecisionCapture.invoke`` owns the prompt, so everything role-specific this MAS needs
    the model to honour has to travel inside the context it serializes.
    """
    context = {
        "question": question,
        "agent_role": f"the {agent_role} in a production incident-response team",
        "model": model_name,
        "instructions": [
            "Use only the supplied evidence and state uncertainties.",
            "Put a one-line description of each alternative in the candidate's content field.",
            "Keep every explanation to one short sentence.",
        ],
    }
    if fixed:
        context["required_candidate_ids"] = list(fixed)
        context["instructions"].append(
            f"Return exactly the candidates {list(fixed)} and nothing else, and assess every one of them."
        )
    else:
        context["instructions"].append(
            "Return at least two genuinely different candidates and assess every one of them."
        )
    return context


def _decision_request(instruction: str, evidence: dict, complaint: str | None = None) -> str:
    """Build the user message for one decision, optionally restating a rejected attempt."""
    request = f"{instruction}\n\nEVIDENCE:\n{json.dumps(evidence, indent=2, default=str)}"
    if complaint:
        request = f"{request}\n\nA previous attempt was rejected: {complaint}\nSatisfy that requirement this time."
    return request


def _describe(content) -> str:
    """Render a candidate's content as one readable line."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for field in ("summary", "description", "title", "name", "strategy"):
            value = content.get(field)
            if isinstance(value, str) and value:
                return value.strip()
    return json.dumps(content, default=str)


def _check_record(record, agent_id: str, fixed: tuple[str, ...] | None) -> None:
    """Apply this MAS's requirements on top of the contract DecisionCapture already enforced.

    These run after ``invoke`` has already published the decision task, so a rejection here
    reports a decision that is present in provenance. That is the cost of letting
    DecisionCapture own the model call; manual capture could validate before recording.
    """
    returned = {candidate.candidate_id for candidate in record.candidates}
    if fixed:
        missing = [option for option in fixed if option not in returned]
        unexpected = sorted(returned - set(fixed))
        if missing or unexpected:
            raise ValueError(
                f"{agent_id} must return exactly the candidates {list(fixed)}; "
                f"missing={missing} unexpected={unexpected}"
            )
    elif len(returned) < 2:
        raise ValueError(f"{agent_id} returned {len(returned)} candidate(s); at least 2 are required")

    if len(record.selected_candidate_ids) != 1:
        raise ValueError(
            f"{agent_id} selected {len(record.selected_candidate_ids)} candidates; this MAS requires exactly one"
        )


def _derive_answer(record, fixed: tuple[str, ...] | None) -> str:
    """Build the narrative passed downstream directly from the recorded decision.

    Deriving it means the visible answer cannot disagree with the recorded decision, and
    it keeps the explicit outcome label the causal analyzer depends on.
    """
    selected = record.selected_candidate_ids[0]
    chosen = next(candidate for candidate in record.candidates if candidate.candidate_id == selected)
    assessment = next(item for item in record.assessments if item.candidate_id == selected)
    rejected = [
        f"{item.candidate_id} ({item.score:g}): {item.explanation}"
        for item in record.assessments
        if item.candidate_id != selected
    ]
    lines = []
    if fixed:
        # The analyzer reads this label to decide whether an intervention changed the outcome.
        lines.append(f"DECISION: {selected.upper()}")
    lines.append(f"{_describe(chosen.content)} (confidence {assessment.score:g})")
    lines.append(assessment.explanation)
    if rejected:
        lines.append("Rejected alternatives: " + "; ".join(rejected))
    return "\n".join(lines)


def run_agent(
    agent_id: str,
    agent_role: str,
    instruction: str,
    evidence: dict,
    input_entity_ids: list[str],
    output_entity_id: str,
    model_name: str,
    workflow_id: str,
    decision_type: str,
    decision_question: str,
    parent_task_id: str | None = None,
    fixed_candidates: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    """Run one local-model agent, capturing its provenance and the choice it made."""
    with FlowceptTask(
        activity_id=agent_id,
        subtype=PROV_AGENT.AGENT_TOOL,
        agent_id=agent_id,
        parent_task_id=parent_task_id,
        used={
            "agent_role": agent_role,
            "evidence": evidence,
            "input_entity_ids": input_entity_ids,
        },
        capture_telemetry=False,
    ) as agent_task:
        # reasoning_effort="none" matters: qwen3 and other reasoning models otherwise emit a
        # long <think> deliberation before answering, which costs minutes per call and can
        # change the outcome. DecisionCapture.invoke wraps this model itself, so these
        # settings are NOT recorded as custom_metadata.model_parameters the way manual
        # capture recorded them, and the causal analyzer cannot read them back.
        model = ChatOpenAI(
            api_key=AGENT_API_KEY,
            base_url=AGENT["llm_server_url"],
            model=model_name,
            temperature=0,
            reasoning_effort="none",
            max_tokens=MAX_TOKENS,
        )
        context = _decision_context(agent_role, decision_question, model_name, fixed_candidates)

        # DecisionCapture drives the model, builds the prompt from the context above, and
        # records the invocation and the decision itself. A capture holds one decision, so a
        # rejected attempt needs a fresh one.
        record = None
        complaint = None
        for attempt in range(2):
            capture = DecisionCapture(
                decision_type=decision_type,
                context=context,
                agent_id=agent_id,
                workflow_id=workflow_id,
                parent_task_id=agent_task.get_id(),
                input_entity_ids=input_entity_ids,
                output_entity_ids=[output_entity_id],
                llm=model,
            )
            record = capture.invoke(_decision_request(instruction, evidence, complaint))
            try:
                _check_record(record, agent_id, fixed_candidates)
                break
            except ValueError as error:
                if attempt == 1:
                    raise
                complaint = str(error)

        selected = record.selected_candidate_ids[0]
        answer = _derive_answer(record, fixed_candidates)

        agent_task.end(
            generated={
                # `response` stays the visible narrative so the causal analyzer contract holds.
                "response": answer,
                "candidates": [candidate.to_dict() for candidate in record.candidates],
                "selected_candidate_id": selected,
                "output_entity_ids": [output_entity_id],
            }
        )
    return answer, agent_task.get_id()


def run_simulation(model_name: str = "qwen3:4b") -> tuple[str, str]:
    """Run five collaborating local-model agents on one realistic incident."""
    with Flowcept(
        workflow_name="Local LLM Incident Response MAS",
        workflow_args={"incident": INCIDENT, "model": model_name},
        start_persistence=True,
        check_safe_stops=False,
    ) as flowcept:
        workflow_id = flowcept.current_workflow_id

        monitoring_report, monitoring_task_id = run_agent(
            "monitoring-agent",
            "monitoring and impact analyst",
            "Summarize the incident, affected capability, business impact, and urgency.",
            {"incident": INCIDENT},
            ["incident:checkout-notifications"],
            "analysis:monitoring",
            model_name,
            workflow_id,
            decision_type="impact_triage",
            decision_question=(
                "Which severity and urgency classification best fits this incident? "
                "Consider at least one higher and one lower severity alternative."
            ),
        )

        investigation, investigation_task_id = run_agent(
            "investigation-agent",
            "root-cause investigator",
            (
                "Develop the most likely root-cause hypothesis and two alternatives. "
                "Cite supporting and contradicting evidence."
            ),
            {"incident": INCIDENT, "monitoring_report": monitoring_report},
            ["incident:checkout-notifications", "analysis:monitoring"],
            "analysis:investigation",
            model_name,
            workflow_id,
            decision_type="root_cause_hypothesis",
            decision_question=(
                "Which root-cause hypothesis best explains the evidence? Each competing hypothesis is a candidate."
            ),
            parent_task_id=monitoring_task_id,
        )

        response_plan, planning_task_id = run_agent(
            "response-planning-agent",
            "incident response planner",
            (
                "Propose a reversible plan that restores confirmations while preserving payment processing. "
                "Include verification and rollback steps."
            ),
            {
                "incident": INCIDENT,
                "monitoring_report": monitoring_report,
                "investigation": investigation,
            },
            ["analysis:monitoring", "analysis:investigation"],
            "plan:incident-response",
            model_name,
            workflow_id,
            decision_type="response_plan_selection",
            decision_question=(
                "Which remediation strategy should be proposed? "
                "Each distinct strategy, such as rollback, configuration fix, or queue drain, is a candidate."
            ),
            parent_task_id=investigation_task_id,
        )

        risk_review, review_task_id = run_agent(
            "risk-review-agent",
            "operational risk reviewer",
            (
                "Critique the plan. Identify customer, payment, data-loss, and rollback risks, then state the "
                "safeguards required before execution."
            ),
            {"incident": INCIDENT, "proposed_plan": response_plan},
            ["incident:checkout-notifications", "plan:incident-response"],
            "assessment:risk-review",
            model_name,
            workflow_id,
            decision_type="risk_disposition",
            decision_question=(
                "What is the risk disposition of the proposed plan? "
                "Candidates are the dispositions you considered, such as acceptable as written, "
                "acceptable with safeguards, or unacceptable."
            ),
            parent_task_id=planning_task_id,
        )

        final_decision, _ = run_agent(
            "incident-commander-agent",
            "incident commander",
            (
                "Make the final operational decision. The answer must begin with exactly DECISION: EXECUTE, "
                "DECISION: MODIFY, or DECISION: REJECT. Then list approved steps and explain how the evidence "
                "and risk review support it."
            ),
            {
                "incident": INCIDENT,
                "investigation": investigation,
                "proposed_plan": response_plan,
                "risk_review": risk_review,
            },
            [
                "incident:checkout-notifications",
                "analysis:investigation",
                "plan:incident-response",
                "assessment:risk-review",
            ],
            "decision:incident-commander",
            model_name,
            workflow_id,
            decision_type="final_operational_decision",
            decision_question="Should the proposed plan be executed, modified, or rejected?",
            parent_task_id=review_task_id,
            fixed_candidates=COMMANDER_CANDIDATES,
        )

    return workflow_id, final_decision


def main() -> None:
    """Run the local-model MAS from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b", help="Model installed in Ollama.")
    args = parser.parse_args()

    workflow_id, final_decision = run_simulation(args.model)
    print("\nFinal incident-commander decision:\n")
    print(final_decision)
    print(f"\nWorkflow ID: {workflow_id}")


if __name__ == "__main__":
    main()
