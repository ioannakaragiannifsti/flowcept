"""Real local-LLM multi-agent incident-response simulation captured by Flowcept.

Every agent must enumerate the alternatives it considered, assess each one, and
select exactly one. Each of those choices is captured as its own Flowcept
decision record, so the UI and the post-hoc analyzers can show not only what each
agent answered but which options it weighed and why it rejected the others.
"""

import argparse
import json
import re

from langchain_openai import ChatOpenAI

from flowcept import DecisionCapture, Flowcept, FlowceptTask
from flowcept.commons.vocabulary import PROV_AGENT
from flowcept.configs import AGENT, AGENT_API_KEY
from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM

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

DECISION_SCHEMA = {
    "candidates": [
        {
            "candidate_id": "short_snake_case_id",
            "summary": "one line naming the alternative",
            "rationale": "why this alternative is plausible",
        }
    ],
    "assessments": [
        {
            "candidate_id": "short_snake_case_id",
            "score": "number between 0 and 1",
            "explanation": "why this alternative scored this way, citing the evidence",
        }
    ],
    "selected_candidate_id": "short_snake_case_id",
    "answer": "the narrative response to the task",
}


def _slug(value: object, fallback: str) -> str:
    """Normalize an arbitrary model-supplied identifier into a stable candidate id."""
    text = str(value).strip().lower() if value is not None else ""
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or fallback


def _build_prompt(agent_role: str, instruction: str, evidence: dict, question: str, fixed: tuple[str, ...] | None):
    """Build a role prompt that requires explicit alternatives, assessments, and one selection."""
    if fixed:
        candidate_rule = (
            f"You MUST return exactly these {len(fixed)} candidates, with candidate_id values "
            f"{list(fixed)} and nothing else, and assess every one of them."
        )
    else:
        candidate_rule = (
            "You MUST return at least two genuinely different candidates and assess every one of them. "
            "Never return a single candidate."
        )
    return [
        {
            "role": "system",
            "content": (
                f"You are the {agent_role} in a production incident-response team. "
                "You do not answer questions directly. You make a decision among explicit alternatives "
                "and report that decision as JSON.\n\n"
                "Your entire reply MUST be one JSON object with EXACTLY these four top-level keys and no "
                'others: "candidates", "assessments", "selected_candidate_id", "answer". '
                "Do not add top-level keys of your own, and do not describe the incident at the top level. "
                'Your prose belongs inside the "answer" string.\n\n'
                f"{candidate_rule}\n\n"
                'Use only the supplied evidence, state uncertainties, keep "answer" under 180 words, and give '
                "concise decision reasons rather than hidden chain-of-thought. "
                'Keep every "rationale" and "explanation" to one short sentence so the JSON stays complete.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"DECISION TO MAKE:\n{question}\n\n"
                f"REQUIRED JSON SHAPE (copy these key names exactly):\n{json.dumps(DECISION_SCHEMA, indent=2)}\n\n"
                f"EVIDENCE:\n{json.dumps(evidence, indent=2, default=str)}\n\n"
                f'What the "answer" field must contain:\n{instruction}\n\n'
                'Reply now with the JSON object whose top-level keys are exactly "candidates", '
                '"assessments", "selected_candidate_id", and "answer".'
            ),
        },
    ]


def _repair_message(error: str, fixed: tuple[str, ...] | None) -> dict:
    """Build a corrective turn asking the model to restate its choice in the required shape."""
    allowed = f" The candidate_id values must be exactly {list(fixed)}." if fixed else ""
    return {
        "role": "user",
        "content": (
            f"That reply was rejected: {error}\n\n"
            'Reply again with ONLY a JSON object whose top-level keys are exactly "candidates", '
            '"assessments", "selected_candidate_id", and "answer". "candidates" must be a JSON array of '
            'objects, each with "candidate_id", "summary", and "rationale". "assessments" must be a JSON '
            'array of objects, each with "candidate_id", "score", and "explanation". '
            '"selected_candidate_id" must equal one of the candidate_id values you listed.' + allowed
        ),
    }


def _parse_decision_payload(response: str, agent_id: str, fixed: tuple[str, ...] | None) -> dict:
    """Parse and normalize a model decision payload, tolerating small-model quirks."""
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"{agent_id} did not return valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise TypeError(f"{agent_id} returned {type(parsed).__name__}, expected a JSON object")

    # Small models sometimes emit a single candidate object instead of a list.
    raw_candidates = parsed.get("candidates")
    if isinstance(raw_candidates, dict):
        raw_candidates = [raw_candidates]
    elif not isinstance(raw_candidates, list):
        raw_candidates = []

    candidates = []
    seen = set()
    for index, item in enumerate(raw_candidates, start=1):
        if not isinstance(item, dict):
            item = {"summary": str(item)}
        candidate_id = _slug(item.get("candidate_id") or item.get("id") or item.get("summary"), f"candidate_{index}")
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidates.append({**item, "candidate_id": candidate_id})

    if fixed:
        # Keep only the agreed vocabulary, and add back any option the model omitted so the
        # rejected alternatives are still recorded rather than silently disappearing.
        by_id = {candidate["candidate_id"]: candidate for candidate in candidates if candidate["candidate_id"] in fixed}
        candidates = [
            by_id.get(option, {"candidate_id": option, "summary": option, "rationale": "not elaborated by the agent"})
            for option in fixed
        ]

    if len(candidates) < 2:
        raise ValueError(f"{agent_id} returned {len(candidates)} candidate(s); at least 2 are required")

    known = {candidate["candidate_id"] for candidate in candidates}

    raw_assessments = parsed.get("assessments")
    if isinstance(raw_assessments, dict):
        raw_assessments = [raw_assessments]
    elif not isinstance(raw_assessments, list):
        raw_assessments = []

    assessments = []
    for item in raw_assessments:
        if not isinstance(item, dict):
            continue
        candidate_id = _slug(item.get("candidate_id") or item.get("id"), "")
        if candidate_id not in known:
            continue  # An assessment of an unknown candidate would be rejected by DecisionRecord.
        try:
            score = float(item["score"]) if item.get("score") is not None else None
        except (TypeError, ValueError):
            score = None
        assessments.append(
            {
                "candidate_id": candidate_id,
                "score": score,
                "explanation": str(item.get("explanation") or "").strip() or None,
            }
        )

    selected = _slug(parsed.get("selected_candidate_id") or parsed.get("selected") or parsed.get("decision"), "")
    if selected not in known:
        raise ValueError(
            f"{agent_id} selected {selected!r}, which is not one of the candidates it proposed: {sorted(known)}"
        )

    answer = str(parsed.get("answer") or "").strip()
    if not answer:
        answer = next(
            (str(c.get("summary") or c["candidate_id"]) for c in candidates if c["candidate_id"] == selected),
            selected,
        )
    if fixed:
        # The causal analyzer keys on this explicit outcome label in the visible answer.
        label = f"DECISION: {selected.upper()}"
        if not answer.upper().startswith("DECISION:"):
            answer = f"{label}\n{answer}"

    return {"candidates": candidates, "assessments": assessments, "selected": selected, "answer": answer}


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
        model = ChatOpenAI(
            api_key=AGENT_API_KEY,
            base_url=AGENT["llm_server_url"],
            model=model_name,
            temperature=0,
            reasoning_effort="none",
            max_tokens=MAX_TOKENS,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        llm = FlowceptLLM(
            model,
            agent_id=agent_id,
            parent_task_id=agent_task.get_id(),
            workflow_id=workflow_id,
        )
        llm.metadata.update(
            {
                "provider": "ollama",
                "model_name": model_name,
                "model_parameters": {
                    "temperature": 0,
                    "reasoning_effort": "none",
                    "max_tokens": MAX_TOKENS,
                    "response_format": {"type": "json_object"},
                },
                "agent_role": agent_role,
            }
        )
        prompt = _build_prompt(agent_role, instruction, evidence, decision_question, fixed_candidates)
        raw_response = llm.invoke(prompt)
        try:
            payload = _parse_decision_payload(raw_response, agent_id, fixed_candidates)
        except (ValueError, TypeError) as error:
            # Small models often miss the shape on the first pass. Ask once more with the
            # specific complaint; the repair call is captured too, so the retry stays visible.
            repair_prompt = [
                *prompt,
                {"role": "assistant", "content": raw_response},
                _repair_message(str(error), fixed_candidates),
            ]
            raw_response = llm.invoke(repair_prompt)
            payload = _parse_decision_payload(raw_response, agent_id, fixed_candidates)
        answer = payload["answer"]

        agent_task.end(
            generated={
                # `response` stays the visible narrative so the causal analyzer contract holds.
                "response": answer,
                "candidates": payload["candidates"],
                "selected_candidate_id": payload["selected"],
                "output_entity_ids": [output_entity_id],
            }
        )
    agent_task_id = agent_task.get_id()

    with DecisionCapture(
        decision_type=decision_type,
        context={
            "question": decision_question,
            "agent_role": agent_role,
            "instruction": instruction,
            "model": model_name,
        },
        agent_id=agent_id,
        workflow_id=workflow_id,
        parent_task_id=agent_task_id,
        input_entity_ids=input_entity_ids,
        output_entity_ids=[output_entity_id],
    ) as decision:
        for rank, candidate in enumerate(payload["candidates"], start=1):
            decision.add_candidate(
                candidate["candidate_id"],
                content=candidate,
                origin_type="model_generation",
                rank=rank,
            )
        for assessment in payload["assessments"]:
            decision.assess(
                assessment["candidate_id"],
                evaluator_id=agent_id,
                score_type="model_self_assessment",
                score=assessment["score"],
                explanation=assessment["explanation"],
                criteria=["evidence support", "operational risk", "reversibility"],
                evidence_ids=input_entity_ids,
            )
        decision.select(payload["selected"])

    return answer, agent_task_id


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
