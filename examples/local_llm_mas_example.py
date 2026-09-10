"""Real local-LLM multi-agent incident-response simulation captured by Flowcept."""

import argparse

from langchain_openai import ChatOpenAI

from flowcept import Flowcept, FlowceptTask
from flowcept.commons.vocabulary import PROV_AGENT
from flowcept.configs import AGENT, AGENT_API_KEY
from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM

INCIDENT = (
    "At 09:10, customers began reporting that the online checkout accepts orders but "
    "does not send confirmations. The order API is healthy, the notification queue "
    "grew from 20 to 18,000 messages, and yesterday's release changed the email worker "
    "configuration. Payment processing must not be interrupted."
)


def run_agent(
    agent_id: str,
    agent_role: str,
    instruction: str,
    evidence: dict,
    input_entity_ids: list[str],
    output_entity_id: str,
    model_name: str,
    workflow_id: str,
    parent_task_id: str | None = None,
) -> tuple[str, str]:
    """Run one local-model agent and capture its complete provenance."""
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
            max_tokens=350,
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
                    "max_tokens": 350,
                },
                "agent_role": agent_role,
            }
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    f"You are the {agent_role} in a production incident-response team. "
                    "Use only the supplied evidence, state uncertainties, and keep the response under 180 words."
                ),
            },
            {"role": "user", "content": f"{instruction}\n\nEvidence:\n{evidence}"},
        ]
        answer = llm.invoke(prompt)
        agent_task.end(
            generated={
                "response": answer,
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
            monitoring_task_id,
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
            investigation_task_id,
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
            planning_task_id,
        )

        final_decision, _ = run_agent(
            "incident-commander-agent",
            "incident commander",
            (
                "Make the final operational decision. State whether to execute, modify, or reject the plan; list "
                "approved steps in order; and explain how the evidence and risk review support the decision."
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
            review_task_id,
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
