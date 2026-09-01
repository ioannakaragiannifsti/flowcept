"""Generic decision-provenance capture example."""

import json

from flowcept import DecisionCapture, Flowcept


def main():
    """Capture alternatives, assessments, and the selected result."""
    with Flowcept(
        workflow_name="Generic Decision Provenance",
        start_persistence=False,
    ), DecisionCapture(
        decision_type="selection",
        context="Select the best output for the supplied request",
        agent_id="example-agent",
        input_entity_ids=["request-001"],
        output_entity_ids=["result-001"],
    ) as decision:
        decision.add_candidate(
            candidate_id="candidate-001",
            content={"text": "First alternative"},
        )
        decision.add_candidate(
            candidate_id="candidate-002",
            content={"text": "Second alternative"},
        )
        decision.assess(
            candidate_id="candidate-001",
            evaluator_id="example-evaluator",
            score_type="evaluator_score",
            score=0.91,
            criteria=["relevance", "completeness"],
            explanation="Best satisfies the declared criteria",
        )
        decision.select("candidate-001")

    print(json.dumps(decision.record.to_dict(), indent=2))


if __name__ == "__main__":
    main()
