"""Decision provenance for production-incident remediation."""

import json

from flowcept import DecisionCapture, Flowcept

INCIDENT = {
    "incident_id": "INC-2048",
    "service": "payment-api",
    "symptoms": [
        "HTTP 500 rate above 18%",
        "Database connection pool exhausted",
    ],
    "constraints": [
        "Avoid data loss",
        "Recovery target below 15 minutes",
        "Require rollback capability",
    ],
}

# These could come from an LLM, planner, human, or several agents.
PROPOSED_ACTIONS = [
    {
        "candidate_id": "restart-service",
        "action": "Restart all payment-api instances",
        "estimated_recovery_minutes": 4,
        "rollback_available": False,
        "data_loss_risk": 0.20,
    },
    {
        "candidate_id": "increase-pool",
        "action": "Increase the connection pool and restart instances gradually",
        "estimated_recovery_minutes": 8,
        "rollback_available": True,
        "data_loss_risk": 0.05,
    },
    {
        "candidate_id": "database-failover",
        "action": "Fail over to the standby database",
        "estimated_recovery_minutes": 12,
        "rollback_available": True,
        "data_loss_risk": 0.15,
    },
]


def safety_score(candidate: dict) -> float:
    """Calculate a deterministic safety assessment."""
    score = 1.0 - candidate["data_loss_risk"]
    if candidate["rollback_available"]:
        score += 0.1
    return min(score, 1.0)


def recovery_score(candidate: dict) -> float:
    """Score whether the action meets the recovery-time objective."""
    recovery_minutes = candidate["estimated_recovery_minutes"]
    return max(0.0, 1.0 - recovery_minutes / 15)


def main():
    """Evaluate remediation alternatives and capture the decision."""
    with Flowcept(
        workflow_name="Production Incident Remediation",
        start_persistence=False,
    ), DecisionCapture(
        decision_type="remediation_selection",
        context={
            "incident_id": INCIDENT["incident_id"],
            "objective": "Restore service safely within 15 minutes",
            "constraints": INCIDENT["constraints"],
        },
        agent_id="incident-response-agent",
        input_entity_ids=[
            "incident:INC-2048",
            "telemetry:payment-api-errors",
            "runbook:database-connections",
        ],
        output_entity_ids=["remediation-plan:INC-2048"],
    ) as decision:
        combined_scores = {}

        for candidate in PROPOSED_ACTIONS:
            candidate_id = candidate["candidate_id"]

            decision.add_candidate(
                candidate_id=candidate_id,
                content=candidate,
                origin_type="planner_search",
            )

            safety = safety_score(candidate)
            recovery = recovery_score(candidate)

            decision.assess(
                candidate_id=candidate_id,
                evaluator_id="safety-policy-engine",
                score_type="validation_score",
                score=safety,
                criteria=["data-loss risk", "rollback availability"],
                evidence_ids=[
                    "policy:production-change-safety",
                    "incident:INC-2048",
                ],
                explanation="Evaluated against production safety controls",
            )

            decision.assess(
                candidate_id=candidate_id,
                evaluator_id="recovery-time-evaluator",
                score_type="validation_score",
                score=recovery,
                criteria=["recovery time objective"],
                evidence_ids=["sla:payment-api"],
                explanation="Compared estimated recovery time with the SLA",
            )

            # Selection logic belongs to the application, not FlowCept.
            combined_scores[candidate_id] = 0.7 * safety + 0.3 * recovery

        eligible_candidates = [
            candidate
            for candidate in PROPOSED_ACTIONS
            if candidate["rollback_available"] and candidate["data_loss_risk"] <= 0.10
        ]

        selected = max(
            eligible_candidates,
            key=lambda candidate: combined_scores[candidate["candidate_id"]],
        )

        decision.assess(
            candidate_id=selected["candidate_id"],
            evaluator_id="incident-response-agent",
            score_type="ranking_score",
            score=combined_scores[selected["candidate_id"]],
            criteria=["safety", "recovery speed"],
            evidence_ids=[
                "policy:production-change-safety",
                "sla:payment-api",
            ],
            explanation="Highest combined score among policy-compliant actions",
        )

        decision.select(selected["candidate_id"])

    print(f"Selected action: {selected['action']}")
    print(json.dumps(decision.record.to_dict(), indent=2))


if __name__ == "__main__":
    main()
