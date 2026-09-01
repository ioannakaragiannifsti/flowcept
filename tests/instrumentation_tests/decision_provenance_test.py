from flowcept.commons.flowcept_dataclasses.decision_provenance import Candidate


def test_candidate_serialization():
    """Serialize a candidate to a provenance dictionary."""
    candidate = Candidate(
        candidate_id="candidate-001",
        content="First alternative",
        status="selected",
    )

    assert candidate.to_dict() == {
        "candidate_id": "candidate-001",
        "content": "First alternative",
        "status": "selected",
        "origin_type": "explicit_generation",
    }


from flowcept import Assessment, DecisionRecord, Flowcept, record_decision


def test_record_decision():
    """Capture a completed decision as a Flowcept task."""
    decision = DecisionRecord(
        decision_type="selection",
        context="Select an alternative",
        candidates=[
            Candidate("candidate-001", "selected", content="First"),
            Candidate("candidate-002", "rejected", content="Second"),
        ],
        assessments=[
            Assessment(
                candidate_id="candidate-001",
                evaluator_id="evaluator-001",
                score_type="evaluator_score",
                score=0.9,
            )
        ],
        selected_candidate_ids=["candidate-001"],
    )

    with Flowcept(start_persistence=False):
        task_id = record_decision(decision, agent_id="agent-001")
        tasks = [message for message in Flowcept.buffer if message.get("type") == "task"]

    captured = tasks[-1]
    assert captured["task_id"] == task_id
    assert captured["subtype"] == "decision"
    assert captured["agent_id"] == "agent-001"
    assert captured["generated"]["decision"]["decision_id"] == decision.decision_id
    assert captured["generated"]["decision"]["selected_candidate_ids"] == ["candidate-001"]


from flowcept import DecisionCapture


def test_decision_capture_context():
    """Capture a decision through the context-manager API."""
    with Flowcept(start_persistence=False):
        with DecisionCapture(
            decision_type="selection",
            context="Select an alternative",
            agent_id="agent-001",
        ) as decision:
            decision.add_candidate("candidate-001", "First")
            decision.add_candidate("candidate-002", "Second")
            decision.assess(
                "candidate-001",
                evaluator_id="evaluator-001",
                score_type="evaluator_score",
                score=0.9,
            )
            decision.select("candidate-001")

        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "decision"]

    assert len(tasks) == 1
    assert tasks[0]["generated"]["decision"]["selected_candidate_ids"] == ["candidate-001"]
