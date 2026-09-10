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


def test_product_engineering_mas_selects_complete_safety_test():
    """The MAS must reject cheaper candidates that miss safety-critical coverage."""
    from examples.decision_provenance_mas import run_pipeline

    result = run_pipeline(review_action="approve")

    assert result["recommendation"]["candidate_id"] == "cpd-test-extended"
    assert result["review"]["action"] == "approve"
    assert result["final_test"]["test_id"] == "CPD-TC-EXTENDED"
    assert result["workflow_id"]

    task_subtypes = {task.get("subtype") for task in result["provenance"]["tasks"]}
    assert {"agent_tool", "decision"} <= task_subtypes
    assert len(result["provenance"]["agents"]) == 6
    assert result["provenance"]["edges"]


def test_product_engineering_mas_preserves_human_override():
    """A human override remains distinct from the AI recommendation."""
    from examples.decision_provenance_mas import run_pipeline

    result = run_pipeline(review_action="override", override_candidate_id="cpd-test-conservative")

    assert result["recommendation"]["candidate_id"] == "cpd-test-extended"
    assert result["review"]["selected_candidate_id"] == "cpd-test-conservative"
    assert result["final_test"]["test_id"] == "CPD-TC-CONSERVATIVE"

    decisions = [task for task in result["provenance"]["tasks"] if task.get("subtype") == "decision"]
    assert [task["agent_id"] for task in decisions] == ["decision-board", "human-reviewer"]


def test_local_agent_prompt_and_response_contract():
    """Local agents exchange reviewable JSON rather than unstructured prose."""
    from examples.decision_provenance_mas import _build_agent_messages, _parse_agent_response

    messages = _build_agent_messages(
        role="Safety critic",
        task="Assess candidate coverage and residual risk.",
        inputs={"candidate_id": "cpd-test-extended"},
        output_schema={"candidate_id": "string", "eligible": "boolean"},
    )
    parsed = _parse_agent_response(
        '{"candidate_id":"cpd-test-extended","eligible":true}',
        required_fields=("candidate_id", "eligible"),
    )

    assert messages[0]["role"] == "system"
    assert "Safety critic" in messages[0]["content"]
    assert '"candidate_id": "cpd-test-extended"' in messages[1]["content"]
    assert parsed == {"candidate_id": "cpd-test-extended", "eligible": True}


def test_local_agent_accepts_single_item_for_collection_handoff():
    """A small local model's single candidate can continue through the MAS."""
    from examples.decision_provenance_mas import _parse_agent_response

    parsed = _parse_agent_response(
        '{"candidate_id":"economical","test_id":"CPD-1"}',
        required_fields=("candidates",),
    )

    assert parsed == {"candidates": [{"candidate_id": "economical", "test_id": "CPD-1"}]}


def test_local_model_observation_does_not_control_pipeline_handoff():
    """Unexpected model JSON remains observable without breaking the test workflow."""
    from examples.decision_provenance_mas import _merge_model_observation

    handoff = {"affected_variants": ["ICE-EU"], "reason": "deterministic scaffold"}
    result = _merge_model_observation(handoff, '{"unexpected":"but captured"}')

    assert result["affected_variants"] == ["ICE-EU"]
    assert result["model_output"] == {"unexpected": "but captured"}
