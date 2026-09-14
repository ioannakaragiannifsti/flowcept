import json

import pytest

from flowcept import Assessment, DecisionCapture, DecisionRecord, Flowcept, record_decision
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


def decision_output():
    """Generate a decision payload to exercise the response boundary."""
    return {
        "candidates": [{"candidate_id": "a", "content": "First"}, {"candidate_id": "b", "content": "Second"}],
        "assessments": [
            {"candidate_id": candidate, "score": score, "criteria": ["relevance"], "explanation": explanation}
            for candidate, score, explanation in [("a", 0.9, "Meets the request"), ("b", 0.4, "Less relevant")]
        ],
        "selected_candidate_ids": ["a"],
    }


def test_capture_structured_response():
    """Map validated output to the real buffer without duplicating context exit records."""
    with Flowcept(start_persistence=False):
        with DecisionCapture("selection", "Select", agent_id="evaluator", input_entity_ids=["request"]) as capture:
            record = capture.record_response(json.dumps(decision_output()), invocation_task_id="invocation")
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "decision"]
    assert len(tasks) == 1
    assert tasks[0]["parent_task_id"] == "invocation"
    assert tasks[0]["generated"]["decision"] == record.to_dict()
    assert record.input_entity_ids == ["request"]
    assert [candidate.status for candidate in record.candidates] == ["selected", "rejected"]
    assert all(assessment.evaluator_id == "evaluator" for assessment in record.assessments)
    assert all(assessment.score_type == "model_reported_confidence" for assessment in record.assessments)
    with pytest.raises(ValueError, match="already"):
        capture.record_response(json.dumps(decision_output()), invocation_task_id="another")


@pytest.mark.parametrize("case", ["score", "nan", "unknown", "duplicate", "unassessed", "empty", "extra", "type"])
def test_invalid_decision_response_is_not_recorded(case):
    """Reject invalid model output without publishing a successful decision."""
    output = decision_output()
    if case == "score":
        output["assessments"][0]["score"] = 1.1
    elif case == "nan":
        output["assessments"][0]["score"] = float("nan")
    elif case == "unknown":
        output["selected_candidate_ids"] = ["missing"]
    elif case == "duplicate":
        output["candidates"][1]["candidate_id"] = "a"
    elif case == "unassessed":
        output["assessments"].pop()
    elif case == "empty":
        output["selected_candidate_ids"] = []
    elif case == "extra":
        output["workflow_id"] = "invented"
    elif case == "type":
        output["assessments"][0]["score"] = "0.9"
    with Flowcept(start_persistence=False):
        with pytest.raises(ValueError), DecisionCapture("selection", "Select", agent_id="evaluator") as capture:
            capture.record_response(json.dumps(output), invocation_task_id="invocation")
        assert not [message for message in Flowcept.buffer if message.get("subtype") == "decision"]


def test_automatic_capture_requires_attachment():
    """Importing or constructing manual capture does not activate an LLM."""
    capture = DecisionCapture("selection", "Select", agent_id="evaluator")
    with pytest.raises(ValueError, match="llm"):
        capture.invoke("Select an output")


def test_decision_prompt_uses_validated_contract():
    """Build the prompt from the same schema used to validate model output."""
    from flowcept.instrumentation.decision_response import DecisionResponse

    prompt = DecisionResponse.build_prompt("selection", {"criteria": ["quality"]})
    assert json.loads(prompt.split("JSON schema: ")[1]) == DecisionResponse.model_json_schema()
    assert json.dumps({"criteria": ["quality"]}) in prompt


def test_provider_response_format_requires_assessments_and_selection():
    """The provider contract must prohibit the empty lists observed in Ollama output."""
    from flowcept.instrumentation.decision_response import DecisionResponse

    response_format = DecisionResponse.response_format()
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    properties = response_format["json_schema"]["schema"]["properties"]
    assert properties["assessments"]["minItems"] == 1
    assert properties["selected_candidate_ids"]["minItems"] == 1


def test_caught_validation_failure_does_not_record_empty_decision():
    """Context exit must not turn a caught parse failure into an empty success."""
    with Flowcept(start_persistence=False):
        with DecisionCapture("selection", "Select", agent_id="evaluator") as capture, pytest.raises(ValueError):
            capture.record_response("not JSON", invocation_task_id="invocation")
        assert not [message for message in Flowcept.buffer if message.get("subtype") == "decision"]


def test_attached_model_without_invocation_does_not_record_decision():
    """Attaching a real client alone must not publish an empty decision on exit."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", api_key="unused-test-credential")
    with Flowcept(start_persistence=False):
        with DecisionCapture("selection", "Select", llm=llm):
            pass
        assert not [message for message in Flowcept.buffer if message.get("subtype") == "decision"]


@pytest.mark.llm
def test_real_llm_decision_capture():
    """Generate and capture a real decision with its invocation and workflow links."""
    from flowcept.configs import AGENT, AGENT_API_KEY

    if not AGENT_API_KEY or AGENT_API_KEY in {"?", "your-api-key-here"}:
        pytest.skip("LLM not configured")
    if AGENT.get("service_provider") != "openai":
        pytest.skip("This integration test requires an OpenAI-compatible provider")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=AGENT["model"], api_key=AGENT_API_KEY, base_url=AGENT.get("llm_server_url"))
    with Flowcept(start_persistence=False):
        capture = DecisionCapture("selection", "Choose an output", agent_id="evaluator", llm=llm)
        record = capture.invoke("Propose two short greetings and select the more formal greeting.")
        tasks = [message for message in Flowcept.buffer if message.get("type") == "task"]
    invocation = next(task for task in tasks if task.get("subtype") == "ai_model_invocation")
    decision = next(task for task in tasks if task.get("subtype") == "decision")
    assert decision["parent_task_id"] == invocation["task_id"]
    assert decision["workflow_id"] == invocation["workflow_id"]
    assert decision["agent_id"] == invocation["agent_id"] == "evaluator"
    assert decision["generated"]["decision"] == record.to_dict()
    assert len(record.candidates) == 2
    assert record.selected_candidate_ids
