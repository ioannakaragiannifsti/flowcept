import json

import pytest

from flowcept import DecisionCapture, Flowcept, Retrieval, RetrievedItem, ToolCapture, record_retrieval


def _web_search(query):
    """Stand in for a real search tool."""
    return [
        {"url": "https://example.com/a", "snippet": f"A result about {query}"},
        {"url": "https://example.com/b", "snippet": f"B result about {query}"},
    ]


def test_retrieval_serialization():
    """Serialize a retrieval with everything the tool returned."""
    retrieval = Retrieval(
        tool_name="web_search",
        tool_type="web_search",
        query="queue backlog",
        items=[RetrievedItem("item-1", content="First hit", source="https://example.com/a", rank=0)],
    )

    serialized = retrieval.to_dict()
    assert serialized["tool_name"] == "web_search"
    assert serialized["query"] == "queue backlog"
    assert serialized["retrieved_count"] == 1
    assert serialized["retrieved"][0]["source"] == "https://example.com/a"


def test_retrieval_rejects_unknown_tool_type():
    """Keep tool_type inside the recorded vocabulary."""
    with pytest.raises(ValueError):
        Retrieval(tool_name="t", query="q", tool_type="telepathy")


def test_record_retrieval_captures_agent_tool_task():
    """A tool call is captured as an agent_tool task carrying query and full results."""
    retrieval = Retrieval(
        tool_name="incident_db",
        tool_type="database",
        query="SELECT * FROM incidents WHERE service = 'checkout'",
        items=[RetrievedItem("row-1", content={"id": 1}), RetrievedItem("row-2", content={"id": 2})],
    )

    with Flowcept(start_persistence=False):
        task_id = record_retrieval(retrieval, agent_id="agent-001")
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]

    captured = tasks[-1]
    assert captured["task_id"] == task_id
    assert captured["activity_id"] == "incident_db"
    assert captured["used"]["tool_type"] == "database"
    assert captured["used"]["query"].startswith("SELECT")
    assert captured["generated"]["retrieved_count"] == 2
    assert captured["generated"]["retrieved_item_ids"] == ["row-1", "row-2"]


def test_tool_capture_context():
    """Capture a live tool call and reuse its retrieval for a decision."""
    with Flowcept(start_persistence=False):
        with ToolCapture(
            "web_search",
            tool_type="web_search",
            query="notification queue backlog",
            agent_id="agent-001",
        ) as tool:
            for hit in _web_search("notification queue backlog"):
                tool.add_item(hit["url"], content=hit["snippet"], source=hit["url"])

        assert tool.retrieval.item_ids == ["https://example.com/a", "https://example.com/b"]

        with DecisionCapture(
            decision_type="selection",
            context="Pick a remediation",
            agent_id="agent-001",
            retrievals=[tool.retrieval],
        ) as decision:
            decision.use_evidence(
                "https://example.com/a", used=True, role="supporting", explanation="Matches the symptom"
            )
            decision.use_evidence("https://example.com/b", used=False, role="irrelevant", explanation="Wrong service")
            decision.add_candidate("rollback", "Roll back the release")
            decision.add_candidate("drain", "Drain the queue")
            decision.assess("rollback", evaluator_id="agent-001", score_type="model_reported_confidence", score=0.8)
            decision.select("rollback")

        tool_tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]
        decision_tasks = [message for message in Flowcept.buffer if message.get("subtype") == "decision"]

    assert len(tool_tasks) == 1
    captured = decision_tasks[-1]
    # The decision task points back at the tool task that produced its evidence.
    assert captured["used"]["retrieval_ids"] == [tool.retrieval.retrieval_id]
    grounding = captured["custom_metadata"]["grounding"]
    assert grounding["tools_used"] == ["web_search"]
    assert grounding["retrieved_count"] == 2
    assert grounding["kept_item_ids"] == ["https://example.com/a"]
    assert grounding["dropped_item_ids"] == ["https://example.com/b"]
    recorded = captured["generated"]["decision"]
    assert recorded["retrievals"][0]["retrieved_count"] == 2
    assert recorded["evidence_uses"][0]["retrieval_id"] == tool.retrieval.retrieval_id


def test_evidence_must_reference_retrieved_items():
    """Reject grounding claims about items no tool returned."""
    with Flowcept(start_persistence=False):
        capture = DecisionCapture(decision_type="selection", context="c", agent_id="agent-001")
        capture.add_candidate("a", "A")
        capture.select("a")
        capture.use_evidence("never-retrieved", used=True)
        with pytest.raises(ValueError, match="unretrieved items"):
            capture._finish(parent_task_id=None)


def test_record_response_captures_evidence_uses():
    """Parse a grounded model response into recorded evidence use."""
    retrieval = Retrieval(
        tool_name="web_search",
        tool_type="web_search",
        query="q",
        items=[RetrievedItem("item-1", content="kept"), RetrievedItem("item-2", content="dropped")],
    )
    response = json.dumps(
        {
            "candidates": [
                {"candidate_id": "a", "content": "A"},
                {"candidate_id": "b", "content": "B"},
            ],
            "assessments": [
                {"candidate_id": "a", "score": 0.9, "criteria": ["fit"], "explanation": "Best fit"},
                {"candidate_id": "b", "score": 0.2, "criteria": ["fit"], "explanation": "Poor fit"},
            ],
            "selected_candidate_ids": ["a"],
            "evidence_uses": [
                {"item_id": "item-1", "used": True, "role": "supporting", "explanation": "Direct evidence"},
                {"item_id": "item-2", "used": False, "role": "unreliable", "explanation": "Unverified source"},
            ],
        }
    )

    with Flowcept(start_persistence=False):
        capture = DecisionCapture(
            decision_type="selection",
            context="c",
            agent_id="agent-001",
            retrievals=[retrieval],
        )
        record = capture.record_response(response, invocation_task_id="llm-task-1")

    assert record.grounding_summary() == {
        "tools_used": ["web_search"],
        "retrieved_count": 2,
        "kept_item_ids": ["item-1"],
        "dropped_item_ids": ["item-2"],
        "unreported_item_ids": [],
    }
