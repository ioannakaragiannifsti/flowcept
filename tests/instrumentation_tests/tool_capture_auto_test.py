"""Automatic tool capture: decorated tools, result normalization, and scope collection."""

import pytest

from flowcept import (
    DecisionCapture,
    Flowcept,
    FlowceptTool,
    current_retrievals,
    flowcept_tool,
    normalize_retrieved_items,
    retrieval_scope,
)


def test_normalizes_list_of_dicts_with_various_id_fields():
    """Pick up whichever identity, source, and score fields a tool happens to use."""
    items = normalize_retrieved_items(
        [
            {"id": "row-1", "title": "First", "score": 0.5},
            {"url": "https://example.com/a", "content": "Second"},
        ]
    )

    assert [item.item_id for item in items] == ["row-1", "https://example.com/a"]
    assert items[0].score == 0.5
    assert items[1].source == "https://example.com/a"
    assert items[1].content == "Second"
    assert [item.rank for item in items] == [0, 1]


def test_normalizes_wrapped_payloads_and_mappings():
    """Unwrap the result list out of the envelope a tool returned it in."""
    assert [item.item_id for item in normalize_retrieved_items({"results": [{"id": "a"}, {"id": "b"}]})] == ["a", "b"]
    assert [item.item_id for item in normalize_retrieved_items({"doc-1": "x", "doc-2": "y"})] == ["doc-1", "doc-2"]


def test_normalizes_documents_rows_and_scalars():
    """Handle LangChain documents, database rows, and plain values without losing them."""

    class FakeDocument:
        def __init__(self):
            self.page_content = "Document body"
            self.metadata = {"source": "kb/a.md"}

    document = normalize_retrieved_items([FakeDocument()])[0]
    assert document.content == "Document body"
    assert document.source == "kb/a.md"

    row = normalize_retrieved_items([("runbook-17", "Config rollback")])[0]
    assert row.item_id == "runbook-17"
    assert row.content == ["runbook-17", "Config rollback"]

    # Nothing is dropped for want of an id; one is derived from the content instead.
    scalar = normalize_retrieved_items(["just a string"])[0]
    assert scalar.item_id.startswith("item-0-")
    assert scalar.content == "just a string"


def test_normalization_keeps_duplicate_results_distinct():
    """Two identical results stay two items, because ids must be unique in a retrieval."""
    items = normalize_retrieved_items([{"id": "same"}, {"id": "same"}])

    assert [item.item_id for item in items] == ["same", "same#1"]


def test_decorated_tool_is_captured_and_returns_its_own_value():
    """The tool runs untouched; the capture happens alongside it."""

    @flowcept_tool(tool_type="database")
    def search_runbooks(query: str, kind: str = "runbook"):
        return [{"id": "runbook-17", "title": f"match for {query}"}]

    with Flowcept(start_persistence=False):
        with retrieval_scope(agent_id="agent-001", workflow_id="wf-1"):
            result = search_runbooks("smtp credentials", kind="runbook")
            collected = current_retrievals()
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]

    # The caller gets exactly what the undecorated function returned.
    assert result == [{"id": "runbook-17", "title": "match for smtp credentials"}]

    captured = tasks[-1]
    assert captured["activity_id"] == "search_runbooks"
    assert captured["used"]["query"] == "smtp credentials"
    assert captured["used"]["tool_type"] == "database"
    assert captured["used"]["tool_args"] == {"kind": "runbook"}
    assert captured["generated"]["retrieved_item_ids"] == ["runbook-17"]
    # The scope collected it, which is what lets a decision find it without being told.
    assert [retrieval.tool_name for retrieval in collected] == ["search_runbooks"]
    assert captured["agent_id"] == "agent-001"


def test_decision_capture_collects_scope_retrievals_automatically():
    """A decision made in the scope is grounded in whatever the tools retrieved there."""

    @flowcept_tool(tool_type="web_search")
    def search(query: str):
        return [{"url": "https://example.com/a"}, {"url": "https://example.com/b"}]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001", workflow_id="wf-1"):
        search("queue backlog")

        with DecisionCapture(decision_type="selection", context="c", agent_id="agent-001") as decision:
            decision.use_evidence("https://example.com/a", used=True, role="supporting")
            decision.use_evidence("https://example.com/b", used=False, role="irrelevant")
            decision.add_candidate("a", "A")
            decision.add_candidate("b", "B")
            decision.select("a")

    summary = decision.record.grounding_summary()
    assert summary["tools_used"] == ["search"]
    assert summary["retrieved_count"] == 2
    assert summary["kept_item_ids"] == ["https://example.com/a"]
    assert summary["dropped_item_ids"] == ["https://example.com/b"]


def test_explicit_empty_retrievals_opts_out_of_grounding():
    """An agent that uses no tools keeps the plain decision contract inside a scope."""

    @flowcept_tool()
    def search(query: str):
        return [{"id": "a"}]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        search("anything")
        capture = DecisionCapture(decision_type="selection", context="c", agent_id="agent-001", retrievals=[])

    assert capture.retrievals == []


def test_scopes_do_not_leak_between_agents():
    """One agent's retrievals never show up in the next agent's decision."""

    @flowcept_tool()
    def search(query: str):
        return [{"id": f"hit-for-{query}"}]

    with Flowcept(start_persistence=False):
        with retrieval_scope(agent_id="agent-001"):
            search("first")
            first = current_retrievals()
        with retrieval_scope(agent_id="agent-002"):
            search("second")
            second = current_retrievals()
        # Outside any scope nothing is collected, and capture still works.
        search("third")
        outside = current_retrievals()

    assert [item.item_id for item in first[0].items] == ["hit-for-first"]
    assert [item.item_id for item in second[0].items] == ["hit-for-second"]
    assert len(first) == len(second) == 1
    assert outside == []


def test_failed_tool_call_records_no_retrieval():
    """A tool that raised has no trustworthy result set, so nothing is recorded."""

    @flowcept_tool()
    def broken(query: str):
        raise RuntimeError("upstream is down")

    with Flowcept(start_persistence=False):
        with retrieval_scope(agent_id="agent-001"):
            with pytest.raises(RuntimeError, match="upstream is down"):
                broken("anything")
            collected = current_retrievals()
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]

    assert collected == []
    assert not [task for task in tasks if task.get("activity_id") == "broken"]


def test_langchain_tool_wrapper_captures_invocations():
    """FlowceptTool captures a LangChain-style tool the same way, passing results through."""

    class FakeTool:
        name = "kb_search"
        description = "Search the knowledge base."

        def invoke(self, tool_input, **kwargs):
            return [{"id": "kb-1", "content": f"about {tool_input['query']}"}]

    with Flowcept(start_persistence=False):
        with retrieval_scope(agent_id="agent-001"):
            wrapped = FlowceptTool(FakeTool(), tool_type="vector_store")
            result = wrapped.invoke({"query": "ack failures"})
            collected = current_retrievals()
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]

    assert result == [{"id": "kb-1", "content": "about ack failures"}]
    assert wrapped.name == "kb_search"
    captured = tasks[-1]
    assert captured["activity_id"] == "kb_search"
    assert captured["used"]["query"] == "ack failures"
    assert collected[0].item_ids == ["kb-1"]


def test_async_tool_is_awaited_before_recording():
    """An async tool records what it resolved to, not the coroutine object."""
    import asyncio

    @flowcept_tool(tool_type="api")
    async def fetch(query: str):
        await asyncio.sleep(0)
        return [{"id": "async-1", "content": f"about {query}"}]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        result = asyncio.run(fetch("latency"))
        collected = current_retrievals()
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]

    assert result == [{"id": "async-1", "content": "about latency"}]
    assert collected[0].item_ids == ["async-1"]
    assert tasks[-1]["generated"]["retrieved_count"] == 1


def test_generator_tool_is_materialized_for_both_sides():
    """A one-shot iterator is recorded and still usable by the caller."""

    @flowcept_tool()
    def stream(query: str):
        return (row for row in [{"id": "g1"}, {"id": "g2"}])

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        result = stream("anything")
        collected = current_retrievals()

    # The caller gets a list rather than an exhausted generator.
    assert result == [{"id": "g1"}, {"id": "g2"}]
    assert collected[0].item_ids == ["g1", "g2"]


def test_scope_can_be_propagated_to_a_worker_thread():
    """A tool dispatched to another thread still reaches its agent's decision."""
    from concurrent.futures import ThreadPoolExecutor

    from flowcept import propagate_scope

    @flowcept_tool()
    def search(query: str):
        return [{"id": f"hit-{query}"}]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001", workflow_id="wf-1"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            runner = propagate_scope(search)
            # Two concurrent submissions of the same wrapper must both work.
            list(pool.map(runner, ["a", "b"]))
        collected = current_retrievals()
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]

    assert sorted(retrieval.items[0].item_id for retrieval in collected) == ["hit-a", "hit-b"]
    assert {task["agent_id"] for task in tasks} == {"agent-001"}


def test_unpropagated_thread_still_records_but_is_not_collected():
    """Without propagation the retrieval is orphaned, which the warning calls out."""
    import threading

    @flowcept_tool()
    def search(query: str):
        return [{"id": "t1"}]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        thread = threading.Thread(target=search, args=("q",))
        thread.start()
        thread.join()
        collected = current_retrievals()

    assert collected == []


def test_result_size_limits_are_applied_and_recorded():
    """Oversized result sets are capped, and the cap is part of the record."""

    @flowcept_tool(max_items=3, max_item_chars=50)
    def search(query: str):
        return [{"id": f"row-{index}", "content": "x" * 200} for index in range(10)]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        search("anything")
        retrieval = current_retrievals()[0]

    assert len(retrieval.items) == 3
    assert retrieval.truncation["dropped_items"] == 7
    assert retrieval.truncation["original_count"] == 10
    assert retrieval.truncation["shortened_item_ids"] == ["row-0", "row-1", "row-2"]
    assert len(retrieval.items[0].content) == 50
    assert retrieval.items[0].metadata["original_chars"] == 200


def test_everything_is_recorded_by_default():
    """Provenance keeps the whole result set; nothing is capped unless a caller asks."""

    @flowcept_tool()
    def search(query: str):
        return [{"id": f"row-{index}", "content": "x" * 5000} for index in range(300)]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        search("anything")
        retrieval = current_retrievals()[0]

    assert len(retrieval.items) == 300
    assert len(retrieval.items[0].content) == 5000
    assert retrieval.truncation == {}


def test_prompt_budget_shortens_only_the_model_copy():
    """The model sees a preview; the record keeps the full content it was made from."""

    @flowcept_tool()
    def search(query: str):
        return [{"id": "doc-1", "content": "y" * 5000}, {"id": "doc-2", "content": "short"}]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        search("anything")
        capture = DecisionCapture(decision_type="selection", context="c", agent_id="agent-001", prompt_item_chars=100)
        rendered = capture._retrieved_for_prompt()

    # Prompt copy is budgeted and says so.
    assert len(rendered[0]["content"]) == 100
    assert rendered[0]["content_is_truncated_preview"] is True
    assert rendered[0]["full_content_chars"] == 5000
    # A short item is untouched, and every id is still present so verdicts cover them all.
    assert rendered[1]["content"] == "short"
    assert "content_is_truncated_preview" not in rendered[1]
    assert [entry["item_id"] for entry in rendered] == ["doc-1", "doc-2"]
    # The stored retrieval still holds the whole document.
    assert len(capture.retrievals[0].items[0].content) == 5000


def test_normalizes_elasticsearch_response():
    """Unwrap a nested hits envelope and keep the relevance score and document body."""
    items = normalize_retrieved_items(
        {
            "took": 5,
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {"_id": "doc-1", "_score": 3.2, "_source": {"title": "Ack failures"}},
                    {"_id": "doc-2", "_score": 1.1, "_source": {"title": "Other"}},
                ],
            },
        }
    )

    assert [item.item_id for item in items] == ["doc-1", "doc-2"]
    assert [item.score for item in items] == [3.2, 1.1]
    assert items[0].content == {"title": "Ack failures"}


def test_normalizes_column_oriented_vector_store():
    """Chroma-style parallel arrays are zipped back into one item per position."""
    items = normalize_retrieved_items(
        {
            "ids": [["v1", "v2"]],
            "documents": [["text one", "text two"]],
            "distances": [[0.1, 0.4]],
        }
    )

    assert [item.item_id for item in items] == ["v1", "v2"]
    assert [item.content for item in items] == ["text one", "text two"]
    # Distance is recorded as given; smaller is closer, and inverting it would be a guess.
    assert [item.score for item in items] == [0.1, 0.4]


def test_normalizes_sparql_and_unknown_envelopes():
    """Descend through a named envelope, and through an unknown single-key wrapper."""
    sparql = normalize_retrieved_items(
        {"results": {"bindings": [{"s": {"value": "http://x/1"}}, {"s": {"value": "http://x/2"}}]}}
    )
    assert len(sparql) == 2

    # Solr nests under "response", which is not a name this knows.
    solr = normalize_retrieved_items({"response": {"numFound": 2, "docs": [{"id": "s1"}, {"id": "s2"}]}})
    assert [item.item_id for item in solr] == ["s1", "s2"]


def test_normalizes_dataframe_rows():
    """A table becomes one item per row, not one item holding the whole table."""
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame([{"id": "r1", "v": 1}, {"id": "r2", "v": 2}])

    items = normalize_retrieved_items(frame)

    assert [item.item_id for item in items] == ["r1", "r2"]
    assert items[0].content == {"id": "r1", "v": 1}


def test_single_record_is_not_split_into_fields():
    """One result stays one item, while an id-to-content mapping still becomes many."""
    single = normalize_retrieved_items({"id": "one", "content": "x"})
    assert [(item.item_id, item.content) for item in single] == [("one", "x")]

    mapping = normalize_retrieved_items({"doc-1": "x", "doc-2": "y"})
    assert [item.item_id for item in mapping] == ["doc-1", "doc-2"]


def test_query_method_is_recorded():
    """How the agent searched is captured alongside what it searched for."""

    @flowcept_tool(tool_type="database", query_method="cypher", query_arg="cypher")
    def graph_db(cypher: str):
        return [{"id": "n1", "name": "Service A"}]

    with Flowcept(start_persistence=False), retrieval_scope(agent_id="agent-001"):
        graph_db(cypher="MATCH (s:Service) RETURN s")
        retrieval = current_retrievals()[0]
        tasks = [message for message in Flowcept.buffer if message.get("subtype") == "agent_tool"]

    assert retrieval.query_method == "cypher"
    assert retrieval.query == "MATCH (s:Service) RETURN s"
    assert tasks[-1]["used"]["query_method"] == "cypher"
    assert retrieval.to_dict()["query_method"] == "cypher"
