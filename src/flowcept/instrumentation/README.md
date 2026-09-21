# `flowcept.instrumentation`

Explicit provenance capture APIs used directly in user code.

## Key Files

- `flowcept_decorator.py`: `@flowcept` workflow decorator.
- `flowcept_task.py`: `FlowceptTask` and task decorators.
- `flowcept_loop.py`: `FlowceptLoop` and lightweight loop capture.
- `flowcept_torch.py`: PyTorch module, epoch, batch, and child-layer capture.
- `flowcept_agent_task.py`: agent-aware task wrapper.
- `decision_provenance.py`: manual and LLM-assisted `DecisionCapture` context manager.
- `tool_provenance.py`: `@flowcept_tool`, `FlowceptTool`, `retrieval_scope`, and `ToolCapture` for agent tool calls and what they retrieved.
- `task_capture.py`: lower-level task capture helpers.

## Choosing An API

- Use `@flowcept_task` for simple function-level tasks.
- Use `@flowcept` for a top-level workflow function.
- Use `with Flowcept():` when the workflow spans multiple steps or files.
- Use `FlowceptTask` when decorators cannot express required fields.
- Use `FlowceptLoop` for loop iterations.
- Use `DecisionCapture` for explicit agent decisions that must record alternatives, assessments, and selections.
- Use `@flowcept_tool` (or `FlowceptTool` for LangChain tools) on agent tool calls — database queries, web searches, retrievers — whose results the decision must be grounded in, and wrap each agent turn in `retrieval_scope`.
- Use `flowcept_torch.py` only for PyTorch-specific instrumentation.

## Automatic Decision Capture

`DecisionCapture` is a context manager, not a decorator. Attach an unwrapped LangChain model and call `invoke()` for the task whose decision provenance should be captured:

```python
from flowcept import DecisionCapture, Flowcept

# Configure my_llm with the provider used by your application.
with Flowcept(start_persistence=False), DecisionCapture(
    decision_type="selection",
    context="Choose the output that best satisfies the request",
    agent_id="my-agent",
    llm=my_llm,
) as decision:
    record = decision.invoke("Write a formal greeting.")

print(record.to_dict())
```

`invoke()` automatically adds domain-neutral decision instructions and an OpenAI-compatible JSON-schema response format. It validates the model's explicitly generated candidates, assessment criteria, model-reported confidence scores, explanations, and selected candidate IDs before storing a `PROV_AGENT.DECISION` task. The decision is linked to the automatically captured LLM invocation.

The attached model must accept the OpenAI-compatible JSON-schema `response_format` parameter. Importing `DecisionCapture` alone has no side effects; capture starts only for an explicit `invoke()` call. Use a fresh `DecisionCapture` instance for each decision. See `docs/prov_capture.rst` and `examples/decision_provenance_example.py` for the complete behavior and runnable example.

## Tool Retrieval And Grounded Decisions

A tool call is recorded as a `PROV_AGENT.AGENT_TOOL` task: `used` holds the tool name, tool type, and the query; `generated` holds *every* item the tool returned. Nothing is filtered, so the recorded result set is the ground truth a later decision can be checked against.

### Automatic capture (what a MAS should use)

Decorate the tool functions the agents already have and open one scope per agent turn. The tool runs untouched and its return value is passed through unchanged; the decision then finds the retrievals by itself.

```python
from flowcept import DecisionCapture, flowcept_tool, retrieval_scope

@flowcept_tool(tool_type="database")
def search_runbooks(query: str) -> list[dict]:
    return [dict(row) for row in cursor.execute(SQL, (query,))]

with retrieval_scope(agent_id="planner", workflow_id=workflow_id, parent_task_id=agent_task_id):
    search_runbooks(query_the_agent_wrote)        # captured as an agent_tool task

    capture = DecisionCapture(..., agent_id="planner", llm=model)   # no retrievals= needed
    record = capture.invoke("Choose a remediation.")
    record.grounding_summary()  # tools_used, retrieved_count, kept_item_ids, dropped_item_ids
```

`normalize_retrieved_items` turns whatever the tool returned into retrieved items, whichever retrieval system produced it:

| Source | Shape handled |
| --- | --- |
| SQL / Mongo / Neo4j | lists of dicts or rows; identity from `item_id`/`id`/`_id`/`url`/… |
| Elasticsearch / OpenSearch | `{"hits": {"hits": [...]}}`, with `_id`, `_score` and `_source` as the body |
| Solr and other envelopes | any single-key wrapper, e.g. `{"response": {"docs": [...]}}` |
| SPARQL | `{"results": {"bindings": [...]}}` |
| Chroma / Weaviate | column-oriented `{"ids": [[...]], "documents": [[...]], "distances": [[...]]}`, zipped back into records |
| Pinecone / Qdrant | `{"matches": [...]}` / `{"result": [...]}` with `score` |
| Web search | lists of dicts keyed by `url`, with `source` taken from it |
| pandas | one item per row, not one item per table |
| LangChain | `Document.page_content` plus `metadata` |

Also handles id-to-content mappings, plain scalars, and a single record (which is kept as one item rather than split per field). A result with no usable identifier gets one derived from its content, and duplicates are suffixed, so every item stays individually addressable by the verdicts a decision reports against it. `query_method` records *how* the agent searched — `"sql"`, `"cypher"`, `"sparql"`, `"vector_similarity"`, `"keyword"`, `"http_get"` — so a recorded query can be interpreted later without inferring it from syntax. A tool that raises records nothing — a failed call has no trustworthy result set.

Scopes are context-local, so concurrent agents keep their own; retrievals never leak between agent turns. `FlowceptTool(tool, tool_type=...)` wraps a LangChain `BaseTool` the same way (`invoke`/`run`/`ainvoke`/`arun`), mirroring `FlowceptLLM` for models. `ToolCapture` remains for callers that add results explicitly.

### Tools that are not a plain sync function

- **`async def` tools** are detected and awaited before recording, so the decorated tool stays a coroutine function.
- **Generators and other one-shot iterators** are materialized into a list. The caller receives that list, because an iterator cannot be both recorded and returned.
- **Tools dispatched to another thread** do not see the scope — `contextvars` do not cross `threading.Thread` — so their retrieval would be published unattributed and never reach the decision. Wrap the callable at submission time with `propagate_scope(tool)`, on the thread that owns the scope, or pass `agent_id=`/`workflow_id=` to `@flowcept_tool`. A capture that finds neither a scope nor an `agent_id` logs a warning rather than failing quietly.
- **Framework-internal dispatch** (LangGraph `ToolNode`, CrewAI, AutoGen) only captures if the wrapped callable is the one registered with the framework.
- **Action tools** — send an email, write a file, run code — are captured correctly as `agent_tool` tasks, but "kept vs dropped evidence" is meaningless for them. Keep them out of the decision's retrievals (call them outside the scope, or pass `retrievals=[]`).

### Result size: what is stored vs what the model sees

These are two different budgets and they are set independently.

**Provenance keeps everything by default.** `max_items` and `max_item_chars` on `@flowcept_tool` default to `0`, meaning no cap: the recorded retrieval holds the whole result set. Set them only if you want to store less — for instance when a tool can return more than a MongoDB document can hold (16MB). Anything they cut is recorded in the retrieval's `truncation`, so a capped result never reads as a complete one, and a retrieval approaching the MongoDB limit logs a warning pointing at `Flowcept.db`'s GridFS-backed blob store for the payload.

**The prompt is budgeted, because a context window is a hard limit.** `DecisionCapture(prompt_item_chars=800)` shortens only the copy of the evidence sent to the model, marking each shortened entry `content_is_truncated_preview` with its `full_content_chars`. Every `item_id` is still present, so the model's verdicts still cover the whole result set, and the stored retrieval is untouched. Raise it for large-context models, or set `0` to send everything — but note that providers truncate an over-long prompt silently, which leaves the model reading a cut-off schema and returning invalid JSON. `prompt_max_items` can drop items entirely; it is off by default, because an item the model never sees is one it cannot report a verdict on.

### What the decision records

Retrievals switch `invoke()` to the `GroundedDecisionResponse` contract: the retrieved items go into the prompt with their `item_id`s, and the model must return an `evidence_uses` entry for each one saying whether it kept the item (`used`), why (`role` from `supporting`, `contradicting`, `irrelevant`, `redundant`, `unreliable`), and a one-sentence explanation. An `item_id` the model invents is rejected, so the grounding trail cannot claim evidence no tool returned. Each verdict may also name a `candidate_id`, which is what turns the record into a per-alternative account: which evidence supported the alternative that was chosen, and which supported the ones rejected.

The stored decision task carries, in one record: `retrievals` (the unfiltered results), `evidence_uses` (kept versus dropped, with reasons), and the usual `candidates`, `assessments`, and `selected_candidate_ids`. `custom_metadata.grounding` holds the summary counts, and `used.retrieval_ids` links the decision back to the `agent_tool` tasks that produced its evidence. Manual capture uses the same fields through `capture.add_retrieval()` and `capture.use_evidence()`.

Passing `retrievals=[]` explicitly opts out inside a scope, which is how an agent that uses no tools keeps the plain decision contract. See `examples/local_llm_tool_grounded_example.py` for a four-agent system where each agent writes its own query, and `examples/tool_provenance_example.py` for the same structure with no model or services.

`tool_type` is restricted to `web_search`, `database`, `vector_store`, `file_system`, `api`, `code_execution`, and `other` so tool usage stays queryable across workflows.

## Extension Rules

- Instrument coarsely by default; tight loops and distributed workers can amplify overhead.
- Preserve `used`, `generated`, `activity_id`, `workflow_id`, and status semantics.
- Keep new instrumentation compatible with the existing interceptor path.
- Tests usually belong in `tests/instrumentation_tests/`.
