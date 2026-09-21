# Agent Retrieval & Decision Provenance

How Flowcept records what an agent searched for, what came back, what it kept, and what it
decided — and how to read the result.

---

## 1. The question this answers

An LLM agent that uses tools produces an answer. Provenance should be able to answer, after
the fact and without re-running anything:

1. **What did it search, and how?** Which tool, and the exact query in that tool's own
   language.
2. **What came back?** Every item the tool returned, unfiltered — including the results
   that turned out to be useless.
3. **What did it keep, and why?** A per-item verdict, with a reason.
4. **What did it decide from what it kept?** The alternatives it weighed, how it scored
   them, and which it selected.

Before this work Flowcept could record (1) partially and (4) fully. `PROV_AGENT.AGENT_TOOL`
and the `agent_flowcept_task` decorator captured a tool's *arguments* and *return value*,
but only for decorated MCP/LangGraph functions, and with no notion of a query, of
individual retrieved items, or of which of them survived into the answer. Steps (2) and (3)
did not exist.

### The separation that matters

The three layers are stored **separately and never merged**:

| Layer | Field | Produced by |
| --- | --- | --- |
| Ground truth | `retrievals` | the tool, verbatim |
| Filtering | `evidence_uses` | the model, one verdict per retrieved item |
| Generation | `candidates`, `assessments`, `selected_candidate_ids` | the model |

This is what makes the record checkable. If what the model generated is not traceable to
items it marked as used, that discrepancy is visible in the data rather than hidden in a
narrative answer.

---

## 2. Data model

### `RetrievedItem` — one result

| Field | Type | Meaning |
| --- | --- | --- |
| `item_id` | str | Identifier the verdicts address. Derived from content if the tool gave none. |
| `content` | any | The payload as returned. |
| `source` | str? | Where it came from (URL, file path, collection). |
| `rank` | int? | Position in the result set. |
| `score` | float? | Relevance or distance, **as the backend reported it** — a distance is not inverted into a similarity, because that would be a guess. |
| `metadata` | dict | Backend extras, plus truncation markers when applicable. |

### `Retrieval` — one tool call

| Field | Type | Meaning |
| --- | --- | --- |
| `tool_name` | str | e.g. `ops_database` |
| `tool_type` | str | Closed vocabulary: `web_search`, `database`, `vector_store`, `file_system`, `api`, `code_execution`, `other` |
| `query_method` | str? | *How* it was expressed: `sql`, `cypher`, `sparql`, `vector_similarity`, `keyword`, `http_get`. Open text — the set of retrieval languages is not closed. |
| `query` | any | The query itself. Typed `Any`, so a SQL string, a Cypher string, or an Elasticsearch DSL dict are all first-class. |
| `tool_args` | dict | Other arguments the call was made with. |
| `items` | list | Every result. |
| `truncation` | dict | Empty unless size limits cut something; see §7. |
| `retrieval_id` | uuid | Links the decision back to this tool call. |

### `EvidenceUse` — one verdict

| Field | Type | Meaning |
| --- | --- | --- |
| `item_id` | str | Must reference an item some tool actually returned. |
| `used` | bool | Kept or dropped. |
| `role` | str? | `supporting`, `contradicting`, `irrelevant`, `redundant`, `unreliable` |
| `explanation` | str? | One sentence of reasoning. |
| `candidate_id` | str? | Which alternative this item bears on — the link that shows *why one option won and another lost*. |
| `retrieval_id` | str? | Which tool call produced the item; filled in automatically. |

**Enforced:** `used: true` cannot combine with `irrelevant`, `redundant` or `unreliable` —
that pair is self-contradictory and inflates the kept count. `contradicting` **is** allowed
with `used: true`: evidence arguing against the chosen option is still evidence that was
used.

### `DecisionRecord` — the decision

Existing fields (`decision_type`, `context`, `candidates`, `assessments`,
`selected_candidate_ids`, `input_entity_ids`, `output_entity_ids`) plus:

- `retrievals` — the full ground truth, embedded in the record so it survives independently.
- `evidence_uses` — the verdicts.
- `grounding_summary()` — derived counts:

```python
{
  "tools_used": ["ops_database", "knowledge_base"],
  "retrieved_count": 9,
  "kept_item_ids": ["runbook-17", "mq-consumer-ack-failures.md"],
  "dropped_item_ids": ["runbook-08", "checkout-scaling.md"],
  "unreported_item_ids": []          # retrieved but never judged
}
```

`unreported_item_ids` is the integrity check: a non-empty list means the model failed to
report on something it was shown.

**Validation:** evidence cannot reference an item no tool returned, and `candidate_id` must
name a candidate the model actually produced. An invented id is rejected, so the trail
cannot claim evidence that never existed.

---

## 3. Capture APIs

### `@flowcept_tool` — the normal path

Decorate a tool function you already have. It runs untouched and **its return value is
passed through unchanged**, so existing agent code is unaffected.

```python
@flowcept_tool(tool_type="database", query_method="sql", query_arg="sql")
def ops_database(sql: str) -> list[dict]:
    return [dict(row) for row in connection.execute(sql)]
```

The query is taken from `query_arg`, or a keyword named `query`/`q`/`question`/`sql`/…, or
the first positional argument.

### `retrieval_scope` — how a decision finds its evidence

```python
with retrieval_scope(agent_id="planner", workflow_id=wf_id, parent_task_id=agent_task_id):
    ops_database(sql=the_sql_the_agent_wrote)     # captured

    capture = DecisionCapture(decision_type="...", context=ctx,
                              agent_id="planner", llm=model)   # no retrievals= needed
    record = capture.invoke("Choose a remediation.")
```

Tools inside the scope inherit the agent's attribution, and the `DecisionCapture` created
there collects their retrievals by itself. **A MAS adopts this by decorating its tools and
opening one scope per agent turn** — nothing is threaded through the call stack.

Scopes are context-local, so concurrent agents keep their own.

### `DecisionCapture` — the decision

With retrievals present, `invoke()` switches from the plain contract to
`GroundedDecisionResponse`: the retrieved items enter the prompt carrying their `item_id`s,
and the JSON schema makes an `evidence_uses` entry for **every** item mandatory, discarded
ones included.

Without retrievals (an agent that used no tools) it falls back to the plain contract and
records no verdicts. Both produce the same kind of record.

Manual use is also supported: `add_retrieval()`, `use_evidence()`, `add_candidate()`,
`assess()`, `select()`.

### Others

| API | Use |
| --- | --- |
| `FlowceptTool(tool)` | A LangChain `BaseTool` — `invoke`/`run`/`ainvoke`/`arun` |
| `ToolCapture` | Context manager for adding results explicitly |
| `propagate_scope(fn)` | Carry the scope into a worker thread (see §7) |
| `normalize_retrieved_items(result)` | The normalizer, usable directly |

---

## 4. What lands in MongoDB

One agent turn produces three task types, all in the `tasks` collection.

### `subtype: "agent_tool"` — one per tool call

```jsonc
{
  "activity_id": "ops_database",
  "subtype": "agent_tool",
  "agent_id": "remediation-planning-agent",
  "used": {
    "tool_name": "ops_database",
    "tool_type": "database",
    "query_method": "sql",
    "query": "SELECT id, title, body FROM ops_records WHERE kind = 'runbook'",
    "tool_args": { "sql": "..." }
  },
  "generated": {
    "retrieved": [ { "item_id": "runbook-17", "content": {...}, "rank": 0, "score": 0.67 } ],
    "retrieved_count": 3,
    "retrieved_item_ids": ["runbook-17", "runbook-42", "runbook-08"]
  },
  "custom_metadata": { "retrieval_id": "…", "tool_type": "database", "query_method": "sql" }
}
```

`generated.retrieved` is the ground truth: unfiltered, in the order the tool returned it.

### `subtype: "ai_model_invocation"` — one per model call

`used.prompt` (complete), `generated.response` (raw), and
`custom_metadata.llm_usage` with model, token counts and `token_count_source`
(`provider` or `estimated_from_chars`). Tool-selection calls are captured too, so the
agent's internal deliberation is in the record, not only its conclusion.

### `subtype: "decision"` — one per decision

```jsonc
{
  "activity_id": "decision",
  "subtype": "decision",
  "used": {
    "context": {...},
    "input_entity_ids": [...],
    "retrieval_ids": ["…"],        // points back at the agent_tool tasks
    "queries": ["SELECT …"]
  },
  "generated": { "decision": { /* the full DecisionRecord */ }, "output_entity_ids": [...] },
  "custom_metadata": { "decision_id": "…", "decision_type": "…", "grounding": { /* summary */ } }
}
```

Tasks are linked by `parent_task_id`: agent span → tool calls and model calls → decision.

---

## 5. Reading the output

A decision record in `runs/<workflow_id>.json`, abbreviated:

```jsonc
"retrievals": [{
  "tool_name": "knowledge_base", "query_method": "keyword",
  "query": "smtp credentials ack failure",
  "retrieved": [
    { "item_id": "mq-consumer-ack-failures.md", "score": 0.667, "content": "..." },
    { "item_id": "checkout-scaling.md",         "score": 0.375, "content": "..." }
  ]
}],
"evidence_uses": [
  { "item_id": "mq-consumer-ack-failures.md", "used": true,  "role": "supporting",
    "candidate_id": "rollback_config", "explanation": "Same failure mode as the incident" },
  { "item_id": "checkout-scaling.md", "used": false, "role": "irrelevant",
    "explanation": "About traffic peaks, not a stalled consumer" }
],
"candidates": [
  { "candidate_id": "rollback_config", "status": "selected" },
  { "candidate_id": "drain_queue",     "status": "rejected" }
],
"assessments": [
  { "candidate_id": "rollback_config", "score": 0.86, "explanation": "Addresses the root cause" },
  { "candidate_id": "drain_queue",     "score": 0.25, "explanation": "Drains symptoms only" }
],
"selected_candidate_ids": ["rollback_config"]
```

Read as: the agent searched the knowledge base with the keywords *smtp credentials ack
failure*; two documents came back; it kept the first as support for `rollback_config` and
discarded the second as irrelevant; it weighed two strategies and chose the rollback at 0.86
confidence over the queue drain at 0.25.

**What each part licenses you to say:**

- `retrievals` — what was *available*. Independent of the model.
- `evidence_uses` — what the model *claims* it did with each item. Self-reported, and
  structurally constrained (ids must exist, verdicts must be coherent) but not verified
  against its internal computation.
- `assessments.score` — model-reported confidence in `[0, 1]`, **not** a calibrated
  probability.
- `grounding.unreported_item_ids` — non-empty means the account is incomplete.

---

## 6. Where to see it

| Surface | What you get |
| --- | --- |
| **UI → Tools tab** | One row per tool call: tool, type, method, agent, query, retrieved count, duration; plus raw tasks |
| **UI → decision graph** | Teal tool nodes showing the query verbatim; evidence nodes showing retrieved content with `kept`/`dropped`, role, source, score, and the candidate it bore on. Green solid = kept, red dashed = dropped. Hover for full text, click for the raw record |
| **`runs/<workflow_id>.json`** | Per-agent decisions plus every captured task; stands alone if the workflow is later deleted |
| **`runs/<workflow_id>.txt`** | Exact mirror of the terminal, including the full trace |
| **MongoDB** | `db.task_query(filter={"workflow_id": ...})` |

---

## 7. Behaviour worth knowing

### Supported retrieval systems

`normalize_retrieved_items` handles, verified by test:

| Source | Shape |
| --- | --- |
| SQL / Mongo / Neo4j | lists of dicts or rows |
| Elasticsearch / OpenSearch | `{"hits": {"hits": [...]}}`, `_id`, `_score`, `_source` |
| Solr and unknown envelopes | any single-key wrapper, e.g. `{"response": {"docs": [...]}}` |
| SPARQL | `{"results": {"bindings": [...]}}` |
| Chroma / Weaviate | column-oriented `{"ids": [[…]], "documents": [[…]]}`, zipped into records |
| Pinecone / Qdrant | `{"matches": […]}` / `{"result": […]}` |
| Web search | lists keyed by `url` |
| pandas | one item per row |
| LangChain | `Document.page_content` + `metadata` |

Duplicate ids are suffixed; a result with no id gets a content-derived one; a single record
is kept whole rather than split per field.

### Storage vs prompt — two budgets

- **Storage keeps everything.** `max_items`/`max_item_chars` default to `0` (no cap). Set
  them only to store less, e.g. when a tool can exceed MongoDB's 16MB document limit —
  whatever they cut is recorded in `truncation`, and a retrieval nearing the limit warns.
- **The prompt is budgeted**, because a context window is a hard limit.
  `DecisionCapture(prompt_item_chars=800)` shortens only the model's copy, marking it
  `content_is_truncated_preview` with `full_content_chars`. All ids remain, so verdicts
  still cover the whole result set. The stored retrieval is untouched.

### Non-obvious cases

- **`async def` tools** are awaited before recording.
- **Generators** are materialized; the caller receives the list, since an iterator cannot be
  both recorded and returned.
- **Worker threads** do not inherit the scope (`contextvars` do not cross threads). Use
  `propagate_scope(tool)` at submission time, or pass `agent_id=` explicitly. A capture with
  neither logs a warning instead of silently orphaning the retrieval.
- **A tool that raises records nothing** — a failed call has no trustworthy result set.
- **Framework-internal dispatch** (LangGraph `ToolNode`, CrewAI, AutoGen) captures only if
  the *wrapped* callable is the one registered. Not yet verified against a live instance.
- **MCP tools** use Flowcept's existing `agent_flowcept_task` path.

### Limits

- Grounding is meaningful only for **retrieval** tools. Action tools (send email, write
  file, run code) are captured correctly as `agent_tool` tasks, but kept/dropped is
  meaningless for them — keep them out of a decision's retrievals.
- The grounded contract needs a provider supporting JSON-schema `response_format`.
- The model must emit a verdict per item; beyond ~100 items small models start dropping ids.
  Invented ids are rejected and retried rather than silently recorded.

---

## 8. Reference

**Modules** — `commons/flowcept_dataclasses/retrieval_provenance.py`,
`commons/flowcept_dataclasses/decision_provenance.py`,
`instrumentation/tool_provenance.py`, `instrumentation/decision_provenance.py`,
`instrumentation/decision_response.py`.

**Examples** — `examples/tool_provenance_example.py` (no model or services needed),
`examples/local_llm_tool_grounded_example.py` (four agents, real tool calls, local Qwen).

**Tests** — `tests/instrumentation_tests/tool_capture_auto_test.py`,
`tool_provenance_test.py`, `decision_provenance_test.py`.

**API docs** — `src/flowcept/instrumentation/README.md`.
