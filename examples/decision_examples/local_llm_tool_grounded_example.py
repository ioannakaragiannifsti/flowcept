r"""Tool-grounded decision capture across a local-Qwen multi-agent incident response.

USE CASE
--------
At 09:10 a production checkout starts accepting orders without sending confirmation
emails. The order API is healthy, the notification queue has grown from 20 to 18,000
messages, and yesterday's release changed the email worker configuration. Payments must
keep running.

Four agents handle it in sequence, each consuming the previous agent's conclusion:

1. ``triage-agent`` queries the operations database and classifies severity and urgency.
2. ``investigation-agent`` searches the knowledge base and the change log to pick a
   root-cause hypothesis.
3. ``remediation-planning-agent`` searches runbooks and the knowledge base, then picks a
   remediation strategy.
4. ``incident-commander-agent`` uses no tools at all -- it decides execute, modify, or
   reject from what the other three concluded.

THE AGENTS ISSUE REAL TOOL CALLS
--------------------------------
The tools are bound to the model with ``bind_tools``, so each agent chooses which tools to
call and writes the arguments itself, in each tool's own query language: a SQL SELECT for
the operations database, keywords for the knowledge base, a search string for the web. The
agent's SQL is executed as SQL. Nothing about which rows come back is fixed here.

Because a model can write a destructive statement, ``ops_database`` accepts exactly one
read-only ``SELECT``; anything else is refused and the refusal is recorded like any other
tool outcome.

The data the tools search (``examples/data/incident_ops.sql`` and ``examples/data/kb/``)
is a *data source*, not a result set: it holds far more records than any one query returns,
including records about payments and checkout scaling, so retrieving the wrong thing is
possible and is exactly what the capture is meant to expose. Point ``--ops-db`` and
``--kb`` at your own database and corpus and everything below works unchanged.

SEEING WHAT THE AGENTS ACTUALLY DID
-----------------------------------
``--trace`` (on by default) reads the finished workflow back out of MongoDB and prints it
in order: for every agent, the full prompt it received, the model's raw reply, the tool
calls it chose with their exact arguments, every row each tool returned, the keep-or-drop
verdict on each of those rows, and the alternatives it weighed before selecting one. That
printout is reconstructed purely from provenance -- nothing is logged on the side -- so it
is a demonstration of what the capture holds, not a separate debug channel.

WHAT MAKES THIS GENERAL
-----------------------
The agents never assemble provenance by hand. Each tool is an ordinary Python function
wearing ``@flowcept_tool``; it runs untouched and returns its normal value, while Flowcept
normalizes whatever it returned into retrieved items and publishes an ``agent_tool`` task.
Each agent turn runs inside ``retrieval_scope(...)``, and the ``DecisionCapture`` created
in that scope picks up the retrievals on its own -- no ``retrievals=`` argument, no
threading of provenance objects through the agent's call stack.

Any MAS can adopt this by decorating its existing tools and opening one scope per agent
turn. ``FlowceptTool`` does the same for a LangChain ``BaseTool``.

WHAT GETS RECORDED
------------------
- One ``agent_tool`` task per tool call: the tool, the arguments the agent wrote, and
  *every* item it returned, unfiltered. That full result set is the ground truth.
- One ``ai_model_invocation`` task per model call, tool-selection calls included, with the
  complete prompt and reply.
- One ``decision`` task per agent, holding ``retrievals``, ``evidence_uses`` (the verdict
  and reason for each retrieved item), and ``candidates`` / ``assessments`` /
  ``selected_candidate_ids``. ``custom_metadata.grounding`` summarizes retrieved vs kept.

HOW TO RUN
----------
Needs Ollama with a tool-capable model, plus Redis and MongoDB for persistence::

    ollama pull qwen3:4b
    $env:FLOWCEPT_SETTINGS_PATH = "agent_sandbox\settings.yaml"
    .\.venv\Scripts\python.exe examples\local_llm_tool_grounded_example.py --model qwen3:4b

Add ``--web-search`` for a live DuckDuckGo search alongside the local tools, and
``--no-trace`` for just the summary.

Every run is saved automatically to ``runs/<workflow_id>.txt`` (the terminal output
verbatim) and ``runs/<workflow_id>.json`` (structured: per-agent decisions plus every
captured task, prompts and retrieved items included). The workflow id is only known once
the run starts, so the names are derived rather than asked for. Use ``--out-dir`` to put
them elsewhere, ``--out-txt``/``--out-json`` for exact paths, or ``--no-save`` for neither.
"""

import argparse
import json
import re
import sqlite3
import sys
import textwrap
from contextvars import ContextVar
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from flowcept import DecisionCapture, Flowcept, FlowceptTask, flowcept_tool, retrieval_scope
from flowcept.configs import AGENT, AGENT_API_KEY
from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM

INCIDENT = (
    "At 09:10, customers began reporting that the online checkout accepts orders but "
    "does not send confirmations. The order API is healthy, the notification queue "
    "grew from 20 to 18,000 messages, and yesterday's release changed the email worker "
    "configuration. Payment processing must not be interrupted."
)

COMMANDER_CANDIDATES = ("execute", "modify", "reject")

MAX_TOKENS = 2400
TOP_K = 5
DATA_DIR = Path(__file__).parent / "data"

# Every run is saved here by default, named after its workflow id. Resolved from this
# file rather than the working directory, so runs land in one place no matter where the
# command was issued.
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "runs"

TABLE_SCHEMA = (
    "ops_records(id TEXT, kind TEXT, service TEXT, title TEXT, body TEXT, recorded_at TEXT). "
    "kind is exactly one of: 'metric', 'change', 'runbook'. "
    "service is exactly one of: 'notifications', 'email-worker', 'orders', 'payments', 'platform', 'checkout'. "
    "recorded_at is an ISO-8601 string such as '2026-09-18T09:48'."
)

# fmt: off
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how", "in",
    "into", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
})
# fmt: on

# Ambient identifiers for the tool-selection call, so gather_evidence keeps a small
# signature while its model call is still attributed to the right agent and turn.
llm_agent_id: ContextVar[str | None] = ContextVar("llm_agent_id", default=None)
llm_workflow_id: ContextVar[str | None] = ContextVar("llm_workflow_id", default=None)
llm_parent_task_id: ContextVar[str | None] = ContextVar("llm_parent_task_id", default=None)


def _terms(query: str) -> list[str]:
    """Split a keyword query into searchable terms."""
    words = re.findall(r"[a-z0-9_]+", str(query).lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


# ---------------------------------------------------------------------------
# Tool argument schemas. These are what the model sees and fills in, so the
# docstrings and field descriptions are the tools' real interface.
# ---------------------------------------------------------------------------


class OpsDatabaseQuery(BaseModel):
    """Query the operations store with SQL for metrics, change-log entries, and runbooks."""

    sql: str = Field(
        description=(
            "One read-only SQLite SELECT statement, no semicolon, against the table "
            f"{TABLE_SCHEMA} Use only the literal kind and service values listed above. "
            "SQLite has no DATE_SUB, CURDATE, NOW or INTERVAL; compare recorded_at as text "
            "or omit time filters entirely. Prefer selecting id, kind, service, title and body, "
            "and prefer LIKE over exact matching on title and body. Example: "
            "SELECT id, kind, service, title, body FROM ops_records "
            "WHERE kind = 'metric' AND service IN ('notifications', 'email-worker')"
        )
    )


class KnowledgeBaseQuery(BaseModel):
    """Search the incident knowledge base of engineering documents by keyword."""

    keywords: str = Field(description="Space-separated search keywords, no punctuation or sentences.")


class WebSearchQuery(BaseModel):
    """Search the public web for background on a failure signature."""

    search_terms: str = Field(description="A short web search query.")


# ---------------------------------------------------------------------------
# Tool implementations. Ordinary functions that really execute; @flowcept_tool
# records the call and its results and passes the return value through untouched.
# ---------------------------------------------------------------------------


def open_ops_database(sql_path: Path) -> sqlite3.Connection:
    """Build the in-memory operations database the tools query."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(sql_path.read_text(encoding="utf-8"))
    return connection


def make_tools(connection: sqlite3.Connection, kb_dir: Path, enable_web: bool) -> dict:
    """Build the executable tools, each closing over its real data source."""

    @flowcept_tool(tool_name="ops_database", tool_type="database", query_arg="sql")
    def ops_database(sql: str) -> list[dict]:
        """Execute the agent's SQL against the operations store."""
        statement = sql.strip().rstrip(";").strip()
        # The model writes this string, so it is untrusted input to a database. Only a
        # single SELECT is allowed through; the refusal is returned as the tool's result,
        # which means it is captured as provenance like any other outcome.
        if not statement.lower().startswith("select") or ";" in statement:
            return [
                {
                    "id": "tool-error:rejected-sql",
                    "content": f"Refused: only a single read-only SELECT is allowed. Received: {sql!r}",
                }
            ]
        try:
            rows = [dict(row) for row in connection.execute(statement).fetchmany(TOP_K)]
        except sqlite3.Error as error:
            return [{"id": "tool-error:sql-error", "content": f"SQL error: {error}. Statement: {statement!r}"}]
        return rows

    @flowcept_tool(tool_name="knowledge_base", tool_type="vector_store", query_arg="keywords")
    def knowledge_base(keywords: str) -> list[dict]:
        """Rank knowledge-base documents by how well they match the agent's keywords."""
        terms = _terms(keywords)
        if not terms:
            return []
        results = []
        for path in sorted(kb_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            lowered = text.lower()
            hits = sum(lowered.count(term) for term in terms)
            if not hits:
                continue
            results.append(
                {
                    "id": path.name,
                    "source": str(path),
                    "title": text.splitlines()[0].lstrip("# ").strip(),
                    "content": " ".join(text.split()),
                    "score": round(hits / (len(terms) + hits), 3),
                }
            )
        results.sort(key=lambda item: (-item["score"], item["id"]))
        return results[:TOP_K]

    tools = {
        "OpsDatabaseQuery": (ops_database, "sql"),
        "KnowledgeBaseQuery": (knowledge_base, "keywords"),
    }

    if enable_web:

        @flowcept_tool(tool_name="web_search", tool_type="web_search", query_arg="search_terms")
        def web_search(search_terms: str) -> list[dict]:
            """Search the live web through DuckDuckGo's HTML endpoint."""
            import requests
            from bs4 import BeautifulSoup

            response = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": search_terms},
                headers={"User-Agent": "Mozilla/5.0 (compatible; flowcept-example/1.0)"},
                timeout=20,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for result in soup.select(".result")[:TOP_K]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if link is None:
                    continue
                results.append(
                    {
                        "url": link.get("href"),
                        "title": link.get_text(strip=True),
                        "content": snippet.get_text(strip=True) if snippet else "",
                    }
                )
            return results

        tools["WebSearchQuery"] = (web_search, "search_terms")

    return tools


SCHEMAS = {
    "OpsDatabaseQuery": OpsDatabaseQuery,
    "KnowledgeBaseQuery": KnowledgeBaseQuery,
    "WebSearchQuery": WebSearchQuery,
}


def _is_failure(results) -> bool:
    """A call that returned nothing, or only the tool's own error rows, found no evidence."""
    if not results:
        return True
    return all(str(item.get("id", "")).startswith("tool-error:") for item in results if isinstance(item, dict))


def gather_evidence(llm, tools: dict, allowed: list[str], instruction: str, context: dict) -> list[dict]:
    """Let the agent choose tools and write their arguments, then execute what it asked for.

    A query that returned nothing usable is handed back to the agent once, with the reason,
    so it can repair it -- which is what a real agent loop does and what makes a rejected
    SQL statement a recoverable step rather than a dead end. Every attempt, including the
    failed one, is captured as its own ``agent_tool`` task.

    Returns one entry per executed call, for the console summary only; the authoritative
    record is the tasks those calls published.
    """
    planner = FlowceptLLM(
        llm.bind_tools([SCHEMAS[name] for name in allowed]),
        agent_id=llm_agent_id.get(),
        workflow_id=llm_workflow_id.get(),
        parent_task_id=llm_parent_task_id.get(),
        return_response_object=True,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Call the tools you need to gather evidence for the task. Fill each tool's arguments "
                "in that tool's own query language. Do not answer the task itself yet."
            ),
        },
        {
            "role": "user",
            "content": f"{instruction}\n\nCONTEXT:\n{json.dumps(context, indent=2, default=str)}",
        },
    ]

    executed: list[dict] = []
    for attempt in range(2):
        response = planner.invoke(messages)
        calls = getattr(response, "tool_calls", None) or []
        failures = []

        for call in calls:
            name = call.get("name")
            if name not in tools or name not in allowed:
                continue
            function, argument = tools[name]
            value = call.get("args", {}).get(argument)
            if not value:
                continue
            results = function(**{argument: value})
            entry = {"tool": name, argument: value, "returned": len(results or [])}
            if _is_failure(results):
                entry["failed"] = True
                failures.append(f"{name}({argument}={value!r}) -> {json.dumps(results, default=str)[:400]}")
            executed.append(entry)

        successful = [entry for entry in executed if not entry.get("failed")]
        if successful or attempt == 1:
            break

        # Nothing usable came back. Give the agent the failures verbatim and one more turn.
        messages.append(
            {
                "role": "user",
                "content": (
                    "Those tool calls returned no usable evidence:\n"
                    + "\n".join(failures if failures else ["no tool was called"])
                    + "\nCall the tools again with corrected arguments."
                ),
            }
        )

    if not [entry for entry in executed if not entry.get("failed")]:
        # Still nothing. Retrieving nothing at all would leave the decision ungrounded and
        # hide the failure, so fall back to a keyword search and record that it was one.
        fallback = tools.get("KnowledgeBaseQuery")
        if fallback and "KnowledgeBaseQuery" in allowed:
            function, argument = fallback
            terms = " ".join(_terms(instruction)[:8])
            results = function(**{argument: terms})
            executed.append({"tool": "KnowledgeBaseQuery", argument: terms, "returned": len(results), "fallback": True})
    return executed


# ---------------------------------------------------------------------------
# Decision plumbing
# ---------------------------------------------------------------------------


def _decision_context(agent_role: str, question: str, model_name: str, fixed: tuple[str, ...] | None) -> dict:
    """Build the decision context DecisionCapture turns into the model's system prompt."""
    context = {
        "question": question,
        "agent_role": f"the {agent_role} in a production incident-response team",
        "model": model_name,
        "instructions": [
            "Base your answer only on the retrieved evidence you marked as used, and state uncertainties.",
            "Put a one-line description of each alternative in the candidate's content field.",
            "Keep every explanation to one short sentence.",
        ],
    }
    if fixed:
        context["required_candidate_ids"] = list(fixed)
        context["instructions"].append(
            f"Return exactly the candidates {list(fixed)} and nothing else, and assess every one of them."
        )
    else:
        context["instructions"].append(
            "Return at least two genuinely different candidates and assess every one of them."
        )
    return context


def _decision_request(instruction: str, evidence: dict, complaint: str | None = None) -> str:
    """Build the user message for one decision, optionally restating a rejected attempt."""
    request = f"{instruction}\n\nCONTEXT:\n{json.dumps(evidence, indent=2, default=str)}"
    if complaint:
        request = f"{request}\n\nA previous attempt was rejected: {complaint}\nSatisfy that requirement this time."
    return request


def _describe(content) -> str:
    """Render a candidate's content as one readable line."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for field in ("summary", "description", "title", "name", "strategy"):
            value = content.get(field)
            if isinstance(value, str) and value:
                return value.strip()
    return json.dumps(content, default=str)


def _check_record(record, agent_id: str, fixed: tuple[str, ...] | None) -> None:
    """Apply this example's requirements on top of the contract DecisionCapture enforced."""
    returned = {candidate.candidate_id for candidate in record.candidates}
    if fixed:
        missing = [option for option in fixed if option not in returned]
        unexpected = sorted(returned - set(fixed))
        if missing or unexpected:
            raise ValueError(
                f"{agent_id} must return exactly the candidates {list(fixed)}; "
                f"missing={missing} unexpected={unexpected}"
            )
    elif len(returned) < 2:
        raise ValueError(f"{agent_id} returned {len(returned)} candidate(s); at least 2 are required")

    if len(record.selected_candidate_ids) != 1:
        raise ValueError(
            f"{agent_id} selected {len(record.selected_candidate_ids)} candidates; this example requires exactly one"
        )


def _derive_answer(record, fixed: tuple[str, ...] | None) -> str:
    """Build the narrative passed downstream directly from the recorded decision."""
    selected = record.selected_candidate_ids[0]
    chosen = next(candidate for candidate in record.candidates if candidate.candidate_id == selected)
    assessment = next(item for item in record.assessments if item.candidate_id == selected)
    lines = []
    if fixed:
        lines.append(f"DECISION: {selected.upper()}")
    lines.append(f"{_describe(chosen.content)} (confidence {assessment.score:g})")
    lines.append(assessment.explanation)
    kept = [use.item_id for use in record.evidence_uses if use.used]
    if kept:
        lines.append("Grounded in: " + ", ".join(kept))
    return "\n".join(lines)


def run_agent(
    agent_id: str,
    agent_role: str,
    instruction: str,
    context_for_model: dict,
    tools: dict,
    allowed_tools: list[str],
    input_entity_ids: list[str],
    output_entity_id: str,
    model_name: str,
    workflow_id: str,
    decision_type: str,
    decision_question: str,
    parent_task_id: str | None = None,
    fixed_candidates: tuple[str, ...] | None = None,
):
    """Run one agent: choose tools, execute them, then decide from what they returned."""
    with FlowceptTask(
        # No subtype on the agent's own span: `agent_tool` means a real tool call, so
        # tagging the agent with it too would make the genuine tool tasks in this workflow
        # indistinguishable from their caller in the UI and in queries.
        activity_id=agent_id,
        agent_id=agent_id,
        parent_task_id=parent_task_id,
        used={
            "agent_role": agent_role,
            "context": context_for_model,
            "input_entity_ids": input_entity_ids,
        },
        capture_telemetry=False,
    ) as agent_task:
        # reasoning_effort="none" matters: qwen3 and other reasoning models otherwise emit
        # a long <think> deliberation before answering, which costs minutes per call.
        model = ChatOpenAI(
            api_key=AGENT_API_KEY,
            base_url=AGENT["llm_server_url"],
            model=model_name,
            temperature=0,
            reasoning_effort="none",
            max_tokens=MAX_TOKENS,
        )

        # Everything captured in this scope belongs to this agent turn: the tools inherit
        # the attribution, and the DecisionCapture below collects their retrievals without
        # being told about them.
        with retrieval_scope(agent_id=agent_id, workflow_id=workflow_id, parent_task_id=agent_task.get_id()):
            executed = []
            if allowed_tools:
                tokens = [
                    llm_agent_id.set(agent_id),
                    llm_workflow_id.set(workflow_id),
                    llm_parent_task_id.set(agent_task.get_id()),
                ]
                try:
                    executed = gather_evidence(model, tools, allowed_tools, instruction, context_for_model)
                finally:
                    llm_agent_id.reset(tokens[0])
                    llm_workflow_id.reset(tokens[1])
                    llm_parent_task_id.reset(tokens[2])

            context = _decision_context(agent_role, decision_question, model_name, fixed_candidates)

            # A capture holds one decision, so a rejected attempt needs a fresh one. Each
            # fresh capture re-collects the same retrievals from this scope, so the tools
            # are not called again. The contract's own validation errors are ValueErrors
            # too, so a contradictory evidence verdict is retried the same way.
            record = None
            complaint = None
            for attempt in range(2):
                capture = DecisionCapture(
                    decision_type=decision_type,
                    context=context,
                    agent_id=agent_id,
                    workflow_id=workflow_id,
                    parent_task_id=agent_task.get_id(),
                    input_entity_ids=input_entity_ids,
                    output_entity_ids=[output_entity_id],
                    llm=model,
                )
                try:
                    record = capture.invoke(_decision_request(instruction, context_for_model, complaint))
                    _check_record(record, agent_id, fixed_candidates)
                    break
                except ValueError as error:
                    if attempt == 1:
                        raise
                    complaint = str(error)

        answer = _derive_answer(record, fixed_candidates)
        agent_task.end(
            generated={
                "response": answer,
                "tool_calls": executed,
                "selected_candidate_id": record.selected_candidate_ids[0],
                "grounding": record.grounding_summary(),
                "output_entity_ids": [output_entity_id],
            }
        )
    return answer, agent_task.get_id(), record, executed


def run_simulation(model_name: str, ops_db: Path, kb_dir: Path, enable_web: bool):
    """Run four collaborating local-model agents, three of which call real tools."""
    connection = open_ops_database(ops_db)
    tools = make_tools(connection, kb_dir, enable_web)
    investigation_tools = ["KnowledgeBaseQuery", "OpsDatabaseQuery"]
    if "WebSearchQuery" in tools:
        investigation_tools.append("WebSearchQuery")

    with Flowcept(
        workflow_name="Tool-Grounded Incident Response MAS",
        workflow_args={"incident": INCIDENT, "model": model_name, "web_search": enable_web},
        start_persistence=True,
        check_safe_stops=False,
    ) as flowcept:
        workflow_id = flowcept.current_workflow_id
        records, calls = {}, {}

        triage, triage_id, records["triage-agent"], calls["triage-agent"] = run_agent(
            "triage-agent",
            "monitoring and impact analyst",
            "Classify the severity and urgency of this incident from current service metrics.",
            {"incident": INCIDENT},
            tools,
            ["OpsDatabaseQuery"],
            ["incident:checkout-notifications"],
            "analysis:triage",
            model_name,
            workflow_id,
            decision_type="impact_triage",
            decision_question=(
                "Which severity and urgency classification best fits this incident? "
                "Consider at least one higher and one lower severity alternative."
            ),
        )

        investigation, investigation_id, records["investigation-agent"], calls["investigation-agent"] = run_agent(
            "investigation-agent",
            "root-cause investigator",
            "Develop the most likely root-cause hypothesis and at least one alternative.",
            {"incident": INCIDENT, "triage_report": triage},
            tools,
            investigation_tools,
            ["incident:checkout-notifications", "analysis:triage"],
            "analysis:investigation",
            model_name,
            workflow_id,
            decision_type="root_cause_hypothesis",
            decision_question=(
                "Which root-cause hypothesis best explains the evidence? Each competing hypothesis is a candidate."
            ),
            parent_task_id=triage_id,
        )

        plan, planning_id, records["remediation-planning-agent"], calls["remediation-planning-agent"] = run_agent(
            "remediation-planning-agent",
            "incident response planner",
            (
                "Propose a reversible remediation that restores confirmations while preserving payment "
                "processing, and say how it would be verified."
            ),
            {"incident": INCIDENT, "triage_report": triage, "investigation": investigation},
            tools,
            ["OpsDatabaseQuery", "KnowledgeBaseQuery"],
            ["analysis:triage", "analysis:investigation"],
            "plan:remediation",
            model_name,
            workflow_id,
            decision_type="response_plan_selection",
            decision_question=(
                "Which remediation strategy should be proposed? "
                "Each distinct strategy, such as rollback, configuration fix, or queue drain, is a candidate."
            ),
            parent_task_id=investigation_id,
        )

        # No tools: this agent decides from what the others concluded, so its capture finds
        # no retrievals in its scope and uses the plain decision contract.
        final, _, records["incident-commander-agent"], calls["incident-commander-agent"] = run_agent(
            "incident-commander-agent",
            "incident commander",
            (
                "Make the final operational decision on the proposed remediation, and explain how the "
                "upstream findings support it."
            ),
            {
                "incident": INCIDENT,
                "triage_report": triage,
                "investigation": investigation,
                "proposed_plan": plan,
            },
            tools,
            [],
            ["analysis:investigation", "plan:remediation"],
            "decision:incident-commander",
            model_name,
            workflow_id,
            decision_type="final_operational_decision",
            decision_question="Should the proposed remediation be executed, modified, or rejected?",
            parent_task_id=planning_id,
            fixed_candidates=COMMANDER_CANDIDATES,
        )

    connection.close()
    return workflow_id, final, records, calls


# ---------------------------------------------------------------------------
# Printing the captured trace back out of MongoDB
# ---------------------------------------------------------------------------


class _Tee:
    """Mirror everything printed to the terminal into a file.

    Teeing rather than re-rendering means the saved text is exactly what was shown, so the
    two can never drift apart.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text):
        """Write to both the terminal and the file."""
        self._stream.write(text)
        self._handle.write(text)
        return len(text)

    def flush(self):
        """Flush both."""
        self._stream.flush()
        self._handle.flush()


def build_report(workflow_id: str, final_decision: str, records: dict, calls: dict, tasks: list[dict]) -> dict:
    """Assemble the run as structured data, for querying rather than reading.

    Holds both the per-agent view and the raw captured tasks, so the file stands on its own
    if the workflow is later deleted from MongoDB.
    """
    return {
        "workflow_id": workflow_id,
        "final_decision": final_decision,
        "agents": [
            {
                "agent_id": agent_id,
                "tool_calls": calls.get(agent_id, []),
                "grounding": record.grounding_summary(),
                "decision": record.to_dict(),
            }
            for agent_id, record in records.items()
        ],
        # Exactly what was persisted, including prompts, replies and every retrieved item.
        "tasks": tasks,
    }


def _block(label: str, body: str, indent: str = "      ") -> str:
    """Render a multi-line field so long prompts stay readable in a terminal."""
    wrapped = textwrap.indent(str(body).strip(), indent)
    return f"{indent}{label}:\n{wrapped}"


def print_trace(tasks: list[dict]) -> None:
    """Print everything the agents did, reconstructed from the captured tasks alone."""
    ordered = sorted(tasks, key=lambda task: task.get("started_at") or 0)
    print("\n" + "=" * 100)
    print("FULL AGENT TRACE, READ BACK FROM MONGODB")
    print("=" * 100)

    for task in ordered:
        # MongoDB returns plain strings, but a task read straight from the in-memory buffer
        # still carries the PROV_AGENT enum, whose str() is "PROV_AGENT.AGENT_TOOL".
        raw_subtype = task.get("subtype")
        subtype = str(getattr(raw_subtype, "value", raw_subtype) or "")
        agent = task.get("agent_id") or "-"
        used = task.get("used") or {}
        generated = task.get("generated") or {}

        if subtype == "ai_model_invocation":
            print(f"\n--- [{agent}] MODEL CALL ---")
            print(_block("prompt sent to the model", used.get("prompt", "")))
            print(_block("raw model reply", generated.get("response", "")))
            usage = (task.get("custom_metadata") or {}).get("llm_usage") or {}
            print(f"      tokens: in={usage.get('input_tokens')} out={usage.get('output_tokens')}")

        elif subtype == "agent_tool":
            print(f"\n--- [{agent}] TOOL CALL: {used.get('tool_name')} ({used.get('tool_type')}) ---")
            print(_block("query the agent wrote", used.get("query", "")))
            if used.get("tool_args"):
                print(_block("arguments", json.dumps(used["tool_args"], indent=2, default=str)))
            retrieved = generated.get("retrieved") or []
            print(f"      returned {generated.get('retrieved_count')} item(s):")
            for item in retrieved:
                print(f"        - {item.get('item_id')}")
                print(_block("content", json.dumps(item.get("content"), default=str), indent="          "))

        elif subtype == "decision":
            decision = generated.get("decision") or {}
            print(f"\n--- [{agent}] DECISION: {decision.get('decision_type')} ---")
            verdicts = decision.get("evidence_uses") or []
            if verdicts:
                print("      evidence verdicts:")
                for use in verdicts:
                    mark = "KEPT   " if use.get("used") else "DROPPED"
                    print(f"        {mark} {use.get('item_id')} [{use.get('role')}] {use.get('explanation')}")
            selected = set(decision.get("selected_candidate_ids") or [])
            scores = {item.get("candidate_id"): item for item in decision.get("assessments") or []}
            print("      alternatives considered:")
            for candidate in decision.get("candidates") or []:
                cid = candidate.get("candidate_id")
                assessment = scores.get(cid, {})
                mark = "SELECTED" if cid in selected else "rejected"
                print(f"        {mark} {cid} (score {assessment.get('score')}): {_describe(candidate.get('content'))}")
                if assessment.get("explanation"):
                    print(f"                 because: {assessment['explanation']}")

        elif task.get("activity_id") and not subtype:
            print(f"\n{'=' * 100}\nAGENT TURN: {task.get('activity_id')}\n{'=' * 100}")


def report(args, workflow_id: str, final_decision: str, records: dict, calls: dict) -> list[dict]:
    """Print the run and return the captured tasks, so the caller can also save them."""
    for agent_id, record in records.items():
        summary = record.grounding_summary()
        print(f"\n=== {agent_id} ===")
        if not summary["tools_used"]:
            print("  no tools; decided from upstream agent reports")
        else:
            for call in calls[agent_id]:
                detail = {key: value for key, value in call.items() if key != "tool"}
                print(f"  called {call['tool']} with {json.dumps(detail, default=str)}")
            print(f"  retrieved {summary['retrieved_count']}, kept {len(summary['kept_item_ids'])}")
            for use in record.evidence_uses:
                verdict = "KEPT" if use.used else "DROPPED"
                print(f"    {verdict:7} {use.item_id} [{use.role}] {use.explanation}")
        print(f"  selected: {record.selected_candidate_ids[0]}")

    print("\nFinal incident-commander decision:\n")
    print(final_decision)
    print(f"\nWorkflow ID: {workflow_id}")

    # Read back once and reuse for both the printed trace and the JSON file.
    tasks = Flowcept.db.task_query(filter={"workflow_id": workflow_id}) or []
    if args.trace:
        print_trace(tasks)
    return tasks


def main() -> None:
    """Run the tool-grounded MAS and show what each agent searched for, kept, and decided."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b", help="Tool-capable model installed in Ollama.")
    parser.add_argument("--ops-db", type=Path, default=DATA_DIR / "incident_ops.sql", help="SQL seed for the ops DB.")
    parser.add_argument("--kb", type=Path, default=DATA_DIR / "kb", help="Directory of knowledge-base documents.")
    parser.add_argument("--web-search", action="store_true", help="Also offer the investigator a live web search.")
    parser.add_argument("--no-trace", dest="trace", action="store_false", help="Print only the summary.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory runs are saved in, named after the workflow id (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument("--out-txt", type=Path, help="Write the terminal output to this exact path instead.")
    parser.add_argument("--out-json", type=Path, help="Write the structured run to this exact path instead.")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not write the run to disk.")
    args = parser.parse_args()

    workflow_id, final_decision, records, calls = run_simulation(args.model, args.ops_db, args.kb, args.web_search)

    # The workflow id only exists once the run has started, so the default filenames are
    # derived here rather than asked for up front.
    out_txt = args.out_txt
    out_json = args.out_json
    if args.save:
        out_txt = out_txt or args.out_dir / f"{workflow_id}.txt"
        out_json = out_json or args.out_dir / f"{workflow_id}.json"

    if out_txt:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        # utf-8 explicitly: the trace carries whatever the tools and the model produced,
        # which the console's default codepage may not be able to encode.
        with out_txt.open("w", encoding="utf-8") as handle:
            original = sys.stdout
            sys.stdout = _Tee(original, handle)
            try:
                tasks = report(args, workflow_id, final_decision, records, calls)
            finally:
                sys.stdout = original
        print(f"Saved terminal output to {out_txt}")
    else:
        tasks = report(args, workflow_id, final_decision, records, calls)

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = build_report(workflow_id, final_decision, records, calls, tasks)
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Saved structured run to {out_json}")


if __name__ == "__main__":
    main()
