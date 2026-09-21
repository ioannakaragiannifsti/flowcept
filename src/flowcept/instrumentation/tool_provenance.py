"""Capture agent tool calls and what they retrieved through Flowcept tasks.

Three ways in, in increasing order of automation:

- :class:`ToolCapture` -- a context manager, when the caller wants to add each result
  explicitly.
- :func:`flowcept_tool` -- a decorator for an existing tool function. The function runs
  untouched and its return value is passed through unchanged; whatever it returned is
  normalized into retrieved items and recorded.
- :class:`FlowceptTool` -- the same, for a LangChain ``BaseTool``.

With :func:`retrieval_scope` open, every retrieval captured inside it is collected, and a
:class:`~flowcept.instrumentation.decision_provenance.DecisionCapture` created in that
scope picks them up on its own. A multi-agent system therefore gets grounded decisions by
decorating its tools and opening one scope per agent turn, without passing retrievals
around by hand.
"""

import hashlib
import inspect
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from flowcept.commons.flowcept_dataclasses.retrieval_provenance import (
    Retrieval,
    RetrievedItem,
)
from flowcept.commons.flowcept_logger import FlowceptLogger
from flowcept.commons.vocabulary import PROV_AGENT
from flowcept.instrumentation.task_capture import FlowceptTask

# Field names tools commonly use for the identity, origin, and ranking of a result. They
# are tried in order, so a result carrying both "id" and "url" is keyed by "id".
ID_FIELDS = ("item_id", "id", "_id", "doc_id", "document_id", "uri", "url", "key", "name", "title", "path")
SOURCE_FIELDS = ("source", "url", "uri", "path", "link", "href")
# "_score" is Elasticsearch/OpenSearch; "distance" is the vector stores, where smaller is
# closer rather than better, which is why it is recorded as given and not inverted.
SCORE_FIELDS = ("score", "_score", "relevance", "relevance_score", "similarity", "rank_score", "distance")

# Keys under which search backends nest their actual result list. Unwrapping recurses,
# because responses like Elasticsearch's {"hits": {"hits": [...]}} and SPARQL's
# {"results": {"bindings": [...]}} bury the list more than one level down.
RESULT_LIST_KEYS = (
    "results",
    "items",
    "documents",
    "docs",
    "data",
    "hits",
    "rows",
    "matches",
    "bindings",
    "records",
    "nodes",
    "chunks",
    "passages",
)
_MAX_UNWRAP_DEPTH = 5

# Fields whose presence means a dict is one result rather than a mapping of many.
RECORD_HINT_FIELDS = (
    set(ID_FIELDS)
    | set(SOURCE_FIELDS)
    | set(SCORE_FIELDS)
    | {"content", "text", "body", "snippet", "metadata", "page_content", "_source", "payload"}
)

# Column-oriented backends (Chroma, Weaviate and similar) return parallel arrays rather
# than a list of records. These names map a column back onto the per-item field it holds.
COLUMN_ALIASES = {
    "ids": "id",
    "documents": "content",
    "texts": "content",
    "contents": "content",
    "metadatas": "metadata",
    "distances": "distance",
    "scores": "score",
    "sources": "source",
    "uris": "uri",
}

# Record-side result limits, off by default: provenance keeps everything a tool returned.
# What the model is shown is budgeted separately, by DecisionCapture, so a complete record
# never costs context window. Set these (or the per-tool arguments) to cap what is stored,
# for instance when a tool can return more than a MongoDB document can hold.
DEFAULT_MAX_ITEMS = 0
DEFAULT_MAX_ITEM_CHARS = 0

# MongoDB refuses documents over 16MB. A retrieval approaching that would fail to persist,
# so warn while there is still room and point at the blob store for genuinely large payloads.
LARGE_RETRIEVAL_WARNING_BYTES = 8_000_000


@dataclass
class _RetrievalScope:
    """Ambient attribution and the retrievals captured inside one agent turn."""

    agent_id: str | None = None
    workflow_id: str | None = None
    parent_task_id: str | None = None
    retrievals: list = field(default_factory=list)


_ACTIVE_SCOPE: ContextVar[_RetrievalScope | None] = ContextVar("flowcept_retrieval_scope", default=None)


@contextmanager
def retrieval_scope(
    agent_id: str | None = None,
    workflow_id: str | None = None,
    parent_task_id: str | None = None,
):
    """Collect every retrieval captured inside this block and supply tool attribution.

    Decorated tools called in the block inherit ``agent_id``, ``workflow_id``, and
    ``parent_task_id`` from here, so a tool function needs no provenance parameters of its
    own. A ``DecisionCapture`` created in the block attaches the collected retrievals
    automatically.

    Scopes are context-local, so concurrent agents in threads or asyncio tasks each keep
    their own.
    """
    scope = _RetrievalScope(agent_id=agent_id, workflow_id=workflow_id, parent_task_id=parent_task_id)
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE_SCOPE.reset(token)


def current_retrievals() -> list[Retrieval]:
    """Return the retrievals captured so far in the active scope."""
    scope = _ACTIVE_SCOPE.get()
    return list(scope.retrievals) if scope else []


def _register(retrieval: Retrieval) -> None:
    scope = _ACTIVE_SCOPE.get()
    if scope is not None:
        scope.retrievals.append(retrieval)


def _stable_id(value: Any, index: int) -> str:
    """Derive a repeatable identifier for a result that carries none of its own."""
    try:
        digest = hashlib.sha1(str(value).encode("utf-8", "replace")).hexdigest()[:12]
    except Exception:  # noqa: BLE001 - a result whose __str__ raises still must not lose its item
        return f"item-{index}"
    return f"item-{index}-{digest}"


def _first_field(record: dict, fields: tuple[str, ...]):
    for name in fields:
        value = record.get(name)
        if value not in (None, ""):
            return name, value
    return None, None


def _is_dataframe(value: Any) -> bool:
    """Detect a pandas-like table without importing pandas."""
    return hasattr(value, "to_dict") and hasattr(value, "columns") and hasattr(value, "index")


def _columns_to_records(payload: dict) -> list | None:
    """Turn parallel arrays into one record per position, or return None if not that shape.

    Chroma and friends answer with ``{"ids": [[...]], "documents": [[...]]}``: the columns
    are aligned by index, and one extra level of nesting holds the batch of queries. Zipped
    back together each position becomes a normal record.
    """
    if len(payload) < 2:
        return None

    columns = {}
    for key, value in payload.items():
        if not isinstance(value, (list, tuple)):
            return None
        values = list(value)
        # A batched response nests one list per query; a single query is the common case.
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = list(values[0])
        columns[COLUMN_ALIASES.get(key, key)] = values

    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1 or lengths == {0}:
        return None

    count = lengths.pop()
    return [{key: values[index] for key, values in columns.items()} for index in range(count)]


def _as_records(result: Any, depth: int = 0) -> list:
    """Unwrap the shapes retrieval backends return into a flat list of results.

    Handles a plain list, a response nesting its list under a key like ``hits`` or
    ``bindings`` at any depth, column-oriented arrays, a pandas table, a mapping of id to
    content, and a single object. Anything unrecognized is treated as one result rather
    than being dropped.
    """
    if result is None:
        return []
    if isinstance(result, RetrievedItem):
        return [result]
    if _is_dataframe(result):
        # One item per row, not one item holding the whole table.
        try:
            return list(result.to_dict("records"))
        except Exception:  # noqa: BLE001 - an exotic table still must not lose its results
            return [result]
    if isinstance(result, dict):
        # Column-oriented first: a vector store's {"ids": [...], "documents": [...]} also
        # contains a key named in RESULT_LIST_KEYS, and unwrapping that would hand back one
        # column instead of the zipped records.
        columns = _columns_to_records(result)
        if columns is not None:
            return columns

        if depth < _MAX_UNWRAP_DEPTH:
            for key in RESULT_LIST_KEYS:
                if key not in result:
                    continue
                nested = result[key]
                if isinstance(nested, (list, tuple)):
                    return list(nested)
                # e.g. Elasticsearch {"hits": {"hits": [...]}}, SPARQL {"results": {...}}.
                if isinstance(nested, dict):
                    unwrapped = _as_records(nested, depth + 1)
                    if unwrapped != [nested]:
                        return unwrapped

            # An envelope whose key this does not know, such as Solr's {"response": {...}}.
            # A single-entry wrapper carries no data of its own, so descend into it.
            if len(result) == 1:
                only = next(iter(result.values()))
                if isinstance(only, (dict, list, tuple)):
                    unwrapped = _as_records(only, depth + 1)
                    if unwrapped != [only]:
                        return unwrapped

        # A mapping of identifier to payload, e.g. {"doc-1": "...", "doc-2": "..."}. Only
        # when the dict carries none of the fields a single record would: otherwise a lone
        # result like {"id": ..., "content": ...} would be split into one item per field.
        if result and all(isinstance(key, str) for key in result) and not (set(result) & RECORD_HINT_FIELDS):
            return [{"item_id": key, "content": value} for key, value in result.items()]
        return [result]
    if isinstance(result, (list, tuple, set)):
        return list(result)
    return [result]


def _to_item(record: Any, index: int) -> RetrievedItem:
    """Convert one tool result into a retrieved item without losing its payload."""
    if isinstance(record, RetrievedItem):
        return record

    # LangChain documents and anything else exposing page_content.
    page_content = getattr(record, "page_content", None)
    if page_content is not None:
        metadata = dict(getattr(record, "metadata", {}) or {})
        _, identifier = _first_field(metadata, ID_FIELDS)
        _, source = _first_field(metadata, SOURCE_FIELDS)
        return RetrievedItem(
            item_id=str(identifier) if identifier is not None else _stable_id(page_content, index),
            content=page_content,
            source=str(source) if source is not None else None,
            rank=index,
            metadata=metadata,
        )

    if isinstance(record, dict):
        payload = dict(record)
        id_field, identifier = _first_field(payload, ID_FIELDS)
        source_field, source = _first_field(payload, SOURCE_FIELDS)
        score_field, score = _first_field(payload, SCORE_FIELDS)
        # The id stays in the content too: dropping it would make the recorded result
        # differ from what the tool actually returned. `_source` is Elasticsearch's
        # document body, and `page_content` the LangChain convention.
        content = payload.get("content", payload.get("_source", payload.get("page_content", payload)))
        try:
            score_value = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_value = None
        return RetrievedItem(
            item_id=str(identifier) if identifier is not None else _stable_id(payload, index),
            content=content,
            source=str(source) if source is not None else None,
            rank=index,
            score=score_value,
            metadata={
                key: value
                for key, value in (("id_field", id_field), ("source_field", source_field), ("score_field", score_field))
                if value
            },
        )

    # Database rows and other sequences of columns.
    if isinstance(record, (list, tuple)) and not isinstance(record, str):
        values = list(record)
        return RetrievedItem(
            item_id=str(values[0]) if values else _stable_id(values, index),
            content=values,
            rank=index,
        )

    return RetrievedItem(item_id=_stable_id(record, index), content=record, rank=index)


def normalize_retrieved_items(result: Any) -> list[RetrievedItem]:
    """Turn whatever a tool returned into retrieved items.

    Handles lists, dicts wrapping a result list, id-to-content mappings, database rows,
    LangChain documents, and plain scalars. A result that carries no usable identifier
    gets one derived from its content, so items stay addressable by the evidence verdicts
    a decision reports against them.
    """
    items = [_to_item(record, index) for index, record in enumerate(_as_records(result))]

    # Identifiers must be unique inside a retrieval; two identical results would otherwise
    # be silently merged into one.
    seen: dict[str, int] = {}
    for item in items:
        if item.item_id in seen:
            seen[item.item_id] += 1
            item.item_id = f"{item.item_id}#{seen[item.item_id]}"
        else:
            seen[item.item_id] = 0
    return items


def materialize(result: Any) -> Any:
    """Turn a one-shot iterator into a list so it can be both recorded and returned.

    A generator can only be consumed once. Recording it without materializing would either
    exhaust it behind the caller's back or store the generator object itself, so it becomes
    a list here and the caller receives that list instead.
    """
    if isinstance(result, (str, bytes, dict, list, tuple, set)) or result is None:
        return result
    if isinstance(result, Iterator) or (hasattr(result, "__iter__") and not hasattr(result, "__len__")):
        return list(result)
    return result


def _apply_limits(items: list[RetrievedItem], max_items: int, max_item_chars: int) -> tuple[list, dict]:
    """Cap how much of a result set is recorded, and say so when capping happened."""
    truncation: dict = {}

    if max_items and len(items) > max_items:
        truncation["dropped_items"] = len(items) - max_items
        truncation["max_items"] = max_items
        truncation["original_count"] = len(items)
        items = items[:max_items]

    if max_item_chars:
        shortened = []
        for item in items:
            try:
                rendered = item.content if isinstance(item.content, str) else json.dumps(item.content, default=str)
            except Exception as error:  # noqa: BLE001 - an unrenderable payload must not fail the capture
                FlowceptLogger().debug(f"Could not measure item {item.item_id} for truncation: {error}")
                continue
            if len(rendered) > max_item_chars:
                item.content = rendered[:max_item_chars]
                item.metadata = {
                    **(item.metadata or {}),
                    "truncated": True,
                    "original_chars": len(rendered),
                }
                shortened.append(item.item_id)
        if shortened:
            truncation["shortened_item_ids"] = shortened
            truncation["max_item_chars"] = max_item_chars

    return items, truncation


def _warn_if_oversized(tool_name: str, items: list[RetrievedItem]) -> None:
    """Warn before a result set grows past what a MongoDB document can hold."""
    try:
        size = len(json.dumps([item.to_dict() for item in items], default=str))
    except Exception:  # noqa: BLE001 - measuring must never break the capture
        return
    if size >= LARGE_RETRIEVAL_WARNING_BYTES:
        FlowceptLogger().warning(
            f"Tool '{tool_name}' returned about {size / 1_000_000:.1f}MB of results. MongoDB rejects "
            "documents over 16MB, so this task may fail to persist. Either cap the capture with "
            "max_items/max_item_chars, or store the payload with Flowcept.db (GridFS-backed) and keep "
            "a reference on the item."
        )


def record_retrieval(
    retrieval: Retrieval,
    agent_id: str | None = None,
    workflow_id: str | None = None,
    parent_task_id: str | None = None,
) -> str:
    """Record a completed tool retrieval and return its Flowcept task identifier.

    The task is an ``agent_tool`` activity, so it shows up beside the agent's model
    invocations rather than as an untyped task: ``used`` carries the query, ``generated``
    carries every item the tool returned.
    """
    task = FlowceptTask(
        activity_id=retrieval.tool_name,
        subtype=PROV_AGENT.AGENT_TOOL,
        agent_id=agent_id,
        workflow_id=workflow_id,
        parent_task_id=parent_task_id,
        used={
            "tool_name": retrieval.tool_name,
            "tool_type": retrieval.tool_type,
            "query_method": retrieval.query_method,
            "query": retrieval.query,
            "tool_args": retrieval.tool_args,
        },
        custom_metadata={
            "retrieval_id": retrieval.retrieval_id,
            "tool_name": retrieval.tool_name,
            "tool_type": retrieval.tool_type,
            "query_method": retrieval.query_method,
            "schema_version": retrieval.schema_version,
        },
        capture_telemetry=False,
    )
    task.end(
        generated={
            # The full result set, before any model filtering: this is the ground truth a
            # decision's kept/dropped evidence is later compared against.
            "retrieved": [item.to_dict() for item in retrieval.items],
            "retrieved_count": len(retrieval.items),
            "retrieved_item_ids": retrieval.item_ids,
        }
    )
    return task.get_id()


class ToolCapture:
    """Capture one tool call as an ``agent_tool`` task, with everything it returned.

    Use it when results are added explicitly::

        with ToolCapture("web_search", tool_type="web_search", query=q) as tool:
            for hit in search(q):
                tool.add_item(hit["url"], content=hit["snippet"], source=hit["url"])

    ``tool.record(results)`` normalizes a tool's return value instead, and
    :func:`flowcept_tool` does that automatically for a whole function.
    """

    def __init__(
        self,
        tool_name: str,
        query,
        tool_type: str = "other",
        query_method: str | None = None,
        tool_args: dict | None = None,
        agent_id: str | None = None,
        workflow_id: str | None = None,
        parent_task_id: str | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
    ):
        scope = _ACTIVE_SCOPE.get()
        self.tool_name = tool_name
        self.query = query
        self.tool_type = tool_type
        self.query_method = query_method
        self.tool_args = tool_args or {}
        self.max_items = max_items
        self.max_item_chars = max_item_chars
        # Explicit attribution wins; otherwise inherit whatever the agent's scope set.
        self.agent_id = agent_id if agent_id is not None else (scope.agent_id if scope else None)
        self.workflow_id = workflow_id if workflow_id is not None else (scope.workflow_id if scope else None)
        self.parent_task_id = (
            parent_task_id if parent_task_id is not None else (scope.parent_task_id if scope else None)
        )
        self.items: list[RetrievedItem] = []
        self.retrieval: Retrieval | None = None
        self.task_id: str | None = None

        if scope is None and self.agent_id is None:
            # Most often this means the tool ran on a worker thread, where the scope's
            # context variable does not follow it. The capture still publishes, but it
            # would be orphaned from its agent and invisible to that agent's decision,
            # so say so instead of failing quietly.
            FlowceptLogger().warning(
                f"Tool '{tool_name}' was captured with no active retrieval_scope and no agent_id. "
                "Its retrieval will not be attached to any decision. If the tool runs on another "
                "thread, wrap the callable with flowcept.propagate_scope, or pass agent_id explicitly."
            )

    def __enter__(self):
        return self

    def add_item(
        self,
        item_id: str,
        content=None,
        source: str | None = None,
        rank: int | None = None,
        score: float | None = None,
        metadata: dict | None = None,
    ) -> RetrievedItem:
        """Record one item the tool returned."""
        item = RetrievedItem(
            item_id=item_id,
            content=content,
            source=source,
            # Default to arrival order so result ranking survives even when a caller
            # does not track it explicitly.
            rank=len(self.items) if rank is None else rank,
            score=score,
            metadata=metadata or {},
        )
        self.items.append(item)
        return item

    def record(self, result: Any) -> Any:
        """Normalize a tool's return value into retrieved items and pass it through.

        A one-shot iterator is materialized first, so the value handed back to the caller
        is the same data that was recorded rather than an exhausted generator.
        """
        result = materialize(result)
        self.items.extend(normalize_retrieved_items(result))
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # A failed tool call has no trustworthy result set; let the error surface
            # rather than recording a partial retrieval as if it were complete.
            return False
        self.finish()
        return False

    def finish(self) -> Retrieval:
        """Build the retrieval record, publish it, and register it with the active scope."""
        if self.retrieval is not None:
            raise ValueError("ToolCapture already recorded a retrieval; use a fresh capture")
        items, truncation = _apply_limits(self.items, self.max_items, self.max_item_chars)
        _warn_if_oversized(self.tool_name, items)
        self.retrieval = Retrieval(
            tool_name=self.tool_name,
            tool_type=self.tool_type,
            query_method=self.query_method,
            query=self.query,
            tool_args=self.tool_args,
            items=items,
            truncation=truncation,
        )
        self.task_id = record_retrieval(
            self.retrieval,
            agent_id=self.agent_id,
            workflow_id=self.workflow_id,
            parent_task_id=self.parent_task_id,
        )
        _register(self.retrieval)
        return self.retrieval


def _derive_query(args: tuple, kwargs: dict, query_arg: str | None) -> Any:
    """Work out what this call's query was, from however the tool takes its arguments."""
    if query_arg is not None:
        if query_arg in kwargs:
            return kwargs[query_arg]
        raise ValueError(f"Tool was called without its declared query argument {query_arg!r}")
    for name in ("query", "q", "question", "search", "text", "prompt", "sql", "statement"):
        if name in kwargs:
            return kwargs[name]
    if args:
        return args[0]
    return kwargs or None


def flowcept_tool(
    func: Callable | None = None,
    *,
    tool_name: str | None = None,
    tool_type: str = "other",
    query_method: str | None = None,
    query_arg: str | None = None,
    agent_id: str | None = None,
    workflow_id: str | None = None,
    parent_task_id: str | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
):
    """Capture every call to a tool function, including everything it returned.

    The wrapped function runs unchanged and its return value is passed through, so existing
    agent code keeps working; the results are additionally normalized into retrieved items
    and published as an ``agent_tool`` task::

        @flowcept_tool(tool_type="database")
        def search_runbooks(query: str) -> list[dict]:
            return cursor.execute(SQL, (query,)).fetchall()

    ``async def`` tools are supported: the wrapper awaits the call and records the awaited
    result, so decorating one gives back a coroutine function, as before. A tool returning
    a generator has it materialized into a list, and the caller receives that list, because
    a one-shot iterator cannot be both recorded and returned.

    Inside a :func:`retrieval_scope`, the capture inherits the agent's attribution and the
    retrieval is offered to that agent's decision automatically.

    Parameters
    ----------
    tool_name : str, optional
        Recorded tool name. Defaults to the function's name.
    tool_type : str
        One of the values in
        :data:`~flowcept.commons.flowcept_dataclasses.retrieval_provenance.TOOL_TYPES`.
    query_arg : str, optional
        Name of the keyword argument holding the query. By default the first positional
        argument, or a keyword named ``query``/``q``/``question``/``sql`` and similar, is
        used.
    agent_id, workflow_id, parent_task_id : str, optional
        Attribution for tools that run where no :func:`retrieval_scope` is visible, such as
        on a worker thread. These override the scope when both are present.
    max_items, max_item_chars : int
        Result-size limits. Pass 0 to disable either one. Whatever they cut is recorded in
        the retrieval's ``truncation``.
    """

    def decorator(target: Callable) -> Callable:
        def build_capture(args, kwargs) -> ToolCapture:
            return ToolCapture(
                tool_name or getattr(target, "__name__", "tool"),
                query=_derive_query(args, kwargs, query_arg),
                tool_type=tool_type,
                query_method=query_method,
                tool_args=kwargs,
                agent_id=agent_id,
                workflow_id=workflow_id,
                parent_task_id=parent_task_id,
                max_items=max_items,
                max_item_chars=max_item_chars,
            )

        if inspect.iscoroutinefunction(target):
            # Recording the coroutine itself would store an un-awaited object and crash on
            # serialization, so the wrapper awaits first and records what it resolved to.
            @wraps(target)
            async def async_wrapper(*args, **kwargs):
                capture = build_capture(args, kwargs)
                with capture:
                    return capture.record(await target(*args, **kwargs))

            return async_wrapper

        @wraps(target)
        def wrapper(*args, **kwargs):
            capture = build_capture(args, kwargs)
            with capture:
                return capture.record(target(*args, **kwargs))

        return wrapper

    return decorator if func is None else decorator(func)


def propagate_scope(target: Callable) -> Callable:
    """Carry the current retrieval scope into another thread.

    ``contextvars`` do not follow a call into ``threading.Thread`` or a thread pool, so a
    tool dispatched there sees no scope: its retrieval is published unattributed and never
    reaches the agent's decision. Wrapping the callable at submission time fixes that::

        pool.submit(propagate_scope(search_runbooks), query)

    The scope is read when this is called, so call it on the thread that owns the scope,
    not inside the worker. The returned callable is safe to submit more than once and from
    several workers at a time, and every retrieval they capture lands in the one scope.
    """
    scope = _ACTIVE_SCOPE.get()

    @wraps(target)
    def runner(*args, **kwargs):
        # Set rather than replay a copied Context: a Context cannot be entered twice at
        # once, which would break the moment a pool ran two of these concurrently.
        token = _ACTIVE_SCOPE.set(scope)
        try:
            return target(*args, **kwargs)
        finally:
            _ACTIVE_SCOPE.reset(token)

    return runner


class FlowceptTool:
    """Capture a LangChain tool's calls, mirroring ``FlowceptLLM`` for models.

    Wrap the tool and use the wrapper wherever the agent would have used the tool::

        wrapped = FlowceptTool(search_tool, tool_type="web_search")
        results = wrapped.invoke({"query": "queue backlog"})

    The underlying tool's return value is passed through unchanged.
    """

    def __init__(
        self,
        tool,
        tool_type: str = "other",
        tool_name: str | None = None,
        query_method: str | None = None,
        agent_id: str | None = None,
        workflow_id: str | None = None,
        parent_task_id: str | None = None,
    ):
        self.tool = tool
        self.tool_type = tool_type
        self.query_method = query_method
        self.tool_name = tool_name or getattr(tool, "name", None) or type(tool).__name__
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.parent_task_id = parent_task_id

    @property
    def name(self) -> str:
        """Expose the wrapped tool's name, as agent frameworks expect."""
        return self.tool_name

    @property
    def description(self) -> str:
        """Expose the wrapped tool's description, as agent frameworks expect."""
        return getattr(self.tool, "description", "")

    def _build_capture(self, tool_input) -> ToolCapture:
        return ToolCapture(
            self.tool_name,
            query=tool_input.get("query", tool_input) if isinstance(tool_input, dict) else tool_input,
            tool_type=self.tool_type,
            query_method=self.query_method,
            tool_args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
            agent_id=self.agent_id,
            workflow_id=self.workflow_id,
            parent_task_id=self.parent_task_id,
        )

    def _call(self, method_name: str, tool_input, **kwargs):
        capture = self._build_capture(tool_input)
        with capture:
            return capture.record(getattr(self.tool, method_name)(tool_input, **kwargs))

    async def _acall(self, method_name: str, tool_input, **kwargs):
        capture = self._build_capture(tool_input)
        with capture:
            return capture.record(await getattr(self.tool, method_name)(tool_input, **kwargs))

    def invoke(self, tool_input, **kwargs):
        """Invoke the wrapped tool, capturing the call and its results."""
        return self._call("invoke", tool_input, **kwargs)

    def run(self, tool_input, **kwargs):
        """Run the wrapped tool, capturing the call and its results."""
        return self._call("run", tool_input, **kwargs)

    async def ainvoke(self, tool_input, **kwargs):
        """Await the wrapped tool's async invoke, capturing the call and its results."""
        return await self._acall("ainvoke", tool_input, **kwargs)

    async def arun(self, tool_input, **kwargs):
        """Await the wrapped tool's async run, capturing the call and its results."""
        return await self._acall("arun", tool_input, **kwargs)

    def __call__(self, tool_input, **kwargs):
        """Call the wrapper like the tool itself."""
        return self.invoke(tool_input, **kwargs)
