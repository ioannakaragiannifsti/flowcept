# `flowcept.instrumentation`

Explicit provenance capture APIs used directly in user code.

## Key Files

- `flowcept_decorator.py`: `@flowcept` workflow decorator.
- `flowcept_task.py`: `FlowceptTask` and task decorators.
- `flowcept_loop.py`: `FlowceptLoop` and lightweight loop capture.
- `flowcept_torch.py`: PyTorch module, epoch, batch, and child-layer capture.
- `flowcept_agent_task.py`: agent-aware task wrapper.
- `decision_provenance.py`: manual and LLM-assisted `DecisionCapture` context manager.
- `task_capture.py`: lower-level task capture helpers.

## Choosing An API

- Use `@flowcept_task` for simple function-level tasks.
- Use `@flowcept` for a top-level workflow function.
- Use `with Flowcept():` when the workflow spans multiple steps or files.
- Use `FlowceptTask` when decorators cannot express required fields.
- Use `FlowceptLoop` for loop iterations.
- Use `DecisionCapture` for explicit agent decisions that must record alternatives, assessments, and selections.
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

## Extension Rules

- Instrument coarsely by default; tight loops and distributed workers can amplify overhead.
- Preserve `used`, `generated`, `activity_id`, `workflow_id`, and status semantics.
- Keep new instrumentation compatible with the existing interceptor path.
- Tests usually belong in `tests/instrumentation_tests/`.
