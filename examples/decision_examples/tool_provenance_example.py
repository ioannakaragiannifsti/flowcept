"""Show exactly what a captured tool call and a grounded decision look like.

No LLM and no running services are needed by default: the tool results and the
keep/drop verdicts are supplied by hand, so the example prints the same task
documents that would otherwise be produced by a model and written to MongoDB.

    python examples/tool_provenance_example.py

With Redis and MongoDB up (``make services``), add ``--persist`` to send the same
tasks through the real pipeline and read them back out of MongoDB:

    python examples/tool_provenance_example.py --persist
"""

import argparse
import json

from flowcept import DecisionCapture, Flowcept, ToolCapture

WEB_RESULTS = [
    {
        "url": "https://status.example-mq.com/incidents/2211",
        "snippet": "A bad SMTP credential in the consumer config causes silent ack failures and queue growth.",
    },
    {
        "url": "https://blog.example.com/black-friday-scaling",
        "snippet": "Capacity planning advice for seasonal checkout traffic peaks.",
    },
    {
        "url": "https://docs.example.com/email-worker/config",
        "snippet": "The email worker reads SMTP credentials at startup only; a bad value fails closed.",
    },
]


def run(persist: bool):
    """Capture one tool call and one decision grounded in what it returned."""
    agent_id = "remediation-agent"
    with Flowcept(
        workflow_name="Tool Provenance Example",
        start_persistence=persist,
        check_safe_stops=False,
    ) as flowcept:
        workflow_id = flowcept.current_workflow_id

        # 1. The tool call. Everything it returned is recorded, unfiltered: this is the
        #    ground truth the decision is later checked against.
        with ToolCapture(
            "web_search",
            tool_type="web_search",
            query="notification queue grows consumer not acking",
            agent_id=agent_id,
            workflow_id=workflow_id,
        ) as tool:
            for rank, hit in enumerate(WEB_RESULTS):
                tool.add_item(hit["url"], content=hit["snippet"], source=hit["url"], rank=rank)

        # 2. The decision. Attaching the retrieval is what makes the keep/drop verdict
        #    part of the record. A real run lets the model fill these in via invoke();
        #    here they are written by hand so the example needs no model.
        with DecisionCapture(
            decision_type="tool_grounded_remediation",
            context={"question": "Which remediation restores order confirmations?"},
            agent_id=agent_id,
            workflow_id=workflow_id,
            input_entity_ids=["incident:checkout-notifications"],
            output_entity_ids=["plan:remediation"],
            retrievals=[tool.retrieval],
        ) as decision:
            decision.use_evidence(
                WEB_RESULTS[0]["url"], used=True, role="supporting", explanation="Same failure mode as the incident"
            )
            decision.use_evidence(
                WEB_RESULTS[1]["url"], used=False, role="irrelevant", explanation="About traffic peaks, not acking"
            )
            decision.use_evidence(
                WEB_RESULTS[2]["url"], used=True, role="supporting", explanation="Confirms config is read at startup"
            )
            decision.add_candidate("rollback_config", "Revert the email worker config and restart workers")
            decision.add_candidate("drain_queue", "Drain the notification queue with the replay tool")
            decision.assess(
                "rollback_config",
                evaluator_id=agent_id,
                score_type="model_reported_confidence",
                score=0.86,
                criteria=["addresses root cause", "reversible"],
                explanation="Removes the bad credential that stopped the acks",
            )
            decision.assess(
                "drain_queue",
                evaluator_id=agent_id,
                score_type="model_reported_confidence",
                score=0.25,
                criteria=["addresses root cause"],
                explanation="Drains symptoms while the consumer is still broken",
            )
            decision.select("rollback_config")

        tool_task_id = tool.task_id
        decision_task_id = decision.task_id
        record = decision.record

        if persist:
            # Read the tasks back out of MongoDB through the same API the UI uses.
            captured = flowcept.db.task_query(filter={"workflow_id": workflow_id})
        else:
            captured = [message for message in Flowcept.buffer if message.get("type") == "task"]

    return workflow_id, tool_task_id, decision_task_id, record, captured


def main() -> None:
    """Print the captured tasks and the retrieved-versus-kept summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Send the tasks through Redis/MongoDB and query them back (requires the services).",
    )
    args = parser.parse_args()

    workflow_id, tool_task_id, decision_task_id, record, captured = run(args.persist)

    by_subtype = {}
    for task in captured or []:
        by_subtype.setdefault(str(task.get("subtype")), []).append(task)

    print("\n=== Captured task subtypes ===")
    for subtype, tasks in sorted(by_subtype.items()):
        print(f"  {subtype}: {len(tasks)}")

    tool_task = next((task for task in captured or [] if task.get("task_id") == tool_task_id), None)
    if tool_task:
        print("\n=== The agent_tool task (the tool record) ===")
        print(json.dumps({key: tool_task.get(key) for key in ("task_id", "activity_id", "subtype")}, indent=2))
        print("used:     ", json.dumps(tool_task.get("used"), indent=2, default=str))
        print("generated:", json.dumps(tool_task.get("generated"), indent=2, default=str))

    decision_task = next((task for task in captured or [] if task.get("task_id") == decision_task_id), None)
    if decision_task:
        print("\n=== The decision task's grounding metadata ===")
        print(json.dumps(decision_task.get("custom_metadata", {}).get("grounding"), indent=2, default=str))
        print("\n=== Evidence verdicts stored in the decision ===")
        for use in decision_task["generated"]["decision"]["evidence_uses"]:
            verdict = "KEPT" if use["used"] else "DROPPED"
            print(f"  {verdict:7} {use['item_id']}  [{use.get('role')}]  {use.get('explanation')}")

    print(f"\nSelected candidate: {record.selected_candidate_ids[0]}")
    print(f"Workflow ID: {workflow_id}")
    if not args.persist:
        print("\n(Nothing was written to MongoDB. Re-run with --persist and the services up to store it.)")


if __name__ == "__main__":
    main()
