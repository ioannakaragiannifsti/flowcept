"""Post-process persisted Flowcept records using message-removal interventions."""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from html import escape
from pathlib import Path

from langchain_openai import ChatOpenAI

from flowcept import Flowcept
from flowcept.configs import AGENT, AGENT_API_KEY


def build_structural_graph(tasks: list[dict]) -> dict:
    """Build agent-message production and consumption relations from Flowcept tasks."""
    nodes = {}
    edges = []
    for task in tasks:
        if not task.get("agent_id"):
            continue
        agent_id = task["agent_id"]
        nodes[agent_id] = {"id": agent_id, "type": "agent"}
        for entity_id in (task.get("generated") or {}).get("output_entity_ids", []):
            nodes[entity_id] = {"id": entity_id, "type": "message"}
            edges.append({"source": agent_id, "target": entity_id, "relation": "produced"})
        for entity_id in (task.get("used") or {}).get("input_entity_ids", []):
            nodes[entity_id] = {"id": entity_id, "type": "message"}
            edges.append({"source": entity_id, "target": agent_id, "relation": "consumedBy"})
    return {"nodes": list(nodes.values()), "edges": edges}


def visible_response(response: str) -> str:
    """Return the user-visible answer after any captured model thinking block."""
    if "</think>" in response:
        return response.rsplit("</think>", maxsplit=1)[1].strip()
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_decision_label(
    response: str,
    allowed_labels: tuple[str, ...] = ("execute", "modify", "reject"),
) -> str:
    """Extract the last explicit allowed decision label from the visible response."""
    allowed = {label.lower() for label in allowed_labels}
    matches = re.findall(
        r"\bDECISION\s*:\s*([A-Z][A-Z0-9_-]*)",
        visible_response(response),
        flags=re.IGNORECASE,
    )
    for match in reversed(matches):
        if match.lower() in allowed:
            return match.lower()
    return "unknown"


def compare_outcomes(
    baseline: str,
    counterfactual: str,
    allowed_labels: tuple[str, ...] = ("execute", "modify", "reject"),
) -> dict:
    """Separate outcome-level causal effects from explanation-text changes."""
    baseline_visible = visible_response(baseline)
    counterfactual_visible = visible_response(counterfactual)
    baseline_label = extract_decision_label(baseline_visible, allowed_labels)
    counterfactual_label = extract_decision_label(counterfactual_visible, allowed_labels)
    labels_observed = baseline_label != "unknown" and counterfactual_label != "unknown"
    outcome_changed = baseline_label != counterfactual_label if labels_observed else None
    outcome_effect = "changed" if outcome_changed else "unchanged" if outcome_changed is False else "undetermined"
    similarity = round(SequenceMatcher(None, baseline_visible, counterfactual_visible).ratio(), 3)
    explanation_change = round(1.0 - similarity, 3)
    explanation_sensitivity = (
        "high" if explanation_change >= 0.5 else "medium" if explanation_change >= 0.2 else "low"
    )
    return {
        "baseline_label": baseline_label,
        "counterfactual_label": counterfactual_label,
        "outcome_changed": outcome_changed,
        "outcome_effect": outcome_effect,
        "outcome_influence_score": 1.0 if outcome_changed else 0.0 if outcome_changed is False else None,
        "explanation_similarity": similarity,
        "explanation_change_score": explanation_change,
        "explanation_sensitivity": explanation_sensitivity,
    }


def summarize_trials(comparisons: list[dict]) -> dict:
    """Aggregate repeated interventions without conflating outcome and text changes."""
    def count_labels(field: str) -> dict[str, int]:
        counts = {}
        for item in comparisons:
            label = item[field]
            counts[label] = counts.get(label, 0) + 1
        return counts

    determinate = [item for item in comparisons if item["outcome_changed"] is not None]
    changed_count = sum(item["outcome_changed"] for item in determinate)
    if not determinate:
        causal_conclusion = "undetermined"
        outcome_change_rate = None
    elif changed_count:
        causal_conclusion = "outcome_change_observed"
        outcome_change_rate = round(changed_count / len(determinate), 3)
    else:
        causal_conclusion = "no_outcome_change_observed"
        outcome_change_rate = 0.0
    mean_explanation_change = round(
        sum(item["explanation_change_score"] for item in comparisons) / len(comparisons),
        3,
    )
    return {
        "causal_conclusion": causal_conclusion,
        "outcome_influence": (
            "observed"
            if causal_conclusion == "outcome_change_observed"
            else "not_observed"
            if causal_conclusion == "no_outcome_change_observed"
            else "undetermined"
        ),
        "outcome_change_count": changed_count,
        "determinate_trial_count": len(determinate),
        "trial_count": len(comparisons),
        "outcome_change_rate": outcome_change_rate,
        "baseline_label_counts": count_labels("baseline_label"),
        "counterfactual_label_counts": count_labels("counterfactual_label"),
        "mean_explanation_change": mean_explanation_change,
        "explanation_sensitivity": (
            "high"
            if mean_explanation_change >= 0.5
            else "medium"
            if mean_explanation_change >= 0.2
            else "low"
        ),
    }


def _find_judge_context(tasks: list[dict], decision_agent_id: str) -> tuple[dict, dict]:
    judge_tasks = [
        task
        for task in tasks
        if task.get("subtype") == "agent_tool" and task.get("agent_id") == decision_agent_id
    ]
    if not judge_tasks:
        raise ValueError(f"No agent_tool task found for decision agent {decision_agent_id!r}")
    judge_task = max(judge_tasks, key=lambda task: task["started_at"])
    invocations = [
        task
        for task in tasks
        if task.get("subtype") == "ai_model_invocation"
        and task.get("parent_task_id") == judge_task["task_id"]
    ]
    if not invocations:
        raise ValueError(f"No ai_model_invocation found below judge task {judge_task['task_id']}")
    return judge_task, max(invocations, key=lambda task: task["started_at"])


def _extract_prompt_parts(formatted_prompt: str) -> tuple[str, str]:
    system_text, user_text = formatted_prompt.removeprefix("System: ").split("\nUser: ", maxsplit=1)
    instruction = user_text.split("\n\nEvidence:\n", maxsplit=1)[0]
    return system_text, instruction


def _find_message_sources(tasks: list[dict], evidence: dict, response_field: str) -> dict:
    sources = {}
    for evidence_key, evidence_value in evidence.items():
        for task in tasks:
            generated = task.get("generated") or {}
            if task.get("subtype") == "agent_tool" and generated.get(response_field) == evidence_value:
                output_ids = generated.get("output_entity_ids") or [task["task_id"]]
                sources[evidence_key] = {
                    "agent_id": task["agent_id"],
                    "message_id": output_ids[0],
                    "message": evidence_value,
                }
                break
    return sources


def _invoke_judge(
    system_text: str,
    instruction: str,
    evidence: dict,
    model_name: str,
    max_tokens: int,
) -> str:
    model = ChatOpenAI(
        api_key=AGENT_API_KEY,
        base_url=AGENT["llm_server_url"],
        model=model_name,
        temperature=0,
        reasoning_effort="none",
        max_tokens=max_tokens,
    )
    response = model.invoke(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": f"{instruction}\n\nEvidence:\n{evidence}"},
        ]
    )
    return response.content


def _semantic_graph(sources: dict, decision_id: str) -> dict:
    nodes = [{"id": decision_id, "type": "decision"}]
    edges = []
    for source in sources.values():
        claim_id = f"claim:{source['message_id']}"
        claim_text = visible_response(source["message"]).split("\n", maxsplit=1)[0][:240]
        nodes.extend(
            [
                {"id": source["message_id"], "type": "message", "agent_id": source["agent_id"]},
                {"id": claim_id, "type": "claim", "text": claim_text},
            ]
        )
        edges.extend(
            [
                {"source": source["message_id"], "target": claim_id, "relation": "expresses"},
                {"source": claim_id, "target": decision_id, "relation": "observedSupportFor"},
            ]
        )
    return {"nodes": nodes, "edges": edges}


def analyze_workflow(
    workflow_id: str,
    decision_agent_id: str = "incident-commander-agent",
    allowed_labels: tuple[str, ...] = ("execute", "modify", "reject"),
    trials: int = 1,
    max_workers: int = 1,
    max_interventions: int | None = None,
    evidence_field: str = "evidence",
    response_field: str = "response",
) -> dict:
    """Read one Flowcept workflow and run message-removal interventions on its judge."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    tasks = Flowcept.db.task_query(filter={"workflow_id": workflow_id})
    if not tasks:
        raise ValueError(f"No persisted Flowcept tasks found for workflow {workflow_id}")
    judge_task, judge_invocation = _find_judge_context(tasks, decision_agent_id)
    evidence = judge_task["used"][evidence_field]
    sources = _find_message_sources(tasks, evidence, response_field)
    if not sources:
        raise ValueError("The judge has no consumed agent messages to remove")
    source_items = list(sources.items())[:max_interventions]

    original_decision = judge_task["generated"]["response"]
    system_text, instruction = _extract_prompt_parts(judge_invocation["used"]["prompt"])
    metadata = judge_invocation.get("custom_metadata") or {}
    model_name = metadata.get("model_name") or AGENT["model"]
    max_tokens = (metadata.get("model_parameters") or {}).get("max_tokens", 350)
    reproduced_baselines = [
        _invoke_judge(system_text, instruction, evidence, model_name, max_tokens)
        for _ in range(trials)
    ]

    counterfactuals = []
    causal_edges = []
    decision_ids = judge_task["generated"].get("output_entity_ids") or [judge_task["task_id"]]
    decision_id = decision_ids[0]
    jobs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for trial_index, reproduced_baseline in enumerate(reproduced_baselines):
            for evidence_key, source in source_items:
                reduced_evidence = {key: value for key, value in evidence.items() if key != evidence_key}
                future = executor.submit(
                    _invoke_judge,
                    system_text,
                    instruction,
                    reduced_evidence,
                    model_name,
                    max_tokens,
                )
                jobs.append((future, trial_index, reproduced_baseline, evidence_key, source))

        trial_results = {evidence_key: [] for evidence_key, _ in source_items}
        for future, trial_index, reproduced_baseline, evidence_key, source in jobs:
            counterfactual_response = future.result()
            comparison = compare_outcomes(reproduced_baseline, counterfactual_response, allowed_labels)
            trial_results[evidence_key].append(
                {
                    "trial": trial_index + 1,
                    "baseline_label": comparison["baseline_label"],
                    "counterfactual_label": comparison["counterfactual_label"],
                    "raw_response": counterfactual_response,
                    "visible_response": visible_response(counterfactual_response),
                    **comparison,
                }
            )

    for evidence_key, source in source_items:
        intervention_trials = trial_results[evidence_key]
        summary = summarize_trials(intervention_trials)
        counterfactual = {
            "intervention": f"remove {source['message_id']}",
            "removed_evidence_key": evidence_key,
            "removed_agent_id": source["agent_id"],
            "removed_message_id": source["message_id"],
            **summary,
            "trials": intervention_trials,
        }
        counterfactuals.append(counterfactual)
        causal_edges.append(
            {
                "source": source["message_id"],
                "target": decision_id,
                "relation": "directRemovalEffectOn",
                **summary,
            }
        )

    baseline_trials = [
        compare_outcomes(original_decision, reproduced, allowed_labels)
        for reproduced in reproduced_baselines
    ]
    baseline_summary = summarize_trials(baseline_trials)

    return {
        "workflow_id": workflow_id,
        "decision_agent_id": decision_agent_id,
        "model": model_name,
        "allowed_outcome_labels": list(allowed_labels),
        "trial_count": trials,
        "max_workers": max_workers,
        "analyzed_intervention_count": len(source_items),
        "available_intervention_count": len(sources),
        "method": "repeated direct judge-input message removal with fixed model parameters",
        "limitations": [
            "This estimates the direct effect of each observed message on the judge, conditional on other messages.",
            "It does not expose hidden chain-of-thought or estimate indirect effects through downstream agents.",
            "No observed outcome change means this run found no necessity; it does not prove zero contribution.",
        ],
        "original_decision": {
            "label": extract_decision_label(original_decision, allowed_labels),
            "visible_response": visible_response(original_decision),
            "raw_response": original_decision,
        },
        "reproduced_baselines": [
            {
                "trial": index + 1,
                "label": extract_decision_label(response, allowed_labels),
                "visible_response": visible_response(response),
                "raw_response": response,
            }
            for index, response in enumerate(reproduced_baselines)
        ],
        "baseline_reproduction": {**baseline_summary, "trials": baseline_trials},
        "observed_judge_inputs": list(sources.values()),
        "structural_provenance": build_structural_graph(tasks),
        "semantic_provenance": _semantic_graph(sources, decision_id),
        "causal_provenance": {"edges": causal_edges, "counterfactuals": counterfactuals},
    }


def _render_report(result: dict) -> str:
    def format_rate(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    def format_counts(counts: dict[str, int]) -> str:
        return ", ".join(f"{label}: {count}" for label, count in counts.items())

    rows = "".join(
        "<tr>"
        f"<td>{escape(item['removed_agent_id'])}</td>"
        f"<td>{escape(item['removed_message_id'])}</td>"
        f"<td>{escape(format_counts(item['baseline_label_counts']))}</td>"
        f"<td>{escape(format_counts(item['counterfactual_label_counts']))}</td>"
        f"<td><strong>{escape(item['outcome_influence'])}</strong></td>"
        f"<td>{item['outcome_change_count']} / {item['determinate_trial_count']}</td>"
        f"<td>{format_rate(item['outcome_change_rate'])}</td>"
        f"<td>{item['mean_explanation_change']:.3f}</td>"
        f"<td>{escape(item['explanation_sensitivity'])}</td>"
        "</tr>"
        for item in result["causal_provenance"]["counterfactuals"]
    )
    observed_inputs = "".join(
        "<tr>"
        f"<td>{escape(item['agent_id'])}</td>"
        f"<td>{escape(item['message_id'])}</td>"
        f"<td>{escape(visible_response(item['message'])[:300])}</td>"
        "</tr>"
        for item in result["observed_judge_inputs"]
    )
    structural = "".join(
        f"<li><code>{escape(edge['source'])}</code> {escape(edge['relation'])} "
        f"<code>{escape(edge['target'])}</code></li>"
        for edge in result["structural_provenance"]["edges"]
    )
    semantic = "".join(
        f"<li><code>{escape(edge['source'])}</code> {escape(edge['relation'])} "
        f"<code>{escape(edge['target'])}</code></li>"
        for edge in result["semantic_provenance"]["edges"]
    )
    intervention_count = result["analyzed_intervention_count"]
    available_count = result["available_intervention_count"]
    reproduced_labels = ", ".join(item["label"] for item in result["reproduced_baselines"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Causal decision provenance</title>
<style>
body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}}
h1,h2{{color:#153a5b}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccd6e0;padding:9px;text-align:left}}
th{{background:#eef4f8}}
code{{background:#eef2f5;padding:2px 5px}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:14px}}
</style></head><body>
<h1>Causal decision provenance</h1>
<p><strong>Workflow:</strong> {escape(result['workflow_id'])}<br>
<strong>Judge:</strong> {escape(result['decision_agent_id'])}<br>
<strong>Method:</strong> {escape(result['method'])}<br>
<strong>Trials per intervention:</strong> {result['trial_count']}<br>
<strong>Interventions analyzed:</strong> {intervention_count} of {available_count}</p>
<p><strong>Original outcome:</strong> {escape(result['original_decision']['label'])}<br>
<strong>Reproduced full-input outcomes:</strong> {escape(reproduced_labels)}<br>
<strong>Baseline reproduction:</strong> {escape(result['baseline_reproduction']['causal_conclusion'])}</p>
<h2>What the judge observed</h2>
<p>These persisted agent messages were present in the judge's input.</p>
<table><thead><tr><th>Source agent</th><th>Message</th><th>Visible excerpt</th></tr></thead>
<tbody>{observed_inputs}</tbody></table>
<h2>Removal-based influence</h2>
<p><strong>Outcome change observed</strong> means removing the message changed the explicit decision label
in at least one trial.
<strong>No outcome change observed</strong> means the label stayed stable in these trials;
this does not prove the message was irrelevant.
<strong>Explanation change</strong> measures wording/content sensitivity separately.</p>
<table><thead><tr>
<th>Removed agent</th><th>Removed message</th><th>Full-input outcomes</th>
<th>Without-message outcomes</th><th>Outcome influence</th>
<th>Outcome changes</th><th>Change rate</th>
<th>Mean explanation change</th><th>Explanation sensitivity</th>
</tr></thead><tbody>{rows}</tbody></table>
<h2>Observed decision</h2><pre>{escape(result['original_decision']['visible_response'])}</pre>
<h2>Structural provenance</h2><ul>{structural}</ul>
<h2>Semantic provenance</h2><ul>{semantic}</ul>
<h2>Method limits</h2><ul>{''.join(f'<li>{escape(item)}</li>' for item in result['limitations'])}</ul>
</body></html>"""


def write_result(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable analysis and a standalone human-readable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis.json"
    html_path = output_dir / "report.html"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    html_path.write_text(_render_report(result), encoding="utf-8")
    return json_path, html_path


def main() -> None:
    """Run causal analysis for one persisted Flowcept workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--decision-agent", default="incident-commander-agent")
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=["execute", "modify", "reject"],
        help="Allowed explicit DECISION labels for this MAS.",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-interventions", type=int)
    parser.add_argument("--evidence-field", default="evidence")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    result = analyze_workflow(
        workflow_id=args.workflow_id,
        decision_agent_id=args.decision_agent,
        allowed_labels=tuple(args.outcomes),
        trials=args.trials,
        max_workers=args.max_workers,
        max_interventions=args.max_interventions,
        evidence_field=args.evidence_field,
        response_field=args.response_field,
    )
    output_dir = args.output_dir or Path("agent_sandbox/causal_decision_provenance") / args.workflow_id
    json_path, html_path = write_result(result, output_dir)
    print(f"Causal analysis: {json_path.resolve()}")
    print(f"Readable report: {html_path.resolve()}")
    for item in result["causal_provenance"]["counterfactuals"]:
        print(
            f"{item['removed_agent_id']}: {item['causal_conclusion']} "
            f"({item['outcome_change_count']}/{item['determinate_trial_count']} outcome changes; "
            f"mean explanation change {item['mean_explanation_change']:.3f})"
        )


if __name__ == "__main__":
    main()
