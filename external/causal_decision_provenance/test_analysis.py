"""Behavior tests for the external causal decision-provenance analyzer."""

from external.causal_decision_provenance.analysis import (
    build_structural_graph,
    compare_outcomes,
    extract_decision_label,
    summarize_trials,
)
from flowcept import Flowcept, FlowceptTask
from flowcept.commons.vocabulary import PROV_AGENT


def test_structural_graph_connects_agent_message_and_judge():
    """Build structural relations from real Flowcept task records."""
    with Flowcept(start_persistence=False, check_safe_stops=False) as flowcept:
        with FlowceptTask(
            activity_id="analyst",
            agent_id="analyst",
            subtype=PROV_AGENT.AGENT_TOOL,
            used={"input_entity_ids": ["request:1"]},
        ) as analyst:
            analyst.end(generated={"response": "Use the database.", "output_entity_ids": ["message:analyst"]})

        with FlowceptTask(
            activity_id="judge",
            agent_id="judge",
            subtype=PROV_AGENT.AGENT_TOOL,
            parent_task_id=analyst.get_id(),
            used={"input_entity_ids": ["message:analyst"]},
        ) as judge:
            judge.end(generated={"response": "DECISION: DATABASE", "output_entity_ids": ["decision:1"]})

        records = list(flowcept.get_buffer())

    for record in records:
        record.pop("type", None)

    graph = build_structural_graph(records)

    assert {"source": "analyst", "target": "message:analyst", "relation": "produced"} in graph["edges"]
    assert {"source": "message:analyst", "target": "judge", "relation": "consumedBy"} in graph["edges"]


def test_counterfactual_comparison_detects_changed_decision():
    """A changed explicit decision receives maximum removal influence."""
    baseline = "DECISION: EXECUTE\nProceed with the guarded rollback."
    counterfactual = "DECISION: REJECT\nThe evidence is insufficient."

    comparison = compare_outcomes(baseline, counterfactual)

    assert extract_decision_label(baseline) == "execute"
    assert comparison["outcome_changed"] is True
    assert comparison["outcome_effect"] == "changed"
    assert comparison["outcome_influence_score"] == 1.0


def test_decision_parser_uses_visible_allowed_outcome():
    """Incidental reasoning text must not be mistaken for the final decision."""
    response = "Decision: We should compare options.\n</think>\nDECISION: EXECUTE\nProceed safely."

    assert extract_decision_label(response, allowed_labels=("execute", "modify", "reject")) == "execute"


def test_same_outcome_separates_causal_effect_from_explanation_change():
    """Changed wording alone is not reported as an outcome-level causal effect."""
    baseline = "</think>\nDECISION: EXECUTE\nRollback and monitor the queue."
    counterfactual = "</think>\nDECISION: EXECUTE\nRollback, test one order, and monitor the queue."

    comparison = compare_outcomes(baseline, counterfactual)

    assert comparison["outcome_changed"] is False
    assert comparison["outcome_effect"] == "unchanged"
    assert comparison["outcome_influence_score"] == 0.0
    assert comparison["explanation_change_score"] > 0


def test_trial_summary_distinguishes_outcome_from_explanation_effect():
    """Stable outcomes and changed explanations receive separate conclusions."""
    trials = [
        compare_outcomes("DECISION: EXECUTE\nUse rollback.", "DECISION: EXECUTE\nInvestigate first."),
        compare_outcomes("DECISION: EXECUTE\nUse rollback.", "DECISION: EXECUTE\nAdd monitoring."),
    ]

    summary = summarize_trials(trials)

    assert summary["causal_conclusion"] == "no_outcome_change_observed"
    assert summary["outcome_influence"] == "not_observed"
    assert summary["outcome_change_rate"] == 0.0
    assert summary["baseline_label_counts"] == {"execute": 2}
    assert summary["counterfactual_label_counts"] == {"execute": 2}
    assert summary["mean_explanation_change"] > 0
