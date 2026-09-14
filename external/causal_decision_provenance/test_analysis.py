"""Behavior tests for the external causal decision-provenance analyzer."""

import json

import pytest

from external.causal_decision_provenance.analysis import (
    apply_evidence,
    build_judge_model,
    build_structural_graph,
    compare_outcomes,
    extract_decision_label,
    find_evidence_rendering,
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


EVIDENCE = {"incident": "queue grew to 18,000", "risk_review": "acceptable with safeguards"}


@pytest.mark.parametrize(
    ("name", "render"),
    [
        ("json_indent_2", lambda e: json.dumps(e, indent=2, default=str)),
        ("json_compact", lambda e: json.dumps(e, default=str)),
        ("python_repr", str),
    ],
)
def test_evidence_rendering_is_detected_for_each_supported_format(name, render):
    """Locate the evidence block whichever way the MAS serialized it."""
    user_text = f"DECISION TO MAKE:\nExecute?\n\nEVIDENCE:\n{render(EVIDENCE)}\n\nReply now."

    detected_name, detected_render = find_evidence_rendering(user_text, EVIDENCE)

    assert detected_name == name
    assert detected_render(EVIDENCE) == render(EVIDENCE)


def test_removed_evidence_leaves_the_rest_of_the_prompt_untouched():
    """An intervention must differ from the original by exactly the removed message."""

    def render(evidence):
        return json.dumps(evidence, indent=2, default=str)

    prefix = "DECISION TO MAKE:\nExecute?\n\nEVIDENCE:\n"
    suffix = "\n\nWhat the answer must contain:\nDecide.\n\nReply now."
    user_text = f"{prefix}{render(EVIDENCE)}{suffix}"
    reduced = {"incident": EVIDENCE["incident"]}

    edited = apply_evidence(user_text, EVIDENCE, reduced, render)

    assert "acceptable with safeguards" not in edited
    assert "queue grew to 18,000" in edited
    assert edited.startswith(prefix)
    assert edited.endswith(suffix)


def test_unlocatable_evidence_is_an_error_rather_than_a_silent_no_op():
    """A prompt the analyzer cannot edit must fail loudly, not report no influence."""
    user_text = "Evidence was summarized by hand and never serialized verbatim."

    with pytest.raises(ValueError, match="Could not locate the evidence block"):
        find_evidence_rendering(user_text, EVIDENCE)


def test_replayed_judge_keeps_the_captured_decoding_configuration():
    """response_format must survive the replay, or the baseline silently stops reproducing."""
    captured = {"temperature": 0, "max_tokens": 1600, "response_format": {"type": "json_object"}}

    model = build_judge_model("qwen3:4b", captured)

    assert model.temperature == 0
    assert model.max_tokens == 1600
    # Constructor arguments and model_kwargs are separate; response_format belongs in the latter.
    assert model.model_kwargs["response_format"] == {"type": "json_object"}


def test_replayed_judge_without_captured_parameters_still_builds():
    """A MAS that recorded no parameters must not crash the replay."""
    model = build_judge_model("qwen3:4b", {})

    assert model.model_name == "qwen3:4b"
    assert not model.model_kwargs
