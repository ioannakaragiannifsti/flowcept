"""Traceable multi-agent test planning for Child Presence Detection."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from flowcept import DecisionCapture, Flowcept, FlowceptTask
from flowcept.commons.vocabulary import PROV_AGENT

REQUIREMENT = {
    "entity_id": "requirement:CPD-017:v2",
    "text": (
        "After the ignition is switched off, detect a child left in any rear seat, "
        "including when the cabin sensor is partly obstructed, and issue escalating "
        "warnings within 30 seconds."
    ),
    "safety_class": "ASIL-B",
    "response_deadline_seconds": 30,
}

PRODUCT_VARIANTS = {
    "entity_id": "configuration:CPD-platform:v5",
    "variants": {
        "ICE-EU": {"rear_seat_sensor": True, "low_power_mode": False},
        "HEV-EU": {"rear_seat_sensor": True, "low_power_mode": True},
        "BEV-EU": {"rear_seat_sensor": True, "low_power_mode": True},
    },
}

EXISTING_TEST = {
    "entity_id": "test:CPD-TC-BASIC:v1",
    "test_id": "CPD-TC-BASIC",
    "covered_variants": ["ICE-EU"],
    "covered_scenarios": ["unobstructed_sensor", "ignition_off"],
}

REQUIRED_SCENARIOS = {
    "ignition_off",
    "unobstructed_sensor",
    "partly_obstructed_sensor",
    "warning_within_30_seconds",
    "low_power_mode",
}


def _capture_agent_activity(
    activity_id: str,
    agent_id: str,
    used: dict,
    operation: Callable[[], dict],
    parent_task_id: str | None = None,
) -> tuple[dict, str]:
    """Execute one agent activity and capture its PROV-AGENT record."""
    with FlowceptTask(
        activity_id=activity_id,
        agent_id=agent_id,
        parent_task_id=parent_task_id,
        subtype=PROV_AGENT.AGENT_TOOL,
        used=used,
        capture_telemetry=False,
    ) as task:
        generated = operation()
        task.end(generated=generated)
    return generated, task.get_id()


def _build_agent_messages(role: str, task: str, inputs: dict, output_schema: dict) -> list[dict[str, str]]:
    """Build a role prompt whose complete input and output contract can be captured."""
    return [
        {
            "role": "system",
            "content": (
                f"You are the {role} in an automotive product-engineering multi-agent system. "
                "Use only the supplied evidence. Return exactly one valid JSON object matching the requested "
                "schema. Give concise decision reasons, but do not provide hidden chain-of-thought."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{task}\n\nInputs:\n{json.dumps(inputs, indent=2)}\n\n"
                f"Required output schema:\n{json.dumps(output_schema, indent=2)}"
            ),
        },
    ]


def _parse_agent_response(response: str, required_fields: tuple[str, ...]) -> dict:
    """Parse JSON and normalize small-model single-item collection responses."""
    parsed = json.loads(response)
    collection_items = {"candidates": "candidate_id", "assessments": "candidate_id"}
    for collection_field, item_id_field in collection_items.items():
        if collection_field in required_fields and collection_field not in parsed and item_id_field in parsed:
            parsed = {collection_field: [parsed]}
    if (
        "selected_candidate_id" in required_fields
        and "selected_candidate_id" not in parsed
        and "candidate_id" in parsed
    ):
        parsed["selected_candidate_id"] = parsed.pop("candidate_id")
    missing = [field for field in required_fields if field not in parsed]
    if missing:
        raise ValueError(f"Local agent response is missing required fields: {missing}")
    return parsed


def _merge_model_observation(pipeline_output: dict, response: str) -> dict:
    """Attach an observed model response without making it a pipeline dependency."""
    try:
        model_output = json.loads(response)
    except json.JSONDecodeError:
        model_output = response
    return {**pipeline_output, "model_output": model_output}


def _capture_local_model_activity(
    activity_id: str,
    agent_id: str,
    agent_role: str,
    task_instruction: str,
    inputs: dict,
    input_entity_ids: list[str],
    output_entity_id: str,
    output_schema: dict,
    pipeline_output: dict,
    parent_task_id: str | None,
    model_name: str,
) -> tuple[dict, str]:
    """Invoke one local foundation-model agent and capture its task and model call."""
    from langchain_openai import ChatOpenAI

    from flowcept.configs import AGENT, AGENT_API_KEY
    from flowcept.instrumentation.flowcept_agent_task import FlowceptLLM

    messages = _build_agent_messages(agent_role, task_instruction, inputs, output_schema)
    with FlowceptTask(
        activity_id=activity_id,
        agent_id=agent_id,
        parent_task_id=parent_task_id,
        subtype=PROV_AGENT.AGENT_TOOL,
        used={"input_entity_ids": input_entity_ids, "inputs": inputs, "agent_role": agent_role},
        capture_telemetry=False,
    ) as task:
        model = ChatOpenAI(
            api_key=AGENT_API_KEY,
            base_url=AGENT["llm_server_url"],
            model=model_name,
            temperature=0,
            reasoning_effort="none",
            max_tokens=1000,
            model_kwargs={
                "response_format": {"type": "json_object"},
            },
        )
        llm = FlowceptLLM(
            model,
            agent_id=agent_id,
        )
        llm.parent_task_id = task.get_id()
        llm.metadata.update(
            {
                "provider": "ollama",
                "model_name": model_name,
                "model_parameters": {
                    "temperature": 0,
                    "max_tokens": 1000,
                    "reasoning_effort": "none",
                    "response_format": {"type": "json_object"},
                },
                "agent_role": agent_role,
            }
        )
        response = llm.invoke(messages)
        generated = _merge_model_observation(pipeline_output, response)
        generated["output_entity_ids"] = [output_entity_id]
        task.end(generated=generated)
    return generated, task.get_id()


def analyse_requirement() -> dict:
    """Extract reviewable safety goals from the changed requirement."""
    return {
        "analysis_id": "analysis:CPD-017:v2",
        "output_entity_ids": ["analysis:CPD-017:v2"],
        "safety_goals": [
            "detect a child after ignition off",
            "remain effective with partial sensor obstruction",
            "warn within 30 seconds",
        ],
        "required_scenarios": sorted(REQUIRED_SCENARIOS),
        "evidence_ids": [REQUIREMENT["entity_id"]],
    }


def analyse_variants(requirement_analysis: dict) -> dict:
    """Identify variants affected by the requirement and their constraints."""
    affected = sorted(PRODUCT_VARIANTS["variants"])
    low_power = sorted(
        variant
        for variant, features in PRODUCT_VARIANTS["variants"].items()
        if features["low_power_mode"]
    )
    return {
        "impact_id": "impact:CPD-platform:v5",
        "output_entity_ids": ["impact:CPD-platform:v5"],
        "affected_variants": affected,
        "low_power_variants": low_power,
        "input_entity_ids": [requirement_analysis["analysis_id"], PRODUCT_VARIANTS["entity_id"]],
    }


def design_candidate_tests(requirement_analysis: dict, variant_impact: dict) -> dict:
    """Generate explicit test alternatives for the affected product variants."""
    variants = variant_impact["affected_variants"]
    required = requirement_analysis["required_scenarios"]
    candidates = [
        {
            "candidate_id": "cpd-test-basic",
            "test_id": "CPD-TC-BASIC",
            "strategy": "reuse existing test unchanged",
            "covered_variants": EXISTING_TEST["covered_variants"],
            "covered_scenarios": EXISTING_TEST["covered_scenarios"],
            "execution_cost": 1,
            "residual_risk": 0.42,
        },
        {
            "candidate_id": "cpd-test-extended",
            "test_id": "CPD-TC-EXTENDED",
            "strategy": "extend the test using boundary and obstruction cases",
            "covered_variants": variants,
            "covered_scenarios": required,
            "execution_cost": 3,
            "residual_risk": 0.08,
        },
        {
            "candidate_id": "cpd-test-conservative",
            "test_id": "CPD-TC-CONSERVATIVE",
            "strategy": "add redundant sensor-fault and false-positive campaigns",
            "covered_variants": variants,
            "covered_scenarios": required + ["sensor_failure", "hot_object_false_positive"],
            "execution_cost": 5,
            "residual_risk": 0.03,
        },
    ]
    return {
        "proposal_id": "proposal:CPD-tests:v2",
        "output_entity_ids": ["proposal:CPD-tests:v2"],
        "input_entity_ids": [requirement_analysis["analysis_id"], variant_impact["impact_id"]],
        "candidates": candidates,
    }


def assess_candidate_tests(proposal: dict, variant_impact: dict) -> dict:
    """Evaluate candidate coverage, safety, and execution feasibility."""
    required_variants = set(variant_impact["affected_variants"])
    assessments = []
    for candidate in proposal["candidates"]:
        scenario_coverage = len(REQUIRED_SCENARIOS & set(candidate["covered_scenarios"])) / len(REQUIRED_SCENARIOS)
        variant_coverage = len(required_variants & set(candidate["covered_variants"])) / len(required_variants)
        safety_score = 1.0 - candidate["residual_risk"]
        feasibility_score = 1.0 - (candidate["execution_cost"] - 1) / 5
        assessments.append(
            {
                "candidate_id": candidate["candidate_id"],
                "scenario_coverage": round(scenario_coverage, 2),
                "variant_coverage": round(variant_coverage, 2),
                "safety_score": round(safety_score, 2),
                "feasibility_score": round(feasibility_score, 2),
                "eligible": scenario_coverage == 1.0
                and variant_coverage == 1.0
                and candidate["residual_risk"] <= 0.10,
            }
        )
    return {
        "assessment_id": "assessment:CPD-tests:v2",
        "output_entity_ids": ["assessment:CPD-tests:v2"],
        "input_entity_ids": [proposal["proposal_id"], REQUIREMENT["entity_id"]],
        "assessments": assessments,
    }


def select_candidate(proposal: dict, assessment_report: dict) -> dict:
    """Select the strongest eligible test using declared weights."""
    candidates = {candidate["candidate_id"]: candidate for candidate in proposal["candidates"]}
    eligible = [assessment for assessment in assessment_report["assessments"] if assessment["eligible"]]
    scores = {
        assessment["candidate_id"]: round(
            0.45 * assessment["scenario_coverage"]
            + 0.20 * assessment["variant_coverage"]
            + 0.25 * assessment["safety_score"]
            + 0.10 * assessment["feasibility_score"],
            3,
        )
        for assessment in eligible
    }
    selected_id = max(scores, key=scores.get)
    return {**candidates[selected_id], "selection_score": scores[selected_id], "all_eligible_scores": scores}


def _record_ai_decision(
    proposal: dict,
    assessment_report: dict,
    parent_task_id: str,
    model_selection: dict | None = None,
) -> tuple[dict, str, str]:
    """Record the decision board's recommendation and supporting assessments."""
    if model_selection is None:
        recommendation = select_candidate(proposal, assessment_report)
        decision_agent_id = "decision-board"
    else:
        candidates = {candidate["candidate_id"]: candidate for candidate in proposal["candidates"]}
        selected_id = model_selection["selected_candidate_id"]
        recommendation = {**candidates[selected_id], "model_selection": model_selection}
        decision_agent_id = "final-selector"
    with DecisionCapture(
        decision_type="test_case_recommendation",
        context={
            "question": "Which test strategy adequately verifies CPD-017 v2?",
            "mandatory_constraints": [
                "complete scenario coverage",
                "complete variant coverage",
                "residual risk <= 0.10",
            ],
            "selection_weights": {"scenario": 0.45, "variant": 0.20, "safety": 0.25, "feasibility": 0.10},
        },
        agent_id=decision_agent_id,
        parent_task_id=parent_task_id,
        input_entity_ids=[proposal["proposal_id"], assessment_report["assessment_id"]],
        output_entity_ids=["recommendation:CPD-tests:v2"],
    ) as decision:
        for rank, candidate in enumerate(proposal["candidates"], start=1):
            decision.add_candidate(
                candidate["candidate_id"],
                content=candidate,
                origin_type="multi_agent_generation",
                rank=rank,
            )
        for assessment in assessment_report["assessments"]:
            decision.assess(
                assessment["candidate_id"],
                evaluator_id="safety-critic",
                score_type="coverage_and_risk_assessment",
                score=assessment["safety_score"],
                criteria=["scenario coverage", "variant coverage", "residual risk"],
                evidence_ids=[REQUIREMENT["entity_id"], PRODUCT_VARIANTS["entity_id"], proposal["proposal_id"]],
                explanation=(
                    f"scenario={assessment['scenario_coverage']}, variant={assessment['variant_coverage']}, "
                    f"safety={assessment['safety_score']}, eligible={assessment['eligible']}"
                ),
            )
        decision.select(recommendation["candidate_id"])
    return recommendation, decision.record.decision_id, decision.task_id


def _record_review(
    proposal: dict,
    ai_decision_id: str,
    ai_candidate_id: str,
    review_action: str,
    override_candidate_id: str | None,
    parent_task_id: str,
) -> tuple[dict, dict | None]:
    """Capture a human approval, rejection, or override as a separate decision."""
    candidate_ids = {candidate["candidate_id"] for candidate in proposal["candidates"]}
    if review_action == "override":
        selected_id = override_candidate_id
        if selected_id not in candidate_ids:
            raise ValueError(f"Unknown override candidate: {selected_id}")
    elif review_action == "approve":
        selected_id = ai_candidate_id
    elif review_action == "reject":
        selected_id = None
    else:
        raise ValueError(f"Unsupported review action: {review_action}")

    output_id = "approved-test:CPD:v2" if selected_id else "review-rejection:CPD:v2"
    with DecisionCapture(
        decision_type="human_review",
        context={"reviewed_decision_id": ai_decision_id, "action": review_action},
        agent_id="human-reviewer",
        parent_task_id=parent_task_id,
        input_entity_ids=[ai_decision_id, REQUIREMENT["entity_id"]],
        output_entity_ids=[output_id],
    ) as review_decision:
        for candidate in proposal["candidates"]:
            review_decision.add_candidate(
                candidate["candidate_id"],
                content=candidate,
                origin_type="reviewed_recommendation",
            )
        review_decision.assess(
            ai_candidate_id,
            evaluator_id="human-reviewer",
            score_type="engineering_review",
            criteria=["safety completeness", "reviewability", "test cost"],
            evidence_ids=[ai_decision_id, REQUIREMENT["entity_id"]],
            explanation=f"Reviewer action: {review_action}",
        )
        if selected_id:
            review_decision.select(selected_id)

    final_test = next(
        (candidate for candidate in proposal["candidates"] if candidate["candidate_id"] == selected_id),
        None,
    )
    review = {
        "action": review_action,
        "selected_candidate_id": selected_id,
        "reviewed_decision_id": ai_decision_id,
        "decision_id": review_decision.record.decision_id,
    }
    return review, final_test


def build_provenance_graph(records: list[dict]) -> dict:
    """Build a queryable node-edge view from Flowcept's captured records."""
    tasks = [record for record in records if record.get("type") == "task"]
    agents = sorted({task["agent_id"] for task in tasks if task.get("agent_id")})
    nodes = []
    edges = []
    known_entities = set()

    for agent_id in agents:
        nodes.append({"id": agent_id, "type": "agent"})
    for task in tasks:
        task_id = task["task_id"]
        nodes.append(
            {
                "id": task_id,
                "type": "activity",
                "subtype": task.get("subtype"),
                "label": task.get("activity_id"),
            }
        )
        if task.get("agent_id"):
            edges.append({"source": task_id, "target": task["agent_id"], "relation": "wasAssociatedWith"})
        if task.get("parent_task_id"):
            edges.append({"source": task_id, "target": task["parent_task_id"], "relation": "wasInformedBy"})

        used = task.get("used") or {}
        generated = task.get("generated") or {}
        input_ids = used.get("input_entity_ids", [])
        output_ids = generated.get("output_entity_ids", [])
        if task.get("subtype") == PROV_AGENT.DECISION.value:
            decision = generated.get("decision", {})
            input_ids = decision.get("input_entity_ids", [])
            output_ids = decision.get("output_entity_ids", [])
        for entity_id in input_ids:
            known_entities.add(entity_id)
            edges.append({"source": task_id, "target": entity_id, "relation": "used"})
        for entity_id in output_ids:
            known_entities.add(entity_id)
            edges.append({"source": entity_id, "target": task_id, "relation": "wasGeneratedBy"})

    nodes.extend({"id": entity_id, "type": "entity"} for entity_id in sorted(known_entities))
    return {"agents": agents, "tasks": tasks, "nodes": nodes, "edges": edges}


def write_run_output(result: dict, output_dir: Path) -> Path:
    """Write the complete MAS result to its output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    run_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return run_path


def _generate_llm_decision_card(recommendation: dict, review: dict, final_test: dict | None) -> str:
    """Generate a grounded review card with the configured, instrumented LLM."""
    from flowcept.agents.llm.builders import build_llm_model

    prompt = (
        "Write a concise engineering decision card using only the JSON below. Include the AI recommendation, "
        "the human action, the final outcome, and residual risk. Do not invent evidence or private reasoning.\n"
        + json.dumps({"recommendation": recommendation, "review": review, "final_test": final_test})
    )
    llm = build_llm_model(agent_id="decision-card-agent")
    return llm.invoke(prompt)


def _run_local_model_agents(model_name: str) -> tuple[dict, dict, dict, dict, dict, str]:
    """Run the five engineering roles with a local OpenAI-compatible model."""
    requirement_handoff = analyse_requirement()
    requirement_analysis, requirement_task_id = _capture_local_model_activity(
        activity_id="analyse_requirement",
        agent_id="requirements-analyst",
        agent_role="Requirement analyst",
        task_instruction=(
            "Extract testable safety goals and the complete set of scenarios required to verify the requirement."
        ),
        inputs={"requirement": REQUIREMENT},
        input_entity_ids=[REQUIREMENT["entity_id"]],
        output_entity_id="analysis:CPD-017:local-model",
        output_schema={
            "safety_goals": ["testable goal"],
            "required_scenarios": ["snake_case scenario"],
            "evidence_ids": [REQUIREMENT["entity_id"]],
            "reason": "concise evidence-based explanation",
        },
        pipeline_output=requirement_handoff,
        parent_task_id=None,
        model_name=model_name,
    )
    requirement_analysis["analysis_id"] = "analysis:CPD-017:local-model"

    variant_handoff = analyse_variants(requirement_analysis)
    variant_impact, variant_task_id = _capture_local_model_activity(
        activity_id="analyse_product_variants",
        agent_id="variant-analyst",
        agent_role="Product-variant analyst",
        task_instruction=(
            "Identify every affected product variant and explain which variant constraints change the test plan."
        ),
        inputs={"requirement_analysis": requirement_analysis, "product_variants": PRODUCT_VARIANTS},
        input_entity_ids=[requirement_analysis["analysis_id"], PRODUCT_VARIANTS["entity_id"]],
        output_entity_id="impact:CPD-platform:local-model",
        output_schema={
            "affected_variants": ["variant name"],
            "low_power_variants": ["variant name"],
            "variant_constraints": [{"variant": "name", "constraint": "test implication"}],
            "reason": "concise evidence-based explanation",
        },
        pipeline_output=variant_handoff,
        parent_task_id=requirement_task_id,
        model_name=model_name,
    )
    variant_impact["impact_id"] = "impact:CPD-platform:local-model"

    proposal_handoff = design_candidate_tests(requirement_analysis, variant_impact)
    proposal, proposal_task_id = _capture_local_model_activity(
        activity_id="design_candidate_tests",
        agent_id="test-designer",
        agent_role="Test-generation agent",
        task_instruction=(
            "Generate exactly three distinct test strategies. Include one economical strategy, one balanced strategy, "
            "and one conservative strategy. Quantify execution cost from 1 to 5 and residual risk from 0 to 1."
        ),
        inputs={
            "requirement_analysis": requirement_analysis,
            "variant_impact": variant_impact,
            "existing_test": EXISTING_TEST,
        },
        input_entity_ids=[requirement_analysis["analysis_id"], variant_impact["impact_id"], EXISTING_TEST["entity_id"]],
        output_entity_id="proposal:CPD-tests:local-model",
        output_schema={
            "candidates": [
                {
                    "candidate_id": "unique stable id",
                    "test_id": "test identifier",
                    "strategy": "description",
                    "covered_variants": ["variant name"],
                    "covered_scenarios": ["scenario"],
                    "execution_cost": "integer 1-5",
                    "residual_risk": "number 0-1",
                    "reason": "concise design rationale",
                }
            ]
        },
        pipeline_output=proposal_handoff,
        parent_task_id=variant_task_id,
        model_name=model_name,
    )
    proposal["proposal_id"] = "proposal:CPD-tests:local-model"

    assessment_handoff = assess_candidate_tests(proposal, variant_impact)
    assessment_report, assessment_task_id = _capture_local_model_activity(
        activity_id="challenge_candidate_tests",
        agent_id="safety-critic",
        agent_role="Safety critic",
        task_instruction=(
            "Challenge every proposed candidate. Score scenario coverage, variant coverage, safety, and feasibility "
            "from 0 to 1. Mark eligible only when all required scenarios and variants are covered and residual risk "
            "is at most 0.10. Preserve each candidate_id exactly."
        ),
        inputs={
            "requirement": REQUIREMENT,
            "required_scenarios": requirement_analysis["required_scenarios"],
            "variant_impact": variant_impact,
            "proposal": proposal,
        },
        input_entity_ids=[proposal["proposal_id"], REQUIREMENT["entity_id"], variant_impact["impact_id"]],
        output_entity_id="assessment:CPD-tests:local-model",
        output_schema={
            "assessments": [
                {
                    "candidate_id": "exact candidate id",
                    "scenario_coverage": "number 0-1",
                    "variant_coverage": "number 0-1",
                    "safety_score": "number 0-1",
                    "feasibility_score": "number 0-1",
                    "eligible": "boolean",
                    "concerns": ["specific concern"],
                    "reason": "concise evidence-based explanation",
                }
            ]
        },
        pipeline_output=assessment_handoff,
        parent_task_id=proposal_task_id,
        model_name=model_name,
    )
    assessment_report["assessment_id"] = "assessment:CPD-tests:local-model"

    selected_handoff = select_candidate(proposal, assessment_report)
    selection_handoff = {
        "selected_candidate_id": selected_handoff["candidate_id"],
        "criteria": ["coverage", "safety", "feasibility"],
        "evidence_ids": [proposal["proposal_id"], assessment_report["assessment_id"]],
        "rationale": "Deterministic scaffold keeps the provenance demonstration executable.",
        "residual_uncertainty": [],
    }
    selection, selection_task_id = _capture_local_model_activity(
        activity_id="select_final_candidate",
        agent_id="final-selector",
        agent_role="Final-selection agent",
        task_instruction=(
            "Select one eligible candidate. Prioritize complete scenario and variant coverage, then safety, then "
            "feasibility. Preserve the selected candidate_id exactly and cite only supplied entity IDs as evidence."
        ),
        inputs={
            "proposal": proposal,
            "assessment_report": assessment_report,
            "selection_policy": {
                "mandatory": ["eligible is true"],
                "priority": ["coverage", "safety", "feasibility"],
            },
        },
        input_entity_ids=[proposal["proposal_id"], assessment_report["assessment_id"]],
        output_entity_id="selection:CPD-tests:local-model",
        output_schema={
            "selected_candidate_id": "exact candidate id",
            "criteria": ["criterion used"],
            "evidence_ids": ["supplied entity id"],
            "rationale": "concise evidence-based explanation",
            "residual_uncertainty": ["remaining uncertainty"],
        },
        pipeline_output=selection_handoff,
        parent_task_id=assessment_task_id,
        model_name=model_name,
    )
    return requirement_analysis, variant_impact, proposal, assessment_report, selection, selection_task_id


def run_pipeline(
    review_action: str = "approve",
    override_candidate_id: str | None = None,
    generate_llm_card: bool = False,
    persist: bool = False,
    agent_mode: str = "deterministic",
    local_model: str = "qwen3:4b",
) -> dict:
    """Run the six-agent engineering workflow and return results plus provenance."""
    with Flowcept(
        workflow_name="CPD Multi-Agent Test Planning",
        workflow_args={"requirement_id": REQUIREMENT["entity_id"]},
        start_persistence=persist,
        check_safe_stops=False,
    ) as flowcept:
        workflow_id = flowcept.current_workflow_id
        if agent_mode == "local-model":
            (
                requirement_analysis,
                variant_impact,
                proposal,
                assessment_report,
                model_selection,
                selection_task_id,
            ) = _run_local_model_agents(local_model)
            recommendation, ai_decision_id, decision_task_id = _record_ai_decision(
                proposal, assessment_report, selection_task_id, model_selection
            )
            agent_outputs = {
                "requirement_analysis": requirement_analysis,
                "variant_impact": variant_impact,
                "test_proposal": proposal,
                "safety_assessment": assessment_report,
                "model_selection": model_selection,
            }
        elif agent_mode == "deterministic":
            requirement_analysis, requirement_task_id = _capture_agent_activity(
                "analyse_requirement",
                "requirements-analyst",
                {"input_entity_ids": [REQUIREMENT["entity_id"]], "requirement": REQUIREMENT},
                analyse_requirement,
            )
            variant_impact, variant_task_id = _capture_agent_activity(
                "analyse_product_variants",
                "variant-analyst",
                {"input_entity_ids": [requirement_analysis["analysis_id"], PRODUCT_VARIANTS["entity_id"]]},
                lambda: analyse_variants(requirement_analysis),
                requirement_task_id,
            )
            proposal, proposal_task_id = _capture_agent_activity(
                "design_candidate_tests",
                "test-designer",
                {"input_entity_ids": [requirement_analysis["analysis_id"], variant_impact["impact_id"]]},
                lambda: design_candidate_tests(requirement_analysis, variant_impact),
                variant_task_id,
            )
            assessment_report, assessment_task_id = _capture_agent_activity(
                "challenge_candidate_tests",
                "safety-critic",
                {"input_entity_ids": [proposal["proposal_id"], REQUIREMENT["entity_id"]]},
                lambda: assess_candidate_tests(proposal, variant_impact),
                proposal_task_id,
            )
            recommendation, ai_decision_id, decision_task_id = _record_ai_decision(
                proposal, assessment_report, assessment_task_id
            )
            agent_outputs = {
                "requirement_analysis": requirement_analysis,
                "variant_impact": variant_impact,
                "test_proposal": proposal,
                "safety_assessment": assessment_report,
            }
        else:
            raise ValueError(f"Unsupported agent mode: {agent_mode}")
        review, final_test = _record_review(
            proposal,
            ai_decision_id,
            recommendation["candidate_id"],
            review_action,
            override_candidate_id,
            decision_task_id,
        )
        decision_card = (
            _generate_llm_decision_card(recommendation, review, final_test) if generate_llm_card else None
        )
        records = list(flowcept.get_buffer())

    return {
        "workflow_id": workflow_id,
        "agent_mode": agent_mode,
        "model": local_model if agent_mode == "local-model" else None,
        "requirement": REQUIREMENT,
        "agent_outputs": agent_outputs,
        "recommendation": recommendation,
        "review": review,
        "final_test": final_test,
        "decision_card": decision_card,
        "provenance": build_provenance_graph(records),
    }


def _interactive_review() -> tuple[str, str | None]:
    print("Review actions: approve, reject, override")
    action = input("Human review action: ").strip().lower()
    override = input("Candidate ID to select: ").strip() if action == "override" else None
    return action, override


def main() -> None:
    """Run the MAS and save its inspectable provenance graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", choices=["approve", "reject", "override", "interactive"], default="approve")
    parser.add_argument("--override-candidate", default="cpd-test-conservative")
    parser.add_argument(
        "--agent-mode",
        choices=["deterministic", "local-model"],
        default="deterministic",
        help="Use deterministic Python agents or five local foundation-model agents.",
    )
    parser.add_argument(
        "--local-model",
        default="qwen3:4b",
        help="Ollama model exposed through its OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--llm-decision-card",
        action="store_true",
        help="Call the configured LLM and capture an ai_model_invocation record.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist the captured workflow through Redis into the configured database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("agent_sandbox/decision_provenance_mas"),
        help="Directory for run.json.",
    )
    args = parser.parse_args()

    review_action, override_id = (
        _interactive_review()
        if args.review == "interactive"
        else (args.review, args.override_candidate if args.review == "override" else None)
    )
    result = run_pipeline(
        review_action=review_action,
        override_candidate_id=override_id,
        generate_llm_card=args.llm_decision_card,
        persist=args.persist,
        agent_mode=args.agent_mode,
        local_model=args.local_model,
    )
    run_path = write_run_output(result, args.output_dir)

    print(f"AI recommendation: {result['recommendation']['candidate_id']}")
    print(f"Human review: {result['review']['action']}")
    print(f"Final test: {result['final_test']['test_id'] if result['final_test'] else 'none'}")
    print(
        f"Captured graph: {len(result['provenance']['nodes'])} nodes, "
        f"{len(result['provenance']['edges'])} edges"
    )
    print(f"Run data: {run_path.resolve()}")


if __name__ == "__main__":
    main()
