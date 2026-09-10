import { describe, expect, it } from "vitest";
import { buildDecisionCandidates } from "../src/lib/decisionCandidates";
import type { Task } from "../src/api/types";

const DECISION_TASK: Task = {
  task_id: "task-decision-1",
  workflow_id: "workflow-1",
  subtype: "decision",
  agent_id: "judge-agent",
  used: {
    input_entity_ids: ["message-analysis"],
  },
  generated: {
    output_entity_ids: ["final-answer"],
    decision: {
      decision_id: "decision-1",
      decision_type: "selection",
      context: "Choose the safest response.",
      candidates: [
        { candidate_id: "rollback", status: "selected", content: "Rollback the release" },
        { candidate_id: "wait", status: "rejected", content: "Wait and monitor" },
      ],
      assessments: [
        {
          assessment_id: "assessment-1",
          candidate_id: "rollback",
          evaluator_id: "risk-agent",
          score_type: "safety",
          score: 0.9,
          explanation: "Rollback is reversible.",
          evidence_ids: ["message-risk"],
        },
      ],
      selected_candidate_ids: ["rollback"],
      input_entity_ids: ["message-analysis"],
      output_entity_ids: ["final-answer"],
    },
  },
};

describe("buildDecisionCandidates", () => {
  it("connects evidence, evaluators, alternatives, the decision, and its output", () => {
    const trace = buildDecisionCandidates([DECISION_TASK]);

    expect(trace.decisions).toHaveLength(1);
    expect(trace.nodes.find((node) => node.kind === "decision")?.label).toBe("selection");
    expect(trace.nodes.find((node) => node.kind === "candidate" && node.selected)?.label).toBe(
      "Rollback the release",
    );
    expect(trace.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ relation: "made", source: "agent:judge-agent" }),
        expect.objectContaining({ relation: "informed", source: "evidence:message-analysis" }),
        expect.objectContaining({ relation: "selected" }),
        expect.objectContaining({ relation: "assessed" }),
        expect.objectContaining({ relation: "supports", source: "evidence:message-risk" }),
        expect.objectContaining({ relation: "generated", target: "output:final-answer" }),
      ]),
    );
  });

  it("ignores ordinary tasks and returns an empty trace", () => {
    const trace = buildDecisionCandidates([{ task_id: "ordinary", subtype: "agent_tool" }]);

    expect(trace).toEqual({ nodes: [], edges: [], decisions: [] });
  });
});
