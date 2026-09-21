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
    const selectedCandidate = trace.nodes.find((node) => node.kind === "candidate" && node.selected);
    // The label is the stable candidate id; the payload becomes a readable sublabel.
    expect(selectedCandidate?.label).toBe("rollback");
    expect(selectedCandidate?.sublabel).toBe("Rollback the release");
    expect(selectedCandidate?.score).toBe(0.9);

    // The winning assessment's explanation is surfaced on the decision node.
    const decisionNode = trace.nodes.find((node) => node.kind === "decision");
    expect(decisionNode?.sublabel).toBe("Rollback is reversible.");
    expect(decisionNode?.details.selected_rationale).toBe("Rollback is reversible.");
    expect(decisionNode?.details.selected_candidate_ids).toEqual(["rollback"]);
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

  it("draws retrieved items as kept or dropped evidence hanging off their tool", () => {
    const groundedTask: Task = {
      task_id: "task-decision-2",
      workflow_id: "workflow-1",
      subtype: "decision",
      agent_id: "remediation-agent",
      used: { retrieval_ids: ["retrieval-1"] },
      generated: {
        decision: {
          decision_id: "decision-2",
          decision_type: "tool_grounded_remediation",
          candidates: [{ candidate_id: "rollback", status: "selected", content: "Roll back" }],
          assessments: [
            {
              assessment_id: "assessment-2",
              candidate_id: "rollback",
              evaluator_id: "remediation-agent",
              score_type: "model_reported_confidence",
              score: 0.8,
              explanation: "Addresses the root cause.",
            },
          ],
          selected_candidate_ids: ["rollback"],
          retrievals: [
            {
              retrieval_id: "retrieval-1",
              tool_name: "runbook_db",
              tool_type: "database",
              query: "SELECT * FROM runbooks",
              retrieved: [
                { item_id: "runbook-17", content: "Config rollback", source: "ops_db", score: 0.9 },
                { item_id: "runbook-08", content: "Payment failover" },
                { item_id: "runbook-99", content: "Never judged" },
              ],
              retrieved_count: 3,
            },
          ],
          evidence_uses: [
            { item_id: "runbook-17", used: true, role: "supporting", explanation: "Matches the failure" },
            { item_id: "runbook-08", used: false, role: "irrelevant", explanation: "Unrelated to notifications" },
          ],
        },
      },
    };

    const trace = buildDecisionCandidates([groundedTask]);

    const toolNode = trace.nodes.find((node) => node.kind === "tool");
    expect(toolNode?.label).toBe("runbook_db");
    // The query is shown verbatim, with the facts worth seeing without clicking.
    expect(toolNode?.preview).toBe("SELECT * FROM runbooks");
    expect(toolNode?.badges).toEqual(["database", "3 retrieved"]);

    const kept = trace.nodes.find((node) => node.id === "evidence:runbook-17");
    const dropped = trace.nodes.find((node) => node.id === "evidence:runbook-08");
    expect(kept?.kept).toBe(true);
    expect(dropped?.kept).toBe(false);
    // The retrieved content stays on the node; the agent's reason sits beside it.
    expect(kept?.preview).toBe("Config rollback");
    expect(kept?.sublabel).toBe("Matches the failure");
    expect(kept?.badges).toEqual(["supporting", "ops_db", "score 0.9"]);
    expect(dropped?.preview).toBe("Payment failover");
    expect(dropped?.sublabel).toBe("Unrelated to notifications");

    // An item the agent never reported on is still drawn, but nothing claims it was judged.
    const unjudged = trace.nodes.find((node) => node.id === "evidence:runbook-99");
    expect(unjudged?.kept).toBeUndefined();
    // It was still retrieved, so its content is still shown.
    expect(unjudged?.preview).toBe("Never judged");
    expect(
      trace.edges.some((edge) => edge.source === "evidence:runbook-99" && edge.target === "decision:decision-2"),
    ).toBe(false);

    expect(trace.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ relation: "performed", source: "agent:remediation-agent", target: "tool:retrieval-1" }),
        expect.objectContaining({ relation: "retrieved", source: "tool:retrieval-1", target: "evidence:runbook-17" }),
        expect.objectContaining({ relation: "kept", source: "evidence:runbook-17", target: "decision:decision-2" }),
        expect.objectContaining({ relation: "dropped", source: "evidence:runbook-08", target: "decision:decision-2" }),
      ]),
    );
  });

  it("ignores ordinary tasks and returns an empty trace", () => {
    const trace = buildDecisionCandidates([{ task_id: "ordinary", subtype: "agent_tool" }]);

    expect(trace).toEqual({ nodes: [], edges: [], decisions: [] });
  });
});
