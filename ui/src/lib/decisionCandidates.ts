import type { Task } from "../api/types";

export type DecisionCandidateNodeKind = "agent" | "evidence" | "assessment" | "candidate" | "decision" | "output";
export type DecisionCandidateRelation =
  | "made"
  | "informed"
  | "performed"
  | "supports"
  | "assessed"
  | "selected"
  | "rejected"
  | "considered"
  | "generated";

export interface DecisionCandidateNode {
  id: string;
  kind: DecisionCandidateNodeKind;
  label: string;
  selected?: boolean;
  details: Record<string, unknown>;
}

export interface DecisionCandidateEdge {
  source: string;
  target: string;
  relation: DecisionCandidateRelation;
}

export interface DecisionCandidates {
  nodes: DecisionCandidateNode[];
  edges: DecisionCandidateEdge[];
  decisions: string[];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asRecords(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function labelValue(value: unknown, fallback: string): string {
  if (typeof value === "string" && value) return value;
  if (value !== undefined && value !== null) return JSON.stringify(value);
  return fallback;
}

/** Convert captured decision tasks into a UI-ready evidence-to-decision trace. */
export function buildDecisionCandidates(tasks: Task[]): DecisionCandidates {
  const nodes = new Map<string, DecisionCandidateNode>();
  const edges = new Map<string, DecisionCandidateEdge>();
  const decisions: string[] = [];

  function addNode(node: DecisionCandidateNode) {
    if (!nodes.has(node.id)) nodes.set(node.id, node);
  }

  function addEdge(edge: DecisionCandidateEdge) {
    edges.set(`${edge.source}|${edge.relation}|${edge.target}`, edge);
  }

  for (const task of tasks.filter((item) => item.subtype === "decision")) {
    const generated = asRecord(task.generated);
    const decision = asRecord(generated.decision);
    if (!Object.keys(decision).length) continue;

    const rawDecisionId = labelValue(decision.decision_id, task.task_id);
    const decisionId = `decision:${rawDecisionId}`;
    decisions.push(decisionId);
    addNode({
      id: decisionId,
      kind: "decision",
      label: labelValue(decision.decision_type, "decision"),
      details: { task_id: task.task_id, agent_id: task.agent_id, ...decision },
    });

    if (task.agent_id) {
      const agentId = `agent:${task.agent_id}`;
      addNode({ id: agentId, kind: "agent", label: task.agent_id, details: { agent_id: task.agent_id } });
      addEdge({ source: agentId, target: decisionId, relation: "made" });
    }

    const inputs = new Set([
      ...asStrings(asRecord(task.used).input_entity_ids),
      ...asStrings(decision.input_entity_ids),
    ]);
    for (const inputId of inputs) {
      const evidenceId = `evidence:${inputId}`;
      addNode({ id: evidenceId, kind: "evidence", label: inputId, details: { entity_id: inputId } });
      addEdge({ source: evidenceId, target: decisionId, relation: "informed" });
    }

    const selectedIds = new Set(asStrings(decision.selected_candidate_ids));
    const candidateIds = new Map<string, string>();
    for (const candidate of asRecords(decision.candidates)) {
      const rawCandidateId = labelValue(candidate.candidate_id, "candidate");
      const candidateId = `${decisionId}:candidate:${rawCandidateId}`;
      const selected = selectedIds.has(rawCandidateId) || candidate.status === "selected";
      const relation = selected ? "selected" : candidate.status === "rejected" ? "rejected" : "considered";
      candidateIds.set(rawCandidateId, candidateId);
      addNode({
        id: candidateId,
        kind: "candidate",
        label: labelValue(candidate.content ?? candidate.content_ref, rawCandidateId),
        selected,
        details: candidate,
      });
      addEdge({ source: candidateId, target: decisionId, relation });
    }

    asRecords(decision.assessments).forEach((assessment, index) => {
      const rawAssessmentId = labelValue(assessment.assessment_id, String(index + 1));
      const assessmentId = `${decisionId}:assessment:${rawAssessmentId}`;
      const score = assessment.score === undefined ? "" : `: ${String(assessment.score)}`;
      addNode({
        id: assessmentId,
        kind: "assessment",
        label: `${labelValue(assessment.score_type, "assessment")}${score}`,
        details: assessment,
      });

      const rawCandidateId = labelValue(assessment.candidate_id, "candidate");
      const candidateId = candidateIds.get(rawCandidateId);
      if (candidateId) addEdge({ source: assessmentId, target: candidateId, relation: "assessed" });

      if (typeof assessment.evaluator_id === "string") {
        const evaluatorId = `agent:${assessment.evaluator_id}`;
        addNode({
          id: evaluatorId,
          kind: "agent",
          label: assessment.evaluator_id,
          details: { agent_id: assessment.evaluator_id },
        });
        addEdge({ source: evaluatorId, target: assessmentId, relation: "performed" });
      }

      for (const rawEvidenceId of asStrings(assessment.evidence_ids)) {
        const evidenceId = `evidence:${rawEvidenceId}`;
        addNode({
          id: evidenceId,
          kind: "evidence",
          label: rawEvidenceId,
          details: { entity_id: rawEvidenceId },
        });
        addEdge({ source: evidenceId, target: assessmentId, relation: "supports" });
      }
    });

    const outputs = new Set([
      ...asStrings(generated.output_entity_ids),
      ...asStrings(decision.output_entity_ids),
    ]);
    for (const outputId of outputs) {
      const traceOutputId = `output:${outputId}`;
      addNode({ id: traceOutputId, kind: "output", label: outputId, details: { entity_id: outputId } });
      addEdge({ source: decisionId, target: traceOutputId, relation: "generated" });
    }
  }

  return { nodes: [...nodes.values()], edges: [...edges.values()], decisions };
}
