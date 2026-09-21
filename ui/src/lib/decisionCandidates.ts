import type { Task } from "../api/types";

export type DecisionCandidateNodeKind =
  | "agent"
  | "tool"
  | "evidence"
  | "assessment"
  | "candidate"
  | "decision"
  | "output";
export type DecisionCandidateRelation =
  | "made"
  | "informed"
  | "performed"
  | "supports"
  | "assessed"
  | "selected"
  | "rejected"
  | "considered"
  | "generated"
  | "retrieved"
  | "kept"
  | "dropped";

export interface DecisionCandidateNode {
  id: string;
  kind: DecisionCandidateNodeKind;
  label: string;
  /** Short human-readable descriptor shown under the label. */
  sublabel?: string;
  /**
   * What the tool actually produced, shown verbatim on the node: the query for a tool,
   * the retrieved content for an evidence item. Kept separate from `sublabel` so the
   * agent's commentary never displaces the data it was commenting on.
   */
  preview?: string;
  /** Short facts worth seeing without clicking: tool type, result count, score, source. */
  badges?: string[];
  /** Headline score shown on the node, when one was assessed. */
  score?: number;
  selected?: boolean;
  /**
   * For evidence retrieved by a tool: whether the decision kept it. `undefined` means the
   * item was never judged, either because it came from an entity id rather than a tool
   * call, or because the agent failed to report a verdict on it.
   */
  kept?: boolean;
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

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function labelValue(value: unknown, fallback: string): string {
  if (typeof value === "string" && value) return value;
  if (value !== undefined && value !== null) return JSON.stringify(value);
  return fallback;
}

/** Short descriptive fields a domain may use to name an alternative. */
const SUMMARY_FIELDS = [
  "strategy",
  "summary",
  "title",
  "name",
  "label",
  "description",
  "test_id",
  // Fields retrieval tools commonly carry their text in, so a search hit shows its text
  // rather than nothing at all.
  "body",
  "snippet",
  "text",
  "abstract",
  "page_content",
  "content",
];

/** Render a payload for display on a node, verbatim but bounded. */
function previewText(value: unknown, maxLength = 260): string | undefined {
  if (value === undefined || value === null) return undefined;
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (!text) return undefined;
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > maxLength ? `${compact.slice(0, maxLength)}…` : compact;
}

/** Pick a short readable descriptor out of an arbitrary candidate payload. */
function summarize(content: unknown): string | undefined {
  if (typeof content === "string") return content;
  const record = asRecord(content);
  for (const field of SUMMARY_FIELDS) {
    const value = record[field];
    if (typeof value === "string" && value) return value;
  }
  return undefined;
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

    // Tool retrievals: every item a tool returned becomes an evidence node, whether or not
    // the agent kept it. Drawing the discarded ones is the point -- they are invisible in
    // the agent's answer and only provenance records that they were ever seen.
    const verdictByItem = new Map<string, Record<string, unknown>>();
    for (const use of asRecords(decision.evidence_uses)) {
      const itemId = asString(use.item_id);
      if (itemId) verdictByItem.set(itemId, use);
    }

    for (const retrieval of asRecords(decision.retrievals)) {
      const toolName = labelValue(retrieval.tool_name, "tool");
      const toolNodeId = `tool:${labelValue(retrieval.retrieval_id, toolName)}`;
      const retrievedRows = asRecords(retrieval.retrieved);
      const toolBadges = [asString(retrieval.tool_type), `${retrievedRows.length} retrieved`].filter(
        (badge): badge is string => Boolean(badge),
      );
      const truncation = asRecord(retrieval.truncation);
      if (Object.keys(truncation).length) toolBadges.push("result capped");
      addNode({
        id: toolNodeId,
        kind: "tool",
        label: toolName,
        // The query verbatim: it is the thing the agent wrote, and paraphrasing it would
        // defeat the point of recording it.
        preview: previewText(retrieval.query),
        badges: toolBadges,
        details: retrieval,
      });
      if (task.agent_id) addEdge({ source: `agent:${task.agent_id}`, target: toolNodeId, relation: "performed" });

      for (const item of asRecords(retrieval.retrieved)) {
        const itemId = asString(item.item_id);
        if (!itemId) continue;
        const evidenceNodeId = `evidence:${itemId}`;
        const verdict = verdictByItem.get(itemId) ?? {};
        const kept = typeof verdict.used === "boolean" ? verdict.used : undefined;
        const role = asString(verdict.role);
        const explanation = asString(verdict.explanation);

        // What the tool returned, shown as-is, and the agent's verdict shown beside it
        // rather than instead of it.
        const content = previewText(item.content ?? summarize(item.content));
        const badges = [
          role,
          asString(item.source),
          item.score === undefined || item.score === null ? undefined : `score ${String(item.score)}`,
        ].filter((badge): badge is string => Boolean(badge));

        const existing = nodes.get(evidenceNodeId);
        if (existing) {
          // An input entity and a retrieved item can share an id; keep one node and let
          // the tool verdict enrich it rather than silently dropping either.
          existing.kept ??= kept;
          existing.preview ??= content;
          existing.sublabel ??= explanation;
          existing.badges = [...(existing.badges ?? []), ...badges];
          existing.details = { ...existing.details, ...item, ...verdict };
        } else {
          addNode({
            id: evidenceNodeId,
            kind: "evidence",
            label: itemId,
            preview: content,
            sublabel: explanation,
            badges,
            kept,
            details: { tool_name: toolName, role, ...item, ...verdict },
          });
        }

        addEdge({ source: toolNodeId, target: evidenceNodeId, relation: "retrieved" });
        // No edge into the decision when the agent never reported a verdict: the item was
        // retrieved but never judged, and inventing an edge would hide that.
        if (kept !== undefined) {
          addEdge({ source: evidenceNodeId, target: decisionId, relation: kept ? "kept" : "dropped" });
        }
      }
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
        label: rawCandidateId,
        sublabel: summarize(candidate.content ?? candidate.content_ref),
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

    // Surface each candidate's headline score on its node, and the winning rationale on the decision.
    for (const assessment of asRecords(decision.assessments)) {
      const rawCandidateId = labelValue(assessment.candidate_id, "candidate");
      const candidateNode = nodes.get(candidateIds.get(rawCandidateId) ?? "");
      if (!candidateNode) continue;
      if (candidateNode.score === undefined && typeof assessment.score === "number") {
        candidateNode.score = assessment.score;
      }
      if (typeof assessment.explanation === "string" && assessment.explanation) {
        if (!candidateNode.sublabel) candidateNode.sublabel = assessment.explanation;
        if (selectedIds.has(rawCandidateId)) {
          const decisionNode = nodes.get(decisionId);
          if (decisionNode) {
            decisionNode.sublabel ??= assessment.explanation;
            decisionNode.details.selected_rationale = assessment.explanation;
          }
        }
      }
    }

    const decisionNode = nodes.get(decisionId);
    if (decisionNode) decisionNode.details.selected_candidate_ids = decision.selected_candidate_ids;

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
