import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, type CSSProperties } from "react";
import {
  Background,
  Controls,
  MarkerType,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
} from "@xyflow/react";
import type { Task } from "../../api/types";
import {
  buildDecisionCandidates,
  type DecisionCandidateNode,
  type DecisionCandidateNodeKind,
  type DecisionCandidateRelation,
} from "../../lib/decisionCandidates";
import { useInspectorStore } from "../../stores/inspectorStore";

const COLUMN: Record<DecisionCandidateNodeKind, number> = {
  agent: 0,
  tool: 0,
  evidence: 1,
  assessment: 2,
  candidate: 3,
  decision: 4,
  output: 5,
};

const NODE_STYLE: Record<DecisionCandidateNodeKind, CSSProperties> = {
  agent: { background: "#E9D5FF", border: "1.5px solid #7E22CE", borderRadius: 18 },
  tool: { background: "#CFFAFE", border: "1.5px solid #0E7490", borderRadius: 18 },
  evidence: { background: "#FFFC87", border: "1.5px solid #808080", borderRadius: 18 },
  assessment: { background: "#FED7AA", border: "1.5px solid #C2410C", borderRadius: 5 },
  candidate: { background: "#F1F5F9", border: "1.5px solid #64748B", borderRadius: 5 },
  decision: { background: "#BFDBFE", border: "2px solid #1D4ED8", borderRadius: 5 },
  output: { background: "#DCFCE7", border: "1.5px solid #15803D", borderRadius: 18 },
};

function edgeStyle(relation: DecisionCandidateRelation): CSSProperties {
  if (relation === "selected" || relation === "kept") return { stroke: "#15803D", strokeWidth: 3 };
  // Dropped evidence keeps the same dashed-red vocabulary as a rejected alternative: in
  // both cases the agent considered something and then set it aside.
  if (relation === "rejected" || relation === "dropped") return { stroke: "#B91C1C", strokeDasharray: "5 4" };
  if (relation === "supports" || relation === "informed") return { stroke: "#A16207" };
  if (relation === "made" || relation === "performed") return { stroke: "#7E22CE" };
  if (relation === "retrieved") return { stroke: "#0E7490" };
  return { stroke: "#64748B" };
}

function nodeLabel(node: DecisionCandidateNode) {
  return (
    <div className="max-w-64 text-center" title={[node.label, node.preview, node.sublabel].filter(Boolean).join("\n\n")}>
      <div className="text-[9px] font-semibold uppercase tracking-wide opacity-65">{node.kind}</div>
      <div className="line-clamp-2 break-all text-[11px] font-semibold">{node.label}</div>
      {node.badges && node.badges.length > 0 && (
        <div className="mt-0.5 flex flex-wrap justify-center gap-1">
          {node.badges.map((badge) => (
            <span key={badge} className="rounded bg-black/10 px-1 text-[8px] leading-4 opacity-80">
              {badge}
            </span>
          ))}
        </div>
      )}
      {node.preview && (
        // The query or the retrieved content, verbatim and monospaced so SQL and JSON stay
        // legible. Hover shows the whole thing; clicking opens the raw record.
        <div className="mt-1 line-clamp-4 break-words text-left font-mono text-[8px] leading-snug opacity-90">
          {node.preview}
        </div>
      )}
      {node.sublabel && (
        <div className="mt-1 line-clamp-2 text-[9px] italic leading-snug opacity-75">{node.sublabel}</div>
      )}
      {node.score !== undefined && (
        <div className="mt-0.5 text-[9px] font-mono font-semibold">score {node.score}</div>
      )}
      {node.kind === "candidate" && (
        <div
          className={`mt-1 text-[9px] font-bold uppercase ${node.selected ? "text-green-800" : "text-red-800"}`}
        >
          {node.selected ? "selected" : "not selected"}
        </div>
      )}
      {node.kind === "evidence" && node.kept !== undefined && (
        <div className={`mt-1 text-[9px] font-bold uppercase ${node.kept ? "text-green-800" : "text-red-800"}`}>
          {node.kept ? "kept" : "dropped"}
        </div>
      )}
    </div>
  );
}

interface Props {
  tasks: Task[];
  height?: string | number;
}

/** Render captured alternatives, assessments, evidence, and selections for decision tasks. */
export function DecisionCandidatesView({ tasks, height }: Props) {
  const trace = useMemo(() => buildDecisionCandidates(tasks), [tasks]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    const rowByColumn = new Map<number, number>();
    setNodes(
      trace.nodes.map((node) => {
        const column = COLUMN[node.kind];
        const row = rowByColumn.get(column) ?? 0;
        rowByColumn.set(column, row + 1);
        const selectedStyle =
          node.kind === "candidate" && node.selected
            ? { background: "#DCFCE7", border: "2px solid #15803D" }
            : node.kind === "evidence" && node.kept === false
              ? // Dropped evidence is faded, not hidden: it stays readable but never
                // competes with the evidence the decision actually rested on.
                { background: "#FEE2E2", border: "1.5px dashed #B91C1C", opacity: 0.75 }
              : node.kind === "evidence" && node.kept === true
                ? { background: "#DCFCE7", border: "1.5px solid #15803D" }
                : {};
        // Tool and evidence nodes carry a verbatim payload, so they get more room; the
        // rest keep the compact size they had.
        const wide = node.kind === "tool" || node.kind === "evidence";
        return {
          id: node.id,
          position: { x: column * 320, y: row * (wide ? 190 : 130) },
          data: { label: nodeLabel(node), traceNode: node },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          style: {
            ...NODE_STYLE[node.kind],
            ...selectedStyle,
            color: "#111827",
            padding: "9px 14px",
            fontSize: 11,
            width: wide ? 250 : 190,
          },
        };
      }),
    );
    setEdges(
      trace.edges.map((edge) => ({
        id: `${edge.source}|${edge.relation}|${edge.target}`,
        source: edge.source,
        target: edge.target,
        label: edge.relation,
        markerEnd: { type: MarkerType.ArrowClosed },
        style: edgeStyle(edge.relation),
        labelStyle: { fontSize: 9, fill: "var(--color-fg-muted)" },
      })),
    );
  }, [trace, setEdges, setNodes]);

  if (!trace.decisions.length) {
    return <div className="text-fg-muted text-xs">No decision provenance records captured for this workflow.</div>;
  }

  return (
    <div className={`space-y-2 ${height === "100%" ? "flex h-full flex-1 flex-col" : ""}`}>
      <p className="text-fg-muted text-[11px]">
        Follow tool retrievals and assessments into alternatives, then see which alternative the decision agent
        selected. Tool nodes show the query the agent wrote; evidence nodes show what came back, marked kept or
        dropped, so discarded results stay visible. Hover a node for the full text, or click it for the complete
        raw record — for a tool that is every item it returned.
      </p>
      <div
        style={{ height: height ?? 440 }}
        className={`rounded border border-border bg-surface-2 ${height === "100%" ? "flex-1" : ""}`}
      >
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodesConnectable={false}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            onNodeClick={(_, graphNode) => {
              const traceNode = graphNode.data.traceNode as DecisionCandidateNode;
              useInspectorStore.getState().set({
                kind: "decision",
                data: { label: traceNode.label, stats: { ...traceNode.details, node_kind: traceNode.kind } },
              });
            }}
          >
            <Background />
            <Controls showInteractive={false} />
          </ReactFlow>
        </ReactFlowProvider>
      </div>
      <div className="flex flex-wrap gap-3 text-[11px] text-fg-muted">
        <span>green = selected / kept</span>
        <span>red dashed edge = rejected / dropped</span>
        <span>teal = tool call and its retrievals</span>
        <span>gold edge = evidence</span>
        <span>purple edge = agent responsibility</span>
      </div>
    </div>
  );
}
