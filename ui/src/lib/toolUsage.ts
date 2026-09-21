import type { Task } from "../api/types";
import { taskDuration } from "./format";

export const AGENT_TOOL_SUBTYPE = "agent_tool";

export interface ToolUsageRow {
  task_id: string;
  tool_name: string;
  tool_type?: string;
  query_method?: string;
  agent_id?: string;
  query_preview: string;
  query: unknown;
  retrieved_count: number;
  retrieved_item_ids: string[];
  duration: number | null;
  started_at?: Task["started_at"];
  task: Task;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function preview(value: unknown, maxLength = 160): string {
  const text = typeof value === "string" ? value : value == null ? "" : JSON.stringify(value);
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > maxLength ? `${compact.slice(0, maxLength)}…` : compact;
}

/**
 * Collect every captured tool call in a workflow.
 *
 * A tool task records the query it ran and the complete, unfiltered result set, so these
 * rows are the ground truth a decision's kept-versus-dropped evidence is compared against.
 */
export function getToolUsageRows(tasks: Task[]): ToolUsageRow[] {
  return tasks
    .filter((task) => task.subtype === AGENT_TOOL_SUBTYPE)
    .map((task) => {
      const used = asRecord(task.used);
      const generated = asRecord(task.generated);
      const metadata = asRecord(task.custom_metadata);
      const retrievedItemIds = asStrings(generated.retrieved_item_ids);
      const retrieved = Array.isArray(generated.retrieved) ? generated.retrieved : [];
      return {
        task_id: task.task_id,
        // Older captures carry the tool name only in activity_id.
        tool_name: asString(used.tool_name) ?? asString(metadata.tool_name) ?? task.activity_id ?? "tool",
        tool_type: asString(used.tool_type) ?? asString(metadata.tool_type),
        query_method: asString(used.query_method) ?? asString(metadata.query_method),
        agent_id: task.agent_id,
        query_preview: preview(used.query),
        query: used.query,
        retrieved_count:
          typeof generated.retrieved_count === "number" ? generated.retrieved_count : retrieved.length,
        retrieved_item_ids: retrievedItemIds,
        duration: taskDuration(task),
        started_at: task.started_at,
        task,
      };
    });
}
