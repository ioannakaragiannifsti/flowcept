import { describe, expect, it } from "vitest";
import { getToolUsageRows } from "../src/lib/toolUsage";
import type { Task } from "../src/api/types";

const TOOL_TASK: Task = {
  task_id: "task-tool-1",
  workflow_id: "workflow-1",
  subtype: "agent_tool",
  activity_id: "runbook_db",
  agent_id: "remediation-agent",
  started_at: 1000,
  ended_at: 1002,
  used: {
    tool_name: "runbook_db",
    tool_type: "database",
    query: "SELECT id, title FROM runbooks WHERE service = 'checkout-notifications'",
  },
  generated: {
    retrieved: [
      { item_id: "runbook-17", content: "Config rollback" },
      { item_id: "runbook-08", content: "Payment failover" },
    ],
    retrieved_count: 2,
    retrieved_item_ids: ["runbook-17", "runbook-08"],
  },
};

describe("getToolUsageRows", () => {
  it("summarizes a captured tool call", () => {
    const [row] = getToolUsageRows([TOOL_TASK]);

    expect(row.tool_name).toBe("runbook_db");
    expect(row.tool_type).toBe("database");
    expect(row.agent_id).toBe("remediation-agent");
    expect(row.query_preview).toContain("SELECT id, title FROM runbooks");
    expect(row.retrieved_count).toBe(2);
    expect(row.retrieved_item_ids).toEqual(["runbook-17", "runbook-08"]);
    expect(row.duration).toBe(2);
  });

  it("falls back to activity_id and counts results when the count is absent", () => {
    const [row] = getToolUsageRows([
      {
        task_id: "task-tool-2",
        subtype: "agent_tool",
        activity_id: "web_search",
        generated: { retrieved: [{ item_id: "a" }] },
      },
    ]);

    expect(row.tool_name).toBe("web_search");
    expect(row.retrieved_count).toBe(1);
    expect(row.query_preview).toBe("");
  });

  it("ignores tasks that are not tool calls", () => {
    expect(getToolUsageRows([{ task_id: "t", subtype: "decision" }])).toEqual([]);
  });
});
