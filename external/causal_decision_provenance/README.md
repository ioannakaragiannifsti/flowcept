# External Causal Decision Provenance

This module analyzes a completed Flowcept workflow after it has been persisted. It
does not change Flowcept source code, instrumentation, or database records. Reports
are written under `agent_sandbox/`.

## What the report means

The report keeps three different claims separate:

- **Structural provenance:** an agent produced a message, and another agent consumed
  that message. These edges come from Flowcept entity IDs.
- **Semantic provenance:** a consumed message expresses a claim that was presented
  as evidence to the decision agent. This shows the observed basis, not causality.
- **Causal provenance:** the decision agent is rerun after one message is removed.
  An outcome influence is observed only when the explicit decision label changes.

The report also measures explanation change. Different wording with the same outcome
is reported as explanation sensitivity and is never presented as outcome influence.
`not_observed` means no outcome change occurred in the tested trials. It does not
prove that the message has no contribution in every possible run.

## Required MAS capture contract

Agent role names are arbitrary. Model names are arbitrary when the model is available
through the configured OpenAI-compatible endpoint. The analyzer needs:

1. A persisted `agent_tool` task for the final decision agent.
2. An `ai_model_invocation` child containing the exact prompt.
3. A mapping of evidence messages in the decision task's `used` data.
4. Upstream `agent_tool` tasks whose generated responses match those messages.
5. A final visible answer containing an explicit label such as `DECISION: EXECUTE`.

Use CLI options when a MAS uses different agent names, field names, or outcomes.
An arbitrary MAS that does not capture this information cannot be reconstructed by
the post-processor.

## Run

Select the same isolated settings used when the workflow was captured:

```powershell
$env:FLOWCEPT_SETTINGS_PATH="$PWD\agent_sandbox\settings.yaml"
```

Run the analyzer with the workflow ID printed by the MAS:

```powershell
.\.venv\Scripts\python.exe -m external.causal_decision_provenance.analysis `
  --workflow-id YOUR_WORKFLOW_ID `
  --decision-agent incident-commander-agent
```

For a different outcome vocabulary or persisted field names:

```powershell
.\.venv\Scripts\python.exe -m external.causal_decision_provenance.analysis `
  --workflow-id YOUR_WORKFLOW_ID `
  --decision-agent final-reviewer `
  --outcomes approve revise reject `
  --evidence-field evidence `
  --response-field response
```

## Repeated trials and larger systems

One analysis makes `trials × (1 + messages)` model calls: one full-input reproduction
and one call per removed message in every trial. Use repeated trials to reduce the
chance that stochastic model variation is mistaken for influence:

```powershell
.\.venv\Scripts\python.exe -m external.causal_decision_provenance.analysis `
  --workflow-id YOUR_WORKFLOW_ID `
  --trials 3 `
  --max-workers 2
```

`--max-workers` bounds concurrent intervention calls. Keep it at `1` for small local
models or limited memory. `--max-interventions 10` limits cost on a large judge input.

Results are written to:

```text
agent_sandbox/causal_decision_provenance/YOUR_WORKFLOW_ID/analysis.json
agent_sandbox/causal_decision_provenance/YOUR_WORKFLOW_ID/report.html
```

Open the report on Windows:

```powershell
Start-Process agent_sandbox\causal_decision_provenance\YOUR_WORKFLOW_ID\report.html
```

The current method estimates direct effects of messages already consumed by the
decision agent. Estimating an upstream agent's total effect requires replaying every
downstream agent affected by its removal.
