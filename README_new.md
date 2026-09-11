# Running Flowcept and Its Official UI on Windows

This guide records the setup used to run the Flowcept repository on this Windows
computer. It covers Flowcept itself, MongoDB, Redis-compatible Memurai, the
official Flowcept web UI, and the decision-provenance multi-agent examples.

## What was configured

- A Python virtual environment at `.venv` instead of Conda.
- Flowcept installed from this repository with MongoDB, Redis, telemetry, and
  webservice dependencies.
- MongoDB as the persistent provenance database.
- Memurai as the Windows-compatible Redis service used by Flowcept.
- A repository-local settings file at `agent_sandbox/settings.yaml`.
- The official React UI built into Flowcept's FastAPI webservice.
- Ollama serving a local model, used by the decision-provenance MAS examples.

Run every command below from the repository root in PowerShell:


## One-time Python setup

Create the virtual environment if `.venv` does not already exist:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Flowcept and the dependencies required by the services and UI:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[extras,webservice]"
```

Conda is not required. Using `.\.venv\Scripts\python.exe` directly also works when
the environment is not activated.

## One-time settings setup

Create the isolated settings file:

```powershell
New-Item -ItemType Directory -Force agent_sandbox | Out-Null
Copy-Item resources\sample_settings.yaml agent_sandbox\settings.yaml
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
```

On Windows with Python 3.14, ensure this section exists in
`agent_sandbox/settings.yaml`:

```yaml
sys_metadata:
  environment_id: laptop
  sys_name: Windows
  node_name: local
```

Then enable the online Redis and MongoDB profile:

```powershell
.\.venv\Scripts\python.exe -m flowcept.cli --config-profile full-online -y
```

The important resulting settings are:

```yaml
project:
  db_flush_mode: online

mq:
  enabled: true
  type: redis
  host: localhost
  port: 6379

kv_db:
  enabled: true
  host: localhost
  port: 6379

web_server:
  host: 127.0.0.1
  port: 8008
  ui_enabled: true

databases:
  mongodb:
    enabled: true
    host: localhost
    port: 27017
    db: flowcept
```

Always set the settings path in each new PowerShell terminal before running
Flowcept:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
```

## MongoDB and Redis on Windows

This computer uses the Windows services named `MongoDB` and `Memurai`. Check them:

```powershell
Get-Service MongoDB, Memurai
```

Their status should be `Running`. Start either stopped service from an Administrator
PowerShell terminal:

```powershell
Start-Service MongoDB
Start-Service Memurai
```

Verify their ports:

```powershell
Test-NetConnection localhost -Port 27017
Test-NetConnection localhost -Port 6379
```

Both commands should show `TcpTestSucceeded : True`.

## One-time official UI build

Node.js and npm are only needed to build the UI. Verify them:

```powershell
node --version
npm --version
```

Install the frontend packages and build the official UI:

```powershell
npm ci --prefix ui
npm run build --prefix ui
```

The build is written to `src/flowcept/webservice/ui_build`. Repeat these commands
only after UI source code changes or after reinstalling dependencies.

## Start Flowcept and open the UI

In the first PowerShell terminal:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
.\.venv\Scripts\python.exe -m flowcept.cli --start --webservice
```

Keep that terminal open. Then open:

- Flowcept UI: <http://127.0.0.1:8008>
- REST API: <http://127.0.0.1:8008/api/v1>
- Swagger API documentation: <http://127.0.0.1:8008/docs>

If Windows reports error `10048`, another Flowcept webservice is already using port
`8008`. Open the UI directly instead of starting a second server.

## Generate ordinary Flowcept provenance

In a second PowerShell terminal, use the same settings and run the repository's
standard instrumentation example:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
.\.venv\Scripts\python.exe examples\instrumented_simple_example.py
```

The example prints its workflow ID. Its tasks are sent through Memurai and stored in
MongoDB by Flowcept.

## Inspect a workflow in the UI

1. Open <http://127.0.0.1:8008>.
2. Select **Workflows**.
3. Open the newest workflow.
4. Use **Tasks** to inspect captured inputs, outputs, timing, and metadata.
5. Use **Graphs** to inspect the workflow's provenance graph and dataflow.

The UI reads the records stored in MongoDB. It does not execute or evaluate the
workflow itself; Flowcept instrumentation captures the workflow while it runs.

## Decision provenance

Ordinary provenance records what each agent *did*. Decision provenance also records
what each agent *could have done*: the alternatives it considered, how it scored
them, and which one it selected. These are stored as tasks with `subtype: "decision"`.

Flowcept never infers decisions. An agent's rejected alternatives exist only inside
its reasoning, so each example must record them explicitly with `DecisionCapture`.
Any new multi-agent system needs the same treatment; the UI side is generic and
needs no per-system work.

`DecisionCapture` is a **context manager, not a decorator**. It is opened around the
point where a choice is made, candidates and assessments are added to it, and the
record is written when the block exits:

```python
from flowcept import DecisionCapture

with DecisionCapture(decision_type="...", context={...}, agent_id="...") as decision:
    for candidate in options:
        decision.add_candidate(candidate["id"], content=candidate)
    decision.assess(candidate_id, evaluator_id="...", score_type="...",
                    score=0.85, explanation="why this scored as it did")
    decision.select(winning_id)
```

A `record_decision(...)` function exists for the case where a complete
`DecisionRecord` has already been built elsewhere.

There is one source of truth. The block above writes a single task with
`subtype: "decision"` into MongoDB. The JSON you query and the **Decision Candidates**
view in the UI both read that same task — the UI adds no capture of its own and
stores nothing extra. Anything missing from the UI is missing from the record.

### One-time Ollama setup

The MAS examples call a local model through Ollama. Install it from
<https://ollama.com/download/windows>, start the Ollama application, then pull the
model:

```powershell
ollama pull qwen3:4b
```

Verify it is serving:

```powershell
Test-NetConnection 127.0.0.1 -Port 11434
ollama list
```

### Run the incident-response MAS

Five agents triage an incident, find a root cause, plan a remediation, review its
risk, and make a final call. Every agent must enumerate alternatives, score each
one, and select one, so the run produces five decision records:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
.\.venv\Scripts\python.exe examples\local_llm_mas_example.py --model qwen3:4b
```

The run prints nothing until it finishes, then prints the final decision and the
workflow ID. It takes a few minutes.

### Run the test-planning MAS

This example decides which test strategy verifies a changed safety requirement, and
adds a human review step that can approve, reject, or override the agents' choice:

```powershell
.\examples\run_decision_provenance_local_mas.ps1 -Model qwen3:4b -Review approve
```

The wrapper script sets the settings path, checks Ollama, and passes `--persist`.
Without `--persist` nothing reaches MongoDB and the UI stays empty. Use
`-Review override` to record a human decision that overrules the agents.

A deterministic variant runs without a model, using weighted scoring instead:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
.\.venv\Scripts\python.exe examples\decision_provenance_mas.py --persist
```

### Inspect decisions in the UI

1. Open <http://127.0.0.1:8008> and select the workflow the example printed.
2. Open the **graph** tab.
3. In the graph-type toggle, choose **Decision Candidates**.

The view reads left to right: agents and evidence, then assessments, then the
candidate alternatives, then the decision, then its output. The selected
alternative is green; rejected ones are marked *not selected* with a red dashed
edge. Click any node to open the inspector panel, which shows that node's captured
record — for an assessment node this includes the explanation of why that
alternative scored as it did.

If the tab reports no decision records, the workflow was produced by an example
that does not call `DecisionCapture`, or it was run without persistence.

### Query decisions directly

Everything the UI shows is in the MongoDB `tasks` collection:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
.\.venv\Scripts\python.exe -c @"
from pymongo import MongoClient
db = MongoClient('localhost', 27017)['flowcept']
for d in db.tasks.find({'subtype': 'decision'}).sort('started_at', 1):
    rec = d['generated']['decision']
    print(d['agent_id'], '->', rec['selected_candidate_ids'])
"@
```

Three task subtypes make up the record of one agent:

| `subtype` | Contents |
| --- | --- |
| `agent_tool` | The agent step: `used.evidence`, `generated.response` |
| `ai_model_invocation` | The exact prompt, raw response, model name, temperature, token counts |
| `decision` | Candidates, assessments with explanations, and the selection |

They are linked by `workflow_id` and `parent_task_id`.

### Reading the incident-response example and its output

Five agents run in a chain. Each reads the incident plus everything the agents
before it produced, then hands its answer to the next. That much is an ordinary
pipeline. What makes it a decision-provenance example is a constraint on every
agent: before it may answer, it must enumerate at least two genuinely different
alternatives, score each, and select exactly one.

The rejected alternatives are the point. They exist nowhere else, because a single
model response carries one answer rather than the option set. If the agent is not
asked to name them, they are gone the moment the call returns.

#### Each agent leaves two records

This is the part that reads as confusing in the raw database: one agent produces two
documents, joined by `parent_task_id`.

- The `agent_tool` document is the agent **doing its job** — `used.evidence` is what
  it saw, `generated.response` is the prose it produced.
- The `decision` document is the **choice inside that job** — `candidates[]` with
  each marked selected or rejected, `assessments[]` with a score and stated reason
  per candidate, and `selected_candidate_ids`.

They are not two agents. They are one agent described twice: the narrative view and
the analytic view. A third document, `ai_model_invocation`, sits under the
`agent_tool` and holds the verbatim prompt, raw response, and token counts.

#### A captured run

One recorded run of the example produced this chain. Scores are each agent's own.

| # | Agent | Selected | Rejected | Scores |
| --- | --- | --- | --- | --- |
| 1 | monitoring | `high_severity_high_urgency` | `medium_severity_low_urgency` | 0.92 / none |
| 2 | investigation | `notification_queue_overflow` | `payment_api_failure` | 0.85 / 0.15 |
| 3 | response-planning | `queue_drain` | `config_fix` | 0.85 / 0.15 |
| 4 | risk-review | `plan_acceptable_safeguards` | `plan_unacceptable` | 0.85 / 0.15 |
| 5 | incident-commander | `execute` | `modify`, `reject` | 0.85 / 0.65 / 0.15 |

Read as a chain: the team triaged the outage as critical, concluded a queue overflow
rather than a payment fault was to blame, chose to drain the queue rather than fix
the configuration, judged that plan acceptable with safeguards, and executed it.

The rejected column is what decision provenance buys. Without it you would know only
that the commander executed. With it you can say `modify` scored 0.65, so this was a
near miss rather than a foregone conclusion.

#### Three things the data warns about

**The scores are not independent.** `0.85 / 0.15` appears in four of the five
decisions. The investigation agent wrote *"Score: 85% for queue overflow, 15% for
payment issues"* into its prose answer, that prose became the `evidence` field of
every downstream agent, and they echoed the numbers back as their own confidence.
This is anchoring propagating along the chain. It is a real finding, and it means
the five scores cannot be treated as five independent judgments.

**Every score is a self-assessment.** `evaluator_id` equals the deciding agent and
`score_type` is `model_self_assessment`: the same model proposed the alternatives and
graded them. A stated rationale is evidence of what the model reported, not evidence
that the reasoning is sound. The test-planning example differs here, where a separate
`safety-critic` evaluates candidates it did not author.

**One rationale can contradict its own option.** In the captured run the `reject`
candidate is summarised as *"Abandon the proposed plan due to unacceptably high
risks"* while its rationale reads *"The risk review indicates the plan has manageable
risks but requires additional safeguards"* — an argument against rejecting. The model
wrote the summary as the option's label and the rationale as why it lost. That is an
artifact of one model authoring all its own alternatives, and is worth reporting
rather than tidying away.

Two smaller reading notes. An agent may score only its winner, leaving a rejected
candidate with no score. And on decision records `started_at` equals `ended_at`,
because the record is written instantaneously when the agent finishes; timing belongs
to the parent `agent_tool`, never to the decision.

#### What the records do and do not support

Supported: *"The commander selected `execute` over `modify`, self-assessed 0.85
against 0.65, citing queue-overflow evidence and the risk review's safeguards."*

Not supported: *"`execute` was chosen because the risk review found the plan
acceptable."* That is the stated reason, not a demonstrated cause. Closing that gap
needs an intervention rather than a record, which is what the analyzer below does.

### Estimate causal influence

The decision records state each agent's own reasons. To test whether a piece of
evidence actually changed an outcome, the post-hoc analyzer re-runs the decision
agent with one evidence message removed at a time and reports influence only when
the outcome label changes:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
.\.venv\Scripts\python.exe -m external.causal_decision_provenance.analysis `
  --workflow-id YOUR_WORKFLOW_ID `
  --decision-agent incident-commander-agent `
  --trials 3
```

Use `--decision-agent final-selector` for the test-planning MAS in local-model mode.
Reports are written to
`agent_sandbox/causal_decision_provenance/YOUR_WORKFLOW_ID/report.html`. See
[the analyzer's README](external/causal_decision_provenance/README.md) for what its
claims do and do not mean.

## Commands needed on a normal day

MongoDB and Memurai start automatically on this computer. Normally only these steps
are required:

```powershell
cd ".../.../dir"
.\.venv\Scripts\Activate.ps1
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
.\.venv\Scripts\python.exe -m flowcept.cli --start --webservice
```

Open <http://127.0.0.1:8008>. Run a Flowcept-instrumented workflow from another
terminal and refresh the workflow list.

Stop the foreground webservice with `Ctrl+C`.

## Troubleshooting

### `conda` is not recognized

Use the `.venv` commands in this guide. Conda is unnecessary.

### Python executable is not found

The correct path begins with one dot:

```powershell
.\.venv\Scripts\python.exe
```

Do not use `..venv\Scripts\python.exe`.

### The UI opens but contains no workflows

Confirm that MongoDB and Memurai are running, use the same
`FLOWCEPT_SETTINGS_PATH` in both terminals, and run an instrumented workflow.

### Port 8008 is already in use

Check the existing listener:

```powershell
Get-NetTCPConnection -LocalPort 8008 -State Listen
```

Usually this means the Flowcept UI is already available at
<http://127.0.0.1:8008>.

### Changes to the UI are not visible

The webservice on port 8008 serves the prebuilt bundle in
`src/flowcept/webservice/ui_build`, which is not rebuilt automatically. Rebuild it,
then hard-refresh the browser with `Ctrl+Shift+R` because the asset filenames change:

```powershell
npm run build --prefix ui
```

To iterate on UI source instead, run the Vite dev server, which reloads on save and
proxies the API to the webservice on 8008:

```powershell
npm run dev --prefix ui
```

It serves <http://localhost:5173>. Use the `localhost` hostname; the dev server binds
IPv6, so `127.0.0.1:5173` refuses the connection.

### `ImportError: cannot import name ... from 'flowcept'`

If the underlying error is `module 'os' has no attribute 'uname'`, the settings path
was not set. `os.uname()` does not exist on Windows, and Flowcept only skips that call
when `sys_metadata.sys_name` is configured. Set the settings path first:

```powershell
$env:FLOWCEPT_SETTINGS_PATH = "$PWD\agent_sandbox\settings.yaml"
```

### A MAS example stops with a candidate or JSON error

Messages such as `monitoring-agent returned 0 candidate(s); at least 2 are required`
mean the local model did not produce the required decision structure. The example
asks the model once more with a corrective message before failing, and both calls are
captured. The example stops rather than recording an invented decision. Retry, or use
a larger model with `--model`.

### The Decision Candidates tab is empty

The workflow contains no `subtype: "decision"` tasks. Either the example does not call
`DecisionCapture`, or it ran without persistence. `examples\instrumented_simple_example.py`
and other ordinary examples never produce decision records.

## Maintained project documentation

- [Setup](docs/setup.rst)
- [Quick start](docs/quick_start.rst)
- [CLI reference](docs/cli-reference.rst)
- [Provenance storage](docs/prov_storage.rst)
- [REST API](docs/rest_api.rst)
- [UI development guide](ui/README.md)
