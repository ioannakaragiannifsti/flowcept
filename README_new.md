# Running Flowcept and Its Official UI on Windows

This guide records the setup used to run the Flowcept repository on this Windows
computer. It covers Flowcept itself, MongoDB, Redis-compatible Memurai, the
official Flowcept web UI, and the `DecisionCapture` extension for application
agents.

## What was configured

- A Python virtual environment at `.venv` instead of Conda.
- Flowcept installed from this repository with MongoDB, Redis, telemetry, and
  webservice dependencies.
- MongoDB as the persistent provenance database.
- Memurai as the Windows-compatible Redis service used by Flowcept.
- A repository-local settings file at `agent_sandbox/settings.yaml`.
- The official React UI built into Flowcept's FastAPI webservice.

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

Flowcept captures decisions through an explicit `DecisionCapture.invoke()` call.
The backend adds a domain-neutral system prompt and requests a structured decision,
so application code no longer needs to append every candidate, assessment, and
selection manually. The UI side is generic and needs no per-system work.

`DecisionCapture` is a **context manager**, not a decorator. Attach an unwrapped
LangChain model and call `invoke()` for the task whose decision provenance should be
captured:

```python
from flowcept import DecisionCapture, Flowcept

# Configure my_llm with the provider used by your application.
with Flowcept(start_persistence=False), DecisionCapture(
    decision_type="selection",
    context="Choose the output that best satisfies the request",
    agent_id="my-agent",
    llm=my_llm,
) as decision:
    record = decision.invoke("Write a formal greeting.")

print(record.to_dict())
```

The same pattern works for more complex, domain-specific agents. For example, a
deployment agent can evaluate operational evidence and choose a rollout strategy
without application code constructing or appending the candidates:

```python
from flowcept import DecisionCapture, Flowcept

# Configure deployment_llm with the provider used by your application.
deployment_request = """
Choose a rollout strategy for release 4.2 using the following evidence:

- The release contains a database migration that is backward compatible.
- Staging tests passed, but the payment-service error rate briefly reached 1.8%.
- The production SLO permits an error rate of at most 1.0%.
- Rollback takes approximately four minutes.
- The release must be available to all customers within 24 hours.

Account for customer impact, rollback risk, observability, and the delivery
deadline. Return the strategy the deployment agent should execute.
"""

with Flowcept(start_persistence=False), DecisionCapture(
    decision_type="deployment_strategy",
    context="Select a safe rollout plan from the available operational choices",
    agent_id="deployment-agent",
    llm=deployment_llm,
) as decision:
    deployment_decision = decision.invoke(deployment_request)

print(deployment_decision.to_dict())
```

Here, `invoke()` instructs the model to generate and compare suitable alternatives
such as a full rollout, canary rollout, staged rollout, or postponement. The caller
provides the task and its evidence; `DecisionCapture` handles the decision-specific
response structure and provenance capture. This makes the same approach reusable
for incident response, test planning, routing, recommendation, approval, and other
agent use cases.

`invoke()` supplies an OpenAI-compatible JSON-schema response format and validates
the model-generated candidates, assessment criteria, confidence scores,
explanations, and selected candidate IDs before storing a `PROV_AGENT.DECISION`
task. That decision is linked to the automatically captured LLM invocation.

The model must accept the OpenAI-compatible JSON-schema `response_format`
parameter. Importing `DecisionCapture` alone has no side effects; capture begins
only when `invoke()` is called. Use a fresh `DecisionCapture` instance for each
decision.

There is one source of truth. The block above writes a single task with
`subtype: "decision"` into MongoDB. The JSON you query and the **Decision Candidates**
view in the UI both read that same task — the UI adds no capture of its own and
stores nothing extra. Anything missing from the UI is missing from the record.

### Inspect decisions in the UI

After running your application with persistence enabled:

1. Open <http://127.0.0.1:8008> and select the workflow created by your application.
2. Open the **graph** tab.
3. In the graph-type toggle, choose **Decision Candidates**.

The view reads left to right: agents and evidence, then assessments, then the
candidate alternatives, then the decision, then its output. The selected
alternative is green; rejected ones are marked *not selected* with a red dashed
edge. Click any node to open the inspector panel, which shows that node's captured
record — for an assessment node this includes the explanation of why that
alternative scored as it did.

If the tab reports no decision records, verify that the application called
`DecisionCapture.invoke()` and ran with persistence enabled.

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

The two records directly associated with an automatic decision are:

| `subtype` | Contents |
| --- | --- |
| `ai_model_invocation` | The exact prompt, raw response, model name, temperature, token counts |
| `decision` | Candidates, assessments with explanations, and the selection |

They share a `workflow_id`, and the decision is linked to the captured model
invocation. A surrounding agent framework or other Flowcept instrumentation may
create additional task records.

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

### `DecisionCapture.invoke()` raises a candidate or JSON-schema error

These errors mean the configured model did not return the structured decision
required by `DecisionCapture.invoke()`. Confirm that the model supports the
OpenAI-compatible JSON-schema `response_format` parameter. Application code should
not repair the response by manually appending candidates. Retry the request or use
a model with reliable structured-output support.

### The Decision Candidates tab is empty

The workflow contains no `subtype: "decision"` tasks. Confirm that the application
calls `DecisionCapture.invoke()` inside the active Flowcept workflow, uses a fresh
capture instance for each decision, and runs with persistence enabled. Importing or
constructing `DecisionCapture` without calling `invoke()` does not create a record.

## Maintained project documentation

- [Setup](docs/setup.rst)
- [Quick start](docs/quick_start.rst)
- [CLI reference](docs/cli-reference.rst)
- [Provenance storage](docs/prov_storage.rst)
- [REST API](docs/rest_api.rst)
- [UI development guide](ui/README.md)
