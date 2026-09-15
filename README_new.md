# Set Up Flowcept Decision Capture and the UI on Windows

This guide takes a new Flowcept user from downloading and installing the project to
adding decision provenance to an existing agent and viewing the captured decisions
in the official UI. Run the commands in PowerShell unless stated otherwise.

## Prerequisites

Install these tools before starting:

- Python 3.10 or newer and `pip`.
- Git, if you want to clone the source repository.
- MongoDB and a Redis-compatible service such as Memurai for persistent capture.
- Node.js and npm if you need to build the official UI from source.
- A LangChain-compatible model that accepts the OpenAI-compatible JSON-schema
  `response_format` parameter.

## Download Flowcept

Clone the official repository and enter its directory:

```powershell
git clone https://github.com/ORNL/flowcept.git
cd flowcept
```

If Git is unavailable, download the repository ZIP from
<https://github.com/ORNL/flowcept/archive/refs/heads/main.zip>, extract it, open
PowerShell in the extracted `flowcept-main` directory, and continue below.

## What this guide configures

- A Python virtual environment at `.venv` instead of Conda.
- Flowcept installed from this repository with MongoDB, Redis, telemetry, and
  webservice dependencies.
- MongoDB as the persistent provenance database.
- Memurai as the Windows-compatible Redis service used by Flowcept.
- A repository-local settings file at `agent_sandbox/settings.yaml`.
- The official React UI built into Flowcept's FastAPI webservice.

## Install Flowcept and its Python dependencies

Create the virtual environment if `.venv` does not already exist:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip`, then install Flowcept from the cloned source with the common runtime
and webservice dependencies used by this guide:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[extras,webservice]"
```

Conda is not required. Using `.\.venv\Scripts\python.exe` directly also works when
the environment is not activated.

If you only need the published package, install from PyPI instead. Clone the
repository as described above if you later need its examples, settings template, or
UI source files.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "flowcept[extras,webservice]"
```

## Add DecisionCapture to your agent

`DecisionCapture` records not only the result an agent selected, but also the
alternatives it considered, its assessments, explanations, confidence scores, and
selected candidate IDs. Each captured decision is stored as a task with
`subtype: "decision"`.

The integration point is the model call. Create a fresh `DecisionCapture` context,
attach the unwrapped LangChain model already used by your application, and replace
that decision-making model call with `decision.invoke(...)`:

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

For a realistic agent, pass the evidence, constraints, and requested outcome in the
same prompt your agent would normally send to its model:

```python
from flowcept import DecisionCapture, Flowcept

deployment_request = """
Choose a rollout strategy for release 4.2 using the following evidence:

- The release contains a backward-compatible database migration.
- Staging tests passed, but the payment-service error rate briefly reached 1.8%.
- The production SLO permits an error rate of at most 1.0%.
- Rollback takes approximately four minutes.
- The release must be available to all customers within 24 hours.

Account for customer impact, rollback risk, observability, and the deadline.
Return the strategy the deployment agent should execute.
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

You only provide the agent task, relevant evidence, and decision context. Do not
append candidates or call separate assessment and selection methods. `invoke()` adds
a domain-neutral decision system prompt, supplies the JSON-schema response format,
and validates the model-generated decision before recording it. The decision is
also linked to the automatically captured LLM invocation.

Use this checklist when adapting the pattern:

1. Pass the model through `llm=` without wrapping it in another agent executor.
2. Give `decision_type` a stable category meaningful to your application.
3. Use `context` to describe the goal and decision boundary.
4. Put current evidence, constraints, and the requested outcome in the `invoke()`
   prompt.
5. Use a new `DecisionCapture` instance for every decision.
6. Keep the call inside the relevant Flowcept workflow so its records share the
   workflow ID.

Importing or constructing `DecisionCapture` does not capture anything by itself;
capture begins only when `invoke()` is called. The configured model must support the
OpenAI-compatible JSON-schema `response_format` parameter.

The examples above demonstrate the code integration. Complete the persistence and
UI setup in the following sections to store and visualize those decisions.

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

## Inspect captured decisions in the UI

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
