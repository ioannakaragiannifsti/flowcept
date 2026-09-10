# Running Flowcept and Its Official UI on Windows

This guide records the setup used to run the Flowcept repository on this Windows
computer. It covers Flowcept itself, MongoDB, Redis-compatible Memurai, and the
official Flowcept web UI. It does not use the custom decision-provenance MAS example.

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

Rebuild it and restart the webservice:

```powershell
npm run build --prefix ui
```

## Maintained project documentation

- [Setup](docs/setup.rst)
- [Quick start](docs/quick_start.rst)
- [CLI reference](docs/cli-reference.rst)
- [Provenance storage](docs/prov_storage.rst)
- [REST API](docs/rest_api.rst)
- [UI development guide](ui/README.md)
