[CmdletBinding()]
param(
    [string]$Model = "qwen3:4b",
    [ValidateSet("approve", "reject", "override", "interactive")]
    [string]$Review = "approve"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$settings = Join-Path $repoRoot "agent_sandbox\settings.yaml"
$example = Join-Path $PSScriptRoot "decision_provenance_mas.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found at $python"
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed. Install it from https://ollama.com/download/windows"
}
if (-not (Test-NetConnection 127.0.0.1 -Port 11434 -InformationLevel Quiet)) {
    throw "Ollama is not running. Start the Ollama application and run this command again."
}

$installedModels = ollama list | Out-String
if ($installedModels -notmatch [regex]::Escape($Model)) {
    throw "Model $Model is missing. Run: ollama pull $Model"
}

$env:FLOWCEPT_SETTINGS_PATH = $settings
& $python $example --agent-mode local-model --local-model $Model --persist --review $Review
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
