$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv-neural\Scripts\python.exe"
$python = if ($env:MEDIGRAPH_PYTHON) {
    $env:MEDIGRAPH_PYTHON
} elseif (Test-Path -LiteralPath $venvPython) {
    $venvPython
} else {
    "C:\Python314\python.exe"
}
$port = 8011

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python not found: $python"
}
$listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($listener) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -like "*mcp_server\server.py*") {
        Write-Host "MediGraph MCP is already running at http://localhost:$port/sse (PID $($process.ProcessId))."
        Write-Host "Nothing else needs to be started."
        exit 0
    }
    throw "Port $port is occupied by another process (PID $($listener.OwningProcess))."
}

$env:MEDIGRAPH_MCP_HOST = "0.0.0.0"
$env:MEDIGRAPH_MCP_PORT = "$port"
$env:FINETUNED_ORCHESTRATOR_URL = "http://127.0.0.1:18088/v1/chat/completions"
$env:FINETUNED_ORCHESTRATOR_MODEL = "qwen3p5-0p8b-orchestrator"

Set-Location -LiteralPath $projectRoot
Write-Host "MediGraph MCP starting at http://localhost:$port/sse"
Write-Host "Python: $python"
Write-Host "Keep this terminal open. Press Ctrl+C to stop."
& $python -X utf8 mcp_server\server.py
