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
$port = 8100

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python not found: $python"
}
$listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($listener) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -like "*integration\a2a\a2a_server.py*") {
        Write-Host "MediGraph A2A agent is already running at http://localhost:$port/.well-known/agent.json (PID $($process.ProcessId))."
        Write-Host "Nothing else needs to be started."
        exit 0
    }
    throw "Port $port is occupied by another process (PID $($listener.OwningProcess))."
}

Set-Location -LiteralPath $projectRoot
Write-Host "MediGraph A2A agent starting at http://localhost:$port/.well-known/agent.json"
Write-Host "Python: $python"
Write-Host "Keep this terminal open. Press Ctrl+C to stop."
& $python -X utf8 integration\a2a\a2a_server.py
