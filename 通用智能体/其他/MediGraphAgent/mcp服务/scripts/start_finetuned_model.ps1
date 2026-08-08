$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

# Python with torch/transformers/peft. Override via MEDIGRAPH_TUNE_PYTHON.
$python = $env:MEDIGRAPH_TUNE_PYTHON
if (-not $python) {
    $condaPython = Join-Path $env:USERPROFILE ".conda\envs\meditune\python.exe"
    if (Test-Path -LiteralPath $condaPython) { $python = $condaPython } else { $python = "python" }
}

# Local Qwen3.5-0.8B base weights (e.g. modelscope cache). Override via ORCHESTRATOR_BASE_MODEL.
$baseModel = $env:ORCHESTRATOR_BASE_MODEL
if (-not $baseModel) {
    $baseModel = Join-Path $env:USERPROFILE ".cache\modelscope\hub\models\Qwen\Qwen3___5-0___8B"
}
$adapter = Join-Path $projectRoot "finetune\outputs\qwen3p5-0p8b-orchestrator"
$port = 18088

if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    throw "Python interpreter not found: $python (set MEDIGRAPH_TUNE_PYTHON)"
}
foreach ($path in @($baseModel, $adapter)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path not found: $path"
    }
}
$listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
if ($listener) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -like "*finetune\openai_compatible_server.py*") {
        Write-Host "Fine-tuned model API is already running at http://localhost:$port/v1/ (PID $($process.ProcessId))."
        Write-Host "Nothing else needs to be started."
        exit 0
    }
    throw "Port $port is occupied by another process (PID $($listener.OwningProcess))."
}

Set-Location -LiteralPath $projectRoot
Write-Host "Fine-tuned Qwen3.5-0.8B API starting at http://localhost:$port/v1/"
Write-Host "First load can take about 1-2 minutes. Keep this terminal open; Ctrl+C stops it."
& $python finetune\openai_compatible_server.py `
    --host 0.0.0.0 `
    --port $port `
    --base $baseModel `
    --adapter $adapter
