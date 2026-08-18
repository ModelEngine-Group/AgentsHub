$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found."
}

& docker compose down
if ($LASTEXITCODE -ne 0) {
    throw "docker compose down failed."
}

Write-Host "MediGraph MCP and A2A containers stopped."
