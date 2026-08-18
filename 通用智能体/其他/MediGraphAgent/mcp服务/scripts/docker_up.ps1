param(
    [switch]$Neural,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install and start Docker Desktop first."
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker daemon is unavailable. Start Docker Desktop and retry."
}

if ($Neural) {
    $env:MEDIGRAPH_INSTALL_NEURAL = "true"
    Write-Host "Neural image requested: torch/transformers will be installed."
    Write-Host "The first build is large; model files remain bind-mounted from data/."
}

$arguments = @("compose", "up", "-d")
if (-not $NoBuild) {
    $arguments += "--build"
}

Write-Host "Starting MediGraph MCP and A2A..."
& docker @arguments
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed. Check whether ports 8011 or 8100 are occupied."
}

function Get-PublishedPort([string]$Service, [int]$ContainerPort) {
    $mapping = [string](& docker compose port $Service $ContainerPort |
        Select-Object -First 1)
    $match = [regex]::Match($mapping.Trim(), ":(\d+)$")
    if (-not $match.Success) {
        throw "Cannot resolve the published port for $Service/$ContainerPort."
    }
    return [int]$match.Groups[1].Value
}

$mcpPort = Get-PublishedPort "mcp" 8011
$a2aPort = Get-PublishedPort "a2a" 8100

function Test-TcpPort([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync("127.0.0.1", $Port)
        if (-not $task.Wait(1000)) {
            return $false
        }
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

$deadline = (Get-Date).AddMinutes(2)
$mcpReady = $false
$a2aReady = $false
while ((Get-Date) -lt $deadline) {
    $mcpReady = Test-TcpPort $mcpPort
    try {
        $card = Invoke-RestMethod `
            -Uri "http://127.0.0.1:$a2aPort/.well-known/agent.json" `
            -TimeoutSec 2
        $a2aReady = [bool]$card.name
    } catch {
        $a2aReady = $false
    }
    if ($mcpReady -and $a2aReady) {
        break
    }
    Start-Sleep -Seconds 2
}

& docker compose ps

if (-not ($mcpReady -and $a2aReady)) {
    & docker compose logs --tail 80
    throw "Services did not become ready within 2 minutes."
}

Write-Host ""
Write-Host "MediGraph Docker services are ready."
Write-Host "MCP (host):   http://127.0.0.1:$mcpPort/sse"
Write-Host "MCP (Nexent): http://host.docker.internal:$mcpPort/sse"
Write-Host "A2A card:     http://127.0.0.1:$a2aPort/.well-known/agent.json"
Write-Host "Stop:         powershell -ExecutionPolicy Bypass -File scripts/docker_down.ps1"
