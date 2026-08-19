param(
    [string]$DatamateComposeDir,
    [string]$ComposeFile = "docker-compose.local.yml",
    [ValidateSet("up", "start")]
    [string]$Mode = "up",
    [string]$Registry = "ghcr.io/modelengine-group/",
    [string]$DatamateUrl = "http://localhost:18000",
    [string]$Distro = "Ubuntu",
    [int]$KeepAliveSeconds = 1800,
    [int]$WaitSeconds = 90,
    [int]$ProbeIntervalSeconds = 10,
    [string]$EvidenceDir,
    [switch]$NoKeepAlive,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-DefaultComposeDir {
    $projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    $workspaceRoot = Resolve-Path (Join-Path $projectRoot "..")
    return Join-Path $workspaceRoot "DataMate\deployment\docker\datamate"
}

function Convert-ToWslPath {
    param([string]$Path)

    if ($Path -match "^/") {
        return $Path
    }

    if ($Path -match "^([A-Za-z]):[\\/](.*)$") {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2] -replace "\\", "/"
        return "/mnt/$drive/$rest"
    }

    throw "Unsupported path format for WSL conversion: $Path"
}

function Quote-Bash {
    param([string]$Value)
    return "'" + ($Value -replace "'", "'\''") + "'"
}

function Write-Log {
    param(
        [string]$Path,
        [string]$Message
    )
    $Message | Tee-Object -FilePath $Path -Append
}

if (-not $DatamateComposeDir) {
    $DatamateComposeDir = Resolve-DefaultComposeDir
}

if (-not $EvidenceDir) {
    $projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    $EvidenceDir = Join-Path $projectRoot "outputs\competition_evidence\datamate-wsl-start"
}

$composeDirWsl = Convert-ToWslPath $DatamateComposeDir
$readinessScript = Join-Path $PSScriptRoot "datamate_readiness.py"
$composeAction = if ($Mode -eq "start") {
    "start datamate-database datamate-runtime datamate-frontend datamate-backend-python datamate-backend datamate-gateway"
} else {
    "up -d"
}
$composeCommand = "cd $(Quote-Bash $composeDirWsl) && REGISTRY=$(Quote-Bash $Registry) docker compose -f $(Quote-Bash $ComposeFile) $composeAction"

if ($DryRun) {
    Write-Host "DataMate compose dir: $DatamateComposeDir"
    Write-Host "DataMate compose dir (WSL): $composeDirWsl"
    Write-Host "WSL distro: $Distro"
    Write-Host "Compose command: $composeCommand"
    Write-Host "Readiness command: python $readinessScript --url $DatamateUrl --timeout 8"
    if ($NoKeepAlive) {
        Write-Host "Keepalive: disabled"
    } else {
        Write-Host "Keepalive command: wsl.exe -d $Distro -- sleep $KeepAliveSeconds"
    }
    exit 0
}

New-Item -ItemType Directory -Force -Path $EvidenceDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $EvidenceDir "start-datamate-wsl-$stamp.log"

Write-Log $logPath "timestamp=$(Get-Date -Format o)"
Write-Log $logPath "compose_dir=$DatamateComposeDir"
Write-Log $logPath "compose_dir_wsl=$composeDirWsl"
Write-Log $logPath "compose_file=$ComposeFile"
Write-Log $logPath "mode=$Mode"
Write-Log $logPath "datamate_url=$DatamateUrl"
Write-Log $logPath "wsl_distro=$Distro"

if (-not $NoKeepAlive) {
    $keepAlive = Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $Distro, "--", "sleep", [string]$KeepAliveSeconds) -WindowStyle Hidden -PassThru
    Write-Log $logPath "keepalive_pid=$($keepAlive.Id)"
    Write-Log $logPath "keepalive_seconds=$KeepAliveSeconds"
} else {
    Write-Log $logPath "keepalive=disabled"
}

Write-Log $logPath "compose_command=$composeCommand"
$composeOutput = & wsl.exe -d $Distro -- bash -lc "$composeCommand 2>&1"
$composeExitCode = $LASTEXITCODE
$composeOutput | Tee-Object -FilePath $logPath -Append
if ($composeExitCode -ne 0) {
    throw "DataMate compose command failed. See $logPath"
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$probeIndex = 0
do {
    Write-Log $logPath ""
    Write-Log $logPath "probe=$probeIndex timestamp=$(Get-Date -Format o)"
    $readinessOutput = & python $readinessScript --url $DatamateUrl --timeout 8 2>&1
    $readinessExitCode = $LASTEXITCODE
    Write-Log $logPath "readiness_exit_code=$readinessExitCode report=$readinessOutput"

    & wsl.exe -d $Distro -- docker ps -a --filter name=datamate --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>&1 |
        Tee-Object -FilePath $logPath -Append

    if ($readinessExitCode -eq 0) {
        Write-Log $logPath "status=ready"
        Write-Host "DataMate is ready. Evidence log: $logPath"
        exit 0
    }

    $probeIndex += 1
    Start-Sleep -Seconds $ProbeIntervalSeconds
} while ((Get-Date) -lt $deadline)

Write-Log $logPath "status=not_ready_after_${WaitSeconds}s"
Write-Host "DataMate did not become ready within $WaitSeconds seconds. Evidence log: $logPath"
exit 1
