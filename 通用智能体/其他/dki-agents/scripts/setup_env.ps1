# Reproducible Python 3.11 environment bootstrap for Windows PowerShell.
#
# Nexent and the DataMate executor are installed with --no-deps so their
# broad upstream dependency declarations cannot replace a target machine's
# fixed torch/torch_npu stack. Their lightweight dependencies are explicit.
$ErrorActionPreference = "Stop"

function Invoke-Uv {
    param(
        [string[]] $UvArguments
    )

    & uv @UvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv failed with exit code $LASTEXITCODE`: uv $($UvArguments -join ' ')"
    }
}

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host "[1/4] Creating the Python 3.11 virtual environment"
Invoke-Uv -UvArguments @("venv", ".venv", "--python", "3.11")
$py = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "[2/4] Installing Nexent 0.1.2 without broad dependency extras"
$nexentDependencies = @(
    "smolagents==1.12.0",
    "huggingface-hub==0.28.1",
    "aiofiles==25.1.0",
    "elasticsearch==8.17.2",
    "exa_py==2.16.2",
    "httpx==0.28.1",
    "openai==2.50.0",
    "openpyxl==3.1.5",
    "pydantic==2.13.4",
    "websockets==16.1.1",
    "aiohttp==3.14.3",
    "jieba==0.42.1",
    "boto3==1.43.57",
    "botocore==1.43.57",
    "python-multipart==0.0.32",
    "markdownify==1.2.3",
    "duckduckgo-search==8.1.1"
)
Invoke-Uv -UvArguments (
    @("pip", "install", "-p", $py) + $nexentDependencies
)

$localNexentSdk = Join-Path $root "..\nexent\sdk"
if (Test-Path -LiteralPath $localNexentSdk -PathType Container) {
    Invoke-Uv -UvArguments @(
        "pip", "install", "-p", $py, "--no-deps", $localNexentSdk
    )
} else {
    Invoke-Uv -UvArguments @(
        "pip", "install", "-p", $py, "--no-deps", "nexent==0.1.2"
    )
}

Write-Host "[3/4] Installing the DataMate executor"
$dataMateRuntime = if ($env:DKI_DATAMATE_RUNTIME) {
    $env:DKI_DATAMATE_RUNTIME
} else {
    Join-Path $root "..\datamate\runtime"
}
$requireRealRaw = if ($env:DKI_REQUIRE_REAL_DATAMATE) {
    $env:DKI_REQUIRE_REAL_DATAMATE
} else {
    $env:DKI_USE_DATAMATE_RUNTIME
}
$requireRealDataMate = @(
    "1", "true", "yes", "on"
) -contains ([string] $requireRealRaw).ToLowerInvariant()
$dataMateOps = Join-Path $dataMateRuntime "ops"
$dataMateExecutor = Join-Path $dataMateRuntime "python-executor"

if (
    (Test-Path -LiteralPath $dataMateOps -PathType Container) -and
    (Test-Path -LiteralPath $dataMateExecutor -PathType Container)
) {
    $dataMateRuntime = (
        Resolve-Path -LiteralPath $dataMateRuntime -ErrorAction Stop
    ).Path
    $env:DKI_DATAMATE_RUNTIME = $dataMateRuntime
    $dataMateExecutor = Join-Path $dataMateRuntime "python-executor"
    Invoke-Uv -UvArguments @(
        "pip", "install", "-p", $py, "--no-deps", $dataMateExecutor
    )
    Invoke-Uv -UvArguments @(
        "pip",
        "install",
        "-p",
        $py,
        "opencv-python-headless",
        "numpy",
        "loguru",
        "pytz",
        "unstructured==0.18.15",
        "sqlalchemy",
        "requests",
        "emoji",
        "zhconv",
        "beautifulsoup4",
        "xmltodict",
        "datasketch"
    )
} elseif ($requireRealDataMate) {
    throw (
        "Required DataMate runtime must contain ops\ and python-executor\: " +
        $dataMateRuntime
    )
} else {
    Write-Warning (
        "DataMate runtime not found at $dataMateRuntime; " +
        "shim mode remains available."
    )
}

Write-Host "[4/4] Installing project dependencies"
Invoke-Uv -UvArguments @(
    "pip",
    "install",
    "-p",
    $py,
    "pyyaml",
    "jinja2",
    "pandas",
    "python-dotenv",
    "pytest",
    "sqlglot",
    "streamlit==1.54.0"
)

Write-Host "Environment ready. Run: .\run.ps1 demo-all"
