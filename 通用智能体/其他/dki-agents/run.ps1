# dki 运行入口（自动激活 venv + UTF-8）
# 用法示例：.\run.ps1 demo-all    /    .\run.ps1 task2-qa --question "高血压有哪些症状？"
param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
$env:PYTHONUTF8 = 1
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[!] 未找到 .venv，请先运行 scripts\setup_env.ps1" -ForegroundColor Yellow
    exit 1
}
Set-Location $PSScriptRoot
& $py -m dki @Rest
