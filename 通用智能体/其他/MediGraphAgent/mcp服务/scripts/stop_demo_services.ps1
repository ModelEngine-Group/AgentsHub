$targets = @(
    @{ Port = 8011; Pattern = "mcp_server\server.py" },
    @{ Port = 18088; Pattern = "finetune\openai_compatible_server.py" },
    @{ Port = 8100; Pattern = "integration\a2a\a2a_server.py" }
)

foreach ($target in $targets) {
    $listeners = Get-NetTCPConnection -State Listen -LocalPort $target.Port -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)" -ErrorAction SilentlyContinue
        if ($process -and $process.CommandLine -like "*$($target.Pattern)*") {
            Stop-Process -Id $process.ProcessId -Force
            Write-Host "Stopped port $($target.Port), PID $($process.ProcessId)"
        } elseif ($process) {
            Write-Warning "Skipped PID $($process.ProcessId): command line did not match $($target.Pattern)"
        }
    }
}
