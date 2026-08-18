$ports = 8011, 18088, 8100

Write-Host "Listening ports:"
$listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $ports -contains $_.LocalPort } |
    Select-Object LocalAddress, LocalPort, OwningProcess
if ($listeners) {
    $listeners | Format-Table -AutoSize
} else {
    Write-Host "  none (demo services are stopped)"
}

Write-Host "Fine-tuned model API:"
try {
    $models = Invoke-RestMethod -Uri "http://localhost:18088/v1/models" -TimeoutSec 3
    Write-Host "  healthy - model:" $models.data[0].id
} catch {
    Write-Host "  stopped/unreachable"
}

Write-Host "A2A collaborator:"
try {
    $card = Invoke-RestMethod -Uri "http://localhost:8100/.well-known/agent.json" -TimeoutSec 3
    Write-Host "  healthy - agent:" $card.name
} catch {
    Write-Host "  stopped/unreachable"
}

Write-Host "Nexent MCP URL (from Docker): http://host.docker.internal:8011/sse"
Write-Host "Fine-tuned model URL (from Docker): http://host.docker.internal:18088/v1/"
Write-Host "A2A Agent Card URL (from Docker): http://host.docker.internal:8100/.well-known/agent.json"
