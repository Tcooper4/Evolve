# Wrapper for Task Scheduler: wait for Docker, then run start-evolve-mobile.ps1.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$logPath = Join-Path $root "data\evolve-mobile-autostart.log"
$startScript = Join-Path $root "scripts\start-evolve-mobile.ps1"
$configPath = Join-Path $root "data\tunnel_bootstrap.json"

function Write-Log($Message) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null
    Add-Content -Path $logPath -Value $line -Encoding UTF8
}

function Wait-DockerReady([int]$Minutes = 12) {
    $deadline = (Get-Date).AddMinutes($Minutes)
    while ((Get-Date) -lt $deadline) {
        try {
            & docker info *> $null
            if ($LASTEXITCODE -eq 0) { return $true }
        } catch { }
        Start-Sleep -Seconds 15
    }
    return $false
}

Write-Log "=== autostart begin ==="

if (-not (Test-Path $configPath)) {
    Write-Log "Skip: missing data\tunnel_bootstrap.json (run setup-tunnel-bootstrap.ps1)"
    exit 0
}

Write-Log "Waiting for Docker..."
if (-not (Wait-DockerReady)) {
    Write-Log "Docker not ready after timeout; aborting."
    exit 1
}
Write-Log "Docker ready; starting mobile stack."

try {
    & $startScript *>> $logPath
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Log "start-evolve-mobile.ps1 exited with code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Log "=== autostart complete ==="
    exit 0
} catch {
    Write-Log "Error: $($_.Exception.Message)"
    exit 1
}
