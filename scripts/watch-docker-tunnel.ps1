# Watches Docker for Evolve container restarts and republishes the mobile tunnel.
# Install via: .\scripts\install-evolve-mobile-autostart.ps1
param(
    [int]$DebounceSeconds = 20
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$logPath = Join-Path $root "data\evolve-tunnel-watcher.log"
$configPath = Join-Path $root "data\tunnel_bootstrap.json"
$startScript = Join-Path $root "scripts\start-evolve-mobile.ps1"
$script:lastRepublish = [DateTime]::MinValue

function Write-Log($Message) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null
    Add-Content -Path $logPath -Value $line -Encoding UTF8
}

function Wait-DockerReady([int]$Minutes = 15) {
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

function Wait-Health([int]$Seconds = 120) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { return $true }
        } catch { }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Invoke-Republish([string]$Reason) {
    $now = Get-Date
    if (($now - $script:lastRepublish).TotalSeconds -lt $DebounceSeconds) {
        Write-Log "Debounced republish ($Reason)"
        return
    }

    try {
        Write-Log "Republish begin: $Reason"
        if (-not (Wait-Health)) {
            Write-Log "Health check failed - skipping republish"
            return
        }
        & $startScript -SkipDocker *>> $logPath
        $script:lastRepublish = Get-Date
        Write-Log "Republish complete (exit $LASTEXITCODE)"
    } catch {
        Write-Log "Republish error: $($_.Exception.Message)"
    }
}

Write-Log "=== tunnel watcher starting ==="

if (-not (Test-Path $configPath)) {
    Write-Log "Missing data\tunnel_bootstrap.json - exit"
    exit 0
}

Write-Log "Waiting for Docker..."
if (-not (Wait-DockerReady)) {
    Write-Log "Docker not ready - exit"
    exit 1
}

Write-Log "Listening for evolve container start/restart (debounce ${DebounceSeconds}s)..."

try {
    & docker events --filter "label=com.docker.compose.service=evolve" --format "{{.Action}}|{{.Actor.Attributes.name}}" |
        ForEach-Object {
            $line = $_.Trim()
            if (-not $line) { return }
            $parts = $line -split '\|', 2
            $action = $parts[0]
            $name = if ($parts.Length -gt 1) { $parts[1] } else { "?" }
            if ($action -in @("start", "restart", "unpause")) {
                Write-Log "Docker event: $action $name"
                Invoke-Republish "$action $name"
            }
        }
} catch {
    Write-Log "Watcher stopped: $($_.Exception.Message)"
    exit 1
}
