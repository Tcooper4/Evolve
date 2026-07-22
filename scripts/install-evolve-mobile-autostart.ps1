# One-time: register Windows Task Scheduler jobs for mobile tunnel + Docker restart watcher.
param(
    [int]$DelayMinutes = 2,
    [int]$WatcherDelayMinutes = 3,
    [switch]$StartWatcherNow,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$taskName = "Evolve Mobile Stack"
$watcherTaskName = "Evolve Tunnel Watcher"
$wrapperScript = Join-Path $root "scripts\start-evolve-mobile-autostart.ps1"
$watcherScript = Join-Path $root "scripts\watch-docker-tunnel.ps1"

function Register-LogonTask(
    [string]$Name,
    [string]$ScriptPath,
    [int]$DelayMin,
    [string]$Description
) {
    $psArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-WindowStyle", "Hidden",
        "-File", "`"$ScriptPath`""
    )
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument ($psArgs -join " ") `
        -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $trigger.Delay = "PT{0}M" -f [Math]::Max(1, $DelayMin)
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    }
    Register-ScheduledTask `
        -TaskName $Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description $Description `
        | Out-Null
}

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $watcherTaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled tasks: $taskName, $watcherTaskName"
    exit 0
}

if (-not (Test-Path (Join-Path $root "data\tunnel_bootstrap.json"))) {
    Write-Error "Run .\scripts\setup-tunnel-bootstrap.ps1 first."
}

Register-LogonTask `
    -Name $taskName `
    -ScriptPath $wrapperScript `
    -DelayMin $DelayMinutes `
    -Description "Start Evolve Docker, cloudflared tunnel, and publish stable mobile URL."

Register-LogonTask `
    -Name $watcherTaskName `
    -ScriptPath $watcherScript `
    -DelayMin $WatcherDelayMinutes `
    -Description "Republish mobile tunnel when Evolve Docker container restarts."

if ($StartWatcherNow) {
    $watcherPsArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-WindowStyle", "Hidden",
        "-File", "`"$watcherScript`""
    )
    Start-Process -FilePath "powershell.exe" -ArgumentList ($watcherPsArgs -join " ") -WorkingDirectory $root -WindowStyle Hidden
    Write-Host "Started tunnel watcher in background (republishes on Docker restart)."
}

Write-Host ""
Write-Host "=== Autostart installed ===" -ForegroundColor Green
Write-Host "Task: $taskName"
Write-Host "  Runs: ${DelayMinutes}m after logon — Docker + tunnel + publish"
Write-Host "Task: $watcherTaskName"
Write-Host "  Runs: ${WatcherDelayMinutes}m after logon — republish when Docker restarts"
Write-Host "Logs: data\evolve-mobile-autostart.log, data\evolve-tunnel-watcher.log"
Write-Host ""
Write-Host "Start watcher now (no reboot): .\scripts\install-evolve-mobile-autostart.ps1 -StartWatcherNow"
Write-Host "Remove with: .\scripts\install-evolve-mobile-autostart.ps1 -Uninstall"
