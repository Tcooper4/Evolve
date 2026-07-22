# One-time: register Windows Task Scheduler job to run start-evolve-mobile on logon.
param(
    [int]$DelayMinutes = 2,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$taskName = "Evolve Mobile Stack"
$wrapperScript = Join-Path $root "scripts\start-evolve-mobile-autostart.ps1"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed scheduled task: $taskName"
    exit 0
}

if (-not (Test-Path (Join-Path $root "data\tunnel_bootstrap.json"))) {
    Write-Error "Run .\scripts\setup-tunnel-bootstrap.ps1 first."
}

$psArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", "`"$wrapperScript`""
)

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument ($psArgs -join " ") `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT{0}M" -f [Math]::Max(1, $DelayMinutes)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Start Evolve Docker, cloudflared tunnel, and publish stable mobile URL." `
    | Out-Null

Write-Host ""
Write-Host "=== Autostart installed ===" -ForegroundColor Green
Write-Host "Task: $taskName"
Write-Host "Runs: ${DelayMinutes}m after you sign in to Windows"
Write-Host "Log:  data\evolve-mobile-autostart.log"
Write-Host ""
Write-Host "Remove with: .\scripts\install-evolve-mobile-autostart.ps1 -Uninstall"
