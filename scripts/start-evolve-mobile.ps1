# Start Docker Evolve + quick Cloudflare tunnel + publish URL to stable bootstrap Worker.
# One-time: run scripts/setup-tunnel-bootstrap.ps1 first.
param(
    [switch]$SkipDocker,
    [switch]$NoPublish
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$configPath = Join-Path $root "data\tunnel_bootstrap.json"
$cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
if (-not (Test-Path $cloudflared)) { $cloudflared = "cloudflared" }
$python = Join-Path $root "evolve_venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$logDir = Join-Path $root "data"
$cfLog = Join-Path $logDir "cloudflared-quick.log"

function Wait-HttpOk($Url, $Seconds = 90) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { return $true }
        } catch { }
        Start-Sleep -Seconds 2
    }
    return $false
}

if (-not (Test-Path $configPath)) {
    Write-Error "Missing data\tunnel_bootstrap.json - run .\scripts\setup-tunnel-bootstrap.ps1 first."
}

$cfg = Get-Content $configPath -Raw | ConvertFrom-Json
$origin = if ($cfg.origin) { $cfg.origin } else { "http://127.0.0.1:8000" }
$bootstrapUrl = $cfg.bootstrap_url

Write-Host "=== Evolve mobile stack ===" -ForegroundColor Cyan

if (-not $SkipDocker) {
    Write-Host "Starting Docker..."
    Push-Location $root
    try {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        docker compose up -d 2>&1 | Out-Host
        $ErrorActionPreference = $prevEap
    } finally {
        Pop-Location
    }
    Write-Host "Waiting for $origin ..."
    if (-not (Wait-HttpOk "$origin/api/health")) {
        Write-Warning "Health check slow - continuing anyway."
    }
}

Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object {
        $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -notmatch "com.docker|docker|wsl") {
            Write-Host "Stopping non-Docker listener PID $($_.OwningProcess) ($($proc.ProcessName))"
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }

Write-Host "Restarting cloudflared quick tunnel -> $origin"
Stop-Process -Name cloudflared -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path $cfLog) { Remove-Item $cfLog -Force }

$cfProc = Start-Process -FilePath $cloudflared -ArgumentList @(
    "tunnel", "--url", $origin, "--protocol", "http2"
) -RedirectStandardError $cfLog -PassThru -WindowStyle Hidden

Write-Host "Waiting for trycloudflare.com URL (pid $($cfProc.Id))..."
$tunnelUrl = $null
$deadline = (Get-Date).AddSeconds(90)
$tunnelPattern = 'https://' + [char]91 + 'a-z0-9-' + [char]93 + '+\.trycloudflare\.com'
while ((Get-Date) -lt $deadline) {
    if ($cfProc.HasExited) {
        Write-Error "cloudflared exited early. Log:`n$(Get-Content $cfLog -Raw -ErrorAction SilentlyContinue)"
    }
    if (Test-Path $cfLog) {
        $text = Get-Content $cfLog -Raw -ErrorAction SilentlyContinue
        if ($text) {
            $tunnelMatch = [regex]::Match($text, $tunnelPattern)
            if ($tunnelMatch.Success) {
                $tunnelUrl = $tunnelMatch.Groups[0].Value
                break
            }
        }
    }
    Start-Sleep -Seconds 1
}

if (-not $tunnelUrl) {
    Write-Error "Timed out waiting for tunnel URL. See $cfLog"
}

Write-Host "Tunnel: $tunnelUrl" -ForegroundColor Green

if (-not $NoPublish) {
    Write-Host "Publishing to bootstrap Worker..."
    & $python (Join-Path $root "scripts\tunnel_bootstrap\publish_url.py") $tunnelUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Publish failed - phone bookmark may be stale until publish succeeds."
    }
}

Write-Host ""
Write-Host "=== Phone URL (stable - bookmark this) ===" -ForegroundColor Cyan
Write-Host "  $bootstrapUrl"
Write-Host ""
Write-Host "Direct tunnel (changes on restart): $tunnelUrl"
Write-Host "Leave cloudflared running (PID $($cfProc.Id))."
