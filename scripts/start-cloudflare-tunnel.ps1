# Start Cloudflare Tunnel for Evolve (localhost:8000).
# Requires: docker compose up, cloudflared installed, config.yml filled in.
param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
if (-not (Test-Path $cloudflared)) {
    $cloudflared = "cloudflared"
}

if ($Quick) {
    Write-Host "Starting quick tunnel (temporary trycloudflare.com URL)..."
    & $cloudflared tunnel --url http://127.0.0.1:8000
    exit $LASTEXITCODE
}

$config = Join-Path $root "cloudflared\config.yml"
if (-not (Test-Path $config)) {
    Write-Error "Missing $config — run setup in cloudflared/README.md first."
}

if ((Get-Content $config -Raw) -match "REPLACE_WITH_TUNNEL_ID") {
    Write-Error @"
config.yml still has placeholder tunnel ID.

One-time setup:
  cloudflared tunnel login
  cloudflared tunnel create evolve
  Edit cloudflared/config.yml with the tunnel UUID from create output.

Or use quick mode: .\scripts\start-cloudflare-tunnel.ps1 -Quick
"@
}

Write-Host "Starting named tunnel 'evolve' -> http://127.0.0.1:8000"
& $cloudflared tunnel --config $config run evolve
