# One-time setup: deploy Cloudflare Worker + save bootstrap URL for start-evolve-mobile.ps1
param(
    [string]$WorkerName = "evolve-tunnel-bootstrap"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$workerDir = Join-Path $root "cloudflared\tunnel-bootstrap\worker"
$configPath = Join-Path $root "data\tunnel_bootstrap.json"
$wranglerToml = Join-Path $workerDir "wrangler.toml"

function New-Secret {
    -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
}

Write-Host ""
Write-Host "=== Evolve mobile bootstrap (stable phone URL) ===" -ForegroundColor Cyan
Write-Host "Free Cloudflare Worker redirects your phone to the latest trycloudflare tunnel."
Write-Host ""

# wrangler (local via npm in worker dir, or global)
$wrangler = Get-Command wrangler -ErrorAction SilentlyContinue
$wranglerCmd = "wrangler"
if (-not $wrangler) {
    if (Test-Path (Join-Path $workerDir "node_modules\.bin\wrangler.cmd")) {
        $wranglerCmd = Join-Path $workerDir "node_modules\.bin\wrangler.cmd"
    } else {
        Write-Host "Installing wrangler (one-time npm install in worker dir)..."
        Push-Location $workerDir
        try {
            npm install 2>&1 | Out-Host
            $wranglerCmd = Join-Path $workerDir "node_modules\.bin\wrangler.cmd"
            if (-not (Test-Path $wranglerCmd)) {
                Write-Host "npm/wrangler not available. Install Node.js from https://nodejs.org" -ForegroundColor Yellow
                $manual = Read-Host "Already deployed a Worker? Enter bootstrap URL (or Enter to exit)"
                if (-not $manual) { exit 1 }
                $secret = New-Secret
                $cfg = @{
                    bootstrap_url = $manual.Trim().TrimEnd("/")
                    update_secret = $secret
                    origin        = "http://127.0.0.1:8000"
                }
                New-Item -ItemType Directory -Force -Path (Split-Path $configPath) | Out-Null
                $cfg | ConvertTo-Json | Set-Content -Path $configPath -Encoding UTF8
                Write-Host "Saved $configPath - set Worker secret UPDATE_SECRET to: $secret"
                exit 0
            }
        } finally {
            Pop-Location
        }
    }
}

Push-Location $workerDir
try {
    Write-Host "Checking Cloudflare login..."
    & $wranglerCmd whoami 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Run: wrangler login (browser opens once)" -ForegroundColor Yellow
        & $wranglerCmd login
    }

    $toml = Get-Content $wranglerToml -Raw
    if ($toml -match "REPLACE_WITH_KV_NAMESPACE_ID") {
        Write-Host "Creating KV namespace TUNNEL..."
        $kvOut = (& $wranglerCmd kv namespace create TUNNEL 2>&1 | Out-String)
        Write-Host $kvOut
        $kvPattern = 'id = "' + [char]40 + [char]91 + 'a-f0-9-' + [char]93 + '+' + [char]41 + '"'
        $kvMatch = [regex]::Match($kvOut, $kvPattern)
        if ($kvMatch.Success) {
            $kvId = $kvMatch.Groups[1].Value
            $toml = $toml -replace "REPLACE_WITH_KV_NAMESPACE_ID", $kvId
            Set-Content -Path $wranglerToml -Value $toml -Encoding UTF8
            Write-Host "Updated wrangler.toml with KV id $kvId"
        } else {
            Write-Error "Could not parse KV namespace id - create manually and edit wrangler.toml"
        }
    }

    $secret = New-Secret
    Write-Host "Setting Worker secret UPDATE_SECRET..."
    $secret | & $wranglerCmd secret put UPDATE_SECRET

    Write-Host "Deploying Worker..."
    $deployOut = (& $wranglerCmd deploy 2>&1 | Out-String)
    Write-Host $deployOut
    $workerPattern = 'https://' + [char]91 + 'a-z0-9-' + [char]93 + '\.workers\.dev'
    $deployMatch = [regex]::Match($deployOut, $workerPattern)
    if (-not $deployMatch.Success) {
        Write-Error "Deploy finished but could not find workers.dev URL in output."
    }
    $bootstrapUrl = $deployMatch.Groups[0].Value

    $cfg = @{
        bootstrap_url = $bootstrapUrl
        update_secret = $secret
        origin        = "http://127.0.0.1:8000"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $configPath) | Out-Null
    $cfg | ConvertTo-Json | Set-Content -Path $configPath -Encoding UTF8

    Write-Host ""
    Write-Host "=== Done ===" -ForegroundColor Green
    Write-Host "Stable phone URL (bookmark / Add to Home Screen):"
    Write-Host "  $bootstrapUrl" -ForegroundColor White
    Write-Host ""
    Write-Host "Config saved: data\tunnel_bootstrap.json (keep private - contains update secret)"
    Write-Host "Start stack:  .\scripts\start-evolve-mobile.ps1"
    Write-Host "Autostart:    .\scripts\install-evolve-mobile-autostart.ps1"
}
finally {
    Pop-Location
}
