# Stable mobile URL (free bootstrap)

Quick Cloudflare tunnels (`*.trycloudflare.com`) **change every restart**.
This bootstrap gives you **one permanent phone URL** that auto-redirects to
the latest tunnel.

## How it works

```
Phone (bookmark)  →  https://you.workers.dev  →  latest trycloudflare URL  →  Evolve on your PC
                              ↑
                    PC script updates this when tunnel starts
```

## One-time setup (~5 min, free)

1. **Cloudflare account** (free) — [dash.cloudflare.com](https://dash.cloudflare.com/sign-up)

2. **Node.js** installed — setup runs `npm install` in the worker folder automatically.
   ```powershell
   wrangler login   # only if not already logged in — browser opens once
   ```

3. **Deploy bootstrap Worker + save config:**
   ```powershell
   cd C:\Users\Thomas\OneDrive\Desktop\Dashboard\evolve_clean
   .\scripts\setup-tunnel-bootstrap.ps1
   ```

4. **On your phone:** open the printed `https://….workers.dev` URL →
   **Add to Home Screen**. Use that icon forever (not the trycloudflare link).

## Daily use

```powershell
.\scripts\start-evolve-mobile.ps1
```

This will:
- Start Docker (`docker compose up -d`)
- Start a new quick tunnel to `127.0.0.1:8000`
- Publish the new tunnel URL to your Worker
- Print your **stable** bootstrap URL again

Keep `cloudflared` running (the script leaves it in the background).

## Files

| File | Purpose |
|------|---------|
| `worker/src/index.js` | Redirect + `/update` API |
| `worker/wrangler.toml` | Worker config (KV id filled by setup) |
| `data/tunnel_bootstrap.json` | Bootstrap URL + secret (**private**, gitignored) |

## Manual Worker deploy

If setup script fails, from `cloudflared/tunnel-bootstrap/worker`:

```powershell
wrangler kv namespace create TUNNEL
# paste id into wrangler.toml
wrangler secret put UPDATE_SECRET
wrangler deploy
```

Then create `data/tunnel_bootstrap.json`:

```json
{
  "bootstrap_url": "https://evolve-tunnel-bootstrap.your-subdomain.workers.dev",
  "update_secret": "same secret as UPDATE_SECRET",
  "origin": "http://127.0.0.1:8000"
}
```

## Security

- Only your PC can update the tunnel URL (needs `X-Evolve-Secret`).
- Only `https://*.trycloudflare.com` URLs are accepted.
- Anyone with the bootstrap URL can **try** to open Evolve when your PC is
  running — same as sharing a trycloudflare link. Use Evolve login
  (`EVOLVE_REQUIRE_LOGIN=1`) if exposed.
