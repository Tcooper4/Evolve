# Deploying Evolve as a live website (friends & family)

This guide takes Evolve from a local app to a small hosted site with real
logins, where each person gets their own persistent workspace — watchlist,
chat history, preferences, and memory that learns from them individually.

## How multi-user mode works

- `EVOLVE_REQUIRE_LOGIN=1` turns on the login gate on every page.
  Unauthenticated visitors see only the sign-in form.
- On sign-in, the username becomes the platform-wide identity
  (`user:<name>`). The memory store, settings, chat learning, and
  watchlist are all keyed on it, so each account is isolated and adapts
  to its own user.
- Accounts live in `data/accounts.db` (bcrypt hashes only). Manage them
  with `python scripts/manage_users.py add|list|passwd|deactivate`.
- First run with no accounts shows a one-time form to create the admin.
- Sessions persist across refreshes via a signed cookie. Set
  `EVOLVE_AUTH_SECRET` to a long random string so restarts don't log
  everyone out (otherwise one is generated to `data/.auth_secret`).

Personal mode is unchanged: with `EVOLVE_REQUIRE_LOGIN` unset, no login
appears and everything keys to `local` exactly as before.

## Recommended setup: one small VPS

A $6–12/month VPS (Hetzner, DigitalOcean, Lightsail) with 4GB RAM is
plenty for a handful of users.

### 1. Server prep (Ubuntu 24)

```bash
sudo apt update && sudo apt install -y python3-venv git caddy
git clone -b codebase-audit-consolidated https://github.com/Tcooper4/Evolve.git
cd Evolve
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY etc.
```

### 2. Environment

Add to `.env` (or the systemd unit):

```
EVOLVE_REQUIRE_LOGIN=1
EVOLVE_AUTH_SECRET=<output of: python3 -c "import secrets;print(secrets.token_hex(32))">
```

### 3. Run as a service

`/etc/systemd/system/evolve.service`:

```ini
[Unit]
Description=Evolve Trading Terminal
After=network.target

[Service]
User=evolve
WorkingDirectory=/home/evolve/Evolve
EnvironmentFile=/home/evolve/Evolve/.env
ExecStart=/home/evolve/Evolve/.venv/bin/streamlit run app.py --server.port 8501 --server.address 127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now evolve
```

### 4. HTTPS with Caddy (automatic TLS)

`/etc/caddy/Caddyfile` — point a domain (or free DuckDNS subdomain) at
the server, then:

```
evolve.yourdomain.com {
    reverse_proxy 127.0.0.1:8501
}
```

```bash
sudo systemctl reload caddy
```

Caddy provisions and renews the certificate automatically. **Never expose
port 8501 directly** — only Caddy should be reachable (e.g.
`ufw allow 80,443/tcp` and nothing else).

### 4b. HTTPS is required for the installable app (PWA)

The React UI (`docker compose`, port 8000) ships as a **Progressive Web
App**: browsers let users "Install" it to the home screen / dock and it
runs in its own standalone window. The service worker that powers this
(precaches the static app shell for fast repeat loads) **only registers
over a secure origin**:

- **HTTPS in production** — the service worker, install prompt, and
  `Add to Home Screen` are silently disabled on plain `http://` origins.
  Terminate TLS at your reverse proxy (the Caddy step above gives you a
  valid cert automatically; nginx/Traefik + Let's Encrypt work too).
- **`http://localhost` is exempt** — browsers treat localhost as a
  secure context, so local dev and `docker compose up` on your own
  machine install fine without a cert.
- **LAN IPs (e.g. `http://192.168.x.x`) are NOT secure contexts** — the
  PWA will not install. Put a tunnel (Cloudflare Tunnel, Tailscale
  Funnel, ngrok) or a reverse proxy with a real cert in front if you want
  to install from another device on your network.

What the PWA does **not** do: it never caches live data. `/api/*` and
`/ws/*` are network-only, so quotes, portfolio, chat, and alerts are
always fresh — the offline shell only serves the static UI, and a small
non-intrusive "Update available — refresh" prompt appears when a new
build is deployed.

If a user opens the site over plain HTTP outside localhost, the app shows
a small dismissible banner explaining that install/offline features need
HTTPS, rather than failing silently.

### 4c. Cloudflare Tunnel (HTTPS without opening ports)

For friends/family access or PWA install from a phone, expose local
Evolve over HTTPS with **cloudflared** — no router port-forwarding.

**Quick test** (temporary URL, no account):

```powershell
cloudflared tunnel --url http://localhost:8000
```

Copy the `https://*.trycloudflare.com` URL from the output. Evolve must
already be running (`docker compose up -d` on port 8000).

**Persistent named tunnel** (recommended):

```powershell
cloudflared tunnel login          # browser once — pick your Cloudflare zone
cloudflared tunnel create evolve  # note the tunnel UUID
# Edit cloudflared/config.yml — paste UUID, optional hostname
.\scripts\start-cloudflare-tunnel.ps1
```

Optional DNS: `cloudflared tunnel route dns evolve evolve.yourdomain.com`

Full steps: `cloudflared/README.md`.

### 5. Create accounts

```bash
python scripts/manage_users.py add thomas --name "Thomas" --admin
python scripts/manage_users.py add mom --name "Mom"
```

Each person signs in at your domain and lands in their own workspace.

## Backups

Everything per-user lives in `data/*.db`. A nightly copy is enough:

```bash
crontab -e
0 3 * * * tar czf ~/evolve-backup-$(date +\%F).tgz -C ~/Evolve data/
```

## Per-user API keys

Each person enters their own API keys under **Settings → API keys** after
signing in. Keys are encrypted at rest (Fernet; set `EVOLVE_ENCRYPTION_KEY`
in `.env` and keep it out of version control) and every request resolves
the **current user's** key — never another account's, and never via the
process environment (which is shared).

Fallback policy: if a user hasn't entered a key, the server's `.env` key
is used by default. To require everyone to bring their own keys (so nobody
can spend yours), set:

```
EVOLVE_SHARED_KEYS=0
```

## Honest security notes

This setup is appropriate for trusted friends and family, not the public
internet at scale: passwords are bcrypt-hashed, per-user API keys are
encrypted at rest and resolved per-request, sessions are signed, TLS
terminates at Caddy, and pages are fully gated — but there is no rate
limiting, no 2FA, no audit logging, and Streamlit itself is not hardened
for adversarial traffic. Don't post the URL publicly, keep the user list
to people you trust, and keep the server patched (`unattended-upgrades`).
