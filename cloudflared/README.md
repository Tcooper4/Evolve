# Cloudflare Tunnel — Evolve
#
# Exposes http://localhost:8000 (docker compose) over HTTPS so the PWA can
# install from phones/tablets on your network or remotely.
#
# ## Quick start (temporary URL, no Cloudflare account)
#
#   cloudflared tunnel --url http://localhost:8000
#
# Prints a https://*.trycloudflare.com URL — good for testing; URL changes
# each run.
#
# ## Named tunnel (persistent)
#
# 1. Log in (opens browser once):
#      cloudflared tunnel login
#
# 2. Create tunnel + credentials:
#      cloudflared tunnel create evolve
#      (note the tunnel UUID printed)
#
# 3. Edit cloudflared/config.yml — replace REPLACE_WITH_TUNNEL_ID with UUID.
#
# 4. Optional — map a domain you manage in Cloudflare:
#      cloudflared tunnel route dns evolve evolve.yourdomain.com
#    Then uncomment the hostname block in config.yml.
#
# 5. Run:
#      .\scripts\start-cloudflare-tunnel.ps1
#
# Or install as a Windows service (runs on boot):
#      cloudflared service install --config cloudflared\config.yml
#      cloudflared service start
#
# ## Verify
#
# - Open the tunnel URL in Chrome → Install app (PWA) should appear.
# - Alerts still use WebSocket; tunnel must stay running.
