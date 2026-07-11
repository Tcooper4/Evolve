# Evolve Web (React + FastAPI) — migration vertical slice

The Streamlit app remains the primary interface; this directory is the
in-progress rewrite. One identity system: accounts created with
`scripts/manage_users.py` work in both frontends.

## Run (two terminals)

```bash
# 1. API (from repo root)
pip install fastapi uvicorn "python-jose[cryptography]"
uvicorn web.backend.main:app --port 8000

# 2. Frontend
cd web/frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api to :8000)
```

## Design system (v2)
Dark-first terminal aesthetic per current fintech practice: near-black
layered surfaces with radial ambient glows; neutral foundation with the
cyan brand accent used precisely; green/red reserved as signals (deltas,
candles, sparklines) never decoration; tabular numerals for all prices;
motion as functional feedback (price flash on change, skeleton loaders,
hover lifts). Layout: watchlist rail with per-symbol sparkline cards ->
KPI strip (hero price + delta pill, prev close, period range, period
change) -> chart card with 1M/3M/6M/1Y segmented control, candlesticks +
volume histogram, crosshair with live OHLC legend. Keyboard: "/" focuses
symbol search.

## What works in this slice
- JWT login against the shared accounts DB (same bcrypt hashes)
- Candlestick chart (lightweight-charts) with the Evolve theme
- Quote card with change coloring; symbol aliasing (SPX -> ^GSPC)
- Per-user watchlist (add/remove/click-to-load), scoped exactly like
  the Streamlit side
- Per-request identity: the multi-user plumbing (API keys, memory)
  scopes to the JWT user

## Not yet built (next sessions)
Analyze/forecast views, backtest UI, scanner, chat, settings,
websocket streaming quotes (the kept streaming_pipeline becomes
relevant), production build serving via FastAPI StaticFiles + Caddy.
