# Evolve Web (React + FastAPI)

Primary day-to-day UI for Evolve. Shares accounts, paper portfolio,
preferences, chat brain, and `trading/*` engines with Streamlit.

Streamlit (`app.py`) remains available for the classic terminal.
Identity: accounts from `scripts/manage_users.py` work in both UIs.

## Run

### Docker (recommended)

From repo root:

```bash
docker compose up -d     # http://localhost:8000 (API + built React)
```

### Two terminals (dev)

```bash
# 1. API (repo root)
uvicorn web.backend.main:app --port 8000

# 2. Frontend
cd web/frontend
npm install
npm run dev        # http://localhost:5173 (proxies /api → :8000)
```

## Pages

| Route | What it covers |
|-------|----------------|
| **Dashboard** | Watchlist / KPI / briefing surface |
| **Analyze** | Chart (volume + news marks, optional overlays), AI Score, forecast, news, options context (GEX/skew), earnings, filings, labs (IC, Diagnostics, patterns, GNN), Monte Carlo |
| **Scanner** | Universe / pairs |
| **Portfolio** | Positions, cashbook, limits, alerts (one-shot / compound / watch|action), account risk (Kelly + stress + concentration) |
| **Backtest** | Strategy runs and comparison hooks |
| **Chat** | Same `chat_turn` stack as Streamlit (tools + skills) |
| **Settings** | Prefs, keys, chart timezone |

## Features worth knowing

- **Chart events** — significant / notable volume with news linkage
  honesty (`same_day` vs `fallback_recent`); robust trimmed-mean volume
  baseline for spike detection.
- **Overlays (default-off)** — strategy signal marks; options-structure
  research guide (IC / PCS / CCS / WAIT from delayed GEX + skew).
- **Background jobs** — limit fills + alerts without an open browser
  (`EVOLVE_BACKGROUND_JOBS`; GEX snapshot log is separate and opt-in).
- **Diagnostics lab** — econometric stationarity / ARCH / breaks (not
  Granger causality). Legacy URL `/api/causal/{symbol}` aliases
  `/api/diagnostics/{symbol}`.
- **Monte Carlo** — i.i.d. bootstrap by default; `method=stationary_block`
  available with documented comparison caveat.

## Design system

Dark terminal aesthetic: near-black surfaces, cyan accent used sparsely,
green/red as signals only, tabular numerals for prices. Chart:
lightweight-charts candlesticks + volume; crosshair OHLC legend with
event mark keys (N / n / E).

## Chat

Both frontends call `trading/services/chat_turn.py` (memory + skills +
platform tools). Tool usage renders as chips above the reply in React.

## Auth & multi-user

JWT against the shared accounts DB. Per-request identity scopes API keys,
memory, paper book, watchlist, and alerts to the logged-in user when
`EVOLVE_REQUIRE_LOGIN=1`.

## Not claimed here

- Live broker routing
- Historical dealer GEX time series from free yfinance (snapshot only)
- OOS-validated options-structure mapping or near_flip threshold
- True tick streaming without a paid feed (websocket polls delayed quotes)
- Engagement/click-based recommendation filtering (stated Settings prefs
  only — `docs/PERSONALIZATION.md`)

Deeper backlog: `docs/NEXT_SESSIONS.md`. Platform MCP tools:
`docs/MCP_SERVER.md`.
