# Evolve — AI Trading Research Platform

A personal SPX/multi-asset **systematic research platform**: AI scoring,
multi-model forecasting, market scanning, strategy backtesting and
parameter optimization, risk analytics, paper trading, options market-
structure research (GEX / skew / structure guides), and an LLM chat agent
with platform tools.

Two UIs share one Python backend and the same accounts:

- **React + FastAPI** — primary day-to-day interface (`docker compose` or
  `uvicorn` + Vite). Dashboard, Analyze, Scanner, Portfolio, Backtest,
  Chat, Settings.
- **Streamlit** — full research terminal still available via `app.py`.

Evolve informs trading decisions; it does not make them — **all
execution is paper/simulated by design**, with the human in the loop for
anything that matters. Research overlays that are not Evolved-OOS-
validated ship **default-off** or with explicit disclosures.

**Branch:** feature work lands on `codebase-audit-consolidated`
(this README tracks that line).

---

## What Evolve does

- **AI Score** — 16-signal composite score (1–10) with grade and
  per-signal breakdown, in Buy or Short mode.
- **Forecasting** — LSTM, XGBoost, Prophet, ARIMA, TCN, Transformer, GNN,
  GARCH, Ridge, CatBoost and ensembles, with explainability.
- **Scanner** — screen S&P 100/500, Nasdaq 100, Russell 1000/3000 or a
  custom list by technical filters, quick score, news score, and short
  interest; pairs-trading cointegration scan.
- **Backtesting & optimization** — walk-forward validation, strategy
  comparison, and a **strategy optimizer** (grid search / genetic / PSO /
  Bayesian) with **out-of-sample validation on by default**.
- **Risk analytics** — Sharpe/Sortino/Calmar, drawdown, VaR/CVaR, Kelly
  family sizing (with sample-size and conditional-vol / options-VIX
  research overlays), account-level stress, and **holding concentration**
  flags from pairwise return correlation.
- **Chart & news** — candlesticks with volume/news event marks, robust
  trimmed-mean volume-spike baseline, optional strategy and options-
  structure overlays (default-off).
- **Options research** — delayed-chain GEX (net / flip / regime), IV skew,
  structure guide (IC / credit / wait), options-aware cost model (ATM
  floor in **% of premium**, not equity 5 bps). Free chains are live
  snapshots only — historical dealer GEX is not reconstructable; opt-in
  `EVOLVE_GEX_SNAPSHOT_LOG=1` builds a forward dataset.
- **Alerts & background jobs** — one-shot price alerts, optional compound
  confirms, watch vs action modes, push throttling; server-side limit
  fills and alert evaluation even with zero browsers open.
- **Labs** — Signal IC, **Diagnostics** (stationarity / ARCH / breaks —
  not Granger causality), patterns, GNN probe; Monte Carlo with i.i.d.
  default and opt-in stationary block bootstrap.
- **Chat agent** — natural-language interface with platform tools and
  skill playbooks (`skills/`), same brain in React and Streamlit.
- **MCP server** — platform tools over the Model Context Protocol. See
  [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md).
- **Paper trading** — simulated execution and portfolio tracking. There
  is no live broker integration.

## Interfaces

| Surface | Entry | Notes |
|---------|--------|--------|
| **React (recommended)** | `docker compose up -d` → http://localhost:8000 | Full page set below |
| **React (dev)** | `uvicorn` + `npm run dev` | See [`web/README.md`](web/README.md) |
| **Streamlit** | `streamlit run app.py` | Classic seven-page terminal |

### React pages

| Page | Purpose |
|------|---------|
| **Dashboard** | Market overview, watchlist, briefing |
| **Analyze** | Chart, AI Score, forecast, news, options context, earnings, filings, labs |
| **Scanner** | Universe screening and pairs |
| **Portfolio** | Paper positions, cashbook, limits, alerts, account risk |
| **Backtest** | Strategy runs, comparison, optimizer hooks |
| **Chat** | LLM + tools (same service as Streamlit) |
| **Settings** | Preferences, API keys, chart timezone |

### Streamlit pages

`app.py` routes: Dashboard, Analyze, Scanner, Trade, Backtest, Chat,
Settings — same research stack, Streamlit chrome.

## Quick start

### Docker (easiest)

```bash
git clone -b codebase-audit-consolidated https://github.com/Tcooper4/Evolve.git
cd Evolve
cp .env.example .env    # add keys as needed
docker compose up -d    # http://localhost:8000
```

Data stays in `./data` on the host. Update with
`git pull && docker compose up -d --build`.

### Local Streamlit

```bash
git clone -b codebase-audit-consolidated https://github.com/Tcooper4/Evolve.git
cd Evolve
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py    # http://localhost:8501
```

Requires Python 3.10+. First launch downloads models/data lazily.

### Local React + API

```bash
uvicorn web.backend.main:app --port 8000
cd web/frontend && npm install && npm run dev   # http://localhost:5173
```

## Multi-user mode (friends & family)

See **docs/DEPLOYMENT.md**. Short version:

```bash
export EVOLVE_REQUIRE_LOGIN=1
export EVOLVE_AUTH_SECRET=$(python3 -c "import secrets;print(secrets.token_hex(32))")
python scripts/manage_users.py add thomas --name "Thomas" --admin
```

Per-user watchlist, memory, preferences, API keys, paper book, and
alerts. Users can enter their own keys under Settings
(`EVOLVE_SHARED_KEYS=0` to require it).

## Where your data lives

| Path | Contents |
|---|---|
| `data/accounts.db` | login accounts (bcrypt hashes only) |
| `data/users.db` | per-user API keys (Fernet) and preferences |
| `data/memory_store.db` | chat learning / long-term memory |
| `data/watchlist.db` | watchlists + alert store |
| `data/paper_portfolio.db` | paper positions + trade ledger |
| `data/options_cache.db` | TTL option-chain cache (not a history archive) |
| `data/gex_regime_snapshots.db` | opt-in forward GEX log (`EVOLVE_GEX_SNAPSHOT_LOG`) |
| `data/leaderboard/` | model/agent performance history |
| `.cache/` / `data/*cache*` | market-data caches (safe to delete) |
| `.env` | API keys + `EVOLVE_ENCRYPTION_KEY` — **never commit** |

**Backup = copy `data/` and `.env`.**

## Research honesty (short)

Several modules are deliberately **scoped**, not hyped:

- Strategy / options-structure chart overlays default **off**.
- GEX `near_flip` (0.5% of spot) and regime→structure mapping are
  research defaults — not Evolve OOS-validated yet.
- Monte Carlo defaults to **i.i.d.** bootstrap; stationary block
  bootstrap is opt-in after a real-data comparison that did not justify
  softening the default tails.
- Options VIX / Kelly overlays are informational unless explicitly wired
  live; equity 5 bps ≠ options costs (see
  `trading/backtesting/options_cost_model.py`).
- Analyze **Diagnostics** is stationarity/structure — not Granger
  causality.

## Tests

```bash
.\evolve_venv\Scripts\python.exe -m pytest tests/test_auth tests/test_data \
  tests/test_volume_baseline_stress.py tests/test_block_bootstrap_mc.py \
  tests/test_diagnostics_naming.py tests/test_concentration.py \
  tests/test_options_cost_model.py -q -p no:cov
```

Full suite coverage varies by optional deps; focused research tests above
are the recent contract.

## Honest scope

All execution is **paper/simulated**; there is no live order routing.
Multi-user mode is appropriate for trusted friends and family, not the
public internet (no rate limiting or 2FA — see DEPLOYMENT.md). Nothing
here is financial advice; predictive accuracy is never promised.
