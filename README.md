# Evolve — AI Trading Research Platform

A personal SPX/multi-asset **systematic research platform**: a Streamlit
terminal with AI scoring, multi-model forecasting, market scanning,
strategy backtesting and parameter optimization, risk analytics, paper
trading, and an LLM chat agent with platform tools. Evolve informs
trading decisions; it does not make them — **all execution is
paper/simulated by design**, with the human in the loop for anything that
matters.

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
  Bayesian) with **out-of-sample validation on by default** and one-click
  apply into backtests.
- **Risk analytics** — Sharpe/Sortino/Calmar, drawdown, VaR/CVaR, and 22
  position-sizing methods (Kelly family, risk parity, mean-variance,
  regime/factor-based, and more).
- **Chat agent** — natural-language interface backed by a configurable
  LLM, with platform tools (scan, score, forecast, news, risk, patterns,
  backtest, options sentiment) and on-demand **skill playbooks**
  (`skills/`).
- **MCP server** — every platform tool exposed over the Model Context
  Protocol for Claude Desktop/Code and other agents. See
  [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md).
- **Paper trading** — simulated execution and portfolio tracking
  (`trading/execution/trade_execution_simulator.py`). There is no live
  broker integration.

## The app

`app.py` boots the terminal and routes to seven pages:

| Page | Purpose |
|------|---------|
| **Dashboard** | Market overview, watchlist, morning briefing |
| **Analyze** | Single-stock deep dive: chart, AI Score, forecast, news, options, earnings |
| **Scanner** | Universe screening, signal breakdown, pairs trading |
| **Trade** | Paper trading and position management |
| **Backtest** | Walk-forward validation, strategy comparison, backtests, **Optimizer** |
| **Chat** | LLM chat with platform tools, news panel, macro context |
| **Settings** | Preferences, API keys, LLM selection |

## Quick start (personal mode — free, local)

```bash
git clone -b codebase-audit-consolidated https://github.com/Tcooper4/Evolve.git
cd Evolve
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # add ANTHROPIC_API_KEY (chat) and any data keys
streamlit run app.py    # open http://localhost:8501
```

That's the whole thing: no login screen, no hosting cost, all data local.
Requires Python 3.10+. First launch downloads models/data lazily, so the
first page render is slower than every one after.

## Multi-user mode (host it for friends & family)

The same codebase becomes a gated multi-user site with one env flag —
see **docs/DEPLOYMENT.md** for the full VPS + HTTPS guide. Short version:

```bash
export EVOLVE_REQUIRE_LOGIN=1
export EVOLVE_AUTH_SECRET=$(python3 -c "import secrets;print(secrets.token_hex(32))")
python scripts/manage_users.py add thomas --name "Thomas" --admin
streamlit run app.py
```

Every account gets its own isolated, persistent workspace: watchlist,
memory/chat learning, preferences, and API keys are all per-user.
Users enter their own API keys under **Settings → API keys** (encrypted
at rest); set `EVOLVE_SHARED_KEYS=0` to require it so nobody spends the
host's quota.

## React frontend (in-progress rewrite — optional)

A modern React + FastAPI interface is being built alongside Streamlit
(same accounts, same backend). Currently: login, dashboard, candlestick
chart with volume, KPI strip, sparkline watchlist.

```bash
# terminal 1 — API
uvicorn web.backend.main:app --port 8000
# terminal 2 — frontend
cd web/frontend && npm install && npm run dev   # http://localhost:5173
```

## Docker (easiest way to run it)

One command, everything included - Python, Node build, the React app:

```bash
docker compose up -d     # then open http://localhost:8000
```

Your data lives in `./data` on the host (mounted as a volume), so
rebuilding or updating the container never touches your accounts, keys,
memory, or watchlists. Set `EVOLVE_REQUIRE_LOGIN=1` in `.env` for
multi-user mode. To update: `git pull && docker compose up -d --build`.

## Where your data lives (and persistence)

Everything persists on disk between sessions, in plain SQLite/JSON under
the repo — no external services:

| Path | Contents |
|---|---|
| `data/accounts.db` | login accounts (bcrypt hashes only) |
| `data/users.db` | per-user API keys (Fernet-encrypted) and preferences |
| `data/memory_store.db` | chat learning, long-term memories, preferences — per-user |
| `data/watchlist.db` | per-user watchlists + alert history |
| `data/paper_portfolio.db` | per-user paper positions + trade ledger |
| `data/leaderboard/` | model/agent performance history |
| `.cache/` / `data/*cache*` | market-data caches (TTL-expired, safe to delete) |
| `.env` | your API keys + `EVOLVE_ENCRYPTION_KEY` — **never commit** |

**Backup = copy `data/` and `.env`.** Deleting `.cache/` only forces
re-fetch. In multi-user mode all of the above is keyed per account.

## Performance (measured)

Compute is not the bottleneck; benchmarks on 1y daily data (single CPU):
feature engineering ~30 ms, regime detection ~24 ms, one strategy
backtest ~50 ms, a default 30-evaluation optimizer run ~1.4 s, a heavy
300-evaluation optimization ~14 s, sentiment scoring 50 headlines ~3 ms.
Perceived latency is dominated by **network fetches** (0.3–2 s per
symbol from Yahoo), which the caching layers absorb: 15 s quote / 60 s
history in-memory caches plus a TTL disk cache. The genuinely slow path
is deep-model training (LSTM/Transformer — minutes by nature); it shows
progress in the UI.

## Tests

```bash
pip install pytest
python -m pytest tests/test_auth tests/test_data tests/test_nlp_depth \
  tests/test_report_depth.py tests/test_agents_depth.py tests/test_web_api.py -q
```

## Honest scope

All execution is **paper/simulated**; there is no live order routing.
Multi-user mode is appropriate for trusted friends and family, not the
public internet (no rate limiting or 2FA — see DEPLOYMENT.md). Nothing
here is financial advice; predictive accuracy is never promised.
