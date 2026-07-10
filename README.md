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

## Quick start

**Prerequisites:** Python 3.10+ recommended, 8GB+ RAM. Optional: GPU for
deep-learning models, Redis for caching.

```bash
git clone https://github.com/Tcooper4/Evolve.git
cd Evolve

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env             # then set ANTHROPIC_API_KEY etc.

streamlit run app.py
```

Open **http://localhost:8501**. Set the active LLM under **Settings**.

**Windows with multiple Pythons:** install and run with the *same*
interpreter, e.g. `py -3.10 -m pip install -r requirements.txt` then
`py -3.10 -m streamlit run app.py`.

## LLM configuration

One active LLM drives Chat, commentary, and intent parsing. Providers:
Claude, GPT-4, Gemini, Ollama, HuggingFace, Kimi
(`config/llm_config.py`; choice stored in MemoryStore via the Settings
page). API keys come from the environment — see `.env.example`.

## MCP server (use Evolve from Claude)

```bash
python -m trading.services.mcp_server
```

exposes `get_ai_score`, `get_forecast`, `scan_universe`, `get_news`,
`get_risk_metrics`, `get_pattern_analysis`, `run_backtest`,
`get_options_sentiment`, `detect_market_regime`, and
`optimize_strategy_params` to any MCP client. Read/analyze only — nothing
executes trades. Setup for Claude Desktop/Code:
[`docs/MCP_SERVER.md`](docs/MCP_SERVER.md).

## Key directories

| Path | Purpose |
|------|---------|
| `app.py` / `pages/` | Streamlit entry point and the seven pages |
| `components/` | Page components and the design system (`theme.py`) |
| `trading/` | Strategies, models, backtesting, optimization, risk, data, memory, services |
| `agents/llm/` | Chat agent, tool executor, LLM interfaces |
| `skills/` | Agent skill playbooks loaded on demand by Chat |
| `config/` | App and LLM configuration |
| `tests/` | Test suites |
| `docs/` | MCP server guide and design notes |
| `_archive/` | Retired code kept for reference (not imported) |

## Project docs

- **Audit & session history:** [`AUDIT_TRACKER.md`](AUDIT_TRACKER.md) —
  every verified bug fix (97 to date) and what each session did.
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md)
- **Known debt:** [`TECHNICAL_DEBT.md`](TECHNICAL_DEBT.md)
- **Config:** [`config/CONFIG_README.md`](config/CONFIG_README.md)
- **Trading module:** [`trading/README.md`](trading/README.md)

## License

MIT — see [LICENSE](LICENSE).
