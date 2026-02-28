# Evolve AI Trading Platform

A production-ready quantitative trading platform with a **Streamlit UI**, natural-language Chat, forecasting, strategy backtesting, execution (paper/live), portfolio and risk management, and configurable **LLM selection** (Claude, GPT-4, Gemini, Ollama, HuggingFace, Kimi).

---

## What Evolve Does

- **Natural language interface** — Chat and intent parsing drive forecasting, backtests, and execution.
- **Forecasting** — Multiple models (LSTM, XGBoost, Prophet, ARIMA, Transformer, etc.) with ensemble and explainability.
- **Strategy testing** — Backtest strategies (RSI, MACD, Bollinger, custom) with walk-forward validation and cost modeling.
- **Trade execution** — Paper and live execution via Alpaca (or other brokers) with a unified execution agent and broker adapter.
- **Portfolio & risk** — Allocation, risk controls, and monitoring.
- **Model lab** — Train, compare, and tune models.
- **Reports** — Export to PDF, Excel, HTML.
- **Admin** — System config, API keys, broker settings, **AI model (LLM) selection** stored in preference memory.
- **Memory** — Preference and performance memory (MemoryStore) for LLM choice and other app-wide settings.

---

## Application Structure

- **Entry point:** Streamlit app at project root.
- **UI:** `app.py` (main app) and `pages/` (Streamlit pages).

| Page | Purpose |
|------|--------|
| **Home** | Dashboard and quick actions |
| **Chat** | Natural-language requests (forecast, backtest, analyze) |
| **Forecasting** | Time-series forecasts and model comparison |
| **Strategy Testing** | Backtest and tune strategies |
| **Trade Execution** | Paper/live orders and execution status |
| **Portfolio** | Positions and allocation |
| **Risk Management** | Risk metrics and controls |
| **Performance** | Strategy and system performance |
| **Model Lab** | Model training and comparison |
| **Reports** | Generate and export reports |
| **Alerts** | Notifications and alerts |
| **Admin** | Config, API keys, broker, **LLM selection** |
| **Memory** | Preference and performance memory |

---

## LLM Configuration

The app uses a **single active LLM** for Chat, commentary, and intent parsing. Configuration is centralized and stored in **MemoryStore** (preference key `active_llm`).

- **Config module:** `config/llm_config.py`  
  - `get_active_llm()` / `set_active_llm()` read/write the active provider and model from MemoryStore.  
  - Supports: Claude, GPT-4, Gemini, Ollama, HuggingFace, Kimi.  
  - See `LLM_PROVIDERS`, `DEFAULT_MODELS`, `PROVIDER_DISPLAY_NAMES`, `HUGGINGFACE_MODES`.

- **Where to set it:** **Admin → Configuration → AI Model Settings.** Choose provider and model, then **Save**. Optional **Test Connection** to verify.

- **Requirements:** `trading.memory.get_memory_store()` (MemoryStore) and `config.llm_config.get_active_llm()` must be available. Ensure project root is on `sys.path` when loading Admin so `config` resolves to the root `config` package (see `config/CONFIG_README.md`).

---

## Quick Start

**Prerequisites:** Python 3.9+ (3.10 recommended), 8GB+ RAM. Optional: GPU for deep learning, Redis for caching.

```bash
# Clone and enter project
git clone <repo-url>
cd evolve_clean

# Virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Dependencies
pip install -r requirements.txt

# Environment (copy and edit)
cp .env.example .env
# Set ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Run Streamlit app
streamlit run app.py
# Or: python main.py streamlit
```

Open **http://localhost:8501**. Use **Admin → Configuration → AI Model Settings** to choose and save the LLM.

---

## Configuration

- **App config:** `config/app_config.py` and `get_config()` (YAML + env). See `config/CONFIG_README.md`.
- **LLM config:** `config/llm_config.py` — API keys from env; active model from MemoryStore (Admin UI).
- **Trading/DB:** `trading.config` for trading-specific settings; `trading.database.connection` for DB URL and shutdown.

---

## Execution and Backtesting

- **Backtest (historical):** `trading/backtesting/backtester.py` (and enhanced_backtester). Do not use execution modules for pure backtests.
- **Paper / live execution:** `execution/live_trading_interface.py` (`mode="simulated"` | `"paper"` | `"live"`) or `execution/execution_agent.py` with broker adapter. See `docs/EXECUTION_AND_BACKTEST_FLOW.md`.

---

## Key Directories

| Path | Purpose |
|------|--------|
| `app.py` | Streamlit entry point |
| `pages/` | Streamlit pages (Home, Chat, Admin, etc.) |
| `config/` | App and LLM config (`app_config`, `llm_config`) |
| `agents/` | LLM agent, prompt routing, active LLM helpers |
| `trading/` | Strategies, backtesting, agents, memory, portfolio, risk, models, report |
| `execution/` | Execution agent, live trading interface, broker adapter |
| `core/` | Orchestrator and shared utilities |
| `docs/` | Execution flow, config, and other docs |

---

## Environment Variables (summary)

See `.env.example` for full list. Common:

- **Data:** `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `POLYGON_API_KEY`
- **LLMs:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` / `GEMINI_API_KEY`, `HUGGINGFACE_API_KEY`, `MOONSHOT_API_KEY` (Kimi)
- **System:** `LOG_LEVEL`, `REDIS_URL`, `DB_*` / `SQLITE_PATH`

---

## Documentation

- **Config:** `config/CONFIG_README.md`
- **Execution & backtest:** `docs/EXECUTION_AND_BACKTEST_FLOW.md`
- **Trading module:** `trading/README.md`
- **Execution module:** `execution/README.md`
- **Production:** `README_PRODUCTION.md`

---

## License

MIT. See [LICENSE](LICENSE) if present.

---

**Evolve** — Streamlit-based AI trading platform with configurable LLM and MemoryStore-driven settings.
