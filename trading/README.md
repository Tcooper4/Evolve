# Trading Module

Core trading logic: strategies, backtesting, agents, memory, portfolio, risk, models, and reporting.

## Structure

```
trading/
├── agents/           # Research, commentary, execution agents; agent registry, loop manager
├── backtesting/      # Backtester, enhanced backtester, position sizing, visualization
├── config/           # Trading-specific config (settings, enhanced_settings)
├── core/             # Shared utilities, history logger
├── data/             # Data providers, preprocessing
├── evaluation/       # Metrics and evaluation
├── execution/        # Execution replay, execution engine (in-process)
├── memory/           # Agent memory, performance memory, goals
├── models/           # Model registry, TCN, neural forecast, timeseries, advanced (transformer, GNN)
├── optimization/     # Strategy optimizer, genetic, PSO, forecasting integration
├── portfolio/        # Allocator, position sizer, risk manager
├── report/           # Report generator, templates, export
├── risk/             # Risk manager, advanced risk, tail risk
├── strategies/       # Strategy runner, RSI, MACD, Bollinger, registry, fallback
├── services/         # Commentary generator, research, home briefing, agent API, etc.
├── market/           # Market data, indicators
├── nlp/              # Sentiment, response formatting
├── monitoring/       # Health check
└── utils/            # Metrics, config utils, safe math, visualization
```

## Components

- **Strategies:** RSI, MACD, Bollinger Bands, pairs trading, strategy runner and registry.
- **Backtesting:** Main and enhanced backtester; cost/slippage; walk-forward and evaluation.
- **Agents:** Research, commentary, execution agents; agent registry and loop manager.
- **Memory:** Agent memory, performance memory, preference storage (used by LLM config).
- **Portfolio:** Allocation, position sizing, risk-aware portfolio logic.
- **Risk:** Risk manager, advanced risk, tail risk controls.
- **Models:** LSTM, XGBoost, Prophet, TCN, Transformer, GNN; model registry and tuning.
- **Config:** Use root `config` when possible; `trading.config` for trading-specific overrides (see `config/CONFIG_README.md`).

## Usage

Import from the trading package; ensure project root is on `sys.path` for `config` and `agents` imports.

```python
from trading.strategies.strategy_runner import StrategyRunner
from trading.backtesting.backtester import Backtester
from trading.memory import get_memory_store
from trading.portfolio.allocator import Allocator
from trading.risk.risk_manager import RiskManager
```

## Testing

```bash
pytest tests/unit/ -v
pytest tests/ -k trading -v
```

## Dependencies

pandas, numpy, scikit-learn, PyTorch (optional), and project requirements in root `requirements.txt`.
