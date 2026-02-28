# Execution and Backtest Entry Points — Ownership and Flow

**P3 fix (AUDIT_REPORT.md 3.3):** This document clarifies which modules own backtest vs paper vs live execution so callers use a consistent path.

## Intended flow

```
Backtest (historical)  →  Paper (broker paper account)  →  Live (real money)
     │                              │                              │
     ▼                              ▼                              ▼
 backtester.py              live_trading_interface         live_trading_interface
 enhanced_backtester.py      mode="paper"                   mode="live"
 trade_execution_simulator  execution_agent                execution_agent
 (simulation only)         + broker_adapter               + broker_adapter
```

## Who owns what

### Backtest (historical simulation only)

| Module | Purpose | Use when |
|--------|--------|----------|
| **`trading/backtesting/backtester.py`** | Main backtest engine (costs, position sizing, metrics). | Running strategy backtests on historical data. |
| **`trading/backtesting/enhanced_backtester.py`** | Extended backtester with extra features. | When you need enhancements over the main backtester. |
| **`trading/core/backtest_common.py`** | Shared backtest utilities. | Used by backtester modules, not a direct entry point. |
| **`trading/execution/trade_execution_simulator.py`** | Simulates execution (slippage, delay) within backtest. | When backtest needs execution simulation. |

**Preferred entry for “run a backtest”:** `trading/backtesting/backtester.py` (or `enhanced_backtester.py` if needed). Do not use execution/ modules for pure backtest; they are for paper/live.

### Paper and live execution

| Module | Purpose | Use when |
|--------|--------|----------|
| **`execution/execution_agent.py`** | Order queue, risk checks, simulation or broker submission. | Paper or live execution with one agent; supports simulation, paper, live via config. |
| **`execution/live_trading_interface.py`** | High-level interface: simulated engine or Alpaca (paper/live). | When you want a single interface that can be simulated, paper, or live (mode=). |
| **`execution/broker_adapter.py`** | Unified broker API (Alpaca, IBKR, Polygon, simulation). | When execution_agent or live_trading_interface needs to submit orders or get market data. |
| **`execution/trade_executor.py`** | **Simulation only.** No broker integration. | Legacy/simple simulation only; do not use for paper/live. Prefer execution_agent or live_trading_interface. |
| **`trading/execution/execution_engine.py`** | In-process execution/simulation helpers. | Used by trading-layer code; not the main entry for “paper/live” flows. |

**Preferred entry for “paper or live”:**  
- **Unified UI/API:** `execution/live_trading_interface.py` with `mode="simulated"` | `"paper"` | `"live"`.  
- **Programmatic/async:** `execution/execution_agent.py` with broker adapter configured for paper or live.

### Summary

- **Backtest only** → `trading/backtesting/backtester.py` (or enhanced_backtester + trade_execution_simulator if needed).  
- **Paper or live** → `execution/live_trading_interface.py` or `execution/execution_agent.py` with broker adapter; do not use `execution/trade_executor.py` for real broker flows.  
- **Single execution facade:** Prefer `LiveTradingInterface` or `ExecutionAgent` and pass mode/config so one path handles simulated / paper / live.
