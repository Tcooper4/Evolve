# Execution Module

Handles **paper and live** trade execution: order submission, broker integration, and simulated execution. For **backtest-only** flows use `trading/backtesting/` instead.

## Structure

```
execution/
├── execution_agent.py           # Order queue, risk checks, simulation or broker submission
├── live_trading_interface.py   # High-level interface: simulated | paper | live (single entry)
├── broker_adapter.py           # Unified broker API (Alpaca, IBKR, Polygon, simulation)
├── trade_executor.py           # Simulation-only legacy; prefer execution_agent / live_trading_interface
├── advanced_order_executor.py  # Advanced order execution logic
├── models.py                   # Execution data models
└── redundant_broker_manager.py # Redundancy / failover (if used)
```

## Entry Points

- **Paper or live execution:** Use `live_trading_interface.LiveTradingInterface(mode="paper"|"live"|"simulated")` or `execution_agent.ExecutionAgent` with broker adapter configured. Do not use `trade_executor` for real broker flows.
- **Backtest (historical):** Use `trading/backtesting/backtester.py` (or enhanced_backtester); execution layer is for paper/live only.

See **`docs/EXECUTION_AND_BACKTEST_FLOW.md`** for ownership and flow details.

## Configuration

Paper vs live is set via `LiveTradingInterface(mode=...)` or broker config (`paper: true/false`). See root `config/CONFIG_README.md` and `.env.example`.

## Testing

```bash
pytest tests/unit/ -k execution -v
```
