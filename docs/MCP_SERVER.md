# Evolve MCP Server

Evolve's platform tools — scanning, AI scoring, forecasting, risk
analytics, pattern analysis, backtesting, regime detection, and
strategy-parameter optimization — exposed over the
[Model Context Protocol](https://modelcontextprotocol.io), the standard
way agents connect to external tools. Any MCP-capable client (Claude
Desktop, Claude Code, Cursor, another agent framework) can now use Evolve
as a research backend.

The in-app chat path is unchanged; this is an additional, standard front
door. Both delegate to the same `trading/services/agent_tools.py`
implementations, so there is exactly one implementation of each
capability.

**Human-in-the-loop by design:** every tool is read/analyze only. Nothing
here places orders or mutates portfolio state.

## Install

```bash
pip install mcp   # already in requirements.txt
```

## Run standalone (stdio)

```bash
cd /path/to/Evolve
python -m trading.services.mcp_server
```

## Connect Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "evolve": {
      "command": "python",
      "args": ["-m", "trading.services.mcp_server"],
      "cwd": "/path/to/Evolve"
    }
  }
}
```

## Connect Claude Code

```bash
claude mcp add evolve -- python -m trading.services.mcp_server
```

(run from the Evolve repo root, or pass `--cwd`.)

## Tools

| Tool | What it does |
|---|---|
| `get_ai_score(symbol)` | Full 16-signal AI score: composite (1–10), grade, per-signal breakdown |
| `get_forecast(symbol, horizon=7)` | Multi-model price forecast: direction, expected move, confidence |
| `scan_universe(universe, min_score, max_results)` | Screen a universe for candidates above a score floor |
| `get_news(symbol, max_items=10)` | Recent headlines with sources and links |
| `get_risk_metrics(symbol, period="1y")` | Volatility, Sharpe, max drawdown, VaR, sizing context |
| `get_pattern_analysis(symbol)` | Chart patterns, support/resistance, breakout posture |
| `run_backtest(symbol, days=90)` | Quick historical strategy backtest |
| `get_options_sentiment(symbol)` | Put/call posture and unusual-activity signals |
| `detect_market_regime(symbol="SPY")` | Bull/bear/sideways/volatile with confidence |
| `optimize_strategy_params(strategy, ...)` | Parameter search (grid/genetic/PSO/Bayesian) with out-of-sample validation on by default |

Failures return `{"success": false, "error": ...}` as tool results rather
than protocol errors, so a client can reason about them (e.g. an unknown
symbol).

## Example (from Claude, once connected)

> "Use evolve to scan the S&P 100 for setups above 7, pull the AI score
> and pattern read on the top two, and optimize the RSI strategy on the
> best one with validation — tell me whether the optimized parameters
> hold up out-of-sample."
