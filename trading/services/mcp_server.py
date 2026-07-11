# -*- coding: utf-8 -*-
"""Evolve MCP server — the platform's tools over the Model Context Protocol.

The handoff's modernization note called for evaluating MCP as the standard
way agents connect to Evolve's capabilities, replacing the hand-rolled
tool registry as the only access path. This server is that evaluation's
conclusion made concrete: every platform tool the chat agent can call is
now also available to ANY MCP-capable client — Claude Desktop, Claude
Code, Cursor, or another agent — with typed parameters and docstrings as
the contract. The in-app chat path is unchanged; this is an additional,
standard front door.

Design notes
------------
* Tools delegate to ``trading.services.agent_tools`` — the same functions
  the in-app executor dispatches to — so there is exactly one
  implementation of each capability.
* Every tool returns the underlying dict as JSON. Failures come back as
  ``{"success": false, "error": ...}`` rather than protocol errors, so a
  client can reason about them.
* ``optimize_strategy_params`` additionally exposes the Fable-session
  optimizer (with out-of-sample validation on by default) — the newest
  capability, available to external agents from day one.
* Human-in-the-loop stance: these tools are read/analyze only. Nothing
  here places orders or mutates portfolio state; that remains a human
  decision in the app, by design.

Running
-------
    python -m trading.services.mcp_server          # stdio transport

Claude Desktop / Claude Code config (claude_desktop_config.json or
`claude mcp add`):

    {
      "mcpServers": {
        "evolve": {
          "command": "python",
          "args": ["-m", "trading.services.mcp_server"],
          "cwd": "/path/to/Evolve"
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "The 'mcp' package is required for the Evolve MCP server: "
        "pip install mcp"
    ) from _e

mcp = FastMCP(
    "evolve",
    instructions=(
        "Evolve is a personal SPX/multi-asset systematic research platform. "
        "These tools provide market scanning, AI scoring, forecasting, risk "
        "analytics, pattern analysis, backtesting, and strategy-parameter "
        "optimization. All tools are read/analyze only — nothing executes "
        "trades. Data comes from live market sources; a tool may return "
        "success=false with an error when a symbol has no data."
    ),
)


def _jsonable(result: Any) -> Dict[str, Any]:
    """Coerce a tool result into something JSON-serializable."""
    try:
        json.dumps(result)
        return result
    except (TypeError, ValueError):
        return json.loads(json.dumps(result, default=str))


@mcp.tool()
def get_ai_score(symbol: str) -> Dict[str, Any]:
    """Full 16-signal AI score for a symbol: composite score (1-10), grade,
    per-signal breakdown with impacts, and a summary of what's driving it."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.get_ai_score(symbol))


@mcp.tool()
def get_forecast(symbol: str, horizon: int = 7) -> Dict[str, Any]:
    """Multi-model price forecast for a symbol over `horizon` trading days:
    direction, expected move, and confidence."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.get_forecast(symbol, horizon=horizon))


@mcp.tool()
def scan_universe(
    universe: str = "default",
    min_score: float = 6.0,
    max_results: int = 15,
) -> Dict[str, Any]:
    """Screen a stock universe (default/sp100/sp500/sp30...) for candidates
    at or above a minimum quick score. Returns ranked rows with price,
    momentum, RSI, and score columns."""
    from trading.services import agent_tools

    return _jsonable(
        agent_tools.scan_universe(
            universe=universe, min_score=min_score, max_results=max_results
        )
    )


@mcp.tool()
def get_news(symbol: str, max_items: int = 10) -> Dict[str, Any]:
    """Recent headlines for a symbol with titles, sources, and links."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.get_news(symbol, max_items=max_items))


@mcp.tool()
def get_risk_metrics(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Risk profile for a symbol over a period: volatility, Sharpe, max
    drawdown, VaR, and Kelly-style sizing context."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.get_risk_metrics(symbol, period=period))


@mcp.tool()
def get_pattern_analysis(symbol: str) -> Dict[str, Any]:
    """Technical pattern read for a symbol: detected chart patterns,
    support/resistance context, and breakout posture."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.get_pattern_analysis(symbol))


@mcp.tool()
def run_backtest(symbol: str, days: int = 90) -> Dict[str, Any]:
    """Quick historical strategy backtest on a symbol over the last `days`
    days: performance metrics and equity-curve summary."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.run_backtest(symbol, days=days))


@mcp.tool()
def get_options_sentiment(symbol: str) -> Dict[str, Any]:
    """Options-flow sentiment for a symbol: put/call posture and unusual
    activity signals."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.get_options_sentiment(symbol))


@mcp.tool()
def detect_market_regime(symbol: str = "SPY", period: str = "1y") -> Dict[str, Any]:
    """Current market regime (bull/bear/sideways/volatile) with confidence
    and which strategy families historically suit it."""
    from trading.services import agent_tools

    return _jsonable(agent_tools.detect_market_regime(symbol, period=period))


@mcp.tool()
def optimize_strategy_params(
    strategy: str,
    symbol: str = "SPY",
    method: str = "pso",
    metric: str = "sharpe_ratio",
    max_evaluations: int = 80,
    lookback_years: int = 2,
    validate: bool = True,
) -> Dict[str, Any]:
    """Search a strategy's parameter space against real history and report
    optimized-vs-default performance.

    strategy: RSIStrategy | MACDStrategy | BollingerStrategy | SMAStrategy
              | ATRStrategy | CCIStrategy
    method:   grid_search | genetic | pso | bayesian
    metric:   sharpe_ratio | sortino_ratio | calmar_ratio | total_return
              | max_drawdown
    validate: when true (default), optimizes on the first 75% of history
    and reports out-of-sample metrics on the held-out remainder — those
    are the numbers to trust; a large train->test drop is the overfit
    signature.
    """
    try:
        import yfinance as yf

        from trading.optimization.strategy_backtest_objective import (
            optimize_strategy,
            optimize_strategy_validated,
        )

        raw = yf.Ticker(symbol).history(period=f"{lookback_years}y")
        if raw is None or raw.empty:
            return {"success": False, "error": f"no price data for {symbol}"}
        if validate:
            run = optimize_strategy_validated(
                strategy, raw, train_fraction=0.75, method=method,
                metric=metric, max_evaluations=max_evaluations,
            )
        else:
            run = optimize_strategy(
                strategy, raw, method=method, metric=metric,
                max_evaluations=max_evaluations,
            )
        out: Dict[str, Any] = {
            "success": True,
            "strategy": run.strategy_name,
            "method": run.method,
            "metric": run.metric,
            "n_evaluations": run.n_evaluations,
            "best_params": run.best_params,
            "default_params": run.baseline_params,
            "in_sample": {
                "optimized": run.best_metrics,
                "defaults": run.baseline_metrics,
            },
        }
        if run.oos_best_metrics is not None:
            out["out_of_sample"] = {
                "optimized": run.oos_best_metrics,
                "defaults": run.oos_baseline_metrics,
                "train_range": run.train_range,
                "test_range": run.test_range,
                "note": (
                    "Out-of-sample numbers are the realistic expectation; "
                    "in-sample results overstate performance."
                ),
            }
        return _jsonable(out)
    except Exception as e:  # noqa: BLE001 - surface as tool result
        logger.warning("optimize_strategy_params failed: %s", e)
        return {"success": False, "error": str(e)}


def main() -> None:
    """Run the server on stdio (the transport Claude Desktop/Code use)."""
    mcp.run()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Connection pass (2026-07): parity with the chat tool surface.
# ---------------------------------------------------------------------------

@mcp.tool()
def get_portfolio_allocation(symbols: str, period: str = "1y") -> dict:
    """Risk-parity allocation across a comma-separated symbol list so each
    holding contributes equal risk (verified engine)."""
    from trading.services.agent_tools import get_portfolio_allocation as _f

    return _f(symbols, period=period)


@mcp.tool()
def retune_strategies(symbols: str = "SPY", method: str = "pso") -> dict:
    """One self-tuning cycle: re-optimize strategies on recent history and
    adopt new parameters only if they beat current ones out-of-sample
    (champion/challenger, journaled in data/self_tune.json)."""
    from trading.services.agent_tools import retune_strategies as _f

    return _f(symbols, method=method)


@mcp.tool()
def get_leaderboard(top_n: int = 10) -> dict:
    """Recorded model/strategy performance on this platform, best first."""
    from trading.services.agent_tools import get_leaderboard as _f

    return _f(top_n=top_n)


@mcp.tool()
def market_research(topic: str, max_results: int = 3) -> dict:
    """Search GitHub/arXiv for strategies, papers, and code on a topic."""
    from trading.services.agent_tools import market_research as _f

    return _f(topic, max_results=max_results)


@mcp.tool()
def critique_backtest(metrics: dict) -> dict:
    """Critique backtest metrics: overfitting signals, robustness, what to
    distrust."""
    from trading.services.agent_tools import critique_backtest as _f

    return _f(metrics)


@mcp.tool()
def get_watchlist() -> dict:
    """The current (local) user's watchlist symbols."""
    from trading.services.agent_tools import get_watchlist as _f

    return _f()


@mcp.tool()
def recommend_strategy(symbol: str = "SPY",
                       risk_tolerance: str = "medium") -> dict:
    """Recommend a trading strategy for current conditions, with reasoning."""
    from trading.services.agent_tools import recommend_strategy as _f

    return _f(symbol=symbol, risk_tolerance=risk_tolerance)


@mcp.tool()
def recommend_model(symbol: str = "SPY", horizon: int = 7) -> dict:
    """Recommend a forecasting model for a symbol/horizon, with reasoning."""
    from trading.services.agent_tools import recommend_model as _f

    return _f(symbol=symbol, horizon=horizon)


@mcp.tool()
def get_position_size(win_rate: float, avg_win_loss_ratio: float = 1.5,
                      account_size: float = 10_000.0) -> dict:
    """Kelly-criterion sizing: full and half Kelly (the practitioner
    reference) with dollar amounts."""
    from trading.services.agent_tools import get_position_size as _f

    return _f(win_rate, avg_win_loss_ratio=avg_win_loss_ratio,
              account_size=account_size)
