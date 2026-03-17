"""
Page-aware context for the lightweight AI assistant sidebar.

Provides get_page_context(page_name, session_state) so the assistant
can describe what the user is currently looking at on each page.
Also get_full_context_summary() for cross-page situational awareness from MemoryStore.
"""

from typing import Any


def get_full_context_summary() -> str:
    """
    Pull last backtests, forecast, trades, and risk snapshot from MemoryStore
    and return a single plain-English paragraph. Used by page assistants for
    cross-page awareness. Returns empty string on any failure.
    """
    try:
        from trading.memory import get_memory_store
        from trading.services.chat_nl_service import get_trading_context_summary
        store = get_memory_store()
        return get_trading_context_summary(store)
    except Exception:
        return ""


def get_page_context(page_name: str, session_state: Any = None) -> str:
    """
    Return a short context string describing the user's current state on the given page.
    Used to build the system prompt for the page assistant. Keeps output concise (~1500 chars).
    """
    try:
        name = page_name.lower().strip()
        if "dashboard" in name:
            return _dashboard_context(session_state)
        if "analyze" in name:
            return _analyze_context(session_state)
        if "scanner" in name:
            return _scanner_context(session_state)
        if "trade" in name:
            return _trade_context(session_state)
        if "backtest" in name:
            return _backtest_context(session_state)
        if "chat" in name:
            return _chat_context(session_state)
        if "settings" in name:
            return _settings_context(session_state)
        return "Evolve algorithmic trading platform."
    except Exception:
        return "Evolve algorithmic trading platform."


def _dashboard_context(ss: Any = None) -> str:
    return (
        "Dashboard: Live market pulse showing SPY/QQQ/IWM/VIX prices, "
        "AI-scored top opportunities, watchlist, and breaking news. "
        "Ask about market conditions, top movers, or your watchlist."
    )


def _analyze_context(ss: Any = None) -> str:
    ticker = ""
    if ss:
        try:
            ticker = ss.get("analyze_ticker", "") or ss.get("symbol", "")
        except Exception:
            ticker = ""
    t = (" for " + str(ticker).upper()) if ticker else ""
    return (
        "Analyze page"
        + t
        + ": Interactive chart with period switcher, AI Score "
        "(technical/momentum/sentiment/fundamental), forecasting models "
        "(ARIMA/XGBoost/Ridge/CatBoost/Prophet), econometric diagnostics, "
        "and news sentiment overlay. Ask about the AI score, what the "
        "forecast means, or buy/sell signals."
    )


def _scanner_context(ss: Any = None) -> str:
    return (
        "Scanner: Screens stocks from S&P 100, S&P 500, NASDAQ 100, or custom lists "
        "for AI Score, momentum, volume spikes, and news sentiment. Ask about top "
        "opportunities or how to interpret scanner results."
    )


def _trade_context(ss: Any = None) -> str:
    return (
        "Trade page: Paper trading execution with slippage model, portfolio positions, "
        "performance history, and risk management. Ask about order types, position "
        "sizing, or how to read performance metrics."
    )


def _backtest_context(ss: Any = None) -> str:
    return (
        "Backtest page: Strategy development and testing with 8 tabs including Quick "
        "Backtest, Strategy Builder, Walk-Forward validation, and RL Training. Ask "
        "about backtest results, strategy parameters, or how to interpret metrics "
        "like Sharpe ratio and max drawdown."
    )


def _settings_context(ss: Any = None) -> str:
    return (
        "Settings page: Watchlist management, alerts configuration, and system admin. "
        "Ask about setting up alerts, managing your watchlist, or system health."
    )


def _strategy_testing_context(session_state: Any) -> str:
    parts = []
    loaded = session_state.get("loaded_data")
    if loaded is not None:
        try:
            n = len(loaded) if hasattr(loaded, "__len__") else 0
            symbol = session_state.get("backtest_symbol", "N/A")
            parts.append(f"Loaded data: {symbol}, {n} rows.")
        except Exception:
            parts.append("Loaded data present.")
    results = session_state.get("backtest_results")
    if results is not None:
        try:
            strategy = session_state.get("backtest_strategy", "N/A")
            parts.append(f"Recent backtest: strategy={strategy}.")
            if isinstance(results, dict):
                total_return = results.get("total_return", results.get("return"))
                if total_return is not None:
                    parts.append(f"Total return: {total_return:.2%}" if isinstance(total_return, (int, float)) else f"Total return: {total_return}")
                sharpe = results.get("sharpe_ratio")
                if sharpe is not None:
                    parts.append(f"Sharpe: {sharpe:.2f}" if isinstance(sharpe, (int, float)) else f"Sharpe: {sharpe}")
            elif hasattr(results, "total_return"):
                parts.append(f"Total return: {getattr(results, 'total_return', 0):.2%}")
        except Exception:
            parts.append("Backtest results available.")
    if not parts:
        return "No data loaded and no backtest run yet. User can load data and run a quick backtest."
    return " ".join(parts)


def _risk_management_context(session_state: Any) -> str:
    parts = []
    rm = session_state.get("risk_manager")
    if rm is not None and getattr(rm, "current_metrics", None) is not None:
        m = rm.current_metrics
        try:
            parts.append(
                f"Current risk: VaR95={getattr(m, 'var_95', 0):.2%}, "
                f"volatility={getattr(m, 'volatility', 0):.2%}, "
                f"max_dd={getattr(m, 'max_drawdown', 0):.2%}, "
                f"Sharpe={getattr(m, 'sharpe_ratio', 0):.2f}."
            )
        except Exception:
            parts.append("Risk metrics computed.")
    limits = session_state.get("risk_limits")
    if limits and isinstance(limits, dict):
        parts.append(
            f"Limits: max_var={limits.get('max_var', 0):.2%}, "
            f"max_volatility={limits.get('max_volatility', 0):.2%}, "
            f"max_drawdown={limits.get('max_drawdown', 0):.2%}."
        )
    history = session_state.get("risk_history") or []
    if history:
        parts.append(f"Risk history: {len(history)} snapshots.")
    if not parts:
        return "No risk snapshot yet. User can load portfolio data to see risk metrics."
    return " ".join(parts)


def _portfolio_context(session_state: Any) -> str:
    pm = session_state.get("portfolio_manager")
    if pm is None:
        return "No portfolio manager. Positions will appear after initialization."
    try:
        summary = pm.get_position_summary()
        if summary is not None and not summary.empty:
            n = len(summary)
            open_pos = summary[summary["status"] == "open"] if "status" in summary.columns else summary
            n_open = len(open_pos) if hasattr(open_pos, "__len__") else 0
            symbols = list(summary["symbol"].unique())[:10] if "symbol" in summary.columns else []
            sym_str = ", ".join(str(s) for s in symbols)
            if len(symbols) > 10:
                sym_str += "..."
            return f"Positions: {n_open} open, {n} total. Symbols: {sym_str or 'none'}."
        all_pos = pm.get_all_positions()
        if all_pos:
            total = sum(len(v) if isinstance(v, list) else 0 for v in all_pos.values())
            return f"Positions: {total} total across symbols."
    except Exception:
        pass
    return "Portfolio loaded; no positions or summary available yet."


def _forecasting_context(session_state: Any) -> str:
    parts = []
    symbol = session_state.get("symbol")
    if symbol:
        parts.append(f"Symbol: {symbol}.")
    data = session_state.get("forecast_data")
    if data is not None:
        try:
            n = len(data) if hasattr(data, "__len__") else 0
            parts.append(f"Forecast data: {n} rows.")
        except Exception:
            parts.append("Forecast data loaded.")
    horizon = session_state.get("forecast_horizon")
    if horizon is not None:
        parts.append(f"Horizon: {horizon} days.")
    model = session_state.get("current_model")
    if model:
        parts.append(f"Last model used: {model}.")
    result = session_state.get("current_forecast_result")
    if result is not None:
        parts.append("Last forecast run result available.")
    if not parts:
        return "No forecast run yet. User can load data and run a quick or advanced forecast."
    return " ".join(parts)


def _chat_context(session_state: Any) -> str:
    """Context for Chat page: recent conversation."""
    parts = []
    messages = session_state.get("messages") or session_state.get("chat_messages") or []
    if messages:
        n = len(messages) if hasattr(messages, "__len__") else 0
        parts.append(f"Recent conversation: {n} messages.")
    if not parts:
        return "Chat page. No recent conversation yet."
    return " ".join(parts)


def _trade_execution_context(session_state: Any) -> str:
    """Context for Trade Execution page: pending orders, execution mode."""
    parts = []
    mode = session_state.get("execution_mode", "paper")
    parts.append(f"Execution mode: {mode}.")
    pending = session_state.get("active_orders") or []
    if pending:
        parts.append(f"Pending orders: {len(pending)}.")
    if not parts:
        return "Trade Execution page. Execution mode and pending orders unknown."
    return " ".join(parts)


def _performance_context(session_state: Any) -> str:
    """Context for Performance page: strategy performance metrics."""
    parts = []
    if session_state.get("strategy_performance"):
        parts.append("Strategy performance metrics loaded.")
    if session_state.get("performance_df") is not None:
        parts.append("Performance data available.")
    if not parts:
        return "Performance page. No strategy performance data loaded yet."
    return " ".join(parts)


def _model_lab_context(session_state: Any) -> str:
    """Context for Model Lab page: current model being trained."""
    parts = []
    model = session_state.get("current_model") or session_state.get("selected_model")
    if model:
        parts.append(f"Current/selected model: {model}.")
    if session_state.get("training_in_progress"):
        parts.append("Training in progress.")
    if not parts:
        return "Model Lab page. No model selected or training in progress."
    return " ".join(parts)


def _home_context(session_state: Any) -> str:
    """Context for Home page: briefing and cards."""
    parts = []
    if session_state.get("home_briefing_text"):
        parts.append("Morning briefing loaded.")
    cards = session_state.get("home_briefing_cards") or []
    if cards:
        parts.append(f"{len(cards)} briefing cards.")
    if not parts:
        return "Home page. Briefing not yet loaded."
    return " ".join(parts)
