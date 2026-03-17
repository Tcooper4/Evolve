# -*- coding: utf-8 -*-
"""
Trade page — Execute, Positions, History, Risk (reorganized from Trade Execution, Performance, Portfolio).
"""
import sys
from pathlib import Path
import runpy

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

st.title("💰 Trade")
st.caption("Execution, positions, history, and risk")

# Summary metrics row (placeholder when no portfolio)
try:
    if "portfolio_manager" not in st.session_state:
        from trading.portfolio.portfolio_manager import PortfolioManager
        st.session_state.portfolio_manager = PortfolioManager()
    pm = st.session_state.portfolio_manager
    positions = pm.get_all_positions() if hasattr(pm, "get_all_positions") else []
    total_val = sum(float(p.get("market_value", 0) or 0) for p in positions) if isinstance(positions, list) else 0
    if not total_val and hasattr(pm, "get_portfolio_value"):
        total_val = float(pm.get_portfolio_value() or 0)
except Exception:
    total_val = 0
    positions = []

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Portfolio Value", f"${total_val:,.2f}")
with c2:
    st.metric("Day P&L", "—", delta=None)
with c3:
    st.metric("Total Return", "—", delta=None)

tab_exec, tab_pos, tab_hist, tab_risk = st.tabs(["Execute", "Positions", "History", "Risk"])

with tab_exec:
    st.subheader("Execute")
    try:
        old_path = project_root / "scripts" / "old_4_Trade_Execution.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

    # Estimated costs / slippage calculator
    try:
        st.markdown("**Estimated Costs**")
        _slippage_bps = st.slider(
            "Slippage (bps)",
            min_value=1,
            max_value=50,
            value=5,
            key="trade_slippage_bps",
            help="Basis points of slippage to assume. "
                 "5bps is typical for liquid large-caps.",
        )
        _shares = st.number_input("Shares (for estimate)", min_value=0, value=0, key="trade_est_shares")
        _price = st.number_input("Price (for estimate)", min_value=0.0, value=0.0, format="%.2f", key="trade_est_price")
        if _shares and _price:
            _notional = float(_shares) * float(_price)
            _slip_cost = _notional * (_slippage_bps / 10000)
            _commission = max(1.0, _notional * 0.0001)
            _total_cost = _slip_cost + _commission
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Notional", f"${_notional:,.2f}")
            with c2:
                st.metric("Est. Slippage", f"${_slip_cost:.2f}", f"{_slippage_bps}bps")
            with c3:
                st.metric("Total Cost", f"${_total_cost:.2f}")
    except Exception:
        pass

with tab_pos:
    st.subheader("Positions")
    try:
        old_path = project_root / "scripts" / "old_5_Portfolio.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_hist:
    st.subheader("History")
    try:
        old_path = project_root / "scripts" / "old_7_Performance.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

with tab_risk:
    st.subheader("Risk")
    try:
        old_path = project_root / "scripts" / "old_6_Risk_Management.py"
        runpy.run_path(str(old_path), run_name="__main__")
    except Exception as e:
        st.caption(f"Feature unavailable: {e}")

    # Advanced Risk Analytics from archive (now wired via trading.risk)
    st.markdown("---")
    st.subheader("📊 Advanced Risk Analytics")
    try:
        from trading.risk.advanced_risk import AdvancedRiskAnalyzer
        from trading.portfolio.portfolio_manager import PortfolioManager

        _pm = st.session_state.get("portfolio_manager")
        if _pm and isinstance(_pm, PortfolioManager):
            _positions = _pm.get_all_positions()
            if _positions:
                # Expect a mapping of ticker -> position/weight
                _analyzer = AdvancedRiskAnalyzer()

                # For now, use synthetic returns built from position notional history if available.
                # If the portfolio manager exposes historical returns, prefer that.
                _hist_returns = getattr(_pm, "get_portfolio_returns", None)
                if callable(_hist_returns):
                    _rets = _hist_returns()
                else:
                    _rets = None

                if _rets is not None and not _rets.empty:
                    _risk_metrics = _analyzer.calculate_comprehensive_risk(_rets)
                    if _risk_metrics:
                        _rc1, _rc2, _rc3 = st.columns(3)
                        with _rc1:
                            st.metric(
                                "Portfolio VaR (95%)",
                                f"{_risk_metrics.var_95:.2%}",
                            )
                        with _rc2:
                            st.metric(
                                "CVaR (95%)",
                                f"{_risk_metrics.cvar_95:.2%}",
                            )
                        with _rc3:
                            st.metric(
                                "Max Drawdown",
                                f"{_risk_metrics.max_drawdown:.2%}",
                            )
                    else:
                        st.caption(
                            "No risk metrics available for current positions."
                        )
                else:
                    st.caption(
                        "Not enough return history to compute advanced risk metrics."
                    )
            else:
                st.caption("No open positions to analyze.")
        else:
            st.caption("Portfolio manager not initialized.")
    except Exception as _re:
        st.caption(f"Advanced risk unavailable: {_re}")


# Page Assistant
try:
    from ui.page_assistant import render_page_assistant
    render_page_assistant("Trade")
except Exception:
    pass
