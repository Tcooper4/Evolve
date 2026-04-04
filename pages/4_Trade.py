# -*- coding: utf-8 -*-
"""
Trade page — Paper trading, portfolio, performance, and risk (inlined; no runpy).
"""
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from components.theme import inject_theme, render_top_bar, keyboard_shortcut_js
from utils.dataframe_utils import normalize_for_display

try:
    st.markdown(keyboard_shortcut_js(), unsafe_allow_html=True)
except Exception:
    pass
inject_theme()
render_top_bar()

logger = logging.getLogger(__name__)

_PAPER_INIT = 100_000.0


def _paper_reset():
    st.session_state.evolve_paper_cash = _PAPER_INIT
    st.session_state.evolve_paper_positions = {}
    st.session_state.evolve_paper_trades = []
    st.session_state.evolve_last_prices = {}
    st.session_state.evolve_equity_log = []


def _ensure_paper():
    if "evolve_paper_cash" not in st.session_state:
        _paper_reset()
    if "evolve_equity_log" not in st.session_state:
        st.session_state.evolve_equity_log = []


def _paper_equity() -> float:
    _ensure_paper()
    cash = float(st.session_state.evolve_paper_cash)
    pos = st.session_state.evolve_paper_positions
    prices = st.session_state.evolve_last_prices
    mv = 0.0
    for sym, row in (pos or {}).items():
        q = float(row.get("qty", 0) or 0)
        if q <= 0:
            continue
        p = float(prices.get(sym) or row.get("avg", 0) or 0)
        mv += q * p
    return cash + mv


def _log_equity(note: str = ""):
    st.session_state.evolve_equity_log.append({
        "time": datetime.now().isoformat(),
        "equity": _paper_equity(),
        "note": note,
    })


_ensure_paper()

st.title("💰 Trade")
st.caption("Paper execution, portfolio, performance, and risk")

eq = _paper_equity()
day_delta = None
_log = st.session_state.evolve_equity_log
if len(_log) >= 2:
    try:
        prev = float(_log[-2]["equity"])
        if prev > 0:
            day_delta = (eq - prev) / prev
    except Exception as e:
        logger.debug("trade: day pnl calc skip: %s", e)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Portfolio value (paper)", f"${eq:,.2f}")
with c2:
    st.metric(
        "Session Δ vs last mark",
        f"{day_delta*100:.2f}%" if day_delta is not None else "—",
        delta=None,
    )
with c3:
    tr = (eq / _PAPER_INIT - 1.0) if _PAPER_INIT else 0.0
    st.metric("Total return (vs $100k start)", f"{tr*100:.2f}%")
with c4:
    if st.button("Reset paper book", key="paper_reset_all"):
        _paper_reset()
        st.rerun()

tab_paper, tab_port, tab_perf, tab_risk = st.tabs(
    ["Paper Trading", "Portfolio", "Performance", "Risk Management"]
)

with tab_paper:
    st.subheader("Paper Trading")
    st.caption(
        "Simulated fills at last close from yfinance. "
        "No live broker — for experimentation only."
    )
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_sym = st.text_input("Symbol", "AAPL", key="paper_sym").strip().upper()
    with pc2:
        p_side = st.selectbox("Side", ["Buy", "Sell"], key="paper_side")
    with pc3:
        p_ot = st.selectbox("Order type", ["Market", "Limit"], key="paper_ot")
    pc4, pc5 = st.columns(2)
    with pc4:
        p_qty = st.number_input("Quantity (shares)", min_value=0.0, value=1.0, step=1.0, key="paper_qty")
    with pc5:
        p_limit = st.number_input(
            "Limit price (limit orders only)",
            min_value=0.0,
            value=0.0,
            format="%.2f",
            key="paper_limit",
        )

    if st.button("Place order", key="paper_place", type="primary"):
        if not p_sym or p_qty <= 0:
            st.warning("Enter a symbol and positive quantity.")
        else:
            try:
                from trading.data.price_cache import get_history

                hist = get_history(p_sym, period="5d")
                if hist.empty:
                    st.warning(f"No quote data for {p_sym}.")
                else:
                    _cm = {c.lower(): c for c in hist.columns}
                    _cc = _cm.get("close", hist.columns[0])
                    last = float(hist[_cc].iloc[-1])
                    fill = last
                    if p_ot == "Limit" and p_limit > 0:
                        if p_side == "Buy":
                            fill = min(last, p_limit)
                        else:
                            fill = max(last, p_limit)
                    fee_rate = 0.0001
                    _ensure_paper()
                    cash = float(st.session_state.evolve_paper_cash)
                    pos = dict(st.session_state.evolve_paper_positions)
                    cur = pos.get(p_sym, {"qty": 0.0, "avg": 0.0})
                    cur_q = float(cur.get("qty", 0) or 0)
                    cur_avg = float(cur.get("avg", 0) or 0)

                    if p_side == "Buy":
                        cost = p_qty * fill * (1.0 + fee_rate)
                        if cost > cash + 1e-6:
                            st.warning("Insufficient paper cash for this order.")
                        else:
                            new_q = cur_q + p_qty
                            new_avg = (
                                (cur_q * cur_avg + p_qty * fill) / new_q
                                if new_q > 0
                                else 0.0
                            )
                            st.session_state.evolve_paper_cash = cash - cost
                            pos[p_sym] = {"qty": new_q, "avg": new_avg}
                            st.session_state.evolve_paper_positions = pos
                            st.session_state.evolve_last_prices[p_sym] = fill
                            st.session_state.evolve_paper_trades.append({
                                "id": str(uuid.uuid4()),
                                "time": datetime.now().isoformat(),
                                "symbol": p_sym,
                                "side": "BUY",
                                "qty": p_qty,
                                "price": fill,
                                "type": p_ot,
                            })
                            _log_equity("buy")
                            st.success(f"Bought {p_qty} {p_sym} @ ~{fill:.2f} (paper).")
                    else:
                        if p_qty > cur_q + 1e-9:
                            st.warning("Cannot sell more than open position.")
                        else:
                            proceeds = p_qty * fill * (1.0 - fee_rate)
                            pnl = (fill - cur_avg) * p_qty
                            st.session_state.evolve_paper_cash = cash + proceeds
                            new_q = cur_q - p_qty
                            if new_q < 1e-9:
                                pos.pop(p_sym, None)
                            else:
                                pos[p_sym] = {"qty": new_q, "avg": cur_avg}
                            st.session_state.evolve_paper_positions = pos
                            st.session_state.evolve_last_prices[p_sym] = fill
                            st.session_state.evolve_paper_trades.append({
                                "id": str(uuid.uuid4()),
                                "time": datetime.now().isoformat(),
                                "symbol": p_sym,
                                "side": "SELL",
                                "qty": p_qty,
                                "price": fill,
                                "type": p_ot,
                                "realized_pnl": round(pnl, 2),
                            })
                            _log_equity("sell")
                            st.success(
                                f"Sold {p_qty} {p_sym} @ ~{fill:.2f} "
                                f"(paper). Realized Δ vs avg: ${pnl:,.2f}"
                            )
            except Exception as e:
                st.caption(f"Order failed: {e}")

    st.markdown("#### Open positions")
    _ensure_paper()
    _pos = st.session_state.evolve_paper_positions
    _prices = st.session_state.evolve_last_prices
    if not _pos:
        st.info("No open paper positions.")
    else:
        if st.button("Refresh market marks", key="paper_refresh_marks"):
            try:
                from trading.data.price_cache import get_history, get_quote

                for _msym in list(
                    st.session_state.evolve_paper_positions.keys()
                ):
                    _mq = get_quote(_msym)
                    _mpx = _mq.get("price")
                    if _mpx is None:
                        _mh = get_history(_msym, period="5d")
                        if _mh is not None and not _mh.empty:
                            _mcm = {
                                c.lower(): c for c in _mh.columns
                            }
                            _mcc = _mcm.get("close", _mh.columns[0])
                            _mpx = float(_mh[_mcc].iloc[-1])
                    if _mpx is not None:
                        st.session_state.evolve_last_prices[_msym] = float(
                            _mpx
                        )
                st.rerun()
            except Exception as e:
                st.caption(f"Refresh failed: {e}")
        rows = []
        for sym, row in _pos.items():
            q = float(row.get("qty", 0) or 0)
            avg = float(row.get("avg", 0) or 0)
            px = float(_prices.get(sym, avg) or avg)
            u_pnl = (px - avg) * q if q else 0.0
            pnl_pct = ((px - avg) / avg * 100.0) if avg else 0.0
            rows.append({
                "Symbol": sym,
                "Qty": q,
                "Avg cost": round(avg, 4),
                "Last mark": round(px, 4),
                "Unrealized P&L": round(u_pnl, 2),
                "P&L %": round(pnl_pct, 2),
            })
        st.dataframe(
            normalize_for_display(pd.DataFrame(rows)),
            use_container_width=True,
            key="paper_positions_df",
        )
        st.caption("Close a full long via market fill at last mark.")
        _c_sym = st.selectbox(
            "Position to close",
            list(_pos.keys()),
            key="paper_close_pick",
        )
        if st.button("Close full position", key="paper_close_btn"):
            try:
                from trading.data.price_cache import get_history, get_quote

                _ensure_paper()
                cash = float(st.session_state.evolve_paper_cash)
                pos = dict(st.session_state.evolve_paper_positions)
                cur = pos.get(_c_sym, {"qty": 0.0, "avg": 0.0})
                cur_q = float(cur.get("qty", 0) or 0)
                cur_avg = float(cur.get("avg", 0) or 0)
                if cur_q <= 0:
                    st.warning("No quantity for selected symbol.")
                else:
                    _cq = get_quote(_c_sym)
                    fill = _cq.get("price")
                    if fill is None:
                        _ch = get_history(_c_sym, period="5d")
                        if _ch is None or _ch.empty:
                            st.warning("No price for close.")
                        else:
                            _ccm = {c.lower(): c for c in _ch.columns}
                            _ccc = _ccm.get("close", _ch.columns[0])
                            fill = float(_ch[_ccc].iloc[-1])
                    if fill is not None:
                        fill = float(fill)
                        fee_rate = 0.0001
                        proceeds = cur_q * fill * (1.0 - fee_rate)
                        pnl = (fill - cur_avg) * cur_q
                        st.session_state.evolve_paper_cash = cash + proceeds
                        pos.pop(_c_sym, None)
                        st.session_state.evolve_paper_positions = pos
                        st.session_state.evolve_last_prices[_c_sym] = fill
                        st.session_state.evolve_paper_trades.append({
                            "id": str(uuid.uuid4()),
                            "time": datetime.now().isoformat(),
                            "symbol": _c_sym,
                            "side": "SELL",
                            "qty": cur_q,
                            "price": fill,
                            "type": "Market",
                            "realized_pnl": round(pnl, 2),
                        })
                        _log_equity("close")
                        st.success(
                            f"Closed {_c_sym} {cur_q} sh @ ~{fill:.2f} (paper)."
                        )
                        st.rerun()
            except Exception as e:
                st.caption(f"Close failed: {e}")

    st.markdown("**Estimated costs**")
    try:
        _slippage_bps = st.slider(
            "Slippage (bps)",
            min_value=1,
            max_value=50,
            value=5,
            key="trade_slippage_bps",
            help="Assumed basis points for quick notional estimates.",
        )
        _shares = st.number_input(
            "Shares (for estimate)", min_value=0, value=0, key="trade_est_shares"
        )
        _price = st.number_input(
            "Price (for estimate)",
            min_value=0.0,
            value=0.0,
            format="%.2f",
            key="trade_est_price",
        )
        if _shares and _price:
            _notional = float(_shares) * float(_price)
            _slip_cost = _notional * (_slippage_bps / 10000)
            _commission = max(1.0, _notional * 0.0001)
            _total_cost = _slip_cost + _commission
            e1, e2, e3 = st.columns(3)
            with e1:
                st.metric("Notional", f"${_notional:,.2f}")
            with e2:
                st.metric("Est. slippage", f"${_slip_cost:.2f}", f"{_slippage_bps}bps")
            with e3:
                st.metric("Total cost", f"${_total_cost:.2f}")
    except Exception as e:
        st.caption(f"Cost estimate unavailable: {e}")

with tab_port:
    st.subheader("Portfolio")
    _ensure_paper()
    cash = float(st.session_state.evolve_paper_cash)
    st.metric("Cash (paper)", f"${cash:,.2f}")
    st.metric("Equity (cash + marks)", f"${_paper_equity():,.2f}")

    st.markdown("#### Holdings (paper)")
    if not st.session_state.evolve_paper_positions:
        st.info("No holdings. Place a paper trade in the Paper Trading tab.")
    else:
        rows = []
        for sym, row in st.session_state.evolve_paper_positions.items():
            q = float(row.get("qty", 0) or 0)
            avg = float(row.get("avg", 0) or 0)
            px = float(st.session_state.evolve_last_prices.get(sym, avg) or avg)
            rows.append({
                "Symbol": sym,
                "Quantity": q,
                "Avg price": avg,
                "Mark": px,
                "Market value": round(q * px, 2),
            })
        st.dataframe(
            normalize_for_display(pd.DataFrame(rows)),
            use_container_width=True,
            key="port_holdings_df",
        )

    st.markdown("#### Platform portfolio (optional)")
    try:
        from trading.portfolio.portfolio_manager import PortfolioManager

        if "portfolio_manager" not in st.session_state:
            st.session_state.portfolio_manager = PortfolioManager()
        pm = st.session_state.portfolio_manager
        ext = pm.get_all_positions() if hasattr(pm, "get_all_positions") else []
        if ext:
            st.caption("Positions from portfolio manager (if configured).")
            st.dataframe(
                normalize_for_display(pd.DataFrame(ext)),
                use_container_width=True,
                key="port_pm_df",
            )
        else:
            st.caption(
                "Portfolio manager has no positions, or module uses external "
                "storage that is not populated."
            )
    except Exception as e:
        st.caption(
            f"Portfolio module unavailable — paper book above is the "
            f"supported path. ({e})"
        )

with tab_perf:
    st.subheader("Performance")
    _ensure_paper()
    trades = list(st.session_state.evolve_paper_trades)
    if not trades:
        st.info("No paper trades yet. Executions appear here after you place orders.")
    else:
        tdf = pd.DataFrame(trades)
        st.markdown("#### Trade history")
        st.dataframe(
            normalize_for_display(tdf),
            use_container_width=True,
            key="perf_trades_df",
        )

        elog = st.session_state.get("evolve_equity_log") or []
        if len(elog) >= 3:
            edf = pd.DataFrame(elog)
            try:
                edf["t"] = pd.to_datetime(edf["time"])
                edf = edf.sort_values("t")
                rets = edf["equity"].astype(float).pct_change().dropna()
                if len(rets) > 5:
                    from utils.risk_metrics import compute_performance_metrics

                    pm = compute_performance_metrics(rets)
                    st.markdown("#### Metrics (from paper equity marks)")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total return (window)", f"{pm.total_return*100:.2f}%")
                    m2.metric("Sharpe (approx)", f"{pm.sharpe_ratio:.2f}")
                    m3.metric("Max drawdown", f"{pm.max_drawdown*100:.2f}%")
                    m4.metric("Win rate (marks)", f"{pm.win_rate*100:.1f}%")
                    with st.expander("Full metrics"):
                        st.json(pm.to_dict())

                    st.markdown("**Alpha Attribution**")
                    try:
                        import yfinance as yf

                        from trading.analytics.alpha_attribution_engine import (
                            AttributionMethod,
                            get_alpha_attribution_engine,
                        )

                        _bench = yf.Ticker("SPY").history(period="1y")
                        if not _bench.empty:
                            _cm = {c.lower(): c for c in _bench.columns}
                            _bc = _cm.get("close", _bench.columns[0])
                            _br = _bench[_bc].pct_change().dropna()
                            _br = _br.reindex(rets.index).fillna(0.0)
                            _pr = rets.reindex(_br.index).fillna(0.0)
                            if len(_pr) > 10:
                                _eng = get_alpha_attribution_engine()
                                _att = _eng.perform_attribution_analysis(
                                    _pr,
                                    {"paper": _pr},
                                    _br,
                                    method=AttributionMethod.STRATEGY_DECOMPOSITION,
                                )
                                _rows = []
                                for k, v in (
                                    _att.strategy_attribution or {}
                                ).items():
                                    _rows.append({"Component": k, "Contribution": v})
                                if _rows:
                                    st.dataframe(
                                        normalize_for_display(pd.DataFrame(_rows)),
                                        use_container_width=True,
                                    )
                                try:
                                    st.markdown("**Alpha Attribution**")
                                    _x1, _x2, _x3 = st.columns(3)
                                    _x1.metric(
                                        "Alpha",
                                        f"{float(_att.excess_return):.2%}",
                                    )
                                    _x2.metric(
                                        "Beta contribution",
                                        f"{float(_att.benchmark_return):.2%}",
                                    )
                                    _sa = _att.strategy_attribution or {}
                                    _sel = (
                                        float(sum(_sa.values()))
                                        if _sa
                                        else float(_att.excess_return)
                                    )
                                    _x3.metric(
                                        "Selection effect",
                                        f"{_sel:.2%}",
                                    )
                                except Exception as _me:
                                    st.caption(
                                        f"Attribution display unavailable: {_me}"
                                    )
                    except Exception as _ae:
                        st.caption(f"Alpha attribution unavailable: {_ae}")
            except Exception as e:
                st.caption(f"Could not compute performance metrics: {e}")
        else:
            st.caption(
                "More equity marks are needed for performance metrics "
                "(place additional trades)."
            )

with tab_risk:
    st.subheader("Risk Management")

    st.markdown("#### Position sizing (Kelly) & VaR")
    rk_sym = st.text_input(
        "Symbol for return history",
        "SPY",
        key="risk_kelly_symbol",
    ).strip().upper()
    rk_pv = st.number_input(
        "Notional for VaR ($)",
        min_value=1000.0,
        value=10000.0,
        key="risk_kelly_pv",
    )
    if st.button("Compute Kelly & VaR", key="risk_kelly_btn"):
        try:
            import yfinance as yf
            from utils.risk_metrics import (
                calculate_var,
                kelly_from_returns,
            )

            h = yf.Ticker(rk_sym).history(period="1y")
            if h.empty:
                st.warning(f"No data for {rk_sym}.")
            else:
                _cm = {c.lower(): c for c in h.columns}
                _cc = _cm.get("close", h.columns[0])
                r = h[_cc].pct_change().dropna()
                kv = kelly_from_returns(r)
                if kv.get("error"):
                    st.caption(f"Kelly: {kv['error']}")
                else:
                    st.json({k: v for k, v in kv.items() if k != "interpretation"})
                    st.caption(kv.get("interpretation", ""))
                vr = calculate_var(
                    r,
                    confidence=0.95,
                    portfolio_value=float(rk_pv),
                )
                if vr.get("error"):
                    st.caption(f"VaR: {vr['error']}")
                else:
                    st.metric("VaR (95%, 1d)", f"${vr.get('var_dollar', 0):,.2f}")
                    st.caption(vr.get("interpretation", ""))
        except Exception as e:
            st.caption(f"Kelly/VaR unavailable: {e}")

    st.markdown("---")
    st.markdown("**Advanced risk (volatility & stress)**")
    try:
        import yfinance as yf

        from trading.risk.risk_manager import RiskManager

        if st.button("Run stress scenarios", key="trade_rm_stress"):
            with st.spinner("Stress tests…"):
                _h = yf.Ticker("SPY").history(period="2y")
                if not _h.empty:
                    _cm = {c.lower(): c for c in _h.columns}
                    _cc = _cm.get("close", _h.columns[0])
                    _r = _h[_cc].pct_change().dropna()
                    _rm = RiskManager()
                    _rm.update_returns(_r)
                    _stress = _rm.run_stress_tests(float(_paper_equity()))
                    if _stress:
                        st.dataframe(
                            normalize_for_display(
                                pd.DataFrame([s.__dict__ for s in _stress])
                            ),
                            use_container_width=True,
                        )
    except Exception as _re:
        st.caption(f"Advanced risk analysis unavailable: {_re}")

    st.markdown("---")
    st.subheader("📊 Advanced Risk Analytics")
    try:
        from trading.risk.advanced_risk import AdvancedRiskAnalyzer
        from trading.portfolio.portfolio_manager import PortfolioManager

        _pm = st.session_state.get("portfolio_manager")
        if _pm and isinstance(_pm, PortfolioManager):
            _positions = _pm.get_all_positions()
            if _positions:
                _analyzer = AdvancedRiskAnalyzer()
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
                st.caption("No open positions from portfolio manager to analyze.")
        else:
            st.caption(
                "Advanced analytics apply when the portfolio manager is initialized "
                "and has positions."
            )
    except Exception as _re:
        st.caption(f"Advanced risk unavailable: {_re}")

    st.markdown("---")
    st.subheader("📉 Risk Analytics")

    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        risk_symbol = st.text_input("Symbol", "AAPL", key="risk_symbol")
    with risk_col2:
        risk_portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000,
            key="risk_portfolio_value",
        )

    if st.button(
        "▶ Calculate Risk Metrics",
        key="risk_calc_btn",
        type="primary",
    ):
        try:
            import yfinance as yf
            from utils.risk_metrics import render_risk_metrics_streamlit

            with st.spinner(f"Calculating risk metrics for {risk_symbol}..."):
                _hist = yf.Ticker(risk_symbol).history(period="1y")
                _spy = yf.Ticker("SPY").history(period="1y")

                if _hist.empty:
                    st.warning(f"No data for {risk_symbol}")
                else:
                    _col_map = {c.lower(): c for c in _hist.columns}
                    _close_col = _col_map.get("close", _hist.columns[0])
                    _returns = _hist[_close_col].pct_change().dropna()

                    _spy_returns = None
                    if not _spy.empty:
                        _spy_col_map = {c.lower(): c for c in _spy.columns}
                        _spy_close = _spy_col_map.get("close", _spy.columns[0])
                        _spy_returns = _spy[_spy_close].pct_change().dropna()

                    render_risk_metrics_streamlit(
                        returns=_returns,
                        symbol=risk_symbol,
                        portfolio_value=float(risk_portfolio_value),
                        benchmark_returns=_spy_returns,
                    )
        except Exception as e:
            st.caption(f"Risk metrics unavailable: {e}")

try:
    from ui.page_assistant import render_page_assistant

    render_page_assistant("Trade")
except Exception:
    pass
