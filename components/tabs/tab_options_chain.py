# -*- coding: utf-8 -*-
"""Analyze page tab body (extracted; logic unchanged)."""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _components
from plotly.subplots import make_subplots

from components.analyze_common import (
    _extract_forecast_values,
    _generate_recommendation,
    _is_english,
    _news_sentiment_score,
)
from trading.data.earnings_calendar import get_upcoming_earnings
from trading.data.insider_flow import get_insider_flow
from trading.data.price_cache import get_history, get_info, get_news, get_quote

try:
    from utils.dataframe_utils import normalize_for_display
except ImportError:
    def normalize_for_display(df):
        return df

logger = logging.getLogger(__name__)


def _get_premium(
    df: pd.DataFrame,
    strike: float,
    opt_type: str,
) -> float:
    """Get mid-price for a strike from calls/puts DataFrame."""
    try:
        _row = df[df["strike"].astype(float) == float(strike)]
        if _row.empty:
            _row = df.iloc[
                (df["strike"].astype(float) - float(strike))
                .abs()
                .argsort()[:1]
            ]
        if _row.empty:
            return 1.0
        _bid = (
            float(_row["bid"].iloc[0])
            if "bid" in _row.columns
            else 0.0
        )
        _ask = (
            float(_row["ask"].iloc[0])
            if "ask" in _row.columns
            else 0.0
        )
        if _bid > 0 and _ask > 0:
            return round((_bid + _ask) / 2, 2)
        _last = (
            float(_row["lastPrice"].iloc[0])
            if "lastPrice" in _row.columns
            else 1.0
        )
        return max(0.01, _last)
    except Exception:
        return 1.0


def render(
    ticker: str,
    hist,
    period: str,
    period_label: str,
    trader_mode: str,
    _interval: str,
    _tf_label: str,
    *,
    backend: dict,
) -> None:
    """Streamlit tab body (legacy Analyze)."""
    DataLoader = backend["DataLoader"]
    DataLoadRequest = backend["DataLoadRequest"]
    YFinanceProvider = backend["YFinanceProvider"]
    LSTMForecaster = backend["LSTMForecaster"]
    XGBoostModel = backend["XGBoostModel"]
    ProphetModel = backend["ProphetModel"]
    ARIMAModel = backend["ARIMAModel"]
    FeatureEngineering = backend["FeatureEngineering"]
    DataPreprocessor = backend["DataPreprocessor"]
    ModelSelectorAgent = backend["ModelSelectorAgent"]
    MarketAnalyzer = backend["MarketAnalyzer"]
    try:
        st.subheader("Options & Short Data")

        try:
            import yfinance as yf

            _t = yf.Ticker(ticker)
            _info = _t.info

            # Short data
            _short_pct = _info.get("shortPercentOfFloat", 0) or 0
            _short_ratio = _info.get("shortRatio", 0) or 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Short float %",
                    f"{_short_pct * 100:.1f}%",
                    help="% of float sold short. >20% = high short interest",
                )
            with c2:
                st.metric(
                    "Days to cover",
                    f"{_short_ratio:.1f}",
                    help=(
                        "Days to cover short positions at avg volume. "
                        ">5 = squeeze potential"
                    ),
                )
            with c3:
                _pc_key = f"put_call_{ticker}"
                _pc_val = st.session_state.get(_pc_key)
                if _pc_val is not None:
                    st.metric(
                        "Put/call ratio",
                        f"{_pc_val:.2f}",
                        help=">1 bearish, <1 bullish",
                    )
                else:
                    st.metric(
                        "Put/call ratio",
                        "Load chain →",
                        help=(
                            "Load options chain below "
                            "to calculate"
                        ),
                    )

            st.markdown("---")

            # Options chain
            if "options_loaded_" + ticker not in st.session_state:
                if st.button("Load options chain", key="load_options_btn"):
                    st.session_state["options_loaded_" + ticker] = True
                else:
                    st.caption("Click to load live options data.")
                    return

            _expiries = _t.options
            if _expiries:
                _sel_exp = st.selectbox(
                    "Expiry date",
                    _expiries[:8],
                    key="options_expiry",
                )
                _cache_key = f"options_chain_{ticker}_{_sel_exp}"
                if _cache_key not in st.session_state:
                    try:
                        import threading as _threading

                        _result = [None]

                        def _fetch_chain() -> None:
                            try:
                                _result[0] = _t.option_chain(_sel_exp)
                            except Exception as _e:
                                logger.warning(
                                    "Analyze: option_chain fetch failed: %s", _e
                                )

                        _th = _threading.Thread(target=_fetch_chain)
                        _th.start()
                        _th.join(timeout=8)
                        if _result[0] is not None:
                            st.session_state[_cache_key] = _result[0]
                    except Exception as _e:
                        logger.warning("Analyze: options chain fetch failed: %s", _e)

                _chain = st.session_state.get(_cache_key)
                if _chain is None:
                    st.warning("Options chain timed out. Try again.")
                    return
                _calls = _chain.calls
                _puts = _chain.puts

                # Put/call ratio from volumes (persist in session_state)
                try:
                    _cv = float(_chain.calls["volume"].fillna(0).sum())
                    _pv = float(_chain.puts["volume"].fillna(0).sum())
                    _pc = round(_pv / _cv, 2) if _cv > 0 else None
                    if _pc is not None:
                        st.session_state[f"put_call_{ticker}"] = _pc
                except Exception:
                    pass

                # IV rank approximation
                # Use ATM options (closest to current price) for IV calculation
                _cur_price = get_quote(ticker).get("price", 0)
                if _cur_price and not _calls.empty:
                    _calls = _calls.copy()
                    _puts = _puts.copy()
                    _calls["dist"] = abs(_calls["strike"] - _cur_price)
                    _atm_iv = float(
                        _calls.nsmallest(5, "dist")["impliedVolatility"].mean()
                    )
                    # Simple IV rank: compare to typical range (annualized)
                    _iv_pct = min(100, _atm_iv * 100)
                    _iv_color = (
                        "#ef5350"
                        if _iv_pct > 50
                        else "#ff9800"
                        if _iv_pct > 30
                        else "#26a69a"
                    )
                    st.markdown(
                        "**ATM IV (approx):** "
                        f'<span style="color:{_iv_color};font-weight:bold">'
                        f"{_iv_pct:.1f}%</span> annualized",
                        unsafe_allow_html=True,
                    )

                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    st.markdown("**Calls**")
                    _c_display = _calls[
                        [
                            "strike",
                            "lastPrice",
                            "impliedVolatility",
                            "volume",
                            "openInterest",
                            "inTheMoney",
                        ]
                    ].copy()
                    _c_display.columns = [
                        "Strike",
                        "Last",
                        "IV",
                        "Vol",
                        "OI",
                        "ITM",
                    ]
                    _c_display["IV"] = (_c_display["IV"] * 100).round(1).astype(str) + "%"
                    st.dataframe(
                        normalize_for_display(_c_display),
                        width='stretch',
                        height=300,
                        key="options_calls_table",
                    )
                with col_opt2:
                    st.markdown("**Puts**")
                    _p_display = _puts[
                        [
                            "strike",
                            "lastPrice",
                            "impliedVolatility",
                            "volume",
                            "openInterest",
                            "inTheMoney",
                        ]
                    ].copy()
                    _p_display.columns = [
                        "Strike",
                        "Last",
                        "IV",
                        "Vol",
                        "OI",
                        "ITM",
                    ]
                    _p_display["IV"] = (_p_display["IV"] * 100).round(1).astype(str) + "%"
                    st.dataframe(
                        normalize_for_display(_p_display),
                        width='stretch',
                        height=300,
                        key="options_puts_table",
                    )

                try:
                    from trading.data.options_flow import get_options_flow

                    flow = get_options_flow(ticker)
                    if flow and not flow.get("error"):
                        st.markdown("**Unusual Options Activity**")
                        net = flow.get("net_flow", "NEUTRAL")
                        color = (
                            "green"
                            if net == "BULLISH"
                            else "red"
                            if net == "BEARISH"
                            else "gray"
                        )
                        st.markdown(
                            f"Net flow: :{color}[**{net}**] · "
                            f"P/C Ratio: {flow.get('put_call_ratio', 0):.2f} · "
                            f"Max pain: ${flow.get('max_pain', 0):.2f}"
                        )
                        calls = flow.get("unusual_calls", [])
                        puts = flow.get("unusual_puts", [])
                        if calls:
                            st.caption(
                                "Unusual calls: "
                                + ", ".join(
                                    f"${c['strike']:.0f} "
                                    f"({c['expiry']})"
                                    for c in calls[:5]
                                )
                            )
                        if puts:
                            st.caption(
                                "Unusual puts: "
                                + ", ".join(
                                    f"${p['strike']:.0f} "
                                    f"({p['expiry']})"
                                    for p in puts[:5]
                                )
                            )
                except Exception as e:
                    st.caption(f"Options flow unavailable: {e}")

                # ── OPTIONS STRATEGY VISUALIZER ──
                st.markdown("---")
                st.markdown("### 📐 Strategy Visualizer")
                st.caption(
                    "Select a strategy to see P&L at expiration "
                    "and key metrics."
                )
                try:
                    _ref_spot = float(_cur_price) if _cur_price else 0.0
                    _all_strikes = sorted(
                        set(
                            list(
                                _calls["strike"]
                                .dropna()
                                .astype(float)
                            )
                            + list(
                                _puts["strike"]
                                .dropna()
                                .astype(float)
                            )
                        )
                    )
                    if not _all_strikes:
                        st.caption("No strikes available for strategies.")
                    else:
                        if _ref_spot <= 0:
                            _ref_spot = float(
                                _all_strikes[len(_all_strikes) // 2]
                            )
                        _strat = st.selectbox(
                            "Strategy",
                            [
                                "Long Call",
                                "Long Put",
                                "Covered Call",
                                "Cash-Secured Put",
                                "Bull Call Spread",
                                "Bear Put Spread",
                                "Long Straddle",
                                "Long Strangle",
                            ],
                            key=f"opt_strat_{ticker}",
                        )
                        _atm_idx = min(
                            range(len(_all_strikes)),
                            key=lambda i: abs(
                                _all_strikes[i] - _ref_spot
                            ),
                        )
                        _col_sv1, _col_sv2 = st.columns(2)
                        _legs: list = []

                        if _strat == "Long Call":
                            with _col_sv1:
                                _k1 = st.selectbox(
                                    "Call Strike",
                                    _all_strikes,
                                    index=_atm_idx,
                                    key=f"k1_{ticker}_{_strat}",
                                )
                            _prem1 = _get_premium(_calls, _k1, "call")
                            with _col_sv2:
                                _prem1 = st.number_input(
                                    "Call Premium ($)",
                                    value=float(_prem1),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                            _legs = [("call", _k1, _prem1, 1)]

                        elif _strat == "Covered Call":
                            with _col_sv1:
                                _k1 = st.selectbox(
                                    "Short Call Strike",
                                    _all_strikes,
                                    index=_atm_idx,
                                    key=f"k1_{ticker}_{_strat}",
                                )
                            _prem1 = _get_premium(_calls, _k1, "call")
                            with _col_sv2:
                                _prem1 = st.number_input(
                                    "Call Premium ($) (received)",
                                    value=float(_prem1),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                            _legs = [
                                ("stock", _ref_spot, _ref_spot, 1),
                                ("call", _k1, _prem1, -1),
                            ]

                        elif _strat == "Long Put":
                            with _col_sv1:
                                _k1 = st.selectbox(
                                    "Put Strike",
                                    _all_strikes,
                                    index=_atm_idx,
                                    key=f"k1_{ticker}_{_strat}",
                                )
                            _prem1 = _get_premium(_puts, _k1, "put")
                            with _col_sv2:
                                _prem1 = st.number_input(
                                    "Put Premium ($)",
                                    value=float(_prem1),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                            _legs = [("put", _k1, _prem1, 1)]

                        elif _strat == "Cash-Secured Put":
                            with _col_sv1:
                                _k1 = st.selectbox(
                                    "Short Put Strike",
                                    _all_strikes,
                                    index=_atm_idx,
                                    key=f"k1_{ticker}_{_strat}",
                                )
                            _prem1 = _get_premium(_puts, _k1, "put")
                            with _col_sv2:
                                _prem1 = st.number_input(
                                    "Put Premium ($) (received)",
                                    value=float(_prem1),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                            _legs = [("put", _k1, _prem1, -1)]

                        elif _strat == "Bull Call Spread":
                            _strikes_list = _all_strikes
                            with _col_sv1:
                                _k1 = st.selectbox(
                                    "Buy Call Strike",
                                    _strikes_list,
                                    index=max(0, _atm_idx - 1),
                                    key=f"k1_{ticker}_{_strat}",
                                )
                                _k2 = st.selectbox(
                                    "Sell Call Strike",
                                    _strikes_list,
                                    index=min(
                                        len(_strikes_list) - 1,
                                        _atm_idx + 1,
                                    ),
                                    key=f"k2_{ticker}_{_strat}",
                                )
                            _p1 = _get_premium(_calls, _k1, "call")
                            _p2 = _get_premium(_calls, _k2, "call")
                            with _col_sv2:
                                _p1 = st.number_input(
                                    "Buy Premium ($)",
                                    value=float(_p1),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                                _p2 = st.number_input(
                                    "Sell Premium ($)",
                                    value=float(_p2),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p2_{ticker}_{_strat}",
                                )
                            _legs = [
                                ("call", _k1, _p1, 1),
                                ("call", _k2, _p2, -1),
                            ]

                        elif _strat == "Bear Put Spread":
                            with _col_sv1:
                                _k1 = st.selectbox(
                                    "Buy Put Strike",
                                    _all_strikes,
                                    index=min(
                                        len(_all_strikes) - 1,
                                        _atm_idx + 1,
                                    ),
                                    key=f"k1_{ticker}_{_strat}",
                                )
                                _k2 = st.selectbox(
                                    "Sell Put Strike",
                                    _all_strikes,
                                    index=max(0, _atm_idx - 1),
                                    key=f"k2_{ticker}_{_strat}",
                                )
                            _p1 = _get_premium(_puts, _k1, "put")
                            _p2 = _get_premium(_puts, _k2, "put")
                            with _col_sv2:
                                _p1 = st.number_input(
                                    "Buy Premium ($)",
                                    value=float(_p1),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                                _p2 = st.number_input(
                                    "Sell Premium ($)",
                                    value=float(_p2),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p2_{ticker}_{_strat}",
                                )
                            _legs = [
                                ("put", _k1, _p1, 1),
                                ("put", _k2, _p2, -1),
                            ]

                        elif _strat == "Long Straddle":
                            with _col_sv1:
                                _k1 = st.selectbox(
                                    "ATM Strike",
                                    _all_strikes,
                                    index=_atm_idx,
                                    key=f"k1_{ticker}_{_strat}",
                                )
                            _pc = _get_premium(_calls, _k1, "call")
                            _pp = _get_premium(_puts, _k1, "put")
                            with _col_sv2:
                                _pc = st.number_input(
                                    "Call Premium ($)",
                                    value=float(_pc),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                                _pp = st.number_input(
                                    "Put Premium ($)",
                                    value=float(_pp),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p2_{ticker}_{_strat}",
                                )
                            _legs = [
                                ("call", _k1, _pc, 1),
                                ("put", _k1, _pp, 1),
                            ]

                        elif _strat == "Long Strangle":
                            with _col_sv1:
                                _k_call = st.selectbox(
                                    "OTM Call Strike",
                                    _all_strikes,
                                    index=min(
                                        len(_all_strikes) - 1,
                                        _atm_idx + 2,
                                    ),
                                    key=f"k1_{ticker}_{_strat}",
                                )
                                _k_put = st.selectbox(
                                    "OTM Put Strike",
                                    _all_strikes,
                                    index=max(0, _atm_idx - 2),
                                    key=f"k2_{ticker}_{_strat}",
                                )
                            _pc = _get_premium(_calls, _k_call, "call")
                            _pp = _get_premium(_puts, _k_put, "put")
                            with _col_sv2:
                                _pc = st.number_input(
                                    "Call Premium ($)",
                                    value=float(_pc),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p1_{ticker}_{_strat}",
                                )
                                _pp = st.number_input(
                                    "Put Premium ($)",
                                    value=float(_pp),
                                    min_value=0.01,
                                    step=0.01,
                                    key=f"p2_{ticker}_{_strat}",
                                )
                            _legs = [
                                ("call", _k_call, _pc, 1),
                                ("put", _k_put, _pp, 1),
                            ]

                        if _legs:
                            import numpy as _np
                            import plotly.graph_objects as _go

                            _price_range = _np.linspace(
                                _ref_spot * 0.7,
                                _ref_spot * 1.3,
                                200,
                            )
                            _pnl = _np.zeros(len(_price_range))

                            for _typ, _k, _p, _dir in _legs:
                                if _typ == "call":
                                    _pnl += (
                                        _dir
                                        * (
                                            _np.maximum(
                                                _price_range - _k,
                                                0,
                                            )
                                            - _p
                                        )
                                        * 100
                                    )
                                elif _typ == "put":
                                    _pnl += (
                                        _dir
                                        * (
                                            _np.maximum(
                                                _k - _price_range,
                                                0,
                                            )
                                            - _p
                                        )
                                        * 100
                                    )
                                elif _typ == "stock":
                                    _pnl += (_price_range - _k) * 100

                            _max_profit = float(_np.max(_pnl))
                            _max_loss = float(_np.min(_pnl))
                            _breakevens = []
                            for _i in range(1, len(_pnl)):
                                if (_pnl[_i - 1] < 0 and _pnl[_i] >= 0) or (
                                    _pnl[_i - 1] >= 0 and _pnl[_i] < 0
                                ):
                                    _be = float(_price_range[_i])
                                    _breakevens.append(round(_be, 2))

                            _pop = 0.0
                            try:
                                _profitable = (_pnl > 0).sum()
                                _pop = _profitable / len(_pnl) * 100
                            except Exception:
                                _pop = 0.0

                            _fig = _go.Figure()
                            _fig.add_trace(
                                _go.Scatter(
                                    x=list(_price_range),
                                    y=list(_pnl),
                                    mode="lines",
                                    line=dict(color="#00D4FF", width=2),
                                    fill="tozeroy",
                                    fillcolor="rgba(0,212,255,0.08)",
                                    name="P&L",
                                )
                            )
                            _fig.add_hline(
                                y=0,
                                line_dash="dash",
                                line_color="#666666",
                                line_width=1,
                            )
                            _fig.add_vline(
                                x=_ref_spot,
                                line_dash="dot",
                                line_color="#FFD700",
                                line_width=1,
                                annotation_text=(
                                    f"Current ${_ref_spot:.2f}"
                                ),
                                annotation_position="top right",
                            )
                            for _be in _breakevens[:2]:
                                _fig.add_vline(
                                    x=_be,
                                    line_dash="dash",
                                    line_color="#FF6B6B",
                                    line_width=1,
                                )

                            _fig.update_layout(
                                title=(
                                    f"{_strat} P&L at Expiration "
                                    f"(per 100 shares)"
                                ),
                                xaxis_title="Stock Price ($)",
                                yaxis_title="P&L ($)",
                                template="plotly_dark",
                                height=380,
                                margin=dict(
                                    l=40,
                                    r=20,
                                    t=40,
                                    b=40,
                                ),
                                showlegend=False,
                            )
                            st.plotly_chart(_fig, width="stretch")

                            _m1, _m2, _m3, _m4 = st.columns(4)
                            with _m1:
                                _mp_show = (
                                    "Unlimited"
                                    if _max_profit >= 1e6
                                    else f"${_max_profit:,.0f}"
                                )
                                st.metric("Max Profit", _mp_show)
                            with _m2:
                                st.metric(
                                    "Max Loss",
                                    f"${abs(_max_loss):,.0f}",
                                )
                            with _m3:
                                _be_str = (
                                    " / ".join(
                                        f"${b:.2f}"
                                        for b in _breakevens[:2]
                                    )
                                    if _breakevens
                                    else "N/A"
                                )
                                st.metric("Breakeven(s)", _be_str)
                            with _m4:
                                st.metric(
                                    "Prob. of Profit",
                                    f"{_pop:.0f}%"
                                    if _pop > 0
                                    else "N/A",
                                )
                except Exception as _sve:
                    st.caption(
                        f"Strategy visualizer unavailable: {_sve}"
                    )

                try:
                    from trading.options.options_forecaster import (
                        OptionsForecaster,
                    )

                    _of = OptionsForecaster()
                    _surf = _of.build_volatility_surface(ticker)
                    _ivm = float(
                        np.mean(_surf.implied_volatilities)
                        if len(_surf.implied_volatilities)
                        else 0.0
                    )
                    st.markdown("**Implied volatility surface**")
                    st.caption(
                        f"Sampled {len(_surf.strikes)} strikes · "
                        f"mean IV ≈ {_ivm * 100:.1f}%"
                    )
                except Exception as _ive:
                    st.caption(f"Options forecasting unavailable: {_ive}")
            else:
                st.info("No options data available for " + ticker)
        except Exception as _oe:
            st.caption(f"Options data unavailable: {_oe}")
    except Exception as e:
        st.caption(f"Tab unavailable: {e}")
