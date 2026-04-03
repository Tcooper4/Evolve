# -*- coding: utf-8 -*-
"""Options chain and short-interest panel."""
import streamlit as st


def render_options(ticker: str) -> None:
    try:
        import yfinance as yf

        from trading.data.price_cache import get_quote

        try:
            from utils.dataframe_utils import normalize_for_display
        except ImportError:
            def normalize_for_display(df):
                return df

        _t = yf.Ticker(ticker)
        _info = _t.info or {}
        _short_pct = _info.get("shortPercentOfFloat", 0) or 0
        _short_ratio = _info.get("shortRatio", 0) or 0
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Short float %", f"{float(_short_pct) * 100:.1f}%")
        with c2:
            st.metric("Days to cover", f"{float(_short_ratio):.1f}")

        _expiries = getattr(_t, "options", None)
        if not _expiries:
            st.info("No options data for this symbol.")
            return
        _sel_exp = st.selectbox("Expiry", list(_expiries)[:8], key=f"dd_opt_exp_{ticker}")
        _chain = _t.option_chain(_sel_exp)
        _calls = _chain.calls
        _puts = _chain.puts
        _cur_price = get_quote(ticker).get("price", 0)
        if _cur_price and not _calls.empty:
            _calls = _calls.copy()
            _calls["dist"] = abs(_calls["strike"] - _cur_price)
            _atm_iv = float(_calls.nsmallest(5, "dist")["impliedVolatility"].mean())
            st.caption(f"ATM IV (approx): {_atm_iv * 100:.1f}% annualized")
        col_a, col_b = st.columns(2)
        with col_a:
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
            _c_display.columns = ["Strike", "Last", "IV", "Vol", "OI", "ITM"]
            _c_display["IV"] = (_c_display["IV"] * 100).round(1).astype(str) + "%"
            st.dataframe(normalize_for_display(_c_display), width="stretch", height=280)
        with col_b:
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
            _p_display.columns = ["Strike", "Last", "IV", "Vol", "OI", "ITM"]
            _p_display["IV"] = (_p_display["IV"] * 100).round(1).astype(str) + "%"
            st.dataframe(normalize_for_display(_p_display), width="stretch", height=280)
    except Exception as e:
        st.caption(f"unavailable: {e}")
