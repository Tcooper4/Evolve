"""
Market Scanner page — screen stocks by technical and AI-driven filters.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Market Scanner", page_icon="🔍", layout="wide")
st.title("🔍 Market Scanner")
st.caption("Screen stocks by technical conditions and AI Score ranking")

try:
    from trading.analysis.market_scanner import scan_market, get_available_filters, DEFAULT_UNIVERSE
    scanner_available = True
except Exception as e:
    st.error(f"Scanner unavailable: {e}")
    scanner_available = False

if not scanner_available:
    st.stop()

# ── Controls ────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    available_filters = get_available_filters()
    selected_filters = st.multiselect(
        "Scan Filters",
        options=list(available_filters.keys()),
        default=["top_ai_score"],
        format_func=lambda k: f"{k}: {available_filters[k]}",
        help="Select one or more filters. Stocks must pass ALL selected filters.",
    )

with col_right:
    max_results = st.slider("Max results", 5, 50, 20)
    custom_universe = st.text_input(
        "Custom universe (optional)",
        placeholder="AAPL,MSFT,NVDA,TSLA",
        help="Comma-separated tickers. Leave blank to use default universe.",
    )

universe = None
if custom_universe.strip():
    universe = [t.strip().upper() for t in custom_universe.split(",") if t.strip()]

if not selected_filters:
    st.warning("Select at least one filter to run a scan.")
    st.stop()

# ── Run Scan ────────────────────────────────────────────────
if st.button("🚀 Run Scan", type="primary", use_container_width=False):
    progress_bar = st.progress(0.0, text="Scanning...")
    status = st.empty()

    def _progress(done, total):
        pct = done / total if total > 0 else 0
        progress_bar.progress(pct, text=f"Scanning {done}/{total}...")

    with st.spinner("Running scan..."):
        scan_result = scan_market(
            filters=selected_filters,
            universe=universe,
            max_results=max_results,
            progress_callback=_progress,
        )

    progress_bar.empty()

    if scan_result.get("error"):
        st.error(f"Scan error: {scan_result['error']}")
        st.stop()

    results = scan_result["results"]
    st.success(
        f"✅ Scanned {scan_result['scanned']} stocks in "
        f"{scan_result['scan_time_s']}s — "
        f"{scan_result['passed']} passed filters"
    )

    if not results:
        st.info("No stocks passed the selected filters. Try relaxing criteria.")
        st.stop()

    # ── Results Table ────────────────────────────────────────
    df = pd.DataFrame(results)

    # Rename for display
    df_display = df.rename(columns={
        "symbol": "Symbol",
        "price": "Price",
        "change_20d": "20d Chg%",
        "rsi": "RSI",
        "vs_sma20": "vs SMA20%",
        "pct_from_52w_high": "vs 52w High%",
        "volume_ratio": "Vol Ratio",
        "ai_score": "AI Score",
        "ai_grade": "Grade",
    })

    # Color AI Score column
    def _color_score(val):
        try:
            v = float(val)
            if v >= 8:
                return "background-color: #d4edda"
            if v >= 6.5:
                return "background-color: #cce5ff"
            if v >= 5:
                return "background-color: #fff3cd"
            return "background-color: #f8d7da"
        except Exception:
            return ""

    try:
        styler = df_display.style.map(_color_score, subset=["AI Score"])
    except Exception:
        styler = df_display.style.applymap(_color_score, subset=["AI Score"])
    st.dataframe(styler, use_container_width=True, height=400)

    # ── Chart: AI Score distribution ────────────────────────
    if len(results) >= 3:
        st.markdown("#### AI Score Distribution")
        fig = px.bar(
            df_display.sort_values("AI Score", ascending=False),
            x="Symbol",
            y="AI Score",
            color="AI Score",
            color_continuous_scale="RdYlGn",
            range_color=[1, 10],
            text="Grade",
            height=300,
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Drill-down ───────────────────────────────────────────
    st.markdown("#### Drill Down")
    selected_sym = st.selectbox(
        "Select stock to analyze",
        options=[r["symbol"] for r in results],
        key="scanner_drilldown",
    )
    if selected_sym:
        try:
            import yfinance as yf
            from trading.analysis.ai_score import compute_ai_score
            _hist = yf.Ticker(selected_sym).history(period="6mo")
            _ai = compute_ai_score(selected_sym, _hist)
            if _ai.get("error") is None:
                st.markdown(f"**{selected_sym}** — {_ai['summary']}")
                _sigs = pd.DataFrame(_ai["signals"])
                if not _sigs.empty:
                    st.dataframe(
                        _sigs[["name", "value", "impact", "description"]],
                        use_container_width=True,
                    )
        except Exception as _e:
            st.caption(f"Drill-down unavailable: {_e}")

else:
    st.info("Configure filters above and click **Run Scan** to screen the market.")

    # Show filter descriptions
    st.markdown("#### Available Filters")
    for k, v in get_available_filters().items():
        st.markdown(f"- **{k}**: {v}")
