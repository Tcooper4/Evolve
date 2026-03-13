"""
Market Scanner page — screen stocks by technical and AI-driven filters.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
st.title("🔍 Market Scanner")
st.caption("Screen stocks by technical conditions and AI Score ranking")

try:
    from trading.analysis.market_scanner import scan_market, get_available_filters, DEFAULT_UNIVERSE
    scanner_available = True
except Exception as e:
    st.error(f"Scanner unavailable: {e}")
    scanner_available = False

RUSSELL_1000_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD",
    "PG","JNJ","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","PEP",
    "TMO","ADBE","ACN","LIN","MCD","CSCO","ABT","GE","TXN","DHR",
    "PM","CAT","ISRG","INTU","AMGN","VZ","NOW","MS","GS","RTX",
]

if not scanner_available:
    st.stop()


@st.cache_data(ttl=86400)
def _load_scanner_universe(universe_label: str) -> list[str]:
    """Load ticker universe for the Market Scanner."""
    universe_label = (universe_label or "").strip()
    try:
        import pandas as pd  # type: ignore

        if universe_label.startswith("S&P 100"):
            # Same S&P 100 universe used on the Home page
            return [
                "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT",
                "AMZN", "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY",
                "BRK.B", "C", "CAT", "CL", "CMCSA", "COF", "COP", "COST", "CRM",
                "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "FDX",
                "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON",
                "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY",
                "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM",
                "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA",
                "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM", "RTX",
                "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA",
                "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC",
                "WMT", "XOM",
            ]

        sp500: list[str] = []
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            if tables:
                table = tables[0]
                sp500 = (
                    table[
                        [c for c in table.columns if str(c).lower() in ("symbol", "ticker")][0]
                    ]
                    .astype(str)
                    .str.replace(".", "-", regex=False)
                    .str.upper()
                    .tolist()
                )
        except Exception:
            sp500 = []

        # Nasdaq 100 universe
        nasdaq100 = [
            "ATVI", "ADBE", "AMD", "ALGN", "AMZN", "ANSS", "AAPL", "AMAT",
            "ASML", "AZN", "TEAM", "ADSK", "BIIB", "BKR", "BKNG", "AVGO",
            "CDNS", "CDW", "CHTR", "CTAS", "CSCO", "CMCSA", "CPRT", "CSX",
            "CTSH", "DDOG", "DXCM", "DOCU", "DLTR", "EA", "EXC", "FAST",
            "FISV", "FTNT", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN",
            "INTC", "INTU", "ISRG", "JD", "KDP", "KLAC", "KHC", "LRCX",
            "LULU", "MAR", "MELI", "META", "MCHP", "MU", "MSFT", "MRNA",
            "MDLZ", "MNST", "NTES", "NFLX", "NVDA", "NXPI", "ORLY", "PCAR",
            "PANW", "PAYX", "PYPL", "PEP", "PDD", "QCOM", "REGN", "ROST",
            "SIRI", "SBUX", "SNPS", "SPLK", "SWKS", "TMUS", "TSLA", "TXN",
            "TTWO", "VRSK", "VRSN", "VRSK", "VRTX", "WBA", "WDAY", "XEL",
            "ZM",
        ]

        if universe_label.startswith("S&P 500 (~500"):
            return sp500

        if universe_label.startswith("S&P 500 + Nasdaq 100"):
            return sorted(set(sp500).union(nasdaq100))

        if "Russell 1000" in universe_label:
            # Approximate Russell 1000 from Russell 3000 components
            try:
                url = "https://en.wikipedia.org/wiki/Russell_3000_Index"
                tables = pd.read_html(url)
                tickers: list[str] = []
                for t in tables:
                    cols = [c.lower() for c in t.columns.astype(str)]
                    if any("ticker" in c or "symbol" in c for c in cols):
                        for col in t.columns:
                            col_lower = str(col).lower()
                            if "ticker" in col_lower or "symbol" in col_lower:
                                series = (
                                    t[col]
                                    .astype(str)
                                    .str.replace(".", "-", regex=False)
                                    .str.upper()
                                )
                                tickers.extend(series.tolist())
                        break
                tickers = [t for t in tickers if t and t != "nan"]
                return tickers[:1000] if tickers else RUSSELL_1000_FALLBACK
            except Exception:
                return sp500

        if "Russell 3000" in universe_label:
            try:
                url = "https://en.wikipedia.org/wiki/Russell_3000_Index"
                tables = pd.read_html(url)
                tickers: list[str] = []
                for t in tables:
                    cols = [c.lower() for c in t.columns.astype(str)]
                    if any("ticker" in c or "symbol" in c for c in cols):
                        for col in t.columns:
                            col_lower = str(col).lower()
                            if "ticker" in col_lower or "symbol" in col_lower:
                                series = (
                                    t[col]
                                    .astype(str)
                                    .str.replace(".", "-", regex=False)
                                    .str.upper()
                                )
                                tickers.extend(series.tolist())
                        break
                tickers = [t for t in tickers if t and t != "nan"]
                return tickers if tickers else RUSSELL_1000_FALLBACK
            except Exception:
                # Fall back to Russell 1000 approximation if 3000 unavailable
                return _load_scanner_universe("Russell 1000 (~1000, slow)")

        # Default: use whatever DEFAULT_UNIVERSE the scanner provides
        return DEFAULT_UNIVERSE or []
    except Exception:
        return DEFAULT_UNIVERSE or []

# ── Controls ────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    available_filters = get_available_filters()

    universe_choice = st.selectbox(
        "Stock Universe",
        [
            "S&P 100 (~100, fastest)",
            "S&P 500 (~500, fast)",
            "S&P 500 + Nasdaq 100 (~600, moderate)",
            "Russell 1000 (~1000, slow)",
            "Russell 3000 (~3000, very slow)",
        ],
    )

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
        help="Comma-separated tickers. Leave blank to use selected stock universe.",
    )

universe = None
if custom_universe.strip():
    universe = [t.strip().upper() for t in custom_universe.split(",") if t.strip()]
else:
    # Map selectbox choice to loader label
    label_map = {
        "S&P 100 (~100, fastest)": "S&P 100",
        "S&P 500 (~500, fast)": "S&P 500 (~500, fast)",
        "S&P 500 + Nasdaq 100 (~600, moderate)": "S&P 500 + Nasdaq 100 (~600, moderate)",
        "Russell 1000 (~1000, slow)": "Russell 1000 (~1000, slow)",
        "Russell 3000 (~3000, very slow)": "Russell 3000 (~3000, very slow)",
    }
    loader_label = label_map.get(universe_choice, "S&P 100")
    universe = _load_scanner_universe(loader_label)

    if "Russell 1000" in universe_choice or "Russell 3000" in universe_choice:
        st.warning("⚠️ Scanning 1000+ stocks may take 2-3 minutes.")

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
