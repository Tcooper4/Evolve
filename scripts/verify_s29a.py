# verify_s29a.py -- Session 29A verification
# Run: .\evolve_venv\Scripts\python.exe scripts\verify_s29a.py
import sys, os, ast, subprocess, io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY   = sys.executable
PASS = []; FAIL = []

def ok(m):   PASS.append(m); print("  PASS  " + m)
def fail(m): FAIL.append(m); print("  FAIL  " + m)

def parse(rel):
    full = os.path.join(ROOT, rel)
    if not os.path.exists(full):
        fail(rel + " NOT FOUND"); return None
    src = open(full, encoding='utf-8', errors='replace').read()
    try:
        ast.parse(src); return src
    except SyntaxError as e:
        fail(rel + " SyntaxError line " + str(e.lineno)
             + ": " + str(e.msg)); return None

def has(src, needle, label):
    if src and needle in src: ok(label)
    else: fail(label)

def gone(src, needle, label):
    if src and needle not in src: ok(label)
    else: fail(label)

def file_exists(rel, label):
    if os.path.exists(os.path.join(ROOT, rel)): ok(label)
    else: fail(label)

print("\n[1] New infrastructure files")
srcs = {}
for p in ["components/theme.py",
          "trading/data/price_cache.py"]:
    s = parse(p); srcs[p] = s
    if s: ok(p + " parses OK")

s = srcs.get("components/theme.py")
has(s, "inject_theme",        "theme: inject_theme()")
has(s, "market_status_html",  "theme: market_status_html()")
has(s, "render_top_bar",      "theme: render_top_bar()")
has(s, "keyboard_shortcut_js","theme: keyboard_shortcut_js()")
has(s, "ev-accent",           "theme: --ev-accent defined")
has(s, "flashUp",             "theme: flashUp animation")
has(s, "market-open",         "theme: market-open badge class")

s = srcs.get("trading/data/price_cache.py")
has(s, "get_quote",     "cache: get_quote()")
has(s, "get_history",   "cache: get_history()")
has(s, "get_news",      "cache: get_news()")
has(s, "batch_quotes",  "cache: batch_quotes()")
has(s, "cache_data",    "cache: st.cache_data used")
has(s, "ttl=15",        "cache: 15s TTL on quotes")
has(s, "ttl=300",       "cache: 300s TTL on batch")

print("\n[2] app.py updated")
s = parse("app.py")
has(s, "global_search_ticker", "app.py: global ticker search")
has(s, "switch_page",          "app.py: st.switch_page to Analyze")
has(s, "inject_theme",         "app.py: inject_theme() called")

print("\n[3] New page files exist and parse")
pages = {
    "pages/1_Dashboard.py": "Dashboard",
    "pages/2_Analyze.py":   "Analyze",
    "pages/3_Scanner.py":   "Scanner",
    "pages/4_Trade.py":     "Trade",
    "pages/5_Backtest.py":  "Backtest",
    "pages/6_Chat.py":      "Chat",
    "pages/7_Settings.py":  "Settings",
}
psrcs = {}
for path, name in pages.items():
    s = parse(path); psrcs[path] = s
    if s: ok(name + " parses OK")

print("\n[4] Dashboard preserves Home features")
s = psrcs.get("pages/1_Dashboard.py")
has(s, "render_top_bar",      "Dashboard: top bar")
has(s, "keyboard_shortcut",   "Dashboard: keyboard shortcut")
has(s, "news_candle_chart",   "Dashboard: news candle chart")
has(s, "watchlist",           "Dashboard: watchlist widget")
has(s, "price_cache",         "Dashboard: uses price_cache")
has(s, "morning_briefing",    "Dashboard: morning briefing")
gone(s, "st_autorefresh",     "Dashboard: no st_autorefresh")
has(s, "fragment",            "Dashboard: uses st.fragment")

print("\n[5] Analyze merges Forecasting + Market Analysis")
s = psrcs.get("pages/2_Analyze.py")
has(s, "analyze_ticker",      "Analyze: ticker session_state key")
has(s, "analyze_trader_mode", "Analyze: trader mode toggle")
has(s, "analyze_period",      "Analyze: period switcher")
has(s, "add_hline",           "Analyze: price hline on chart")
has(s, "Partial consensus",   "Analyze: graceful consensus")
has(s, "compute_ai_score",    "Analyze: AI Score")
has(s, "forecast_router",     "Analyze: forecast router")
has(s, "earnings_reaction",   "Analyze: earnings tracker")
has(s, "multi_timeframe",     "Analyze: multi-timeframe chart")
has(s, "news_candle_chart",   "Analyze: news candle chart")
has(s, "What drives this",    "Analyze: AI Score tooltips")
has(s, "keyboard_shortcut",   "Analyze: keyboard shortcut")
has(s, "HOT",                 "Analyze: news sentiment HOT")

print("\n[6] Scanner enhanced")
s = psrcs.get("pages/3_Scanner.py")
has(s, "render_top_bar",  "Scanner: top bar")
has(s, "batch_quotes",    "Scanner: batch_quotes cache")
has(s, "fragment",        "Scanner: st.fragment")
has(s, "HOT",             "Scanner: news score HOT")
has(s, "scanner_filter",  "Scanner: filter radio key")
has(s, "universes",       "Scanner: universe JSON loader")

print("\n[7] Trade merges 3 pages")
s = psrcs.get("pages/4_Trade.py")
has(s, "render_top_bar",    "Trade: top bar")
has(s, "st.tabs",           "Trade: tab structure")
has(s, "execution_engine",  "Trade: execution engine")
has(s, "slippage",          "Trade: slippage estimate")
has(s, "st.metric",         "Trade: summary metrics")
has(s, "attribution",       "Trade: performance attribution")

print("\n[8] Backtest merges 2 pages")
s = psrcs.get("pages/5_Backtest.py")
has(s, "render_top_bar",   "Backtest: top bar")
has(s, "backtester",       "Backtest: backtester")
has(s, "walk_forward",     "Backtest: walk-forward")
has(s, "st.tabs",          "Backtest: tab structure")

print("\n[9] Chat merges 2 pages")
s = psrcs.get("pages/6_Chat.py")
has(s, "render_top_bar",        "Chat: top bar")
has(s, "call_active_llm",       "Chat: LLM routing")
has(s, "EnhancedPromptRouter",  "Chat: prompt router")
has(s, "arXiv",                 "Chat: arXiv research")

print("\n[10] Settings merges 3 pages")
s = psrcs.get("pages/7_Settings.py")
has(s, "render_top_bar",      "Settings: top bar")
has(s, "NotificationSystem",  "Settings: alerts")
has(s, "psutil",              "Settings: system health")
has(s, "user_store",          "Settings: user store")

print("\n[11] Old pages still intact (not deleted)")
for old in ["pages/0_Home.py", "pages/2_Forecasting.py",
            "pages/13_Scanner.py"]:
    s = parse(old)
    if s: ok(old + " still exists")

print("\n[12] Smoke tests")
r = subprocess.run([PY, "tests/model_smoke_test.py"],
    capture_output=True, text=True, cwd=ROOT)
if r.returncode == 0:
    ok("All 12 models passed smoke tests")
else:
    for l in (r.stdout + r.stderr).splitlines()[-15:]:
        print("       " + l)
    fail("Smoke tests failed")

print("\n" + "="*52)
print("  PASSED: " + str(len(PASS))
      + "   FAILED: " + str(len(FAIL)))
if FAIL:
    print("  Failed:")
    for f in FAIL: print("    - " + f)
else:
    print("  29A complete. Confirm app runs, then do 29B.")
print("="*52)