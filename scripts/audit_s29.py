# audit_s29.py -- Pre-session 29 architecture audit
# Run: .\evolve_venv\Scripts\python.exe scripts\audit_s29.py
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def section(title):
    print("")
    print("=" * 60)
    print("  " + title)
    print("=" * 60)

def show_grep(filepath, patterns, context=2):
    """Show lines matching any pattern with line numbers."""
    full = os.path.join(ROOT, filepath)
    if not os.path.exists(full):
        print("  FILE NOT FOUND: " + filepath)
        return
    with open(full, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for pat in patterns:
            if pat.lower() in line.lower():
                start = max(0, i - context)
                end = min(len(lines), i + context + 1)
                print("  -- match '" + pat + "' at line " + str(i+1) + " --")
                for j in range(start, end):
                    marker = ">>" if j == i else "  "
                    print("  " + marker + " " + str(j+1).rjust(4) + "  " + lines[j].rstrip())
                break

# ── 1. AI Score composition ──────────────────────────────────────
section("1. AI Score -- weights and signal composition")
show_grep("trading/analysis/ai_score.py",
    ["weight", "sentiment", "news", "technical", "momentum", "fundamental", "score"],
    context=1)

# ── 2. News sentiment pipeline ───────────────────────────────────
section("2. News sentiment -- what exists")
for f in ["trading/data/news_aggregator.py",
          "trading/analysis/market_scanner.py",
          "pages/0_Home.py"]:
    full = os.path.join(ROOT, f)
    if os.path.exists(full):
        with open(full, encoding='utf-8', errors='replace') as fh:
            src = fh.read()
        has_sentiment = "sentiment" in src.lower()
        has_news = "news" in src.lower()
        size = len(src.splitlines())
        print("  " + f + " (" + str(size) + " lines) -- sentiment=" + str(has_sentiment) + " news=" + str(has_news))
    else:
        print("  MISSING: " + f)

# ── 3. Forecasting page -- tabs and news presence ────────────────
section("3. Forecasting page -- tabs, news, sentiment")
show_grep("pages/2_Forecasting.py",
    ["tab", "news", "sentiment", "live", "refresh", "consensus"],
    context=1)

# ── 4. Chart setup -- live updates, refresh ──────────────────────
section("4. Charts -- live update / refresh logic")
for f in ["components/multi_timeframe_chart.py",
          "pages/2_Forecasting.py",
          "pages/0_Home.py"]:
    show_grep(f, ["refresh", "live", "autorefresh", "auto_refresh",
                  "rerun", "interval", "sleep", "period", "1d", "1mo", "1y"],
              context=1)

# ── 5. yfinance call patterns -- caching ─────────────────────────
section("5. yfinance -- caching and call patterns")
import subprocess
result = subprocess.run(
    ["grep", "-rn", "yf.download\|yf.Ticker\|st.cache\|cache_data\|cache_resource",
     "trading/", "pages/", "components/"],
    capture_output=True, text=True, cwd=ROOT
)
lines = result.stdout.strip().splitlines()
print("  Total yfinance/cache hits: " + str(len(lines)))
cache_lines = [l for l in lines if "cache" in l.lower()]
yf_lines = [l for l in lines if "yf." in l.lower()]
print("  Cache decorators: " + str(len(cache_lines)))
print("  yf calls: " + str(len(yf_lines)))
for l in cache_lines[:15]:
    print("    " + l)
print("  ...")
for l in yf_lines[:15]:
    print("    " + l)

# ── 6. Slippage -- exists anywhere? ──────────────────────────────
section("6. Slippage -- current state")
result2 = subprocess.run(
    ["grep", "-rn", "slippage", "trading/"],
    capture_output=True, text=True, cwd=ROOT
)
if result2.stdout.strip():
    for l in result2.stdout.strip().splitlines():
        print("  " + l)
else:
    print("  No slippage references found anywhere in trading/")

# ── 7. Page reset on click -- st.session_state usage ─────────────
section("7. Session state -- widget stability")
result3 = subprocess.run(
    ["grep", "-rn", "session_state", "pages/"],
    capture_output=True, text=True, cwd=ROOT
)
lines3 = result3.stdout.strip().splitlines()
print("  Total session_state references in pages/: " + str(len(lines3)))
for l in lines3[:20]:
    print("  " + l)

# ── 8. Quick Forecast -- consensus failure path ───────────────────
section("8. Quick Forecast -- consensus failure path")
show_grep("pages/2_Forecasting.py",
    ["consensus forecast failed", "load data", "consensus"],
    context=3)

# ── 9. Auto-refresh grey-out -- st.rerun usage ───────────────────
section("9. Auto-refresh -- st.rerun / experimental_rerun")
result4 = subprocess.run(
    ["grep", "-rn", "rerun\|time.sleep\|autorefresh",
     "pages/"],
    capture_output=True, text=True, cwd=ROOT
)
for l in result4.stdout.strip().splitlines()[:20]:
    print("  " + l)

# ── 10. File sizes -- biggest pages ──────────────────────────────
section("10. Page file sizes")
pages_dir = os.path.join(ROOT, "pages")
sizes = []
for fn in os.listdir(pages_dir):
    if fn.endswith(".py"):
        fp = os.path.join(pages_dir, fn)
        with open(fp, encoding='utf-8', errors='replace') as f:
            n = len(f.readlines())
        sizes.append((n, fn))
sizes.sort(reverse=True)
for n, fn in sizes:
    print("  " + str(n).rjust(5) + " lines  " + fn)

print("")
print("=" * 60)
print("  Audit complete.")
print("=" * 60)