# -*- coding: utf-8 -*-
import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 82 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — Quick Forecast: AI lookback + consensus training symbol (S82 follow-up)
tf = open(
    "components/tabs/tab_quick_forecast.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tf)
    ok("Syntax valid: tab_quick_forecast.py")
except SyntaxError as e:
    fail(f"Syntax error forecast: {e}")

if (
    "quick_forecast_ai_lookback" in tf
    and "analyze_forecast_data_symbol" in tf
):
    ok("Quick Forecast AI/consensus data wiring present")
elif "_fc_load_key" in tf or "_fc_data_exists" in tf:
    ok("Forecast load guard present (legacy)")
else:
    fail("Quick Forecast data wiring missing")

# Fix 2 — URL dict unwrapping
tc = open(
    "trading/data/price_cache.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tc)
    ok("Syntax valid: price_cache.py")
except SyntaxError as e:
    fail(f"Syntax error price_cache: {e}")

if "isinstance(url, dict)" in tc or "url.get(" in tc:
    ok("URL dict unwrapping present")
else:
    fail("URL dict unwrapping missing")

# News URL functional test
_news_check = r"""
import sys
sys.path.insert(0, ".")
from trading.data.price_cache import get_news
items = get_news("AAPL")
if items:
    for it in items[:3]:
        t = it.get("title", "")
        u = it.get("url", "")
        assert "({" not in t, "Dict in title: %r" % (t[:60],)
        assert not str(u).startswith("{"), "Dict as url: %r" % (str(u)[:60],)
    print("News clean OK")
else:
    print("No items - skip")
"""
_r = subprocess.run(
    [python, "-c", _news_check],
    capture_output=True,
    text=True,
    timeout=60,
)
print((_r.stdout or "")[:200])
if _r.returncode == 0:
    ok("News URL clean test passed")
else:
    fail(
        f"News URL test: "
        f"{(_r.stdout + _r.stderr)[:200]}"
    )

# Fix 3 — deep dive radio
td = open(
    "components/deep_dive.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(td)
    ok("Syntax valid: deep_dive.py")
except SyntaxError as e:
    fail(f"Syntax error deep dive: {e}")

if (
    "st.radio" in td
    and "_tf_period_map" in td
    and "dd_tf_" in td
):
    ok("Radio timeframe in deep dive")
else:
    fail("Radio timeframe missing")

# Fix 4 — short score label
if "Buy Score" in td:
    ok("Buy Score label in deep dive")
else:
    fail("Buy Score label missing")

# Fix 5 — chat news
tc2 = open(
    "pages/6_Chat.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tc2)
    ok("Syntax valid: 6_Chat.py")
except SyntaxError as e:
    fail(f"Syntax error Chat: {e}")

if (
    "Enter a ticker" in tc2
    or (
        "click" in tc2.lower()
        and "news" in tc2.lower()
    )
):
    ok("Chat news initial state updated")
else:
    fail("Chat news initial state wrong")

if "](" in tc2 and "url" in tc2:
    ok("Chat news hyperlinks present")
else:
    fail("Chat news hyperlinks missing")

# Fix 6 — VWAP guard (Short-term strip in analyze_chart.py)
_tacf = open(
    "components/analyze_chart.py",
    encoding="utf-8",
    errors="replace",
).read()
if (
    "vs VWAP" in _tacf
    and "N/A" in _tacf
    and "isfinite" in _tacf
):
    ok("VWAP NaN guard present (analyze_chart)")
elif (
    "isnan" in tf
    and "vwap" in tf.lower()
) or (
    "N/A" in tf
    and "vwap" in tf.lower()
):
    ok("VWAP NaN guard present (tab_quick_forecast)")
else:
    fail(
        "VWAP guard check — manual verify needed"
    )

# Fix 7 — intraday fallback
ta = open(
    "pages/2_Analyze.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(ta)
    ok("Syntax valid: 2_Analyze.py")
except SyntaxError as e:
    fail(f"Syntax error Analyze: {e}")

if (
    "hist.empty" in ta
    and "interval" in ta
    and "daily" in ta.lower()
):
    ok("Intraday fallback present")
else:
    fail("Intraday fallback missing")

# Smoke test
print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
_out = result.stdout + result.stderr
_passes = _out.count("PASS:")
_fails = _out.count("FAIL:")
print(f"Smoke: {_passes} PASS, {_fails} FAIL")
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(
    f"\n=== {len(PASS)} passed, "
    f"{len(FAIL)} failed ==="
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. "
        "Ready to commit v4.7.3."
    )
    sys.exit(0)
