"""Session 16 verification."""
import sys

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 16 VERIFICATION")
print("=" * 60)

# 1. Market scanner module
print("\n[1] Market scanner module...")
try:
    from trading.analysis.market_scanner import (
        scan_market,
        get_available_filters,
        DEFAULT_UNIVERSE,
    )
    filters = get_available_filters()
    print(f"  Filters available: {list(filters.keys())}")
    print(f"  Default universe size: {len(DEFAULT_UNIVERSE)}")
    print("  PASS: market_scanner.py imports correctly")
except Exception as e:
    print(f"  FAIL: {e}")

# 2. Quick scan functional test (small universe)
print("\n[2] Quick scan functional test (5 stocks)...")
try:
    from trading.analysis.market_scanner import scan_market
    result = scan_market(
        filters=["momentum"],
        universe=["AAPL", "MSFT", "NVDA", "JPM", "SPY"],
        max_results=10,
    )
    assert result.get("error") is None, f"Scan error: {result.get('error')}"
    assert result["scanned"] == 5
    print(
        f"  Scanned: {result['scanned']}, Passed: {result['passed']}, "
        f"Time: {result['scan_time_s']}s"
    )
    print(f"  Results: {[r['symbol'] for r in result['results']]}")
    print("  PASS: scan_market() runs without error")
except Exception as e:
    print(f"  FAIL: {e}")

# 3. Scanner page exists
print("\n[3] Scanner page...")
import os

if os.path.exists("pages/13_Scanner.py"):
    c = open("pages/13_Scanner.py", encoding="utf-8", errors="replace").read()
    has_scan = "scan_market" in c
    has_df = "dataframe" in c
    print("  File exists: True")
    print(f"  scan_market wired: {has_scan}")
    print(f"  Table display: {has_df}")
    print("  PASS" if has_scan and has_df else "  FAIL")
else:
    print("  FAIL: pages/13_Scanner.py not found")

# 4. Outstanding fixes audit
print("\n[4] Outstanding fixes audit...")

# Reports
try:
    c = open("pages/9_Reports.py", encoding="utf-8", errors="replace").read()
    fake = "12.5" in c and "1.85" in c
    gate = "backtest_results" in c and "st.stop" in c
    print(f"  Reports — fake data: {fake}, session gate: {gate}")
except Exception as e:
    print(f"  Reports check error: {e}")

# Walk-forward
try:
    c = open("pages/3_Strategy_Testing.py", encoding="utf-8", errors="replace").read()
    print(f"  WFV import: {'walk_forward_validator' in c}, call: {'walk_forward_test' in c}")
except Exception as e:
    print(f"  Strategy Testing check error: {e}")

# AI Score in Forecasting
try:
    c = open("pages/2_Forecasting.py", encoding="utf-8", errors="replace").read()
    print(f"  AI Score in Forecasting: {'compute_ai_score' in c or 'ai_score' in c}")
except Exception as e:
    print(f"  Forecasting check error: {e}")

# 5. Smoke test
print("\n[5] Model smoke test...")
import subprocess

r = subprocess.run(
    [sys.executable, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
    timeout=180,
    cwd=".",
)
out = (r.stdout or "") + (r.stderr or "")
last_lines = out.strip().split("\n")[-5:]
for line in last_lines:
    if "PASS" in line or "FAIL" in line or "All" in line:
        print(f"  {line.strip()}")

print("\n" + "=" * 60)
print("Done.")
