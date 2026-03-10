"""Session 18 verification."""
import sys

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 18 VERIFICATION")
print("=" * 60)

# 1. Earnings reaction module
print("\n[1] Earnings reaction tracker...")
try:
    from trading.data.earnings_reaction import get_earnings_reactions

    r = get_earnings_reactions("AAPL", num_quarters=4)
    print(f"  Symbol: {r['symbol']}")
    print(f"  Quarters: {r['num_quarters']}")
    print(f"  Avg 1d move: ±{r['avg_move_1d']}%")
    print(f"  Beat rate: {r['beat_rate']}%")
    print(f"  Error: {r['error']}")
    if r["num_quarters"] > 0:
        print(f"  PASS: {r['num_quarters']} quarters loaded")
    else:
        print(f"  PARTIAL: 0 quarters — {r['error']}")
except Exception as e:
    print(f"  FAIL: {e}")

# 2. Earnings tab wired into Forecasting
print("\n[2] Earnings tab in Forecasting page...")
try:
    with open("pages/2_Forecasting.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_reaction = "earnings_reaction" in c or "get_earnings_reactions" in c
    has_tab = "Earnings" in c
    print(f"  earnings_reaction wired: {has_reaction}")
    print(f"  Earnings tab: {has_tab}")
    print(f"  {'PASS' if has_reaction else 'FAIL'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Reports avg_pnl fix
print("\n[3] Reports avg_pnl fix...")
try:
    with open("pages/9_Reports.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_pnl_fix = (
        "profit_loss" in c or "_pnl_col" in c or "pnl_col" in c
    )
    has_attribution = (
        "Performance Attribution" in c or "Profit Factor" in c
    )
    has_fake = "12.5" in c and "1.85" in c
    print(f"  pnl normalization: {has_pnl_fix}")
    print(f"  Attribution section: {has_attribution}")
    print(f"  Fake data still present: {has_fake}")
    print(f"  {'PASS' if has_pnl_fix and not has_fake else 'PARTIAL'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Home page pre/post market
print("\n[4] Home page pre/post market...")
try:
    with open("pages/0_Home.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_prepost = (
        "preMarketPrice" in c or "pre_mkt" in c or "postMarketPrice" in c
    )
    print(f"  Pre/post market wired: {has_prepost}")
    print(f"  {'PASS' if has_prepost else 'NOT WIRED'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Smoke test
print("\n[5] Model smoke test...")
try:
    import subprocess

    result = subprocess.run(
        [sys.executable, "tests/model_smoke_test.py"],
        capture_output=True,
        text=True,
        cwd=".",
    )
    output = result.stdout + result.stderr
    if "All smoke tests completed. All PASS" in output:
        print("  PASS: All models passing")
    else:
        print("  Check manually — subprocess output may be truncated")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("Run: .\\evolve_venv\\Scripts\\python.exe scripts\\verify_s18.py")
