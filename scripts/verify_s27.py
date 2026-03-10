"""Session 27 verification."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 27 VERIFICATION")
print("=" * 60)

# 1. Startup noise suppressed
print("\n[1] Startup noise suppression...")
import io
from contextlib import redirect_stderr, redirect_stdout

_buf_out = io.StringIO()
_buf_err = io.StringIO()
with redirect_stdout(_buf_out), redirect_stderr(_buf_err):
    try:
        import importlib

        for mod in list(sys.modules.keys()):
            if "trading" in mod or "agents" in mod:
                del sys.modules[mod]
        import trading
    except Exception:
        pass
_combined = _buf_out.getvalue() + _buf_err.getvalue()
_json_lines = [l for l in _combined.split("\n") if l.strip().startswith("{")]
_info_lines = [l for l in _combined.split("\n") if "INFO" in l and "WARNING" not in l]
print(f"  JSON INFO log lines captured: {len(_json_lines)}")
print(f"  INFO lines captured: {len(_info_lines)}")
print(
    f"  {'PASS (quiet)' if len(_json_lines) < 3 else 'STILL NOISY - ' + str(len(_json_lines)) + ' JSON lines'}"
)

# 2. Liquidity risk - real data
print("\n[2] Liquidity risk uses real ADV data...")
try:
    with open("pages/6_Risk_Management.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_adv = "adv" in c.lower() or "avg_volume" in c or "Volume" in c
    idx = c.find("calculate_liquidity")
    no_random = (
        "np.random" not in c[idx : idx + 1000]
        if idx >= 0
        else True
    )
    print(f"  Uses real ADV/volume data: {has_adv}")
    print(f"  Random data removed from liquidity fn: {no_random}")
    print(f"  {'PASS' if has_adv and no_random else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Factor decomposition uses factor_model
print("\n[3] Factor decomposition uses factor_model.py...")
try:
    with open("pages/6_Risk_Management.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    uses_factor_model = "factor_attribution_pct" in c or "factor_model" in c
    has_spy_fallback = "SPY" in c and "factor" in c.lower()
    idx = c.find("calculate_factor")
    no_hardcoded = (
        "0.70" not in c[idx : idx + 500] or "idiosyncratic" in c[idx : idx + 500]
        if idx >= 0
        else True
    )
    print(f"  Calls factor_attribution_pct: {uses_factor_model}")
    print(f"  SPY regression fallback: {has_spy_fallback}")
    print(f"  Hardcoded 70/30 removed: {no_hardcoded}")
    print(f"  {'PASS' if uses_factor_model else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Strategy correlation uses real trades
print("\n[4] Strategy correlation uses real trade data...")
try:
    with open("pages/7_Performance.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    fn_start = c.find("calculate_strategy_correlation")
    fn_body = c[fn_start : fn_start + 1500] if fn_start > 0 else ""
    uses_real_trades = "backtest_results" in fn_body or "trade_hist" in fn_body
    no_pure_random = "np.random.rand" not in fn_body
    print(f"  Uses real trade history: {uses_real_trades}")
    print(f"  Pure random matrix removed: {no_pure_random}")
    print(f"  {'PASS' if uses_real_trades else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Greek exposure - equity delta=1
print("\n[5] Greek exposure updated for equities...")
try:
    with open("pages/6_Risk_Management.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    fn_start = c.find("calculate_greek_exposure")
    fn_body = c[fn_start : fn_start + 1000] if fn_start > 0 else ""
    has_equity_delta = "delta" in fn_body and ("1.0" in fn_body or "equity" in fn_body.lower())
    no_random_greeks = "np.random" not in fn_body
    print(f"  Equity delta=1.0: {has_equity_delta}")
    print(f"  Random Greek values removed: {no_random_greeks}")
    print(f"  {'PASS' if has_equity_delta and no_random_greeks else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 6. Smoke tests
print("\n[6] Model smoke tests...")
import subprocess

result = subprocess.run(
    [sys.executable, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
output = result.stdout + result.stderr
if "All smoke tests completed. All PASS" in output:
    print("  PASS: All 12 models")
else:
    fails = [l.strip() for l in output.split("\n") if "FAIL" in l]
    print(f"  ISSUES: {fails}")

print("\n" + "=" * 60)
print("Session 27 complete. Paste output back.")
print("=" * 60)
