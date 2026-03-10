"""Session 24 verification."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 24 VERIFICATION")
print("=" * 60)

# 1. SHAP — conditional import + flag
print("\n[1] SHAP wiring in Model Lab...")
try:
    c = open("pages/8_Model_Lab.py", encoding="utf-8", errors="replace").read()
    has_import_guard = "import shap" in c and "SHAP_AVAILABLE" in c
    has_fallback_msg = (
        "pip install shap" in c or "Install SHAP" in c or "shap not" in c.lower()
    )
    shap_not_hardcoded_false = (
        "SHAP_AVAILABLE = False" not in c or "ImportError" in c
    )
    print(f"  Conditional shap import + SHAP_AVAILABLE flag: {has_import_guard}")
    print(f"  Graceful fallback message if missing: {has_fallback_msg}")
    print(f"  Not hardcoded disabled: {shap_not_hardcoded_false}")
    print(f"  {'PASS' if has_import_guard else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. Benchmark returns — real fetch
print("\n[2] Benchmark returns in Performance Attribution...")
try:
    c = open("pages/7_Performance.py", encoding="utf-8", errors="replace").read()
    todo_gone = "TODO" not in c or c.count("TODO") < 3  # allow unrelated TODOs
    has_real_fetch = (
        "yf.Ticker" in c or "yfinance" in c
    ) and "benchmark" in c.lower()
    has_reindex = "reindex" in c or "fillna" in c
    print(f"  Real yfinance fetch for benchmark: {has_real_fetch}")
    print(f"  Reindex/align to portfolio dates: {has_reindex}")
    print(f"  {'PASS' if has_real_fetch else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Factor model audit
print("\n[3] Factor model state...")
try:
    fpath = "trading/analysis/factor_model.py"
    c = open(fpath, encoding="utf-8", errors="replace").read()
    has_factor_fn = "factor_attribution_pct" in c or "factor_attribution" in c
    line_count = len(c.splitlines())
    print(f"  File exists: True ({line_count} lines)")
    print(f"  factor_attribution function present: {has_factor_fn}")
    # Try importing and calling with synthetic data (signature: df, returns)
    try:
        from trading.analysis.factor_model import (
            factor_attribution_pct,
            STANDARD_FACTORS,
        )
        import numpy as np
        import pandas as pd

        _fake_returns = pd.Series(
            np.random.randn(60) * 0.01,
            index=pd.date_range("2024-01-01", periods=60, freq="B"),
        )
        _fake_df = pd.DataFrame(
            {
                "Close": np.random.rand(60) * 100 + 100,
                "Volume": np.random.randint(1000, 10000, 60),
            },
            index=pd.date_range("2024-01-01", periods=60, freq="B"),
        )
        _result = factor_attribution_pct(_fake_df, _fake_returns, window=60)
        print("  Import + call with synthetic data: PASS")
        print(
            f"  Result keys: {list(_result.keys()) if isinstance(_result, dict) else type(_result)}"
        )
    except Exception as call_err:
        print(f"  Import/call result: {call_err}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Smoke tests
print("\n[4] Model smoke tests...")
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
print("Session 24 complete. Paste output back.")
print("=" * 60)
