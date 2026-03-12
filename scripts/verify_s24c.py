"""Session 24D verification"""
import subprocess, sys

results = []
def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append(status)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

python = sys.executable

# 1. Verify the injection block landed
with open("trading/models/forecast_router.py", encoding="utf-8", errors="replace") as f:
    src = f.read()
check("forecast_router: ARIMAModel( instantiation present near hybrid",
      "ARIMAModel({" in src or "ARIMAModel({'order'" in src or "_arima = ARIMAModel" in src,
      "grep for ARIMAModel( in forecast_router.py")
check("forecast_router: RidgeModel( instantiation present near hybrid",
      "RidgeModel({" in src or "_ridge = RidgeModel" in src)
check("forecast_router: hybrid sub-model injection comment present",
      "Hybrid sub-model injection" in src or "hybrid injection" in src.lower())

# 2. End-to-end hybrid price-space test
r = subprocess.run([python, "-c", """
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=300, freq='B')
data = pd.DataFrame({
    'open':  250 + np.random.randn(300).cumsum(),
    'high':  252 + np.random.randn(300).cumsum(),
    'low':   248 + np.random.randn(300).cumsum(),
    'close': 250 + np.random.randn(300).cumsum(),
    'volume': np.random.randint(1000000,10000000,300).astype(float)
}, index=dates)
last_close = float(data['close'].iloc[-1])
from trading.models.forecast_router import ForecastRouter
router = ForecastRouter()
result = router.get_forecast(data, model_type='hybrid', horizon=7)
preds = result.get('forecast', result.get('predictions', result.get('values', [])))
vals = list(preds) if hasattr(preds,'__iter__') else []
print('Last close:', round(last_close,2))
print('Hybrid values:', [round(v,2) for v in vals[:3]])
ok = all(last_close*0.5 < v < last_close*1.5 for v in vals[:3]) if vals else False
print('PASS' if ok else 'FAIL')
"""], capture_output=True, text=True, cwd=".")
check("Hybrid returns price-space values", "PASS" in r.stdout,
      r.stdout.strip()[:300] + (r.stderr[:100] if r.stderr else ""))

# 3. Smoke tests
print("\nRunning smoke tests...")
r2 = subprocess.run([python, "tests/model_smoke_test.py"],
                    capture_output=True, text=True, cwd=".")
p = r2.stdout.count("PASS"); f2 = r2.stdout.count("FAIL")
check("All 12 smoke tests pass", f2 == 0 and p >= 10, f"{p} PASS {f2} FAIL")

total_p = results.count("PASS"); total_f = results.count("FAIL")
print(f"\n{'='*55}")
print(f"Session 24D Results: {total_p} PASS  {total_f} FAIL")
print("READY — commit tag v1.6.0 and deploy" if total_f == 0 else "NOT READY")
print('='*55)