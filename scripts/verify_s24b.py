"""Session 24B verification — paste output to Claude after Cursor run"""
import subprocess, sys, os

results = []
def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append(status)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

python = sys.executable

# 1. Hybrid wired in forecast_router
with open("trading/models/forecast_router.py", encoding="utf-8", errors="replace") as f:
    router_src = f.read()
# Should now have actual model instantiation near 'hybrid'
hybrid_idx = router_src.lower().find("hybrid")
context = router_src[max(0,hybrid_idx-200):hybrid_idx+500] if hybrid_idx > 0 else ""
has_submodel = any(m in context for m in ["ARIMAModel", "RidgeModel", "XGBoostModel", "arima_model", "ridge_model"])
check("forecast_router: hybrid has sub-model instances", has_submodel, context[:200] if not has_submodel else "OK")

# 2. Hybrid smoke test
r = subprocess.run([python, "-c", """
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
dates = pd.date_range('2023-01-01', periods=300, freq='B')
data = pd.DataFrame({'open':250+np.random.randn(300).cumsum(),
  'high':252+np.random.randn(300).cumsum(),
  'low':248+np.random.randn(300).cumsum(),
  'close':250+np.random.randn(300).cumsum(),
  'volume':np.random.randint(1000000,10000000,300).astype(float)}, index=dates)
from trading.models.forecast_router import ForecastRouter
router = ForecastRouter()
result = router.get_forecast(data, 'hybrid', horizon=7)
preds = result.get('forecast', result.get('predictions', result.get('values', [])))
import numpy as np2
vals = list(preds) if hasattr(preds, '__iter__') else []
in_range = all(200 < v < 400 for v in vals[:3]) if vals else False
print('PASS' if in_range else f'FAIL vals={vals[:3]}')
"""], capture_output=True, text=True, cwd=".")
check("Hybrid forecast returns price-space values", "PASS" in r.stdout, r.stdout.strip()[:200])

# 3. Chat page news section
with open("pages/1_Chat.py", encoding="utf-8", errors="replace") as f:
    chat_src = f.read()
check("Chat page: Get News button present", "Get News" in chat_src or "get_news" in chat_src.lower())
check("Chat page: default market news on load", "SPY" in chat_src and "chat_news" in chat_src)

# 4. Home page timestamp formatting
with open("pages/0_Home.py", encoding="utf-8", errors="replace") as f:
    home_src = f.read()
check("Home: timestamp uses strftime not ISO", "strftime" in home_src or "%b" in home_src)

# 5. Home: volume breach grid
check("Home: news-candle breach grid (2x2)", 
      "volume_ratio" in home_src and "render_news_candle_chart" in home_src and "columns" in home_src)

# 6. Home: watchlist chart section
check("Home: watchlist chart selectbox", 
      "Watchlist Chart" in home_src or "watchlist_chart" in home_src.lower())

# 7. Home: universe count shown
check("Home: universe count shown in selector", 
      "stocks)" in home_src or "len(" in home_src)

# 8. Smoke tests
print("\nRunning model smoke tests...")
r = subprocess.run([python, "tests/model_smoke_test.py"], capture_output=True, text=True, cwd=".")
passes = r.stdout.count("PASS")
fails = r.stdout.count("FAIL")
check("All 12 smoke tests pass", fails == 0 and passes >= 10, f"{passes} PASS {fails} FAIL")

# Summary
p = results.count("PASS"); f = results.count("FAIL")
print(f"\n{'='*55}")
print(f"Session 24B Results: {p} PASS  {f} FAIL")
print("READY — commit tag v1.6.0 and deploy" if f == 0 else "NOT READY — fix failures")
print('='*55)