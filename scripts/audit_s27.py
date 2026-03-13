"""
audit_s27.py — Check S27 fix status
Run: .\evolve_venv\Scripts\python.exe scripts\audit_s27.py > scripts\audit_s27.txt 2>&1
"""
import py_compile

# ── 1. Show Forecasting.py syntax error context ──────────────
print("=" * 60)
print("Forecasting.py — lines around error (line 539)")
print("=" * 60)
lines = open('pages/2_Forecasting.py', encoding='utf-8', errors='replace').read().splitlines()
for i, line in enumerate(lines[525:555], start=526):
    marker = " >>>" if i == 539 else "    "
    print(f"{marker} {i:5}: {line}")

# ── 2. Check GNN denormalization fix ─────────────────────────
print()
print("=" * 60)
print("GNN model — price-space conversion check")
print("=" * 60)
src = open('trading/models/advanced/gnn/gnn_model.py', encoding='utf-8', errors='replace').read()
for keyword in ['last_price', 'predicted_return', 'predicted_price', '1 + ', 'denorm', 'price_space', 'Close']:
    if keyword in src:
        lines2 = src.splitlines()
        for i, l in enumerate(lines2, 1):
            if keyword in l:
                print(f"  {i:5}: {l.rstrip()}")
        break
else:
    print("  [WARNING] No price-space conversion found in GNN model")
    # Show forecast() method
    in_forecast = False
    for i, l in enumerate(src.splitlines(), 1):
        if 'def forecast' in l:
            in_forecast = True
        if in_forecast:
            print(f"  {i:5}: {l.rstrip()}")
        if in_forecast and i > 0 and l.strip().startswith('def ') and 'forecast' not in l:
            break

# ── 3. Check SHAP explainer routing ──────────────────────────
print()
print("=" * 60)
print("ForecastExplainability — explainer routing check")
print("=" * 60)
src = open('trading/analytics/forecast_explainability.py', encoding='utf-8', errors='replace').read()
for keyword in ['TreeExplainer', 'KernelExplainer', 'DeepExplainer', 'model_type', 'xgboost', 'neural']:
    hits = [(i+1, l.rstrip()) for i, l in enumerate(src.splitlines()) if keyword in l]
    if hits:
        print(f"\n  [{keyword}]")
        for lineno, l in hits:
            print(f"    {lineno:5}: {l}")

# ── 4. Check caching in Forecasting.py ───────────────────────
print()
print("=" * 60)
print("Forecasting.py — caching check")
print("=" * 60)
src = open('pages/2_Forecasting.py', encoding='utf-8', errors='replace').read()
for keyword in ['cache_resource', 'cache_data', 'lru_cache', 'get_cached_model']:
    hits = [(i+1, l.rstrip()) for i, l in enumerate(src.splitlines()) if keyword in l]
    if hits:
        print(f"\n  [{keyword}]")
        for lineno, l in hits[:5]:
            print(f"    {lineno:5}: {l}")

# ── 5. Check trade_models.py to_dict ─────────────────────────
print()
print("=" * 60)
print("trade_models.py — to_dict() check")
print("=" * 60)
src = open('trading/backtesting/trade_models.py', encoding='utf-8', errors='replace').read()
in_to_dict = False
for i, l in enumerate(src.splitlines(), 1):
    if 'def to_dict' in l:
        in_to_dict = True
    if in_to_dict:
        print(f"  {i:5}: {l.rstrip()}")
    if in_to_dict and i > 0 and l.strip().startswith('def ') and 'to_dict' not in l:
        break
    if in_to_dict and i > 50:
        break