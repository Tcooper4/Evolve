"""
audit_s27b.py — Deeper audit of S27 remaining issues
Run: .\evolve_venv\Scripts\python.exe scripts\audit_s27b.py > scripts\audit_s27b.txt 2>&1
"""

# ── 1. Forecasting.py — show full broken block (lines 520-560) ─
print("=" * 60)
print("1. Forecasting.py lines 520-565 (syntax error context)")
print("=" * 60)
lines = open('pages/2_Forecasting.py', encoding='utf-8', errors='replace').read().splitlines()
for i, l in enumerate(lines[519:565], start=520):
    print(f"  {i:5}: {l}")

# ── 2. GNN — show full forecast() method ──────────────────────
print()
print("=" * 60)
print("2. GNN forecast() method")
print("=" * 60)
src = open('trading/models/advanced/gnn/gnn_model.py', encoding='utf-8', errors='replace').read()
lines2 = src.splitlines()
in_fn = False
depth = 0
for i, l in enumerate(lines2, 1):
    if 'def forecast' in l:
        in_fn = True
        depth = 0
    if in_fn:
        print(f"  {i:5}: {l}")
        stripped = l.strip()
        if stripped.endswith(':') and any(k in stripped for k in ['if ','else:','try:','except','for ','with ','def ']):
            depth += 1
        if in_fn and i > 470 and depth == 0 and stripped.startswith('def ') and 'forecast' not in stripped:
            break
        if i > 560:
            break

# ── 3. SHAP — show full explain_forecast() method ─────────────
print()
print("=" * 60)
print("3. ForecastExplainability explain_forecast() method")
print("=" * 60)
src = open('trading/analytics/forecast_explainability.py', encoding='utf-8', errors='replace').read()
lines3 = src.splitlines()
in_fn = False
for i, l in enumerate(lines3, 1):
    if 'def explain_forecast' in l or (in_fn):
        in_fn = True
        print(f"  {i:5}: {l}")
    if in_fn and i > 5 and l.strip().startswith('def ') and 'explain_forecast' not in l:
        break
    if in_fn and i > 340:
        print("  ... (truncated at 340)")
        break

# ── 4. trade_models.py — full to_dict() ───────────────────────
print()
print("=" * 60)
print("4. trade_models.py — Trade class + to_dict()")
print("=" * 60)
src = open('trading/backtesting/trade_models.py', encoding='utf-8', errors='replace').read()
lines4 = src.splitlines()
in_class = False
for i, l in enumerate(lines4, 1):
    if 'class Trade' in l or in_class:
        in_class = True
        print(f"  {i:5}: {l}")
    if in_class and i > 5 and l.strip().startswith('class ') and 'Trade' not in l:
        break
    if in_class and i > 140:
        print("  ... (truncated)")
        break