"""
Audit XGBoost feature preparation failure inside EnsembleModel
Run: .\evolve_venv\Scripts\python.exe scripts\audit_ensemble_xgb.py
"""

# ── 1. Show XGBoost prepare_features around line 493 ──────────────────────
print("=" * 55)
print("xgboost_model.py — prepare_features (lines 460-510)")
print("=" * 55)
with open("trading/models/xgboost_model.py",
          encoding="utf-8", errors="replace") as f:
    xgb_lines = f.readlines()

for i in range(459, min(515, len(xgb_lines))):
    print(f"{i+1}: {xgb_lines[i].rstrip()}")

# ── 2. Show how EnsembleModel calls XGBoost predict ───────────────────────
print()
print("=" * 55)
print("ensemble_model.py — how it calls sub-model predict")
print("=" * 55)
with open("trading/models/ensemble_model.py",
          encoding="utf-8", errors="replace") as f:
    ens_lines = f.readlines()

# Find predict method and sub-model calls
for i, line in enumerate(ens_lines):
    if "def predict" in line or ".predict(" in line or "sub_model" in line.lower() or "model.predict" in line:
        start = max(0, i-1)
        end = min(len(ens_lines), i+3)
        for j in range(start, end):
            print(f"{j+1}: {ens_lines[j].rstrip()}")
        print()

# ── 3. Show XGBoost predict method ────────────────────────────────────────
print("=" * 55)
print("xgboost_model.py — predict method (lines 590-620)")
print("=" * 55)
for i in range(589, min(625, len(xgb_lines))):
    print(f"{i+1}: {xgb_lines[i].rstrip()}")

# ── 4. Show ARIMA predict to find DataFrame->Timestamp issue ──────────────
print()
print("=" * 55)
print("arima_model.py — predict method")
print("=" * 55)
with open("trading/models/arima_model.py",
          encoding="utf-8", errors="replace") as f:
    arima_lines = f.readlines()

in_predict = False
for i, line in enumerate(arima_lines):
    if "def predict" in line:
        in_predict = True
    if in_predict:
        print(f"{i+1}: {arima_lines[i].rstrip()}")
        if i > 0 and line.strip() == "" and in_predict:
            # stop after first blank line after method content
            count = sum(1 for l in arima_lines[i-5:i] if l.strip())
            if count > 3:
                pass  # still in method
        if "def " in line and i > 0 and "def predict" not in line:
            if in_predict:
                break
    if in_predict and i > 200:
        break