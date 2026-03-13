"""
audit_s27c.py — Show exact broken sections of the three failing files
Run: .\evolve_venv\Scripts\python.exe scripts\audit_s27c.py > scripts\audit_s27c.txt 2>&1
"""

# 1. Forecasting.py — lines 515-570
print("=" * 60)
print("1. pages/2_Forecasting.py lines 515-580")
print("=" * 60)
lines = open('pages/2_Forecasting.py', encoding='utf-8', errors='replace').read().splitlines()
for i, l in enumerate(lines[514:580], start=515):
    print(f"{i:5}|{l}")

# 2. forecast_router.py — first 80 lines + any 'def ' lines
print()
print("=" * 60)
print("2. trading/models/forecast_router.py — structure")
print("=" * 60)
src = open('trading/models/forecast_router.py', encoding='utf-8', errors='replace').read()
lines2 = src.splitlines()
# Show first 60 lines
for i, l in enumerate(lines2[:60], start=1):
    print(f"{i:5}|{l}")
print("...")
# Show all def lines
for i, l in enumerate(lines2, start=1):
    if l.strip().startswith('def ') or l.strip().startswith('class '):
        print(f"{i:5}|{l}")

# 3. forecast_explainability.py — full file if short, else key sections
print()
print("=" * 60)
print("3. trading/analytics/forecast_explainability.py — full content")
print("=" * 60)
lines3 = open('trading/analytics/forecast_explainability.py', encoding='utf-8', errors='replace').read().splitlines()
print(f"  Total lines: {len(lines3)}")
for i, l in enumerate(lines3, start=1):
    print(f"{i:5}|{l}")