"""
audit_s27d.py
Run: .\evolve_venv\Scripts\python.exe scripts\audit_s27d.py > scripts\audit_s27d.txt 2>&1
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 1. Forecasting.py lines 555-590
print("=" * 60)
print("1. pages/2_Forecasting.py lines 540-595")
print("=" * 60)
lines = open('pages/2_Forecasting.py', encoding='utf-8', errors='replace').read().splitlines()
for i, l in enumerate(lines[539:595], start=540):
    print(f"{i:5}|{l}")

# 2. Home.py universe loading function (full)
print()
print("=" * 60)
print("2. pages/0_Home.py load_universe_tickers function")
print("=" * 60)
src = open('pages/0_Home.py', encoding='utf-8', errors='replace').read()
lines2 = src.splitlines()
in_fn = False
for i, l in enumerate(lines2, 1):
    if 'def load_universe_tickers' in l or 'def _load_universe' in l:
        in_fn = True
    if in_fn:
        print(f"{i:5}|{l}")
    if in_fn and i > 5 and l.strip().startswith('def ') and 'universe' not in l.lower():
        break
    if in_fn and i > 260:
        print("...(truncated)")
        break

# 3. Scanner universe loading function (full)
print()
print("=" * 60)
print("3. pages/13_Scanner.py _load_scanner_universe function")
print("=" * 60)
src3 = open('pages/13_Scanner.py', encoding='utf-8', errors='replace').read()
lines3 = src3.splitlines()
in_fn = False
for i, l in enumerate(lines3, 1):
    if 'def _load_scanner_universe' in l or 'def load_universe' in l:
        in_fn = True
    if in_fn:
        print(f"{i:5}|{l}")
    if in_fn and i > 5 and l.strip().startswith('def ') and 'universe' not in l.lower():
        break
    if in_fn and i > 160:
        print("...(truncated)")
        break