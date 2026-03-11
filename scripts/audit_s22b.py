"""
Session 22 Audit — find actual file locations
Run: .\evolve_venv\Scripts\python.exe scripts\audit_s22.py
"""

import os
import glob

print("=" * 55)
print("PAGES DIRECTORY")
print("=" * 55)
for f in sorted(os.listdir("pages")):
    print(f"  {f}")

print()
print("=" * 55)
print("TRADING/MODELS DIRECTORY")
print("=" * 55)
for f in sorted(os.listdir("trading/models")):
    print(f"  {f}")

print()
print("=" * 55)
print("SEARCH: GNN anywhere in codebase")
print("=" * 55)
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in
               [".git", "__pycache__", "evolve_venv", ".cache"]]
    for f in files:
        if "gnn" in f.lower() and f.endswith(".py"):
            print(f"  {os.path.join(root, f)}")

print()
print("=" * 55)
print("SEARCH: SHAP flag in forecast_explainability or similar")
print("=" * 55)
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in
               [".git", "__pycache__", "evolve_venv", ".cache"]]
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                if "shap" in content.lower() and "HAS_SHAP" in content:
                    print(f"  {fpath}")
            except Exception:
                pass

print()
print("=" * 55)
print("SEARCH: Trade class to_dict in backtesting")
print("=" * 55)
for root, dirs, files in os.walk("trading/backtesting"):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                if "to_dict" in content and ("Trade" in content or "trade" in content):
                    print(f"  {fpath}")
                    # Show the to_dict method
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if "to_dict" in line:
                            start = max(0, i-1)
                            end = min(len(lines), i+15)
                            print(f"    --- to_dict at line {i+1} ---")
                            for j, l in enumerate(lines[start:end], start+1):
                                print(f"    {j}: {l}")
                            print()
            except Exception:
                pass

print()
print("=" * 55)
print("SEARCH: Forecasting page actual filename")
print("=" * 55)
for f in os.listdir("pages"):
    if "forecast" in f.lower() or "2_" in f:
        print(f"  pages/{f}")

print()
print("=" * 55)
print("SEARCH: Strategy Research Agent error phrase location")
print("=" * 55)
try:
    with open("pages/3_Strategy_Testing.py",
              encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines, 1):
        if "Strategy Research Agent" in line or "not available" in line.lower():
            print(f"  line {i}: {line.rstrip()}")
except FileNotFoundError:
    print("  pages/3_Strategy_Testing.py not found")

print()
print("=" * 55)
print("SEARCH: Model discovery error phrase location")
print("=" * 55)
try:
    with open("pages/8_Model_Lab.py",
              encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines, 1):
        if "Model discovery" in line or "not available" in line.lower():
            print(f"  line {i}: {line.rstrip()}")
except FileNotFoundError:
    print("  pages/8_Model_Lab.py not found")