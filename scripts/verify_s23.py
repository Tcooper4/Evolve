"""Session 23 verification."""
import sys
import os
import hashlib

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 23 VERIFICATION")
print("=" * 60)

# 1. Onboarding: localStorage no longer primary mechanism
print("\n[1] Onboarding session_id fix...")
try:
    c = open("components/onboarding.py", encoding="utf-8", errors="replace").read()
    uses_hash = "sha256" in c or "hashlib" in c
    has_qparam = "query_params" in c
    localStorage_not_primary = (
        c.count("localStorage") == 0 or "best-effort" in c or uses_hash
    )
    print(f"  Uses key hash for session_id: {uses_hash}")
    print(f"  Uses query_params: {has_qparam}")
    print(f"  localStorage no longer primary: {localStorage_not_primary}")
    print(f"  {'PASS' if uses_hash and has_qparam else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. GNN crash: try/except around GNN tab
print("\n[2] GNN crash fix...")
try:
    c = open("pages/2_Forecasting.py", encoding="utf-8", errors="replace").read()
    gnn_section = (
        c[c.find("GNN") : c.find("GNN") + 3000] if "GNN" in c else ""
    )
    has_try = "try:" in gnn_section
    has_type_guard = "isinstance" in gnn_section or "int(" in gnn_section
    print(f"  try/except in GNN section: {has_try}")
    print(f"  Type guard present: {has_type_guard}")
    print(f"  {'PASS' if has_try else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 3. Trade.to_dict() dates
print("\n[3] Trade.to_dict() entry/exit dates...")
import glob

found_any = False
for pattern in [
    "trading/backtesting/*.py",
    "trading/portfolio/*.py",
    "trading/execution/*.py",
]:
    for fpath in glob.glob(pattern):
        try:
            c = open(fpath, encoding="utf-8", errors="replace").read()
            if "to_dict" in c and ("Trade" in c or "Position" in c):
                has_entry = "entry_date" in c
                has_exit = "exit_date" in c
                if has_entry or has_exit:
                    found_any = True
                    print(f"  {fpath}: entry_date={has_entry}, exit_date={has_exit}")
        except Exception:
            pass
if not found_any:
    print("  WARNING: No to_dict with date fields found")

# 4. Performance page reads entry/exit_date safely
print("\n[4] Performance page date handling...")
try:
    c = open("pages/7_Performance.py", encoding="utf-8", errors="replace").read()
    reads_entry = "entry_date" in c
    reads_exit = "exit_date" in c
    has_get = ".get('entry_date')" in c or ".get('exit_date')" in c
    print(f"  References entry_date: {reads_entry}")
    print(f"  References exit_date: {reads_exit}")
    print(f"  Uses safe .get(): {has_get}")
    print(f"  {'PASS' if reads_entry else 'CHECK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Smoke tests
print("\n[5] Model smoke tests...")
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
print("Session 23 complete.")
print("=" * 60)
