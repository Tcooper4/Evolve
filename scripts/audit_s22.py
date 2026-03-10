import ast, os

# 1. Check app.py for set_page_config
src = open("app.py", encoding="utf-8", errors="replace").read()
lines = src.split("\n")
print("=== app.py set_page_config ===")
for i, line in enumerate(lines):
    if "set_page_config" in line:
        start = max(0, i-2)
        end = min(len(lines), i+6)
        for j in range(start, end):
            print(f"  {j+1:4d}: {lines[j]}")

# 2. Check Reports page for what's broken
src9 = open("pages/9_Reports.py", encoding="utf-8", errors="replace").read()
lines9 = src9.split("\n")
print("\n=== 9_Reports.py — first 80 lines ===")
for i, line in enumerate(lines9[:80]):
    print(f"  {i+1:4d}: {line}")

# 3. Check for tab/section definitions in Reports
print("\n=== 9_Reports.py — tab/section definitions ===")
for i, line in enumerate(lines9):
    if any(k in line for k in ["tab", "Tab", "custom", "Custom", "schedule", "Schedule", "library", "Library", "st.error", "except"]):
        print(f"  {i+1:4d}: {line}")

# 4. Quick size check on all pages with set_page_config still present
import glob
print("\n=== Pages still containing set_page_config ===")
for p in sorted(glob.glob("pages/*.py")):
    src = open(p, encoding="utf-8", errors="replace").read()
    if "set_page_config" in src:
        count = src.count("set_page_config")
        print(f"  {p}: {count} occurrence(s)")
