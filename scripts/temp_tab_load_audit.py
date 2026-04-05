# scripts/temp_tab_load_audit.py
import os

# Check what the Risk, Technical, Research tabs actually are
# and what errors they might throw on import
tab_dir = "components/tabs"
print("=== All tab files ===")
for fname in sorted(os.listdir(tab_dir)):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join(tab_dir, fname)
    lines = open(fpath, encoding="utf-8", 
                 errors="replace").readlines()
    print(f"\n{fname} ({len(lines)} lines)")
    # Show first 10 lines
    for i, line in enumerate(lines[:10], 1):
        print(f"  {i:3}: {line.rstrip()}")

# Check analyze_tabs_sections to see how tabs are wired
print("\n\n=== components/analyze_tabs_sections.py ===")
path = "components/analyze_tabs_sections.py"
lines = open(path, encoding="utf-8", errors="replace").readlines()
print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines[:80], 1):
    print(f"  {i:3}: {line.rstrip()}")

# Check the trading.ui import in tab_quick_forecast
print("\n\n=== trading.ui import in tab_quick_forecast ===")
path = "components/tabs/tab_quick_forecast.py"
lines = open(path, encoding="utf-8", errors="replace").readlines()
for i, line in enumerate(lines, 1):
    if "trading.ui" in line or "registry" in line.lower():
        # show context
        start = max(0, i-3)
        end = min(len(lines), i+3)
        print(f"  Line {i}: {line.rstrip()}")
        for j in range(start, end):
            if j != i-1:
                print(f"    context {j+1}: {lines[j].rstrip()}")
        print()

# Check what registry.py is supposed to contain
print("\n=== _archive for registry.py ===")
for root, dirs, files in os.walk("_archive"):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for fname in files:
        if "registry" in fname.lower():
            fpath = os.path.join(root, fname)
            print(f"  FOUND: {fpath}")