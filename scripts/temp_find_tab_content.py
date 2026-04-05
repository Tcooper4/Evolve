# scripts/temp_find_tab_content.py
import os

# 1. Show the FULL analyze_tabs_sections.py
print("=== components/analyze_tabs_sections.py (FULL) ===")
lines = open("components/analyze_tabs_sections.py",
             encoding="utf-8", errors="replace").readlines()
print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines, 1):
    print(f"  {i:3}: {line.rstrip()}")

# 2. Check pages/2_Analyze.py for any tab rendering
print("\n\n=== pages/2_Analyze.py (FULL) ===")
lines = open("pages/2_Analyze.py",
             encoding="utf-8", errors="replace").readlines()
print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines, 1):
    print(f"  {i:3}: {line.rstrip()}")

# 3. Check if there are other analyze-related components
print("\n\n=== components/ files mentioning Risk/Technical/Research ===")
for fname in os.listdir("components"):
    if not fname.endswith(".py"):
        continue
    fpath = os.path.join("components", fname)
    text = open(fpath, encoding="utf-8", errors="replace").read()
    if any(x in text for x in 
           ["tab_risk", "tab_technical", "tab_research",
            "tab_market", "Risk", "Technical", "Research"]):
        lines = open(fpath, encoding="utf-8", 
                     errors="replace").readlines()
        print(f"\n  {fname} ({len(lines)} lines) — mentions relevant tabs")