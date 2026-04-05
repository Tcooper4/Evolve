# scripts/temp_tab_structure.py
import os

# Find all tab files and check their imports
tab_dir = "components/tabs"
for fname in sorted(os.listdir(tab_dir)):
    if not fname.endswith(".py") or fname == "__init__.py":
        continue
    fpath = os.path.join(tab_dir, fname)
    lines = open(fpath, encoding="utf-8", errors="replace").readlines()
    errors = [l.strip() for l in lines[:30] 
              if "trading.ui" in l or "registry" in l.lower()]
    if errors:
        print(f"\n{fname}:")
        for e in errors:
            print(f"  {e}")