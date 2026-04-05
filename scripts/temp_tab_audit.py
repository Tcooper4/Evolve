# scripts/temp_tab_audit.py
import os

files_to_check = [
    "trading/ui/__init__.py",
    "trading/ui/config/__init__.py", 
    "trading/ui/config/registry.py",
]

print("=== trading.ui module audit ===\n")
for f in files_to_check:
    exists = os.path.exists(f)
    print(f"{'EXISTS' if exists else 'MISSING'}  {f}")

# Check what trading/ui actually contains
print("\n--- trading/ui directory ---")
if os.path.exists("trading/ui"):
    for root, dirs, files in os.walk("trading/ui"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            print(f"  {os.path.join(root, fname)}")
else:
    print("  trading/ui does not exist")

# Find where trading.ui.config.registry is imported
print("\n--- Who imports trading.ui.config.registry ---")
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in 
               ("__pycache__", "evolve_venv", ".git", ".cache")]
    for fname in files:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(root, fname)
        try:
            text = open(fpath, encoding="utf-8", errors="replace").read()
            if "trading.ui" in text or "trading/ui" in text:
                print(f"  {fpath}")
        except:
            pass