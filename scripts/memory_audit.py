"""
Audits Evolve's data storage and memory usage.
Run: .\evolve_venv\Scripts\python.exe scripts/memory_audit.py
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())


def fmt_bytes(b: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


print("=" * 60)
print("EVOLVE MEMORY & STORAGE AUDIT")
print("=" * 60)

# 1. Disk usage by directory
print("\n[DISK USAGE]")
dirs_to_check = ["data", ".cache", "logs", "trading", "agents", "pages"]
for d in dirs_to_check:
    if os.path.exists(d):
        total = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(d)
            for f in files
        )
        num_files = sum(len(files) for _, _, files in os.walk(d))
        print(f"  {d:20s}: {fmt_bytes(total):>10} ({num_files} files)")

# 2. Cache files
print("\n[CACHE FILES]")
cache_dirs = [".cache", "data/cache", "pycache"]
for cd in cache_dirs:
    if os.path.exists(cd):
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(cd)
            for f in files
        )
        print(f"  {cd}: {fmt_bytes(size)}")

# 3. SQLite databases
print("\n[DATABASES]")
for r, _, files in os.walk("data"):
    for f in files:
        if f.endswith(".db") or f.endswith(".sqlite"):
            path = os.path.join(r, f)
            print(f"  {path}: {fmt_bytes(os.path.getsize(path))}")

# 4. Log files
print("\n[LOG FILES]")
for r, _, files in os.walk("."):
    for f in files:
        if f.endswith(".log"):
            path = os.path.join(r, f)
            size = os.path.getsize(path)
            if size > 1024:
                print(f"  {path}: {fmt_bytes(size)}")

# 5. Python process memory (if psutil available)
print("\n[PROCESS MEMORY]")
try:
    import os as _os

    import psutil

    proc = psutil.Process(_os.getpid())
    mem = proc.memory_info()
    print(f"  RSS (resident): {fmt_bytes(mem.rss)}")
    print(f"  VMS (virtual):  {fmt_bytes(mem.vms)}")
except ImportError:
    print("  psutil not installed — run: pip install psutil")

# 6. yfinance cache
print("\n[YFINANCE CACHE]")
import appdirs

try:
    yf_cache = os.path.join(appdirs.user_cache_dir(), "py-yfinance")
    if os.path.exists(yf_cache):
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(yf_cache)
            for f in files
        )
        print(f"  {yf_cache}: {fmt_bytes(size)}")
    else:
        print("  yfinance cache not found (clean)")
except Exception as e:
    print(f"  Could not check yfinance cache: {e}")

print("\n[RECOMMENDATIONS]")
# Check for known memory traps
issues = []
if os.path.exists("logs"):
    log_size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk("logs")
        for f in files
    )
    if log_size > 50 * 1024 * 1024:  # >50MB
        issues.append(f"  Logs are {fmt_bytes(log_size)} — add log rotation")

if not issues:
    print("  No major issues found.")
else:
    for i in issues:
        print(i)

print("\n" + "=" * 60)

