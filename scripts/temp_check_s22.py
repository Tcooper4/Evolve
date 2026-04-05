import os

search_dirs = ["components", "pages", "agents", "trading"]
remaining = []
for d in search_dirs:
    for root, dirs, files in os.walk(d):
        dirs[:] = [
            x
            for x in dirs
            if x not in ("__pycache__", "evolve_venv", ".git")
        ]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            lines = open(
                fpath, encoding="utf-8", errors="replace"
            ).readlines()
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if "ForecastRouter()" not in s:
                    continue
                if "return ForecastRouter()" in s:
                    continue
                if s.startswith("#"):
                    continue
                remaining.append(f"{fpath}:{i}  {s}")
for r in remaining:
    print(r)
print(f"Total remaining: {len(remaining)}")
