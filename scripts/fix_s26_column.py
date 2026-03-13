"""
fix_s26_column.py — Fix hardcoded 'Ticker'/'Symbol' column in Wikipedia scrape
Run: .\evolve_venv\Scripts\python.exe scripts\fix_s26_column.py
"""
import re

RESILIENT = "[c for c in tables[0].columns if c.lower() in (\"symbol\", \"ticker\")][0]"

for path in ['pages/13_Scanner.py', 'pages/0_Home.py']:
    src = open(path, encoding='utf-8', errors='replace').read()
    original = src

    # Replace tables[0]["Ticker"] or tables[0]['Ticker'] or same with Symbol
    src = re.sub(
        r'tables\[0\]\[["\'](?:Ticker|Symbol)["\']\]',
        f'tables[0][{RESILIENT}]',
        src
    )

    if src != original:
        open(path, 'w', encoding='utf-8').write(src)
        count = len(re.findall(r'tables\[0\]\[' + re.escape(RESILIENT.replace('"', r'"')), src))
        print(f"[PATCHED] {path}")
    else:
        print(f"[NO MATCH] {path} — checking what's actually there:")
        for i, line in enumerate(src.splitlines(), 1):
            if 'tables[0]' in line:
                print(f"  {i:5}: {line.strip()}")

# Verify
print()
for path in ['pages/13_Scanner.py', 'pages/0_Home.py']:
    src = open(path, encoding='utf-8', errors='replace').read()
    import py_compile
    try:
        py_compile.compile(path, doraise=True)
        resilient = 'c.lower() in' in src
        bare = bool(re.search(r'tables\[0\]\[["\'](?:Ticker|Symbol)["\']\]', src))
        status = "PASS" if resilient and not bare else "FAIL"
        print(f"[{status}] {path}: compiles=OK resilient={resilient} bare_column={bare}")
    except py_compile.PyCompileError as e:
        print(f"[FAIL] {path}: {e}")