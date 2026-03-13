"""
fix_s26_syntax.py  —  Fix the syntax error introduced by patch_s26_universe.py
Run with: .\evolve_venv\Scripts\python.exe scripts\fix_s26_syntax.py

The previous patch inserted RUSSELL_1000_FALLBACK inside a try/except block.
This script moves it to the correct location (after imports, before any functions).
"""

import sys, re
sys.path.insert(0, '.')

FALLBACK_CONST = (
    '\n# Fallback universe when Russell Wikipedia scrape returns 0 stocks\n'
    'RUSSELL_1000_FALLBACK = [\n'
    '    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",\n'
    '    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD",\n'
    '    "PG","JNJ","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","PEP",\n'
    '    "TMO","ADBE","ACN","LIN","MCD","CSCO","ABT","GE","TXN","DHR",\n'
    '    "PM","CAT","ISRG","INTU","AMGN","VZ","NOW","MS","GS","RTX",\n'
    ']\n'
)

def read(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()

def write(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def fix_file(path):
    src = read(path)

    # Step 1: Remove ALL existing RUSSELL_1000_FALLBACK constant definitions
    # (wherever the previous patch inserted them, possibly mid-block)
    # Match the full constant block:
    #   \n# Fallback universe...\nRUSSELL_1000_FALLBACK = [\n    ...\n]\n
    pattern = r'\n# Fallback universe when Russell Wikipedia scrape returns 0 stocks\nRUSSELL_1000_FALLBACK = \[[\s\S]*?\]\n'
    cleaned = re.sub(pattern, '\n', src)

    if cleaned == src:
        # Try simpler pattern
        pattern2 = r'RUSSELL_1000_FALLBACK = \[[\s\S]*?\]\n'
        cleaned = re.sub(pattern2, '', src)

    removed = cleaned != src
    print(f"  Removed existing RUSSELL_1000_FALLBACK block: {removed}")

    # Step 2: Find the correct insertion point — after all top-level imports
    # but before the first function/class definition or non-import code block.
    lines = cleaned.splitlines(keepends=True)

    # Find the last import line
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            last_import_idx = i

    # Insert the constant block right after the last import
    lines.insert(last_import_idx + 1, FALLBACK_CONST)
    result = ''.join(lines)

    # Step 3: Verify it compiles
    try:
        compile(result, path, 'exec')
        write(path, result)
        print(f"  Compiles cleanly — written.")
        return True
    except SyntaxError as e:
        print(f"  Still has syntax error after fix attempt: {e}")
        print(f"  Showing lines around the error:")
        err_lines = result.splitlines()
        start = max(0, e.lineno - 5)
        end   = min(len(err_lines), e.lineno + 5)
        for i in range(start, end):
            marker = " >>>" if i == e.lineno - 1 else "    "
            print(f"  {marker} {i+1:4}: {err_lines[i]}")
        return False

print("=" * 60)
print("FIX S26 SYNTAX")
print("=" * 60)

results = []
for path in ['pages/13_Scanner.py', 'pages/0_Home.py']:
    print(f"\n--- {path} ---")
    ok = fix_file(path)
    results.append(ok)

print()
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

checks = []
def check(name, passed, detail=''):
    checks.append(passed)
    print(('[PASS] ' if passed else '[FAIL] ') + name
          + (f'\n       {detail}' if detail else ''))

for path in ['pages/13_Scanner.py', 'pages/0_Home.py']:
    src = read(path)
    fname = path.split('/')[-1]

    # Compiles
    try:
        compile(src, path, 'exec')
        check(f"{fname}: compiles cleanly", True)
    except SyntaxError as e:
        check(f"{fname}: compiles cleanly", False, str(e))

    # Has resilient column lookup
    check(f"{fname}: resilient column lookup",
          'c.lower() in' in src)

    # Has fallback constant
    check(f"{fname}: RUSSELL_1000_FALLBACK present",
          'RUSSELL_1000_FALLBACK' in src)

    # Fallback is at module level (not inside a function/try block)
    # Check that it appears before any 'def ' line
    fallback_pos = src.find('RUSSELL_1000_FALLBACK = [')
    first_def    = src.find('\ndef ')
    check(f"{fname}: RUSSELL_1000_FALLBACK is at module level (before first def)",
          fallback_pos != -1 and fallback_pos < first_def,
          f"fallback at char {fallback_pos}, first def at char {first_def}")

print()
total  = len(checks)
passed = sum(checks)
print(f"RESULTS: {passed}/{total} PASS   {total-passed} FAIL")
if total - passed == 0:
    print()
    print("All fixes confirmed. Commit with:")
    print("  git add pages/13_Scanner.py pages/0_Home.py")
    print("  git commit -m \"fix: S26 -- resilient S&P 500 column, Russell fallback\"")
    print("  git tag v1.8.0")
    print("  git push origin main --tags")
print("=" * 60)
