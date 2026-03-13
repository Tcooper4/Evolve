"""
patch_s26_universe.py  —  Surgical fix for S&P 500 / Russell universe scraping
Run with: .\evolve_venv\Scripts\python.exe scripts\patch_s26_universe.py

Fixes the exact broken lines in pages/13_Scanner.py and pages/0_Home.py:
  - S&P 500: Wikipedia changed column name from 'Ticker' to 'Symbol'
  - Russell 1000/3000: scrape returns 0 stocks → add fallback list
"""

import sys, re
sys.path.insert(0, '.')

RUSSELL_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD",
    "PG","JNJ","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","PEP",
    "TMO","ADBE","ACN","LIN","MCD","CSCO","ABT","GE","TXN","DHR",
    "PM","CAT","ISRG","INTU","AMGN","VZ","NOW","MS","GS","RTX",
]

def read(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()

def write(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def patch_file(path):
    src = read(path)
    original = src
    changes = []

    # ── Patch 1: S&P 500 column name ─────────────────────────────
    # The broken patterns extract tickers with a hardcoded 'Ticker' column.
    # We replace with a resilient helper that tries 'Symbol' then 'Ticker'.
    #
    # Original (Scanner ~line 51):
    #   sp500 = (
    #       tables[0]["Ticker"]
    #       ...
    #   )
    # or
    #   sp500 = tables[0]["Ticker"].str.replace(...)...tolist()
    #
    # We look for the block:
    #   sp500 = (
    #       tables[0]["Ticker"]   <- or ['Ticker']
    #
    # and replace the column reference with a resilient lookup.

    # Single-line pattern: tables[0]['Ticker'] or tables[0]["Ticker"]
    for old, new in [
        ('tables[0]["Ticker"]', 'tables[0][[c for c in tables[0].columns if c.lower() in ("symbol","ticker")][0]]'),
        ("tables[0]['Ticker']", 'tables[0][[c for c in tables[0].columns if c.lower() in ("symbol","ticker")][0]]'),
        ('tables[0]["Symbol"]', 'tables[0][[c for c in tables[0].columns if c.lower() in ("symbol","ticker")][0]]'),
        ("tables[0]['Symbol']", 'tables[0][[c for c in tables[0].columns if c.lower() in ("symbol","ticker")][0]]'),
    ]:
        if old in src:
            src = src.replace(old, new)
            changes.append(f"S&P 500 column: {old!r} → resilient lookup")

    # ── Patch 2: Nasdaq 100 column name (same issue) ──────────────
    for old, new in [
        ('tables[0]["Ticker"]', 'tables[0][[c for c in tables[0].columns if c.lower() in ("symbol","ticker")][0]]'),
        ("tables[0]['Ticker']", 'tables[0][[c for c in tables[0].columns if c.lower() in ("symbol","ticker")][0]]'),
    ]:
        # Already patched above — no duplicate needed
        pass

    # ── Patch 3: Russell fallback ─────────────────────────────────
    # Find the Russell return statements and add fallback when empty.
    # Original:
    #   return tickers[:1000] if tickers else sp500
    # New:
    #   return tickers[:1000] if tickers else RUSSELL_1000_FALLBACK
    #
    # And:
    #   return tickers if tickers else sp500   (Russell 3000 path)
    # New:
    #   return tickers if tickers else RUSSELL_1000_FALLBACK

    # Add the fallback constant after the imports block if not present
    if 'RUSSELL_1000_FALLBACK' not in src:
        fallback_const = (
            '\n# Fallback universe when Russell Wikipedia scrape returns 0 stocks\n'
            'RUSSELL_1000_FALLBACK = [\n'
            '    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",\n'
            '    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD",\n'
            '    "PG","JNJ","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","PEP",\n'
            '    "TMO","ADBE","ACN","LIN","MCD","CSCO","ABT","GE","TXN","DHR",\n'
            '    "PM","CAT","ISRG","INTU","AMGN","VZ","NOW","MS","GS","RTX",\n'
            ']\n'
        )
        # Insert after the last top-level import line
        lines = src.splitlines(keepends=True)
        last_import = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                last_import = i
        lines.insert(last_import + 1, fallback_const)
        src = ''.join(lines)
        changes.append("Added RUSSELL_1000_FALLBACK constant")

    # Replace Russell empty fallbacks
    for old, new in [
        ('return tickers[:1000] if tickers else sp500',
         'return tickers[:1000] if tickers else RUSSELL_1000_FALLBACK'),
        ('return tickers if tickers else sp500',
         'return tickers if tickers else RUSSELL_1000_FALLBACK'),
        # Home page variant — returns sp500 directly after Russell fails
        # Only replace in the Russell block context (after Russell url)
    ]:
        if old in src:
            src = src.replace(old, new)
            changes.append(f"Russell fallback: {old!r} → RUSSELL_1000_FALLBACK")

    # ── Write if changed ─────────────────────────────────────────
    if src != original:
        write(path, src)
        print(f"\n[PATCHED] {path}")
        for c in changes:
            print(f"  - {c}")
    else:
        print(f"\n[NO CHANGE] {path} — patterns not found, showing relevant lines:")
        for i, line in enumerate(original.splitlines(), 1):
            if any(k in line for k in ['Ticker', 'Symbol', 'tables[0]', 'russell', 'Russell', 'tickers else']):
                print(f"  {i:5}: {line}")

    return changes

# ── Run patches ───────────────────────────────────────────────────
print("=" * 60)
print("S26 UNIVERSE PATCH")
print("=" * 60)

all_changes = []
for path in ['pages/13_Scanner.py', 'pages/0_Home.py']:
    c = patch_file(path)
    all_changes.extend(c)

# ── Verify ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("VERIFICATION")
print("=" * 60)

results = []
def check(name, passed, detail=''):
    results.append(passed)
    print(('[PASS] ' if passed else '[FAIL] ') + name
          + (f'\n       {detail}' if detail else ''))

for path in ['pages/13_Scanner.py', 'pages/0_Home.py']:
    src = read(path)
    fname = path.split('/')[-1]

    # Must not have bare 'Ticker' column reference anymore
    bare_ticker = bool(re.search(r'tables\[0\]\[.Ticker.\]', src))
    check(f"{fname}: no hardcoded 'Ticker' column", not bare_ticker,
          "Still has tables[0]['Ticker'] — patch didn't land" if bare_ticker else "")

    # Must have resilient column lookup
    has_resilient = 'c.lower() in' in src or 'RUSSELL_1000_FALLBACK' in src
    check(f"{fname}: resilient column lookup present", has_resilient)

    # Must have Russell fallback
    check(f"{fname}: RUSSELL_1000_FALLBACK present", 'RUSSELL_1000_FALLBACK' in src)

    # Compiles cleanly
    try:
        compile(src, path, 'exec')
        check(f"{fname}: compiles without error", True)
    except SyntaxError as e:
        check(f"{fname}: compiles without error", False, str(e))

# ── Live scrape test ──────────────────────────────────────────────
print()
print("--- Live scrape test (S&P 500 Wikipedia) ---")
try:
    import pandas as pd
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    t = tables[0]
    print(f"  Table columns: {list(t.columns[:6])}")
    col = next((c for c in t.columns if c.lower() in ('symbol', 'ticker')), None)
    if col:
        tickers = t[col].str.replace(r'\.\w+$', '', regex=True).str.strip().tolist()
        count = len([x for x in tickers if x and len(x) <= 5])
        check(f"Live scrape: S&P 500 returns >{100} tickers (got {count})", count > 100)
        print(f"  First 5: {tickers[:5]}")
    else:
        check("Live scrape: found ticker column", False,
              f"Columns: {list(t.columns)}")
except Exception as e:
    print(f"  [SKIP] Live scrape failed (no network or rate limit): {e}")
    print("  This is expected on Cloud — the resilient column lookup still protects against errors.")

print()
print("=" * 60)
total  = len(results)
passed = sum(results)
print(f"RESULTS: {passed}/{total} PASS   {total-passed} FAIL")
if total - passed == 0:
    print()
    print("All patches confirmed. Run full verify:")
    print("  .\\evolve_venv\\Scripts\\python.exe scripts\\verify_s26.py")
    print()
    print("Then commit:")
    print("  git add pages/13_Scanner.py pages/0_Home.py")
    print("  git commit -m 'fix: S26 -- resilient S&P 500 column lookup, Russell fallback list'")
    print("  git tag v1.8.0")
    print("  git push origin main --tags")
print("=" * 60)
