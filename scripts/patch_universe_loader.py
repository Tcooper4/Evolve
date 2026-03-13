"""
patch_universe_loader.py — Rewire universe loading to use local JSON files first.
Run AFTER fetch_universes.py has been run and data/universes/*.json files exist.

  .\evolve_venv\Scripts\python.exe scripts\patch_universe_loader.py
"""
import json, os, re, py_compile

def read(p): return open(p, encoding='utf-8', errors='replace').read()
def write(p, s): open(p, 'w', encoding='utf-8').write(s)

def compile_check(p):
    try:
        py_compile.compile(p, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

# ── Shared loader utility to inject into both page files ──────
# This replaces the Wikipedia scraping with a 3-tier approach:
#   1. Load from local JSON file (fast, works on Cloud)
#   2. Scrape Wikipedia (works locally, may fail on Cloud)
#   3. Emergency 50-stock hardcoded list (always works)

LOADER_CODE = '''
# ── Universe loader (3-tier: local JSON → Wikipedia → emergency fallback) ──
import os as _os, json as _json

_UNIVERSE_DIR = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'universes')

_EMERGENCY_50 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD",
    "PG","JNJ","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","PEP",
    "TMO","ADBE","ACN","LIN","MCD","CSCO","ABT","GE","TXN","DHR",
    "PM","CAT","ISRG","INTU","AMGN","VZ","NOW","MS","GS","RTX",
]

def _load_json_universe(filename):
    """Load a pre-fetched universe JSON file."""
    for base in [_UNIVERSE_DIR,
                 _os.path.join('data', 'universes'),
                 _os.path.join(_os.path.dirname(__file__), 'data', 'universes')]:
        path = _os.path.join(base, filename)
        if _os.path.exists(path):
            try:
                return _json.load(open(path))
            except Exception:
                pass
    return None

def _scrape_sp500():
    """Scrape S&P 500 from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=10)
        t = tables[0]
        col = next((c for c in t.columns if str(c).lower() in ('symbol', 'ticker')), None)
        if col:
            return (t[col].astype(str)
                          .str.replace(r'\\.\\w+$', '', regex=True)
                          .str.replace('.', '-', regex=False)
                          .str.upper().str.strip().tolist())
    except Exception:
        pass
    return None

def _scrape_russell(url):
    """Scrape Russell index from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html(url, timeout=10)
        tickers = []
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                col = next(c for c in t.columns if 'ticker' in str(c).lower() or 'symbol' in str(c).lower())
                series = (t[col].astype(str)
                                .str.replace('.', '-', regex=False)
                                .str.upper().str.strip().tolist())
                tickers.extend(series)
        return [t for t in tickers if t and t != 'NAN' and len(t) <= 6] or None
    except Exception:
        pass
    return None

def get_universe(name):
    """
    Load a stock universe by name. Returns a list of ticker strings.
    name: 'sp500' | 'nasdaq100' | 'russell1000' | 'russell3000' | 'sp500_nasdaq100'
    """
    file_map = {
        'sp500':          'sp500.json',
        'nasdaq100':      'nasdaq100.json',
        'russell1000':    'russell1000.json',
        'russell3000':    'russell3000.json',
        'sp500_nasdaq100':'sp500_nasdaq100.json',
    }
    filename = file_map.get(name)

    # Tier 1: local JSON
    if filename:
        tickers = _load_json_universe(filename)
        if tickers and len(tickers) > 50:
            return tickers

    # Tier 2: Wikipedia scrape
    if name == 'sp500':
        tickers = _scrape_sp500()
        if tickers and len(tickers) > 100:
            return tickers
    elif name in ('russell1000', 'russell3000'):
        url = "https://en.wikipedia.org/wiki/Russell_3000_Index"
        tickers = _scrape_russell(url)
        if tickers:
            return tickers[:1000] if name == 'russell1000' else tickers
    elif name == 'sp500_nasdaq100':
        sp = get_universe('sp500')
        nq = get_universe('nasdaq100')
        return sorted(set(sp) | set(nq))

    # Tier 3: emergency fallback
    return _EMERGENCY_50
# ── End universe loader ────────────────────────────────────────
'''

# ── Patch pages/0_Home.py ─────────────────────────────────────
path = 'pages/0_Home.py'
src = read(path)

# Check if already patched
if '_load_json_universe' in src:
    print(f"[SKIP] {path} already has new loader")
else:
    # Find the load_universe_tickers function and replace its body
    # with calls to get_universe()
    NEW_LOAD_FN = '''def load_universe_tickers(universe: str) -> list[str]:
    """Load ticker universe. Uses local JSON files, falls back to Wikipedia scraping."""
    u = universe.lower()
    if 's&p 500' in u and 'nasdaq' in u:
        return get_universe('sp500_nasdaq100')
    elif 'russell 3000' in u or 'russell3000' in u:
        return get_universe('russell3000')
    elif 'russell 1000' in u or 'russell1000' in u:
        return get_universe('russell1000')
    elif 's&p 500' in u or 'sp500' in u:
        return get_universe('sp500')
    elif 'nasdaq' in u:
        return get_universe('nasdaq100')
    else:
        return get_universe('sp500')

'''
    # Replace the old function
    pattern = re.compile(r'def load_universe_tickers\(.*?\n(?=def |\nclass |\Z)', re.DOTALL)
    if pattern.search(src):
        src = pattern.sub(NEW_LOAD_FN, src, count=1)
    else:
        # Just append the new function
        src = src + '\n' + NEW_LOAD_FN

    # Inject the loader utility before load_universe_tickers
    src = src.replace(NEW_LOAD_FN, LOADER_CODE + NEW_LOAD_FN, 1)

    write(path, src)
    ok, err = compile_check(path)
    print(f"[{'PASS' if ok else 'FAIL'}] {path}: {'compiles OK' if ok else err}")

# ── Patch pages/13_Scanner.py ─────────────────────────────────
path2 = 'pages/13_Scanner.py'
src2 = read(path2)

if '_load_json_universe' in src2:
    print(f"[SKIP] {path2} already has new loader")
else:
    NEW_SCAN_FN = '''def _load_scanner_universe(universe_label: str) -> list[str]:
    """Load ticker universe for scanner. Uses local JSON, falls back to Wikipedia."""
    u = universe_label.lower()
    if 's&p 500' in u and 'nasdaq' in u:
        return get_universe('sp500_nasdaq100')
    elif 'russell 3000' in u or 'russell3000' in u:
        return get_universe('russell3000')
    elif 'russell 1000' in u or 'russell1000' in u:
        return get_universe('russell1000')
    elif 's&p 500' in u or 'sp500' in u:
        return get_universe('sp500')
    elif 'nasdaq' in u:
        return get_universe('nasdaq100')
    else:
        return get_universe('sp500')

'''
    pattern2 = re.compile(r'def _load_scanner_universe\(.*?\n(?=def |\nclass |\Z)', re.DOTALL)
    if pattern2.search(src2):
        src2 = pattern2.sub(NEW_SCAN_FN, src2, count=1)
    else:
        src2 = src2 + '\n' + NEW_SCAN_FN

    src2 = src2.replace(NEW_SCAN_FN, LOADER_CODE + NEW_SCAN_FN, 1)

    write(path2, src2)
    ok2, err2 = compile_check(path2)
    print(f"[{'PASS' if ok2 else 'FAIL'}] {path2}: {'compiles OK' if ok2 else err2}")

# ── Verify data/universes/ files exist ────────────────────────
print()
print("=" * 50)
print("Universe JSON files:")
for fname in ['sp500.json', 'nasdaq100.json', 'russell1000.json', 'russell3000.json', 'sp500_nasdaq100.json']:
    found = False
    for base in ['data/universes', os.path.join('data', 'universes')]:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            data = json.load(open(p))
            print(f"  [FOUND] {fname}: {len(data)} tickers")
            found = True
            break
    if not found:
        print(f"  [MISSING] {fname} — run fetch_universes.py first!")
print("=" * 50)
