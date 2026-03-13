"""
fetch_universes.py — One-time fetch of all stock universe lists.
Saves to data/universes/*.json for use as static fallbacks.

Run locally (not on Cloud) where Wikipedia is accessible:
  .\evolve_venv\Scripts\python.exe scripts\fetch_universes.py
"""
import json, os, time
import pandas as pd

OUT_DIR = 'data/universes'
os.makedirs(OUT_DIR, exist_ok=True)

EMERGENCY_50 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "TSLA","WMT","JPM","V","XOM","UNH","ORCL","MA","COST","HD",
    "PG","JNJ","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","PEP",
    "TMO","ADBE","ACN","LIN","MCD","CSCO","ABT","GE","TXN","DHR",
    "PM","CAT","ISRG","INTU","AMGN","VZ","NOW","MS","GS","RTX",
]

def save(name, tickers):
    path = os.path.join(OUT_DIR, f'{name}.json')
    tickers = sorted(set(t for t in tickers if t and len(t) <= 6 and t != 'nan' and t != 'NAN'))
    with open(path, 'w') as f:
        json.dump(tickers, f, indent=2)
    print(f"  Saved {len(tickers)} tickers -> {path}")
    return tickers

def get_col(df):
    for c in df.columns:
        if str(c).lower() in ('symbol', 'ticker'):
            return c
    for c in df.columns:
        if 'symbol' in str(c).lower() or 'ticker' in str(c).lower():
            return c
    return None

def clean(series):
    return (series.astype(str)
                  .str.replace(r'\.\w+$', '', regex=True)
                  .str.replace('.', '-', regex=False)
                  .str.upper()
                  .str.strip()
                  .tolist())

# ── 1. S&P 500 ────────────────────────────────────────────────
print("\n[1/6] Fetching S&P 500...")
try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=15)
    col = get_col(tables[0])
    if col:
        sp500 = clean(tables[0][col])
        sp500 = save('sp500', sp500)
        print(f"  OK: {len(sp500)} tickers")
    else:
        print(f"  WARN: No ticker column. Columns: {list(tables[0].columns[:6])}")
        sp500 = EMERGENCY_50
except Exception as e:
    print(f"  FAIL: {e}")
    sp500 = EMERGENCY_50
time.sleep(2)

# ── 2. Nasdaq 100 ─────────────────────────────────────────────
print("\n[2/6] Fetching Nasdaq 100...")
nasdaq = []
try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", timeout=15)
    for t in tables:
        col = get_col(t)
        if col:
            tickers = [x for x in clean(t[col]) if x and len(x) <= 5]
            if len(tickers) > 50:
                nasdaq = tickers
                break
    if nasdaq:
        nasdaq = save('nasdaq100', nasdaq)
        print(f"  OK: {len(nasdaq)} tickers")
    else:
        print(f"  WARN: Could not find Nasdaq 100 table with ticker column")
except Exception as e:
    print(f"  FAIL: {e}")
time.sleep(2)

# ── 3. S&P 100 ────────────────────────────────────────────────
print("\n[3/6] Fetching S&P 100...")
sp100 = []
try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/S%26P_100", timeout=15)
    for t in tables:
        col = get_col(t)
        if col:
            tickers = [x for x in clean(t[col]) if x and len(x) <= 6]
            if len(tickers) > 50:
                sp100 = tickers
                break
    if sp100:
        sp100 = save('sp100', sp100)
        print(f"  OK: {len(sp100)} tickers")
    else:
        print(f"  WARN: Falling back to hardcoded S&P 100")
        raise ValueError("no table found")
except Exception as e:
    print(f"  Using hardcoded S&P 100 ({e})")
    sp100 = [
        "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
        "AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","BRK-B","C",
        "CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS",
        "CVX","D","DE","DHR","DIS","DOW","DUK","EMR","EXC","F",
        "FDX","GD","GE","GILD","GM","GOOG","GOOGL","GS","HD","HON",
        "IBM","INTC","INTU","JNJ","JPM","KHC","KO","LIN","LLY","LMT",
        "LOW","MA","MCD","MDLZ","MDT","MET","META","MMM","MO","MRK",
        "MS","MSFT","NEE","NFLX","NKE","NVDA","ORCL","PEP","PFE","PG",
        "PM","PYPL","QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT",
        "TMO","TMUS","TXN","UNH","UNP","UPS","USB","V","VZ","WFC",
        "WMT","XOM",
    ]
    sp100 = save('sp100', sp100)
    print(f"  Saved hardcoded: {len(sp100)} tickers")
time.sleep(2)

# ── 4. Russell 3000 (approximated) ────────────────────────────
print("\n[4/6] Fetching Russell 3000...")
russell3000 = []
try:
    tables = pd.read_html("https://en.wikipedia.org/wiki/Russell_3000_Index", timeout=15)
    for t in tables:
        col = get_col(t)
        if col:
            tickers = [x for x in clean(t[col]) if x and len(x) <= 6 and x != 'NAN']
            if len(tickers) > 100:
                russell3000.extend(tickers)
    if russell3000:
        print(f"  Wikipedia returned {len(russell3000)} tickers")
    else:
        print(f"  Wikipedia returned 0 tickers — using S&P 500 + mid-cap extension")
except Exception as e:
    print(f"  Wikipedia FAIL: {e}")

if len(russell3000) < 500:
    # Extend S&P 500 with known mid-caps as best approximation
    mid_caps = [
        "PARA","VTRS","HII","MOS","AIZ","BEN","IVZ","PNW","RL","VFC",
        "HRL","LKQ","AOS","SEE","FRT","IPG","SNA","TAP","PVH","LEG",
        "NWSA","NWS","DXC","XRX","PBCT","NAVI","WDC","STX","ETSY","PENN",
        "RCL","CCL","MGM","LVS","WYNN","MAR","HLT","CHH","TNL","VAC",
        "AAL","DAL","UAL","LUV","ALK","SAVE","HA","JBLU","ALGT","SKYW",
        "KR","ACI","SFM","WINN","PFGC","USFD","SPTN","CASY","WDFC","CHEF",
        "MPW","SBRA","OHI","WELL","VTR","HR","DOC","PEAK","CTRE","LTC",
        "AMC","CNK","IMAX","MCS","NCMI","RGC","MKGP","ATER","PRCH","PAYO",
        "CHWY","PETS","WOOF","FRPT","HIMS","NTRA","RXRX","SEER","ACMR","BEAM",
        "AFRM","UPST","LC","CACC","WRLD","ENVA","CURO","QFIN","TREE","RATE",
    ]
    russell3000 = list(sp500) + mid_caps
    print(f"  Approximated: S&P500 ({len(sp500)}) + mid-caps ({len(mid_caps)}) = {len(russell3000)}")

russell3000 = save('russell3000', russell3000)
russell1000 = save('russell1000', russell3000[:1000])
print(f"  Russell 3000: {len(russell3000)}, Russell 1000: {len(russell1000)}")

# ── 5. S&P 500 + Nasdaq 100 combined ─────────────────────────
print("\n[5/6] Building S&P 500 + Nasdaq 100 combined...")
combined = save('sp500_nasdaq100', list(set(sp500) | set(nasdaq)))
print(f"  Combined: {len(combined)} tickers")

# ── 6. Summary ────────────────────────────────────────────────
print("\n[6/6] Summary:")
for fname in ['sp100', 'sp500', 'nasdaq100', 'sp500_nasdaq100', 'russell1000', 'russell3000']:
    path = os.path.join(OUT_DIR, f'{fname}.json')
    if os.path.exists(path):
        data = json.load(open(path))
        print(f"  {fname:25}: {len(data):5} tickers")
    else:
        print(f"  {fname:25}: MISSING")

print()
print("Next: commit these files to git.")
print("  git add data/universes/")
print("  git commit -m 'data: static universe JSON files'")
print("Then run: .\\evolve_venv\\Scripts\\python.exe scripts\\patch_universe_loader.py")
