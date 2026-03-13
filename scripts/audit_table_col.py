"""
audit_table_col.py — Show the Wikipedia scrape blocks in both page files
Run: .\evolve_venv\Scripts\python.exe scripts\audit_table_col.py
"""
for path in ['pages/13_Scanner.py', 'pages/0_Home.py']:
    print(f"\n{'='*60}")
    print(f"{path}")
    print('='*60)
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    for i, line in enumerate(lines, 1):
        if any(k in line for k in ['table', 'Ticker', 'Symbol', 'ticker', 'symbol', 'wikipedia', 'read_html', 'tickers', 'sp500', 'russell', 'Russell', 'tolist', 'str.replace']):
            print(f"{i:5}: {line}")