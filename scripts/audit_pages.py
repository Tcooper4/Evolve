import os, glob

pages = sorted(glob.glob("pages/*.py"))
print("=== ALL PAGES ===")
for p in pages:
    src = open(p, encoding="utf-8", errors="replace").read()
    lines = src.split("\n")
    print(f"\n{p} ({len(lines)} lines)")
    # Key signals
    signals = {
        "set_page_config": "set_page_config" in src,
        "fast_info": "fast_info" in src,
        "hardcoded tickers": any(f"'{t}'" in src for t in ["'AMD'","'NVDA'","'AAPL'","'GOOGL'","'MSFT'"]),
        "st.cache_data": "st.cache_data" in src,
        "try/except": "try:" in src,
    }
    for k, v in signals.items():
        print(f"  {k}: {v}")
