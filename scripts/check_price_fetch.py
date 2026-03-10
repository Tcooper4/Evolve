src = open("pages/6_Risk_Management.py", encoding="utf-8", errors="replace").read()
lines = src.split("\n")

# Find lines related to price fetching
keywords = ["price", "fetch", "yfinance", "ticker", "history", "fast_info", "download", "regularMarket"]
for i, line in enumerate(lines):
    if any(k.lower() in line.lower() for k in keywords):
        print(f"{i+1:4d}: {line}")
