# -*- coding: utf-8 -*-
import ast

files = [
    "components/tabs/tab_market_analysis.py",
    "components/tabs/tab_diagnostics.py",
    "components/tabs/tab_multi_asset_gnn.py",
    "components/tabs/tab_causal.py",
    "components/tabs/tab_earnings.py",
    "components/tabs/tab_monte_carlo.py",
]
for f in files:
    try:
        ast.parse(open(f, encoding="utf-8", errors="replace").read())
        print(f"OK  {f}")
    except SyntaxError as e:
        print(f"FAIL  {f}: {e}")
