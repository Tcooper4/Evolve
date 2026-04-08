import sys

sys.path.insert(0, ".")
from trading.data.analyst_signals import get_analyst_signals

result = get_analyst_signals("AAPL")
print("Symbol:", result.get("symbol"))
print("Rec:", result.get("recommendation"))
print("N analysts:", result.get("n_analysts"))
print("Target mean:", result.get("target_mean"))
print("Upside %:", result.get("upside_pct"))
print("Signal:", result.get("signal"))
print("Strength:", result.get("signal_strength"))
print("Success:", result.get("success"))
