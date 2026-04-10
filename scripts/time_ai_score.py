import sys, time
sys.path.insert(0, '.')
from trading.data.price_cache import get_history
from trading.analysis.ai_score import compute_ai_score

hist = get_history('AAPL', period='6mo')
t0 = time.time()
result = compute_ai_score('AAPL', hist)
elapsed = time.time() - t0
print(f'Single AI score: {elapsed:.2f}s')
print(f'Score: {result.get("overall_score")}')
print(f'Signals computed: {len(result.get("signals", []))}')

for sig in result.get('signals', []):
    print(f'  {sig.get("name","?")}: {sig.get("value","?")}')
