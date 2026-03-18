import sys
from pathlib import Path

def read(p):
    return Path(p).read_text(encoding='utf-8', errors='replace')

results = []
src = read('trading/analysis/ai_score.py')

results.append(('SECTOR_PE dict present',        'SECTOR_PE' in src))
results.append(('Valuation overlay present',      'pe_premium' in src and 'val_label' in src))
results.append(('Short squeeze tiered',           'si_sentiment = 9.0' in src and 'squeeze_tier' in src))
results.append(('Momentum bonus for short squeeze','momentum_score + 2.0' in src or 'momentum_score + 1.5' in src))
results.append(('earnings_near flag initialised', 'earnings_near = False' in src))
results.append(('Earnings conviction cap',        'min(overall, 7.5)' in src))
results.append(('Earnings typo fixed',            'Earningss' not in src))
results.append(('Insider No Activity fix',        'No Activity' in src and '_insider_val' in src))

prophet = read('trading/models/prophet_model.py')
results.append(('Prophet hardcode 30 removed',    'periods=30' not in prophet or 'horizon' in prophet))
results.append(('Prophet accepts horizon param',  'horizon' in prophet))

print()
for name, passed in results:
    status = 'PASS' if passed else 'FAIL'
    print(f"{status}  {name}")
print()
