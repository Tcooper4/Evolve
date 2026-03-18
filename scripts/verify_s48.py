from pathlib import Path

def read(p):
    return Path(p).read_text(encoding='utf-8', errors='replace')

results = []

td = read('trading/agents/task_dashboard.py')
results.append(('task_dashboard experimental_rerun gone', 'experimental_rerun' not in td))
results.append(('task_dashboard st.rerun present', 'st.rerun()' in td))

ww = read('components/watchlist_widget.py')
results.append(('watchlist experimental_rerun gone', 'experimental_rerun' not in ww))
results.append(('watchlist st.rerun present', 'st.rerun()' in ww))

bm = read('trading/models/base_model.py')
results.append(('base_model DatetimeIndex guard', 'DatetimeIndex' in bm and 'errors=\'coerce\'' in bm))

xg = read('trading/models/xgboost_model.py')
results.append(('xgboost date fix applied', 'isinstance(data.index, pd.DatetimeIndex)' in xg))
results.append(('xgboost wrong freq check gone', 'hasattr(data.index[-1], "freq")' not in xg))

lm = read('trading/models/lstm_model.py')
results.append(('lstm date fix applied', 'isinstance(data.index, pd.DatetimeIndex)' in lm))
results.append(('lstm wrong freq check gone', 'hasattr(data.index, "freq")' not in lm))

fr = read('trading/models/forecast_router.py')
results.append(('router NaT guard present', 'nat_mask' in fr and 'isna()' in fr))

print()
for name, passed in results:
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print()
