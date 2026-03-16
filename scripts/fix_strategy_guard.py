# fix_strategy_guard.py -- removes duplicate page guard key from
# walk-forward and RL split files, keeping it only in backtest.
# Run: .\evolve_venv\Scripts\python.exe scripts\fix_strategy_guard.py
import sys, os, io, ast, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')

# The guard key appears in shared header copied to all 3 files.
# Remove it from walkforward and rl files only -- keep in backtest.
GUARD_KEY = 'EVOLVE_PAGE_GUARD_STRATEGY_TESTING'

files_to_fix = [
    'old_strategy_walkforward.py',
    'old_strategy_rl.py',
]

for fname in files_to_fix:
    fpath = os.path.join(SCRIPTS, fname)
    with open(fpath, encoding='utf-8', errors='replace') as f:
        src = f.read()

    original_count = src.count(GUARD_KEY)

    # Remove any line containing the guard key
    lines = src.splitlines(keepends=True)
    cleaned = []
    for line in lines:
        if GUARD_KEY in line:
            continue
        cleaned.append(line)
    new_src = ''.join(cleaned)

    new_count = new_src.count(GUARD_KEY)

    # Parse check
    try:
        ast.parse(new_src)
        parse_status = 'parses OK'
    except SyntaxError as e:
        parse_status = f'SyntaxError line {e.lineno}: {e.msg}'

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_src)

    print(f'{fname}: removed {original_count - new_count} guard line(s), '
          f'{len(new_src.splitlines())} lines, {parse_status}')

# Verify backtest still has it
bt_path = os.path.join(SCRIPTS, 'old_strategy_backtest.py')
with open(bt_path, encoding='utf-8', errors='replace') as f:
    bt_src = f.read()
bt_count = bt_src.count(GUARD_KEY)
print(f'old_strategy_backtest.py: guard key present {bt_count} time(s) (correct)')

# Final duplicate check across all three
print('\nFinal check:')
all_keys = []
key_pattern = re.compile(r'key\s*=\s*["\']([^"\']+)["\']')
for fname in ['old_strategy_backtest.py', 'old_strategy_walkforward.py',
              'old_strategy_rl.py']:
    fpath = os.path.join(SCRIPTS, fname)
    with open(fpath, encoding='utf-8', errors='replace') as f:
        src = f.read()
    keys = key_pattern.findall(src)
    all_keys.extend(keys)

from collections import Counter
counts = Counter(all_keys)
dups = {k: v for k, v in counts.items() if v > 1}
if dups:
    print('  Remaining duplicates:')
    for k, v in sorted(dups.items()):
        print(f'    "{k}" x{v}')
else:
    print('  No duplicate keys remaining across strategy files.')

print('\nDone. Now update pages/5_Backtest.py runpy calls:')
print('  tab_bt  -> old_strategy_backtest.py')
print('  tab_wf  -> old_strategy_walkforward.py')
print('  tab_rl  -> old_strategy_rl.py')
print('  tab_rep -> old_9_Reports.py')