import os, sys, ast, re
from collections import Counter
from pathlib import Path

ROOT = Path(r'C:\Users\Thomas\OneDrive\Desktop\Dashboard\evolve_clean')
SCRIPTS = ROOT / 'scripts'
SRC = SCRIPTS / 'old_3_Strategy_Testing.py'

with open(SRC, encoding='utf-8', errors='replace') as f:
    all_lines = f.readlines()

HEADER = """import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

"""

def dedent_fixed(lines, amount):
    out = []
    for line in lines:
        if line.startswith(' ' * amount):
            out.append(line[amount:])
        elif line.strip() == '':
            out.append('\n')
        else:
            out.append(line)
    return out

def dedent_auto(lines):
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return lines
    min_ind = min(len(l) - len(l.lstrip()) for l in non_empty)
    out = []
    for line in lines:
        if line.strip():
            out.append(line[min_ind:] if len(line) > min_ind else line)
        else:
            out.append('\n')
    return out

def write_and_check(fname, content):
    path = SCRIPTS / fname
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    try:
        ast.parse(content)
        n = len(content.splitlines())
        print('OK  ' + fname + ' (' + str(n) + ' lines)')
        return True
    except SyntaxError as e:
        print('FAIL ' + fname + ' line ' + str(e.lineno) + ': ' + str(e.msg))
        return False

# FILE 1: backtest -- body of "with tab1:" lines 210-934 (0-indexed 209-933)
bt_lines = dedent_fixed(all_lines[209:934], 4)
bt_content = HEADER + ''.join(bt_lines)
write_and_check('old_strategy_backtest.py', bt_content)

# FILE 2: walkforward -- lines 2525-2889 (0-indexed), dedent auto
wf_lines = dedent_auto(all_lines[2525:2890])
wf_content = HEADER + ''.join(wf_lines)
write_and_check('old_strategy_walkforward.py', wf_content)

# FILE 3: rl trainer -- body of "with tab_rl:" lines 3465 to end (0-indexed 3464+1)
rl_lines = dedent_auto(all_lines[3465:])
rl_content = HEADER + ''.join(rl_lines)
write_and_check('old_strategy_rl.py', rl_content)

# Duplicate form key check
all_forms = []
for fname in ['old_strategy_backtest.py', 'old_strategy_walkforward.py', 'old_strategy_rl.py']:
    src = open(SCRIPTS / fname, encoding='utf-8', errors='replace').read()
    all_forms += re.findall(r'st\.form\s*\(\s*["\']([^"\']+)["\']', src)

counts = Counter(all_forms)
dups = {k: v for k, v in counts.items() if v > 1}
print('Duplicate form keys: ' + (str(dups) if dups else 'none'))
print('Done.')