# split_strategy.py -- splits old_3_Strategy_Testing.py into
# three focused files for the Backtest page tabs.
# Run: .\evolve_venv\Scripts\python.exe scripts\split_strategy.py
import sys, os, io, ast
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')
SRC = os.path.join(SCRIPTS, 'old_3_Strategy_Testing.py')

with open(SRC, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

total = len(lines)
print(f"Source file: {total} lines")

# ── Extract the shared header (imports + setup, lines 1-192) ─────
# Line 193 is st.title(...) -- we want everything before that
header_end = 192  # 0-indexed: lines 0..191

header_lines = lines[:header_end]

# Clean header: remove set_page_config, inject_theme, st.title,
# the tab creation block (line 197), and sys.path manipulation
skip_patterns = [
    'st.set_page_config(',
    'inject_theme(',
    'st.title(',
    'tab1, tab2,',
    'tab_research',
    'tab_rl',
]

def clean_header(raw_lines):
    out = []
    for line in raw_lines:
        if any(p in line for p in skip_patterns):
            continue
        if 'sys.path' in line and ('insert' in line or 'append' in line):
            continue
        if 'project_root' in line and ('Path(' in line or 'dirname' in line):
            continue
        out.append(line)
    return out

clean_hdr = clean_header(header_lines)

# ── Section boundaries (1-indexed line numbers from Select-String)
# Backtest section: line 210 to 935 (tab1 Quick Backtest through end of tab2)
# The structure inside old page:
#   line 210: st.header("Quick Backtest")  -- inside "with tab1:"
#   line 741: walk-forward subsection inside tab1
#   line 936: st.header("Strategy Builder") -- tab2 starts
# We take lines 209..935 (0-indexed) for the backtest tab
# but we need the "with tab1:" wrapper too -- it's around line 209
# Let's take from line 205 (the with tab1: block) to line 935

# Find exact line indices for key markers
def find_line(pattern, start=0):
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

tab1_start    = find_line('with tab1:')
tab2_start    = find_line('with tab2:')
tab6_start    = find_line('with tab6:')   # Advanced Analysis
tab_rl_start  = find_line('with tab_rl:')

print(f"tab1 (Quick Backtest) starts at line {tab1_start+1}")
print(f"tab2 (Strategy Builder) starts at line {tab2_start+1}")
print(f"tab6 (Advanced Analysis) starts at line {tab6_start+1}")
print(f"tab_rl starts at line {tab_rl_start+1}")

# Walk-forward is inside tab6 (Advanced Analysis) around line 2531
wf_start = find_line('Walk-For', tab6_start) if tab6_start >= 0 else -1
mc_start = find_line('Monte Carlo', tab6_start) if tab6_start >= 0 else -1
print(f"Walk-Forward subheader at line {wf_start+1}")
print(f"Monte Carlo subheader at line {mc_start+1}")

# ── FILE 1: old_strategy_backtest.py ────────────────────────────
# Content: imports/setup + tab1 body (Quick Backtest, lines tab1_start..tab2_start-1)
# We extract the BODY of tab1 (dedented by 4 spaces) to avoid
# the "with tab1:" wrapper since it'll be inside "with tab_bt:"

def extract_tab_body(start_idx, end_idx):
    """Extract lines inside a 'with tabN:' block, dedented."""
    body = lines[start_idx+1:end_idx]  # skip the "with tabN:" line itself
    # Dedent by 4 spaces if all lines are indented
    out = []
    for line in body:
        if line.startswith('    '):
            out.append(line[4:])
        else:
            out.append(line)
    return out

if tab1_start >= 0 and tab2_start >= 0:
    bt_body = extract_tab_body(tab1_start, tab2_start)
else:
    bt_body = []
    print("WARNING: Could not find tab1/tab2 boundaries")

bt_content = ''.join(clean_hdr) + '\n' + ''.join(bt_body)
bt_path = os.path.join(SCRIPTS, 'old_strategy_backtest.py')
with open(bt_path, 'w', encoding='utf-8') as f:
    f.write(bt_content)
bt_lines = len(bt_content.splitlines())
print(f"\nWrote old_strategy_backtest.py ({bt_lines} lines)")
try:
    ast.parse(bt_content)
    print("  Parses OK")
except SyntaxError as e:
    print(f"  SyntaxError line {e.lineno}: {e.msg}")

# ── FILE 2: old_strategy_walkforward.py ─────────────────────────
# Walk-Forward subsection from tab6 (Advanced Analysis)
# Lines from wf_start to mc_start-1 (just the WF part, not Monte Carlo)
if wf_start >= 0 and mc_start >= 0:
    wf_raw = lines[wf_start:mc_start]
elif wf_start >= 0:
    wf_raw = lines[wf_start:tab_rl_start if tab_rl_start >= 0 else len(lines)]
else:
    wf_raw = []
    print("WARNING: Could not find Walk-Forward section")

# Dedent to top level (these lines are deeply indented inside tab6)
def dedent_to_toplevel(raw):
    if not raw:
        return []
    # Find minimum indentation
    min_indent = 999
    for line in raw:
        stripped = line.lstrip()
        if stripped:
            min_indent = min(min_indent, len(line) - len(stripped))
    if min_indent == 999:
        min_indent = 0
    return [line[min_indent:] if len(line) > min_indent else line for line in raw]

wf_body = dedent_to_toplevel(wf_raw)
wf_content = ''.join(clean_hdr) + '\n' + ''.join(wf_body)
wf_path = os.path.join(SCRIPTS, 'old_strategy_walkforward.py')
with open(wf_path, 'w', encoding='utf-8') as f:
    f.write(wf_content)
wf_lines = len(wf_content.splitlines())
print(f"\nWrote old_strategy_walkforward.py ({wf_lines} lines)")
try:
    ast.parse(wf_content)
    print("  Parses OK")
except SyntaxError as e:
    print(f"  SyntaxError line {e.lineno}: {e.msg}")

# ── FILE 3: old_strategy_rl.py ───────────────────────────────────
# RL Trainer: from tab_rl_start to end of file
if tab_rl_start >= 0:
    rl_raw = lines[tab_rl_start+1:]  # skip "with tab_rl:" line
    rl_body = dedent_to_toplevel(rl_raw)
else:
    rl_body = []
    print("WARNING: Could not find tab_rl section")

rl_content = ''.join(clean_hdr) + '\n' + ''.join(rl_body)
rl_path = os.path.join(SCRIPTS, 'old_strategy_rl.py')
with open(rl_path, 'w', encoding='utf-8') as f:
    f.write(rl_content)
rl_lines = len(rl_content.splitlines())
print(f"\nWrote old_strategy_rl.py ({rl_lines} lines)")
try:
    ast.parse(rl_content)
    print("  Parses OK")
except SyntaxError as e:
    print(f"  SyntaxError line {e.lineno}: {e.msg}")

print("\nDone. Update pages/5_Backtest.py to use:")
print("  tab_bt  -> scripts/old_strategy_backtest.py")
print("  tab_wf  -> scripts/old_strategy_walkforward.py")
print("  tab_rl  -> scripts/old_strategy_rl.py")
print("  tab_rep -> scripts/old_9_Reports.py  (unchanged)")