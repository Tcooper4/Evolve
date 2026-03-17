"""
Clean codebase audit - excludes venv, focuses on actual code.
Run: .\evolve_venv\Scripts\python.exe scripts\audit_clean.py
Output: scripts\audit_clean_report.txt
"""
import os
import ast
from pathlib import Path
from collections import defaultdict

ROOT = Path(r'C:\Users\Thomas\OneDrive\Desktop\Dashboard\evolve_clean')
REPORT = ROOT / 'scripts' / 'audit_clean_report.txt'

# Directories that are NOT our code
SKIP_DIRS = {
    'evolve_venv', '__pycache__', '.git', '.venv',
    'venv', 'env', 'node_modules', '.cache',
    'dist', 'build', '.pytest_cache', 'htmlcov',
    'site-packages', '.tox',
}

# Directories that ARE our code
OUR_DIRS = {
    'trading', 'agents', 'components', 'config',
    'core', 'pages', 'scripts', 'tests', 'ui',
    'utils', 'data', 'execution', 'features',
    'interface', 'llm', 'memory', 'meta',
    'meta_learning', 'models', 'monitoring', 'nlp',
    'portfolio', 'reporting', 'risk', 'rl',
    'routing', 'strategies', 'system', 'testing',
    'tools', 'visualization', 'automate', 'causal',
    'dashboard', 'evaluation', 'market_analysis',
}

out = []

def w(line=''):
    out.append(str(line))
    print(line)

def section(title):
    w()
    w('=' * 70)
    w('  ' + title)
    w('=' * 70)

def is_our_file(path):
    """Check if this file belongs to our codebase."""
    parts = path.relative_to(ROOT).parts
    # Skip if any part is a venv/cache dir
    for part in parts:
        if part in SKIP_DIRS:
            return False
        if part.endswith('.egg-info'):
            return False
    # Root level files (app.py etc)
    if len(parts) == 1:
        return True
    # Must be in one of our dirs
    return parts[0] in OUR_DIRS or parts[0] == 'pages'

def get_our_files():
    result = []
    for path in ROOT.rglob('*.py'):
        if is_our_file(path):
            result.append(path)
    return sorted(result)

def parse_file(path):
    try:
        src = path.read_text(encoding='utf-8', errors='replace')
        tree = ast.parse(src)
        return src, tree, None
    except SyntaxError as e:
        return None, None, f'line {e.lineno}: {e.msg}'
    except Exception as e:
        return None, None, str(e)

def get_imports(tree):
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports

def get_classes(tree):
    return [n.name for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef)]

def get_public_functions(tree):
    return [n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef,
                               ast.AsyncFunctionDef))
            and not n.name.startswith('_')]

# ── COLLECT FILES ─────────────────────────────────────────────────
w('Collecting files...')
all_files = get_our_files()
w('Total files in codebase: ' + str(len(all_files)))

section('1. FILES BY DIRECTORY')
by_dir = defaultdict(list)
for f in all_files:
    rel = f.relative_to(ROOT)
    top = str(rel.parts[0]) if len(rel.parts) > 1 else 'root'
    by_dir[top].append(f)

total_kb = sum(f.stat().st_size for f in all_files) / 1024
w('Total codebase size: ' + str(round(total_kb, 1)) + ' KB')
w()
for d, files in sorted(by_dir.items(),
                        key=lambda x: -len(x[1])):
    size_kb = sum(f.stat().st_size for f in files) / 1024
    w('  ' + d.ljust(25) + str(len(files)).rjust(5) +
      ' files  ' + str(round(size_kb, 1)).rjust(8) + ' KB')

# ── SYNTAX ERRORS ─────────────────────────────────────────────────
section('2. SYNTAX ERRORS IN OUR CODE')
syntax_errors = []
for f in all_files:
    src, tree, err = parse_file(f)
    if err:
        rel = str(f.relative_to(ROOT))
        syntax_errors.append((rel, err))

if syntax_errors:
    for rel, err in syntax_errors:
        w('  FAIL  ' + rel)
        w('        ' + err)
else:
    w('  No syntax errors in codebase.')
w()
w('Total: ' + str(len(syntax_errors)) + ' syntax errors')

# ── EMPTY FILES ────────────────────────────────────────────────────
section('3. EMPTY FILES IN OUR CODE')
empty_files = []
for f in all_files:
    if f.stat().st_size == 0:
        rel = str(f.relative_to(ROOT))
        empty_files.append(rel)
        w('  ' + rel)
if not empty_files:
    w('  No empty files.')
w('Total: ' + str(len(empty_files)))

# ── SCRIPTS INVENTORY ─────────────────────────────────────────────
section('4. SCRIPTS DIRECTORY - WHAT TO KEEP VS DELETE')
scripts_dir = ROOT / 'scripts'
keep = []
delete_candidates = []
runpy_sources = []

if scripts_dir.exists():
    for f in sorted(scripts_dir.glob('*.py')):
        size = f.stat().st_size
        name = f.name

        if size == 0:
            tag = 'DELETE - empty'
            delete_candidates.append(name)
        elif 'null bytes' in name or name == 'old_8_Model_Lab.py':
            tag = 'DELETE - corrupt'
            delete_candidates.append(name)
        elif name.startswith('old_') and not any(
                x in name for x in ['strategy_backtest',
                                     'strategy_walkforward',
                                     'strategy_rl']):
            tag = 'KEEP - runpy source for pages'
            runpy_sources.append(name)
            keep.append(name)
        elif name.startswith('old_strategy'):
            tag = 'KEEP - runpy source (split)'
            runpy_sources.append(name)
            keep.append(name)
        elif name.startswith('verify_s'):
            tag = 'KEEP - verification script'
            keep.append(name)
        elif name.startswith('audit_'):
            tag = 'KEEP - audit script'
            keep.append(name)
        elif name.startswith(('patch_', 'fix_',
                               'rebuild_', 'split_',
                               'fetch_', 'diag_',
                               'build_', 'check_')):
            tag = 'DELETE - one-time fix script'
            delete_candidates.append(name)
        else:
            tag = 'REVIEW - other'

        w('  ' + name.ljust(48) +
          str(round(size/1024, 1)).rjust(7) +
          ' KB  [' + tag + ']')

w()
w('Scripts to DELETE (' +
  str(len(delete_candidates)) + '):')
for n in delete_candidates:
    w('    del scripts\\' + n)
w()
w('Scripts to KEEP (' + str(len(keep)) + '):')
for n in keep:
    w('    ' + n)

# ── TRADING MODULES FULL MAP ───────────────────────────────────────
section('5. TRADING MODULE FULL MAP')
trading_dir = ROOT / 'trading'
if trading_dir.exists():
    trading_files = sorted([
        f for f in trading_dir.rglob('*.py')
        if '__pycache__' not in str(f)
    ])
    w('Total trading modules: ' + str(len(trading_files)))
    w()
    for f in trading_files:
        src, tree, err = parse_file(f)
        rel = str(f.relative_to(ROOT))
        size = f.stat().st_size
        if err:
            w('  [ERROR] ' + rel + ' -- ' + err)
            continue
        classes = get_classes(tree) if tree else []
        funcs = get_public_functions(tree) if tree else []
        lines = len(src.splitlines()) if src else 0
        w('  ' + rel)
        w('    ' + str(lines) + ' lines, ' +
          str(round(size/1024, 1)) + ' KB')
        if classes:
            w('    Classes: ' + ', '.join(classes))
        if funcs:
            shown = funcs[:6]
            extra = len(funcs) - 6
            w('    Fns: ' + ', '.join(shown) +
              (' +' + str(extra) + ' more' if extra > 0
               else ''))

# ── BROKEN IMPORTS ────────────────────────────────────────────────
section('6. BROKEN INTERNAL IMPORTS')
broken = []
check_files = list(all_files)
seen = set()

for f in check_files:
    src, tree, err = parse_file(f)
    if not tree:
        continue
    rel = str(f.relative_to(ROOT))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        mod = node.module or ''
        if not mod.startswith(('trading.', 'agents.',
                               'components.', 'config.',
                               'ui.', 'core.')):
            continue
        # Check if module file/dir exists
        mod_file = ROOT / Path(mod.replace('.', os.sep) + '.py')
        mod_dir = ROOT / Path(mod.replace('.', os.sep))
        if not mod_file.exists() and not mod_dir.exists():
            key = mod
            if key not in seen:
                seen.add(key)
                broken.append((rel, mod))
                w('  ' + rel)
                w('    -> missing: ' + mod)

if not broken:
    w('  No broken internal imports detected.')
w('Total: ' + str(len(broken)) + ' broken imports')

# ── PAGES WHAT THEY USE ───────────────────────────────────────────
section('7. PAGES - FULL IMPORT MAP')
pages_dir = ROOT / 'pages'
if pages_dir.exists():
    for f in sorted(pages_dir.glob('*.py')):
        src, tree, err = parse_file(f)
        if not tree:
            w('  ' + f.name + ': PARSE ERROR - ' + str(err))
            continue
        size = round(f.stat().st_size / 1024, 1)
        lines = len(src.splitlines()) if src else 0
        w()
        w('  ' + f.name + ' (' + str(lines) +
          ' lines, ' + str(size) + ' KB)')
        imports = sorted(set(get_imports(tree)))
        for imp in imports:
            if imp.startswith(('trading.', 'agents.',
                               'components.', 'config.',
                               'ui.')):
                # Check if it exists
                mod_file = ROOT / Path(
                    imp.replace('.', os.sep) + '.py')
                mod_dir = ROOT / Path(
                    imp.replace('.', os.sep))
                exists = mod_file.exists() or mod_dir.exists()
                marker = '  OK' if exists else '  MISSING'
                w('    [' + marker.strip() + '] ' + imp)

# ── TESTS INVENTORY ───────────────────────────────────────────────
section('8. TESTS DIRECTORY')
tests_dir = ROOT / 'tests'
if tests_dir.exists():
    test_files = sorted([
        f for f in tests_dir.rglob('*.py')
        if '__pycache__' not in str(f)
    ])
    w('Total test files: ' + str(len(test_files)))
    syntax_ok = 0
    syntax_fail = 0
    for f in test_files:
        src, tree, err = parse_file(f)
        rel = str(f.relative_to(ROOT))
        if err:
            syntax_fail += 1
            w('  FAIL  ' + rel + '  -- ' + err)
        else:
            syntax_ok += 1
    w()
    w('Tests parsing OK:   ' + str(syntax_ok))
    w('Tests with errors:  ' + str(syntax_fail))

# ── ALL FILES BY SIZE ─────────────────────────────────────────────
section('9. ALL OUR FILES BY SIZE')
sized = sorted(
    [(f.stat().st_size, f) for f in all_files],
    reverse=True
)
for size, f in sized:
    rel = str(f.relative_to(ROOT))
    w('  ' + str(round(size/1024, 1)).rjust(9) +
      ' KB  ' + rel)

# ── SUMMARY ───────────────────────────────────────────────────────
section('AUDIT SUMMARY')
w('Total files in our codebase: ' + str(len(all_files)))
w('Total size:                  ' +
  str(round(total_kb, 1)) + ' KB')
w('Syntax errors:               ' + str(len(syntax_errors)))
w('Empty files:                 ' + str(len(empty_files)))
w('Broken imports:              ' + str(len(broken)))
w('Scripts to delete:           ' +
  str(len(delete_candidates)))
w()
w('KEY ACTIONS:')
if syntax_errors:
    w('  1. Fix ' + str(len(syntax_errors)) +
      ' syntax errors (esp. tcn_model.py, shared_utilities.py)')
if empty_files:
    w('  2. Delete ' + str(len(empty_files)) +
      ' empty files')
if delete_candidates:
    w('  3. Delete ' + str(len(delete_candidates)) +
      ' one-time fix scripts')
if broken:
    w('  4. Investigate ' + str(len(broken)) +
      ' broken imports')
w()
w('Report: ' + str(REPORT))

with open(REPORT, 'w', encoding='utf-8') as rep:
    rep.write('\n'.join(out))

print('\nDone. Report: ' + str(REPORT))