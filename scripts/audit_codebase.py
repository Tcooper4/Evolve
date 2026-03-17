"""
Full codebase audit for Evolve platform - Session 34
Run: .\evolve_venv\Scripts\python.exe scripts\audit_codebase.py
Output: scripts\audit_codebase_report.txt
"""
import os
import ast
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(r'C:\Users\Thomas\OneDrive\Desktop\Dashboard\evolve_clean')
REPORT = ROOT / 'scripts' / 'audit_codebase_report.txt'

SKIP_DIRS = {
    'evolve_venv', '__pycache__', '.git', '.streamlit',
    'node_modules', '.cache', 'dist', 'build', '.pytest_cache'
}

out_lines = []

def w(line=''):
    out_lines.append(str(line))
    print(line)

def section(title):
    w()
    w('=' * 70)
    w('  ' + title)
    w('=' * 70)

def get_all_py_files():
    result = []
    for path in ROOT.rglob('*.py'):
        parts = set(path.parts)
        if any(skip in parts for skip in SKIP_DIRS):
            continue
        result.append(path)
    return sorted(result)

def parse_file(path):
    try:
        src = path.read_text(encoding='utf-8', errors='replace')
        tree = ast.parse(src)
        return src, tree, None
    except SyntaxError as e:
        return None, None, str(e)
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

def get_functions(tree):
    return [node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef,
                                  ast.AsyncFunctionDef))]

def get_classes(tree):
    return [node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)]

# ── COLLECT ALL FILES ─────────────────────────────────────────────
w('Starting codebase audit...')
all_files = get_all_py_files()

section('1. COMPLETE FILE INVENTORY')
w('Total Python files found: ' + str(len(all_files)))
w()

by_dir = defaultdict(list)
for f in all_files:
    rel = f.relative_to(ROOT)
    top = str(rel.parts[0]) if len(rel.parts) > 1 else 'root'
    by_dir[top].append(f)

w('Files by top-level directory:')
for d, files in sorted(by_dir.items()):
    total_size = sum(f.stat().st_size for f in files)
    w('  ' + d.ljust(30) + str(len(files)).rjust(5) +
      ' files  ' + str(round(total_size/1024, 1)).rjust(8) + ' KB')

# ── SYNTAX ERRORS ─────────────────────────────────────────────────
section('2. SYNTAX ERRORS')
syntax_errors = []
for f in all_files:
    src, tree, err = parse_file(f)
    if err:
        rel = str(f.relative_to(ROOT))
        syntax_errors.append((rel, err))
        w('  FAIL  ' + rel)
        w('        ' + err)
if not syntax_errors:
    w('  No syntax errors found.')
w('Total files with syntax errors: ' + str(len(syntax_errors)))

# ── EMPTY FILES ────────────────────────────────────────────────────
section('3. EMPTY FILES (0 bytes - safe to delete)')
empty = []
for f in all_files:
    if f.stat().st_size == 0:
        empty.append(str(f.relative_to(ROOT)))
        w('  ' + str(f.relative_to(ROOT)))
if not empty:
    w('  No empty files.')
w('Total empty: ' + str(len(empty)))

# ── VERY SMALL FILES ──────────────────────────────────────────────
section('4. VERY SMALL FILES under 50 bytes (likely stubs)')
small = []
for f in all_files:
    size = f.stat().st_size
    if 0 < size < 50:
        src = f.read_text(encoding='utf-8', errors='replace')
        small.append(str(f.relative_to(ROOT)))
        w('  ' + str(f.relative_to(ROOT)).ljust(60) +
          str(size).rjust(6) + ' bytes')
        w('    Content: ' + src.strip()[:80])
w('Total small stubs: ' + str(len(small)))

# ── SCRIPTS DIRECTORY ─────────────────────────────────────────────
section('5. SCRIPTS DIRECTORY - full inventory')
scripts_dir = ROOT / 'scripts'
if scripts_dir.exists():
    script_files = sorted(scripts_dir.glob('*.py'))
    w('Total scripts: ' + str(len(script_files)))
    w()
    for f in script_files:
        size = f.stat().st_size
        name = f.name
        if size == 0:
            tag = 'EMPTY'
        elif 'verify_s' in name:
            tag = 'verify script - keep'
        elif 'audit_' in name:
            tag = 'audit script - keep'
        elif 'patch_' in name:
            tag = 'patch script - can delete post-deploy'
        elif name.startswith('old_'):
            tag = 'old page backup - needed by runpy'
        elif 'fetch_' in name:
            tag = 'fetch script - can delete post-deploy'
        elif 'fix_' in name:
            tag = 'fix script - can delete post-deploy'
        elif 'rebuild_' in name:
            tag = 'rebuild script - can delete post-deploy'
        elif 'split_' in name:
            tag = 'split script - can delete post-deploy'
        elif 'check_' in name:
            tag = 'check script - keep or delete'
        elif 'diag_' in name:
            tag = 'diagnostic script - can delete'
        elif 'build_' in name:
            tag = 'build script - can delete post-deploy'
        else:
            tag = 'other - review manually'
        w('  ' + name.ljust(48) +
          str(round(size/1024, 1)).rjust(7) + ' KB  [' + tag + ']')

# ── PAGES INVENTORY ───────────────────────────────────────────────
section('6. PAGES DIRECTORY')
pages_dir = ROOT / 'pages'
if pages_dir.exists():
    for f in sorted(pages_dir.glob('*.py')):
        size = f.stat().st_size
        src, tree, err = parse_file(f)
        if err:
            status = 'SYNTAX ERROR'
        elif size == 0:
            status = 'EMPTY'
        else:
            lines = len(src.splitlines()) if src else 0
            status = str(lines) + ' lines'
        w('  ' + f.name.ljust(35) +
          str(round(size/1024, 1)).rjust(8) + ' KB  ' + status)

# ── TRADING MODULE MAP ────────────────────────────────────────────
section('7. TRADING MODULE MAP')
trading_dir = ROOT / 'trading'
if trading_dir.exists():
    trading_files = [f for f in trading_dir.rglob('*.py')
                     if '__pycache__' not in str(f)]
    w('Total trading modules: ' + str(len(trading_files)))
    w()
    for f in sorted(trading_files):
        src, tree, err = parse_file(f)
        if not tree:
            continue
        classes = get_classes(tree)
        funcs = [fn for fn in get_functions(tree)
                 if not fn.startswith('_')]
        rel = str(f.relative_to(ROOT))
        size = f.stat().st_size
        w('  ' + rel)
        w('    Size: ' + str(round(size/1024, 1)) + ' KB')
        if classes:
            w('    Classes: ' + ', '.join(classes))
        if funcs:
            display = funcs[:8]
            more = len(funcs) - 8
            w('    Public fns: ' + ', '.join(display) +
              ('..+' + str(more) + ' more' if more > 0 else ''))

# ── WHAT PAGES USE ────────────────────────────────────────────────
section('8. WHAT EACH PAGE IMPORTS FROM TRADING')
if pages_dir.exists():
    for f in sorted(pages_dir.glob('*.py')):
        src, tree, err = parse_file(f)
        if not tree:
            continue
        imports = get_imports(tree)
        trading_imports = sorted(set(
            i for i in imports
            if i.startswith(('trading.', 'agents.',
                             'components.', 'config.',
                             'ui.'))))
        w('  ' + f.name)
        for imp in trading_imports:
            w('    -> ' + imp)

# ── BROKEN IMPORTS ────────────────────────────────────────────────
section('9. POTENTIALLY BROKEN IMPORTS')
w('Checking if imported internal modules exist on disk...')
broken = []
check_files = list(pages_dir.glob('*.py')) if pages_dir.exists() else []
check_files.append(ROOT / 'app.py')
for tf in trading_dir.rglob('*.py') if trading_dir.exists() else []:
    if '__pycache__' not in str(tf):
        check_files.append(tf)

seen_broken = set()
for f in check_files:
    src, tree, err = parse_file(f)
    if not tree:
        continue
    rel = str(f.relative_to(ROOT))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if mod.startswith(('trading.', 'agents.',
                               'components.', 'config.',
                               'ui.')):
                mod_as_path = ROOT / Path(
                    mod.replace('.', os.sep) + '.py')
                mod_as_dir = ROOT / Path(
                    mod.replace('.', os.sep))
                if (not mod_as_path.exists()
                        and not mod_as_dir.exists()):
                    key = (rel, mod)
                    if key not in seen_broken:
                        seen_broken.add(key)
                        broken.append(key)
                        w('  ' + rel)
                        w('    missing: from ' +
                          mod + ' import ...')

if not broken:
    w('  No broken internal imports found.')
w('Total broken imports: ' + str(len(broken)))

# ── LARGEST FILES ─────────────────────────────────────────────────
section('10. ALL FILES BY SIZE (largest first)')
sized = sorted(
    [(f.stat().st_size, f) for f in all_files],
    reverse=True
)
for size, f in sized:
    rel = str(f.relative_to(ROOT))
    w('  ' + str(round(size/1024, 1)).rjust(10) +
      ' KB  ' + rel)

# ── SUMMARY ───────────────────────────────────────────────────────
section('AUDIT SUMMARY')
w('Total Python files:          ' + str(len(all_files)))
w('Syntax errors:               ' + str(len(syntax_errors)))
w('Empty files (0 bytes):       ' + str(len(empty)))
w('Stub files (<50 bytes):      ' + str(len(small)))
w('Broken internal imports:     ' + str(len(broken)))
w()
w('ACTION REQUIRED:')
if syntax_errors:
    w('  Fix ' + str(len(syntax_errors)) + ' syntax errors')
if empty:
    w('  Delete ' + str(len(empty)) + ' empty files')
if broken:
    w('  Investigate ' + str(len(broken)) +
      ' broken imports')
w()
w('Report saved to:')
w('  ' + str(REPORT))

# Write report file
with open(REPORT, 'w', encoding='utf-8') as rep:
    rep.write('\n'.join(out_lines))

print()
print('Done. Report written to: ' + str(REPORT))