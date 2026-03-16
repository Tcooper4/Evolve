# check_duplicate_keys.py -- finds duplicate st.form keys and
# widget keys across the old source files used by each page.
# Run: .\evolve_venv\Scripts\python.exe scripts\check_duplicate_keys.py
import sys, os, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')

# Map each new page to the old files it runs via runpy
PAGE_SOURCES = {
    'pages/4_Trade.py': [
        'old_4_Trade_Execution.py',
        'old_5_Portfolio.py',
        'old_7_Performance.py',
        'old_6_Risk_Management.py',
    ],
    'pages/5_Backtest.py': [
        'old_strategy_backtest.py',
        'old_strategy_walkforward.py',
        'old_strategy_rl.py',
        'old_9_Reports.py',
    ],
    'pages/7_Settings.py': [
        'old_10_Alerts.py',
        'old_11_Admin.py',
    ],
}

KEY_PATTERN = re.compile(r'key\s*=\s*["\']([^"\']+)["\']')
FORM_PATTERN = re.compile(r'st\.form\s*\(\s*["\']([^"\']+)["\']')

def extract_keys(src):
    """Return list of all key= values found in source."""
    return KEY_PATTERN.findall(src)

def extract_form_keys(src):
    """Return list of all st.form() key values."""
    return FORM_PATTERN.findall(src)

print("=" * 60)
print("Duplicate key check across runpy source files")
print("=" * 60)

all_clean = True

for page, source_files in PAGE_SOURCES.items():
    print(f"\n{page}")
    print("-" * 50)

    all_keys = []
    all_form_keys = []
    file_keys = {}

    for fname in source_files:
        fpath = os.path.join(SCRIPTS, fname)
        if not os.path.exists(fpath):
            print(f"  MISSING: {fname}")
            continue
        with open(fpath, encoding='utf-8', errors='replace') as f:
            src = f.read()
        keys = extract_keys(src)
        form_keys = extract_form_keys(src)
        file_keys[fname] = {'keys': keys, 'forms': form_keys}
        all_keys.extend(keys)
        all_form_keys.extend(form_keys)
        print(f"  {fname}: {len(keys)} widget keys, {len(form_keys)} form keys")

    # Find duplicates across files
    from collections import Counter
    key_counts = Counter(all_keys)
    form_counts = Counter(all_form_keys)

    dup_keys = {k: v for k, v in key_counts.items() if v > 1}
    dup_forms = {k: v for k, v in form_counts.items() if v > 1}

    if dup_forms:
        all_clean = False
        print(f"\n  DUPLICATE FORM KEYS ({len(dup_forms)}):")
        for k, count in sorted(dup_forms.items()):
            print(f"    '{k}' appears {count} times")
            # Show which files
            for fname, data in file_keys.items():
                if k in data['forms']:
                    print(f"      -> {fname}")
    else:
        print("  No duplicate form keys")

    if dup_keys:
        # Only show keys that appear 3+ times or are form-adjacent
        # 2 appearances might be legitimate (e.g. reading and writing)
        serious_dups = {k: v for k, v in dup_keys.items() if v >= 3}
        if serious_dups:
            all_clean = False
            print(f"\n  SERIOUS DUPLICATE WIDGET KEYS (3+ occurrences):")
            for k, count in sorted(serious_dups.items()):
                print(f"    '{k}' appears {count} times")
        minor_dups = {k: v for k, v in dup_keys.items() if v == 2}
        if minor_dups:
            print(f"\n  Minor duplicate widget keys (2 occurrences, may be OK):")
            for k, count in sorted(minor_dups.items()):
                print(f"    '{k}'")
    else:
        print("  No duplicate widget keys")

print("\n" + "=" * 60)
if all_clean:
    print("  All pages clean -- no duplicate form keys found.")
else:
    print("  ACTION REQUIRED -- fix duplicate keys before running app.")
print("=" * 60)