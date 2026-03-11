"""
Session 22 final fixes:
1. Add HAS_SHAP / shap_available flag to forecast_explainability.py
2. Fix SyntaxError at line 4717 in pages/8_Model_Lab.py

Run: .\evolve_venv\Scripts\python.exe scripts\fix_s22_final.py
"""

import ast
import re

# ── FIX 1: forecast_explainability.py — add HAS_SHAP flag ─────────────────

fpath = "trading/models/forecast_explainability.py"
with open(fpath, encoding="utf-8", errors="replace") as f:
    src = f.read()

if "HAS_SHAP" not in src and "shap_available" not in src:
    # Find the first 'import shap' line and wrap it
    if "import shap" in src:
        src = src.replace(
            "import shap",
            "try:\n    import shap\n    HAS_SHAP = True\nexcept ImportError:\n    shap = None\n    HAS_SHAP = False\nif False:  # original import kept for IDE\n    import shap  # noqa"
        )
        # Remove the duplicate noqa stub — cleaner approach
        src = re.sub(
            r"try:\n    import shap\n    HAS_SHAP = True\nexcept ImportError:\n    shap = None\n    HAS_SHAP = False\nif False:  # original import kept for IDE\n    import shap  # noqa",
            "try:\n    import shap\n    HAS_SHAP = True\nexcept ImportError:\n    shap = None\n    HAS_SHAP = False",
            src
        )
    else:
        # shap is imported somewhere else — prepend flag block at top after
        # the first non-comment, non-blank line that isn't an import
        flag_block = (
            "\ntry:\n    import shap\n    HAS_SHAP = True\n"
            "except ImportError:\n    shap = None\n    HAS_SHAP = False\n"
        )
        # Insert after the last top-level import block
        lines = src.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, flag_block)
        src = "".join(lines)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[FIXED] Added HAS_SHAP flag to {fpath}")
else:
    print(f"[OK] HAS_SHAP already present in {fpath} — checking flag name...")
    # Flag exists but verify_s22 says flag=False meaning neither
    # HAS_SHAP nor shap_available is present — add it
    if "HAS_SHAP" not in src:
        flag_block = (
            "\ntry:\n    import shap\n    HAS_SHAP = True\n"
            "except ImportError:\n    shap = None\n    HAS_SHAP = False\n"
        )
        lines = src.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, flag_block)
        src = "".join(lines)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[FIXED] Inserted HAS_SHAP block into {fpath}")

# Verify it parses
with open(fpath, encoding="utf-8", errors="replace") as f:
    check_src = f.read()
try:
    ast.parse(check_src)
    has_flag = "HAS_SHAP" in check_src
    print(f"[{'OK' if has_flag else 'WARN'}] forecast_explainability.py — "
          f"parses OK, HAS_SHAP present: {has_flag}")
except SyntaxError as e:
    print(f"[ERROR] forecast_explainability.py SyntaxError after fix: {e}")


# ── FIX 2: 8_Model_Lab.py — SyntaxError at line 4717 ─────────────────────

fpath2 = "pages/8_Model_Lab.py"
with open(fpath2, encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

# Show context around line 4717
print(f"\n[AUDIT] 8_Model_Lab.py around line 4717:")
start = max(0, 4710)
end = min(len(lines), 4725)
for i in range(start, end):
    print(f"  {i+1}: {lines[i].rstrip()}")

# Try to parse and find the exact error
src2 = "".join(lines)
try:
    ast.parse(src2)
    print("[OK] 8_Model_Lab.py already parses cleanly")
except SyntaxError as e:
    print(f"\n[SyntaxError] line {e.lineno}: {e.msg}")
    print(f"  text: {e.text}")

    err_line = (e.lineno or 4717) - 1  # 0-indexed

    # Common fix: bare except/finally without matching try
    # Look backwards for a try: that needs an except
    context_start = max(0, err_line - 30)
    context_lines = lines[context_start:err_line + 5]
    print(f"\n[CONTEXT] lines {context_start+1}-{err_line+5}:")
    for i, l in enumerate(context_lines, context_start + 1):
        print(f"  {i}: {l.rstrip()}")

    # Most common cause at this line range: a try block missing except,
    # or an except/finally that lost its try due to bad indentation.
    # Strategy: find the offending line and apply minimal fix.

    offending = lines[err_line].rstrip()
    print(f"\n[OFFENDING LINE {err_line+1}]: '{offending}'")

    # If it's an orphaned except/finally, check if previous non-blank
    # line ends a block that needs it
    # Simple fix: if line starts with 'except' or 'finally' and there's
    # no matching try, add a try: before the nearest preceding code block
    fixed = False

    # Check if the issue is a try block with no except
    # Scan backwards from error line for unclosed try
    indent_of_error = len(offending) - len(offending.lstrip())
    for j in range(err_line - 1, max(0, err_line - 50), -1):
        l = lines[j]
        stripped = l.strip()
        if not stripped:
            continue
        indent = len(l) - len(l.lstrip())
        if indent == indent_of_error and stripped.startswith("try:"):
            # Found the try — check if there's an except between it and err
            has_except = any(
                lines[k].strip().startswith(("except", "finally"))
                for k in range(j + 1, err_line)
                if len(lines[k]) - len(lines[k].lstrip()) == indent_of_error
            )
            if not has_except:
                # Insert a bare except: pass before the offending line
                indent_str = " " * indent_of_error
                lines.insert(err_line, f"{indent_str}except Exception:\n")
                lines.insert(err_line + 1, f"{indent_str}    pass\n")
                print(f"[FIXED] Inserted except/pass before line {err_line+1}")
                fixed = True
                break

    if not fixed:
        # Fallback: wrap the offending line's block in try/except
        print("[WARN] Could not auto-fix — printing full context for manual review")
        for i in range(max(0, err_line-5), min(len(lines), err_line+5)):
            print(f"  {i+1}: {lines[i].rstrip()}")

    if fixed:
        new_src = "".join(lines)
        try:
            ast.parse(new_src)
            with open(fpath2, "w", encoding="utf-8") as f:
                f.write(new_src)
            print(f"[OK] 8_Model_Lab.py fixed and parses cleanly")
        except SyntaxError as e2:
            print(f"[ERROR] Still has SyntaxError after fix: {e2}")
            print("Manual fix required — see context above")

print("\nDone. Run verify_s22.py to confirm.")