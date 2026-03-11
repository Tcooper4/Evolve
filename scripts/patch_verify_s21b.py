"""
Patches verify_s21.py to find the FIRST movers assignment, not the last.
Run: .\evolve_venv\Scripts\python.exe scripts\patch_verify_s21b.py
"""

with open("scripts/verify_s21.py", encoding="utf-8", errors="replace") as f:
    src = f.read()

old = '''    movers_assign = None
    movers_use = None
    for i, line in enumerate(home_lines, 1):
        stripped = line.strip()
        if "movers" in stripped and "=" in stripped and "generate_morning_briefing" not in stripped and not stripped.startswith("#"):
            movers_assign = i
        if "generate_morning_briefing" in stripped and "movers" in stripped:
            movers_use = i'''

new = '''    movers_assign = None  # first assignment
    movers_use = None
    for i, line in enumerate(home_lines, 1):
        stripped = line.strip()
        # capture only the FIRST assignment (don't overwrite with later ones)
        if (movers_assign is None and "movers" in stripped and "=" in stripped
                and "generate_morning_briefing" not in stripped
                and not stripped.startswith("#")):
            movers_assign = i
        if "generate_morning_briefing" in stripped and "movers" in stripped:
            movers_use = i'''

if old in src:
    src = src.replace(old, new)
    with open("scripts/verify_s21.py", "w", encoding="utf-8") as f:
        f.write(src)
    print("verify_s21.py patched successfully")
else:
    print("WARNING: exact block not found — printing movers-related lines:")
    for i, line in enumerate(src.splitlines(), 1):
        if "movers_assign" in line or "movers_use" in line:
            print(f"  line {i}: {line}")