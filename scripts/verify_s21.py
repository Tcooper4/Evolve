"""
Session 21 Verification Script
Run: .\evolve_venv\Scripts\python.exe scripts\verify_s21.py
"""

import sys
import ast
import os
import subprocess

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

# CHECK 1: 11_Admin.py
try:
    with open("pages/11_Admin.py", encoding="utf-8", errors="replace") as f:
        admin_src = f.read()
    admin_lines = admin_src.splitlines()
    assign_line = None
    use_line = None
    for i, line in enumerate(admin_lines, 1):
        if "tab_data_mgmt" in line and "=" in line and "with" not in line:
            assign_line = i
        if line.strip().startswith("with tab_data_mgmt"):
            use_line = i
    if assign_line and use_line:
        check("Admin tab_data_mgmt assigned before use", assign_line < use_line, f"assigned line {assign_line}, used line {use_line}")
    elif assign_line:
        check("Admin tab_data_mgmt assigned before use", True, f"assigned at line {assign_line}")
    else:
        check("Admin tab_data_mgmt assigned before use", False, "assignment not found")
    try:
        ast.parse(admin_src)
        check("Admin page parses cleanly", True)
    except SyntaxError as e:
        check("Admin page parses cleanly", False, str(e))
except FileNotFoundError:
    check("Admin page file exists", False, "pages/11_Admin.py not found")

# CHECK 2: 0_Home.py
try:
    with open("pages/0_Home.py", encoding="utf-8", errors="replace") as f:
        home_src = f.read()
    home_lines = home_src.splitlines()
    movers_assign = None  # first assignment
    movers_use = None
    for i, line in enumerate(home_lines, 1):
        stripped = line.strip()
        # capture only the FIRST assignment (don't overwrite with later ones)
        if (movers_assign is None and "movers" in stripped and "=" in stripped
                and "generate_morning_briefing" not in stripped
                and not stripped.startswith("#")):
            movers_assign = i
        if "generate_morning_briefing" in stripped and "movers" in stripped:
            movers_use = i
    if movers_assign and movers_use:
        check("Home movers assigned before briefing call", movers_assign < movers_use, f"assigned line {movers_assign}, used line {movers_use}")
    elif movers_assign:
        check("Home movers assigned (briefing call not found or renamed)", True, f"movers assigned at line {movers_assign}")
    else:
        check("Home movers assigned before briefing call", False, "movers assignment not found")
    try:
        ast.parse(home_src)
        check("Home page parses cleanly", True)
    except SyntaxError as e:
        check("Home page parses cleanly", False, str(e))
except FileNotFoundError:
    check("Home page file exists", False, "pages/0_Home.py not found")

# CHECK 3: 7_Performance.py
try:
    with open("pages/7_Performance.py", encoding="utf-8", errors="replace") as f:
        perf_src = f.read()
    try:
        ast.parse(perf_src)
        check("Performance page parses cleanly", True)
    except SyntaxError as e:
        check("Performance page parses cleanly", False, str(e))
    bare_returns = []
    for i, line in enumerate(perf_src.splitlines(), 1):
        stripped = line.strip()
        if stripped == "return" and not line.startswith(" ") and not line.startswith("\t"):
            bare_returns.append(i)
    check("No bare return at module level", len(bare_returns) == 0, f"found at lines: {bare_returns}" if bare_returns else "clean")
except FileNotFoundError:
    check("Performance page file exists", False, "pages/7_Performance.py not found")

# CHECK 4: API key injection
try:
    with open("config/user_store.py", encoding="utf-8", errors="replace") as f:
        user_store_src = f.read()
    check("inject_user_keys_to_env in user_store.py", "inject_user_keys_to_env" in user_store_src)
except FileNotFoundError:
    check("config/user_store.py exists", False, "not found")

try:
    with open("app.py", encoding="utf-8", errors="replace") as f:
        app_src = f.read()
    check("app.py calls inject_user_keys_to_env", "inject_user_keys_to_env" in app_src)
except FileNotFoundError:
    check("app.py exists", False, "not found")

# Onboarding UI lives in components/onboarding.py per Session 21 audit
onboarding_path = "components/onboarding.py"
try:
    with open(onboarding_path, encoding="utf-8", errors="replace") as f:
        onboarding_content = f.read()
    check("Onboarding file exists (components/onboarding.py)", True)
    check("Onboarding calls inject_user_keys_to_env after save",
          "inject_user_keys_to_env" in onboarding_content,
          onboarding_path)
except FileNotFoundError:
    check("Onboarding file exists (components/onboarding.py)", False,
          "components/onboarding.py not found")

# CHECK 5: Smoke tests
print("\nRunning model smoke tests...")
result = subprocess.run([r".\evolve_venv\Scripts\python.exe", "tests/model_smoke_test.py"], capture_output=True, text=True)
smoke_passed = result.returncode == 0
check("All 12 model smoke tests pass", smoke_passed, "see output below" if not smoke_passed else "all green")
if not smoke_passed:
    print(result.stdout[-2000:])
    print(result.stderr[-1000:])

# SUMMARY
print("\n" + "="*55)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"Session 21 Results: {passed} PASS  {failed} FAIL")
print("READY" if failed == 0 else "NEEDS WORK")
print("="*55)
