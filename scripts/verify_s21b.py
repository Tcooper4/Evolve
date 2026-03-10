import ast, sys, os

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

# 1. Home page
try:
    src = open("pages/0_Home.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    check("Home.py parses", True)
    check("Home.py uses fast_info", "fast_info" in src)
    check("Home.py no hardcoded movers", not ("'AMD'" in src or "'NVDA'" in src or "'GOOGL'" in src))
    check("Home.py stray chart removed", "Live Market Chart" not in src)
    check("Home.py dynamic movers present", "gainers" in src.lower() or "top_movers" in src.lower() or "pct_change" in src.lower())
except SyntaxError as e:
    check("Home.py parses", False, str(e))

# 2. Scanner
try:
    src = open("pages/13_Scanner.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    check("Scanner.py parses", True)
    check("Scanner.py has universe selectbox", "Stock Universe" in src or "universe_choice" in src)
    check("Scanner.py has slow scan warning", "slow" in src.lower() or "minutes" in src.lower())
except SyntaxError as e:
    check("Scanner.py parses", False, str(e))

# 3. Admin
try:
    src = open("pages/11_Admin.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    check("Admin.py parses", True)
    check("Admin.py has Top Movers Universe setting", "Top Movers Universe" in src or "top_movers_universe" in src.lower())
except SyntaxError as e:
    check("Admin.py parses", False, str(e))

# 4. Position sizing — find the file first
pos_page = None
for candidate in ["pages/6_Position_Sizing.py", "pages/5_Position_Sizing.py",
                  "pages/7_Position_Sizing.py", "pages/4_Position_Sizing.py"]:
    if os.path.exists(candidate):
        pos_page = candidate
        break

# also try glob search
if not pos_page:
    import glob
    matches = glob.glob("pages/*osition*.py") + glob.glob("pages/*izing*.py")
    if matches:
        pos_page = matches[0]

if pos_page:
    print(f"  [INFO] Found position sizing page: {pos_page}")
    try:
        src = open(pos_page, encoding="utf-8", errors="replace").read()
        ast.parse(src)
        check(f"{pos_page} parses", True)
        check(f"{pos_page} uses fast_info", "fast_info" in src)
    except SyntaxError as e:
        check(f"{pos_page} parses", False, str(e))
else:
    # list all pages so we can see what exists
    import glob
    all_pages = glob.glob("pages/*.py")
    print(f"  [INFO] All pages found: {sorted(all_pages)}")
    check("Position sizing page found", False, "could not locate — see page list above")

# 5. Smoke test
check("smoke test file exists", os.path.exists("tests/model_smoke_test.py"))

print("\n" + "=" * 50)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"TOTAL: {passed} passed, {failed} failed")
if failed == 0:
    print("SUCCESS: Session 21B complete")
else:
    print("FAILED: Fix failures above")
    sys.exit(1)
