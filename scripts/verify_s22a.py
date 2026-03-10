import ast, os, glob, sys

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

# app.py must still have set_page_config
src = open("app.py", encoding="utf-8", errors="replace").read()
check("app.py still has set_page_config", "set_page_config" in src)

# These two were fixed in 21A — confirm still clean
for p in ["pages/2_Forecasting.py", "pages/3_Strategy_Testing.py"]:
    src = open(p, encoding="utf-8", errors="replace").read()
    check(f"{p} still clean", "set_page_config" not in src)

# All 12 pages must now have no set_page_config and must parse
pages_to_check = [
    "pages/0_Home.py", "pages/1_Chat.py", "pages/4_Trade_Execution.py",
    "pages/5_Portfolio.py", "pages/6_Risk_Management.py", "pages/7_Performance.py",
    "pages/8_Model_Lab.py", "pages/9_Reports.py", "pages/10_Alerts.py",
    "pages/11_Admin.py", "pages/12_Memory.py", "pages/13_Scanner.py"
]

for p in pages_to_check:
    if not os.path.exists(p):
        check(f"{p} exists", False)
        continue
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
        ast.parse(src)
        check(f"{p} parses", True)
        check(f"{p} no set_page_config", "set_page_config" not in src)
    except SyntaxError as e:
        check(f"{p} parses", False, str(e))

print("\n" + "=" * 50)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"TOTAL: {passed} passed, {failed} failed")
if failed == 0:
    print("SUCCESS: Session 22A complete — ready for 22B")
else:
    print("FAILED: Fix failures above before 22B")
    sys.exit(1)
