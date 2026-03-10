import ast, os, sys

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

pages = [
    "pages/5_Portfolio.py",
    "pages/6_Risk_Management.py",
    "pages/7_Performance.py",
    "pages/9_Reports.py",
    "pages/10_Alerts.py",
    "pages/13_Scanner.py",
]

for p in pages:
    if not os.path.exists(p):
        check(f"{p} exists", False)
        continue
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
        ast.parse(src)
        check(f"{p} parses", True)
        # Check for bad patterns
        has_info_price = ".info.get('regularMarketPrice'" in src or "regularMarketPrice" in src
        has_info_dict_price = "ticker.info[" in src.lower() or ".info['price']" in src.lower()
        check(f"{p} no regularMarketPrice", not has_info_price)
        check(f"{p} no .info price lookup", not has_info_dict_price)
    except SyntaxError as e:
        check(f"{p} parses", False, str(e))

# Confirm already-correct pages untouched
for p in ["pages/0_Home.py", "pages/4_Trade_Execution.py"]:
    try:
        src = open(p, encoding="utf-8", errors="replace").read()
        ast.parse(src)
        check(f"{p} still intact", "fast_info" in src)
    except SyntaxError as e:
        check(f"{p} still intact", False, str(e))

check("smoke test exists", os.path.exists("tests/model_smoke_test.py"))

print("\n" + "=" * 50)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"TOTAL: {passed} passed, {failed} failed")
if failed == 0:
    print("SUCCESS: Session 22B complete")
else:
    print("FAILED: Fix above before proceeding")
    sys.exit(1)
