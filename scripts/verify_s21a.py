import ast
import sys
import os

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

try:
    src = open("agents/llm/agent.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    check("agent.py syntax", True)
except SyntaxError as e:
    check("agent.py syntax", False, str(e))

try:
    src = open("pages/2_Forecasting.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    has_spc = "set_page_config" in src
    check("Forecasting.py parses", True)
    check("Forecasting.py no set_page_config", not has_spc, "still contains set_page_config" if has_spc else "")
except SyntaxError as e:
    check("Forecasting.py parses", False, str(e))

try:
    src = open("pages/3_Strategy_Testing.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    has_spc = "set_page_config" in src
    check("Strategy_Testing.py parses", True)
    check("Strategy_Testing.py no set_page_config", not has_spc, "still contains set_page_config" if has_spc else "")
except SyntaxError as e:
    check("Strategy_Testing.py parses", False, str(e))

try:
    src = open("pages/10_Alerts.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    check("Alerts.py parses", True)
except SyntaxError as e:
    check("Alerts.py parses", False, str(e))

try:
    src = open("pages/11_Admin.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    check("Admin.py parses", True)
except SyntaxError as e:
    check("Admin.py parses", False, str(e))

try:
    src = open("pages/8_Model_Lab.py", encoding="utf-8", errors="replace").read()
    ast.parse(src)
    check("Model_Lab.py parses", True)
    has_df_wrap = "pd.DataFrame" in src
    check("Model_Lab.py wraps arrays in DataFrame", has_df_wrap, "no pd.DataFrame wrapping found" if not has_df_wrap else "")
except SyntaxError as e:
    check("Model_Lab.py parses", False, str(e))

exists = os.path.exists("tests/model_smoke_test.py")
check("smoke test file exists", exists)

print("\n" + "=" * 50)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"TOTAL: {passed} passed, {failed} failed")
if failed == 0:
    print("SUCCESS: Session 21A complete — ready for 21B")
else:
    print("FAILED: Fix failures above before proceeding to 21B")
    sys.exit(1)
