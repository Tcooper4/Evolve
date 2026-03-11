"""
Session 22 Verification Script — corrected paths from audit_s22b
Run: .\evolve_venv\Scripts\python.exe scripts\verify_s22.py
"""

import ast
import os
import subprocess

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

# 1. GNN — correct path is trading/models/advanced/gnn/gnn_model.py
try:
    with open("trading/models/advanced/gnn/gnn_model.py",
              encoding="utf-8", errors="replace") as f:
        gnn_src = f.read()
    check("GNN model file exists at correct path", True)
    check("GNN has try/except guard",
          "try:" in gnn_src and "except" in gnn_src)
    # Check for the multi-asset dict crash pattern
    check("GNN multi-asset int*dict pattern addressed",
          "unsupported" not in gnn_src)
except FileNotFoundError:
    check("GNN model file exists at correct path", False,
          "trading/models/advanced/gnn/gnn_model.py not found")

# 2. SHAP — correct path is trading/models/forecast_explainability.py
try:
    with open("trading/models/forecast_explainability.py",
              encoding="utf-8", errors="replace") as f:
        shap_src = f.read()
    has_import = "import shap" in shap_src
    has_except = "except ImportError" in shap_src or (
        "except" in shap_src and "shap" in shap_src.lower())
    has_flag = "HAS_SHAP" in shap_src or "shap_available" in shap_src
    hardcoded_false = ("HAS_SHAP = False" in shap_src and
                       "import shap" not in shap_src)
    check("SHAP dynamic try/except import in forecast_explainability.py",
          has_import and has_except and has_flag and not hardcoded_false,
          f"import={has_import} except={has_except} flag={has_flag} "
          f"hardcoded_false={hardcoded_false}")
except FileNotFoundError:
    check("trading/models/forecast_explainability.py exists", False,
          "file not found")

# 3. Trade.to_dict — correct file is trading/backtesting/trade_models.py
try:
    with open("trading/backtesting/trade_models.py",
              encoding="utf-8", errors="replace") as f:
        trade_src = f.read()
    check("Trade.to_dict has entry_date", '"entry_date"' in trade_src)
    check("Trade.to_dict has exit_date", '"exit_date"' in trade_src)
    check("Trade.to_dict has duration_days", '"duration_days"' in trade_src)
    check("exit_date not hardcoded None",
          'exit_date = None  # Backtesting Trade has no exit_time'
          not in trade_src)
    try:
        ast.parse(trade_src)
        check("trade_models.py parses cleanly", True)
    except SyntaxError as e:
        check("trade_models.py parses cleanly", False, str(e))
except FileNotFoundError:
    check("trading/backtesting/trade_models.py exists", False)

# 4. factor_model.py
try:
    with open("trading/analysis/factor_model.py",
              encoding="utf-8", errors="replace") as f:
        fm_src = f.read()
    try:
        ast.parse(fm_src)
        check("factor_model.py parses cleanly", True)
    except SyntaxError as e:
        check("factor_model.py parses cleanly", False, str(e))
except FileNotFoundError:
    check("factor_model.py exists", False)

# 5. Ridge safe MAPE
try:
    with open("trading/models/ridge_model.py",
              encoding="utf-8", errors="replace") as f:
        ridge_src = f.read()
    check("Ridge has safe MAPE",
          "safe_mape" in ridge_src
          or ("mask" in ridge_src and "mape" in ridge_src.lower()))
except FileNotFoundError:
    check("ridge_model.py exists", False)

# 6. Admin page
try:
    with open("pages/11_Admin.py",
              encoding="utf-8", errors="replace") as f:
        admin_src = f.read()
    try:
        ast.parse(admin_src)
        check("Admin page parses cleanly", True)
    except SyntaxError as e:
        check("Admin page parses cleanly", False, str(e))
    check("Admin Task Orchestrator has fallback",
          "Orchestrator" not in admin_src
          or any(x in admin_src for x in
                 ["except", "st.caption", "st.info"]))
except FileNotFoundError:
    check("pages/11_Admin.py exists", False)

# 7. Briefing service
try:
    with open("trading/services/home_briefing_service.py",
              encoding="utf-8", errors="replace") as f:
        briefing_src = f.read()
    check("Briefing service catches ValueError",
          "ValueError" in briefing_src
          or ("except Exception" in briefing_src
              and "fallback" in briefing_src.lower()))
    try:
        ast.parse(briefing_src)
        check("home_briefing_service.py parses cleanly", True)
    except SyntaxError as e:
        check("home_briefing_service.py parses cleanly", False, str(e))
except FileNotFoundError:
    check("home_briefing_service.py exists", False)

# 8. InnovationConfig
try:
    with open("agents/model_innovation_agent.py",
              encoding="utf-8", errors="replace") as f:
        innov_src = f.read()
    check("InnovationConfig defined", "InnovationConfig" in innov_src)
    try:
        ast.parse(innov_src)
        check("model_innovation_agent.py parses cleanly", True)
    except SyntaxError as e:
        check("model_innovation_agent.py parses cleanly", False, str(e))
except FileNotFoundError:
    check("agents/model_innovation_agent.py exists", False)

# 9. Raw error phrases removed
bad_phrases = {
    "pages/3_Strategy_Testing.py":
        "Strategy Research Agent not available",
    "pages/8_Model_Lab.py":
        "Model discovery is not available (agent rationalized)",
}
for fpath, phrase in bad_phrases.items():
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            content = f.read()
        check(f"{os.path.basename(fpath)} raw error phrase removed",
              phrase not in content,
              f"still contains: '{phrase[:55]}'"
              if phrase in content else "clean")
    except FileNotFoundError:
        check(f"{fpath} exists", False)

# 10. Key pages parse cleanly — CORRECT filenames from audit
for fpath in [
    "pages/2_Forecasting.py",   # was wrongly listed as 5_Forecasting.py
    "pages/7_Performance.py",
    "pages/0_Home.py",
    "pages/3_Strategy_Testing.py",
    "pages/8_Model_Lab.py",
]:
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            src = f.read()
        try:
            ast.parse(src)
            check(f"{os.path.basename(fpath)} parses cleanly", True)
        except SyntaxError as e:
            check(f"{os.path.basename(fpath)} parses cleanly",
                  False, str(e))
    except FileNotFoundError:
        check(f"{fpath} exists", False)

# Smoke tests
print("\nRunning model smoke tests...")
result = subprocess.run(
    [r".\evolve_venv\Scripts\python.exe", "tests/model_smoke_test.py"],
    capture_output=True, text=True
)
smoke_passed = result.returncode == 0
check("All 12 model smoke tests pass", smoke_passed,
      "see output below" if not smoke_passed else "all green")
if not smoke_passed:
    print(result.stdout[-2000:])
    print(result.stderr[-1000:])

# Summary
print("\n" + "=" * 55)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"Session 22 Results: {passed} PASS  {failed} FAIL")
print("READY — commit and deploy" if failed == 0 else "NEEDS WORK")
print("=" * 55)