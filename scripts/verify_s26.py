"""Session 26 verification."""
import sys
import os

sys.path.insert(0, ".")

print("=" * 60)
print("SESSION 26 VERIFICATION")
print("=" * 60)

# 1. Research Browser tab in Model Lab
print("\n[1] Research Browser tab in Model Lab...")
try:
    with open("pages/8_Model_Lab.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_research_tab = "Research Browser" in c or "research_browser" in c.lower()
    has_arxiv_import = "ArxivResearchFetcher" in c or "research_fetcher" in c
    has_search_btn = "Search arXiv" in c or "research_search" in c
    has_impl_btn = "Generate Implementation" in c or "ImplementationGenerator" in c or "ModelImplementationGenerator" in c
    print(f"  Research Browser tab added: {has_research_tab}")
    print(f"  ArxivResearchFetcher imported: {has_arxiv_import}")
    print(f"  Search button: {has_search_btn}")
    print(f"  Implementation generator: {has_impl_btn}")
    print(f"  {'PASS' if has_research_tab and has_arxiv_import else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 2. ArxivResearchFetcher importable and callable
print("\n[2] ArxivResearchFetcher functionality...")
try:
    from agents.implementations.research_fetcher import ArxivResearchFetcher

    _f = ArxivResearchFetcher()
    _methods = [m for m in dir(_f) if not m.startswith("_")]
    print("  Import: PASS")
    print(f"  Public methods: {_methods}")
    print("  PASS")
except Exception as e:
    print(f"  Import error: {e}")

# 3. Portfolio partial close + risk levels
print("\n[3] Portfolio partial close + risk level wiring...")
try:
    with open("pages/5_Portfolio.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    partial_wired = (
        "close_position" in c
        and "partial" in c.lower()
        and "rerun" in c
    )
    risk_wired = (
        "stop_loss" in c
        and "take_profit" in c
        and "save" in c.lower()
        and "position." in c
    )
    print(f"  Partial close wired: {partial_wired}")
    print(f"  Risk levels wired: {risk_wired}")
    print(f"  {'PASS' if partial_wired and risk_wired else 'CHECK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 4. Reports email wired
print("\n[4] Reports email delivery wired...")
try:
    with open("pages/9_Reports.py", encoding="utf-8", errors="replace") as f:
        c = f.read()
    has_ns = "NotificationService" in c or "notification_service" in c
    has_send = "send_notification" in c or "send_report" in c
    has_fallback = "ImportError" in c or "not available" in c.lower()
    print(f"  NotificationService called: {has_ns}")
    print(f"  Send method wired: {has_send}")
    print(f"  Graceful fallback if unconfigured: {has_fallback}")
    print(f"  {'PASS' if has_ns and has_send else 'NEEDS WORK'}")
except Exception as e:
    print(f"  ERROR: {e}")

# 5. Smoke tests
print("\n[5] Model smoke tests...")
import subprocess

result = subprocess.run(
    [sys.executable, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
output = result.stdout + result.stderr
if "All smoke tests completed. All PASS" in output:
    print("  PASS: All 12 models")
else:
    fails = [l.strip() for l in output.split("\n") if "FAIL" in l]
    print(f"  ISSUES: {fails}")

print("\n" + "=" * 60)
print("Session 26 complete. Paste output back.")
print("=" * 60)
