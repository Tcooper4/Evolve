import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 53 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


files = {
    "components/analyze_ai_score.py":
        ["astype(str)", "hide_index"],
    "pages/3_Scanner.py":
        ["pd.to_numeric", "fillna(50)"],
    "trading/models/advanced/transformer/"
    "time_series_transformer.py":
        ['hasattr(self, "model")'],
    "components/analyze_diagnostics.py":
        ["_ss_data", "hasattr(_ss_data"],
    "components/analyze_chart.py":
        ['period in ("1d", "5d")',
         "_period_thresh"],
    "pages/6_Chat.py":
        ["aggregator empty",
         "yfinance"],
}

for fpath, patterns in files.items():
    try:
        t = open(fpath, encoding="utf-8",
                 errors="replace").read()
        try:
            ast.parse(t)
            ok(f"Syntax valid: "
               f"{fpath.split('/')[-1]}")
        except SyntaxError as e:
            fail(f"Syntax error "
                 f"{fpath.split('/')[-1]}"
                 f": {e}")
        for pat in patterns:
            if pat in t:
                ok(f"{pat[:40]} present "
                   f"in {fpath.split('/')[-1]}")
            else:
                fail(f"{pat[:40]} MISSING "
                     f"from "
                     f"{fpath.split('/')[-1]}")
    except FileNotFoundError:
        fail(f"File not found: {fpath}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True, text=True)
print((result.stdout + result.stderr)
      [-600:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, "
      f"{len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. "
          "Ready to commit v4.4.6.")
    sys.exit(0)
