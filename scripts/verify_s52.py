import ast, subprocess, sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 52 Verification ===\n")
PASS, FAIL = [], []
def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")
def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")

# insider_flow.py
t = open(
    "trading/data/insider_flow.py",
    encoding="utf-8",
    errors="replace").read()
try:
    ast.parse(t)
    ok("Syntax valid: insider_flow.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

for name, pat in [
    ("get_insider_cluster_signal",
     "get_insider_cluster_signal"),
    ("Cluster window logic",
     "cluster_window_days"),
    ("Unique buyer detection",
     "unique"),
    ("STRONG_BUY signal",
     "STRONG_BUY"),
    ("Neutral fallback",
     "_neutral_cluster"),
]:
    if pat in t:
        ok(f"{name} present")
    else:
        fail(f"{name} missing")

# ai_score.py
t2 = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8",
    errors="replace").read()
try:
    ast.parse(t2)
    ok("Syntax valid: ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

for name, pat in [
    ("Cluster fetch in worker",
     "get_insider_cluster_signal"),
    ("Cluster merge",
     "insider_cluster"),
    ("Cluster signal append",
     "Insider Cluster"),
    ("STRONG_BUY merge",
     "STRONG_BUY"),
]:
    if pat in t2:
        ok(f"{name} present")
    else:
        fail(f"{name} missing")

# Functional test
print("\n--- Cluster neutral test ---")
_r = subprocess.run(
    [python, "-c",
     "import sys; sys.path.insert(0,'.');"
     "from trading.data.insider_flow"
     " import _neutral_cluster,"
     " get_insider_cluster_signal;"
     "r = _neutral_cluster('TEST');"
     "assert r['success'] == False;"
     "assert r['cluster_signal'] =="
     " 'NEUTRAL';"
     "assert 'cluster_buy_count' in r;"
     "assert 'recent_buyers' in r;"
     "print('Neutral cluster OK')"],
    capture_output=True, text=True,
    timeout=10)
print(_r.stdout[:200])
if _r.returncode == 0:
    ok("Cluster neutral functional")
else:
    fail(f"Error: {_r.stderr[:200]}")

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
          "Ready to commit v4.4.5.")
    sys.exit(0)
