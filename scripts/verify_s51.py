import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 51 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# HRP in optimizer
t = open(
    "trading/optimization/portfolio_optimizer.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: portfolio_optimizer")
except SyntaxError as e:
    fail(f"Syntax error optimizer: {e}")

for name, pat in [
    ("HRP method", "hierarchical_risk_parity"),
    ("Recursive bisection", "_recursive_bisect"),
    ("Quasi-diagonal reorder", "leaves_list"),
    ("Distance matrix", "squareform"),
    ("Cluster variance", "_get_cluster_var"),
]:
    if pat in t:
        ok(f"{name} present")
    else:
        fail(f"{name} missing")

# Trade page UI
t2 = open(
    "pages/4_Trade.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t2)
    ok("Syntax valid: 4_Trade.py")
except SyntaxError as e:
    fail(f"Syntax error Trade: {e}")

for name, pat in [
    ("Portfolio Optimizer section", "Portfolio Optimizer"),
    ("HRP option in UI", "Hierarchical Risk Parity"),
    ("Run Optimization button", "run_port_opt"),
    ("Results display", "port_opt_result"),
    ("Weights bar chart", "Optimal Weights"),
]:
    if pat in t2:
        ok(f"{name} present")
    else:
        fail(f"{name} missing")

# Functional test - HRP math
print("\n--- HRP functional test ---")
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys, numpy as np, pandas as pd;"
        "sys.path.insert(0,'.');"
        "from trading.optimization"
        ".portfolio_optimizer import "
        "PortfolioOptimizer;"
        "np.random.seed(42);"
        "ret = pd.DataFrame("
        "np.random.randn(100,4)*0.01,"
        "columns=['AAPL','MSFT','GOOG','AMZN']);"
        "opt = PortfolioOptimizer();"
        "r = opt.hierarchical_risk_parity(ret);"
        "assert 'weights' in r, 'No weights';"
        "assert abs(sum(r['weights'].values())"
        "-1.0) < 0.01, 'Weights dont sum to 1';"
        "print('HRP weights:', r['weights']);"
        "print('Sharpe:', r['sharpe_ratio']);"
        "print('HRP OK')",
    ],
    capture_output=True,
    text=True,
    timeout=15,
)
print(_r.stdout[:400])
if _r.returncode == 0:
    ok("HRP math functional")
else:
    fail(f"HRP error: {_r.stderr[:300]}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-600:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.4.4.")
    sys.exit(0)
