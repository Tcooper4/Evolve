import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 50 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


t = open(
    "trading/analysis/ai_score.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(t)
    ok("Syntax valid: ai_score.py")
except SyntaxError as e:
    fail(f"Syntax error: {e}")

checks = [
    ("_apply_regime_tilt function", "_apply_regime_tilt"),
    ("regime_label parameter", "regime_label"),
    ("vix_level parameter", "vix_level"),
    ("RISK_ON handling", "RISK_ON"),
    ("RISK_OFF handling", "RISK_OFF"),
    ("MAX_TILT cap", "_MAX_TILT"),
    ("Renormalize after tilt", "_blend_total"),
    ("Regime signal appended", "Market Regime"),
    ("Call site updated", "regime_label=_regime_label"),
]
for name, pat in checks:
    if pat in t:
        ok(f"{name} present")
    else:
        fail(f"{name} missing")

# Functional test
print("\n--- Regime tilt test ---")
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.analysis.ai_score"
        " import _apply_regime_tilt;"
        "base = {'technical':0.30,"
        "'momentum':0.35,'sentiment':0.20,"
        "'fundamental':0.15};"
        "risk_on = _apply_regime_tilt("
        "base, 'RISK_ON', 12.0);"
        "risk_off = _apply_regime_tilt("
        "base, 'RISK_OFF', 28.0);"
        "assert risk_on['momentum'] >"
        " base['momentum'], "
        "'Risk-on should tilt momentum up';"
        "assert risk_off['fundamental'] >"
        " base['fundamental'], "
        "'Risk-off should tilt fundamental';"
        "print('RISK_ON weights:', risk_on);"
        "print('RISK_OFF weights:', risk_off);"
        "print('Tilt logic correct')",
    ],
    capture_output=True,
    text=True,
    timeout=10,
)
print(_r.stdout[:400])
if _r.returncode == 0:
    ok("Regime tilt logic correct")
else:
    fail(f"Tilt test failed: {_r.stderr[:200]}")

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
    print("All checks passed. Ready to commit v4.4.3.")
    sys.exit(0)
