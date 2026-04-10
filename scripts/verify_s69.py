import ast
import os
import re
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 69 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — Transformer callable
tt = open(
    "trading/models/advanced/transformer/time_series_transformer.py",
    encoding="utf-8",
    errors="replace",
).read()
try:
    ast.parse(tt)
    ok("Syntax valid: transformer.py")
except SyntaxError as e:
    fail(f"Syntax error transformer: {e}")

if "__call__" in tt:
    ok("__call__ defined in transformer")
else:
    fail("__call__ missing from transformer")

# Functional test — transformer callable
_r = subprocess.run(
    [
        python,
        "-c",
        "import sys; sys.path.insert(0,'.');"
        "from trading.models.advanced.transformer.time_series_transformer "
        "import TransformerForecaster;"
        "t = TransformerForecaster();"
        "print('callable:', callable(t));"
        "assert callable(t), 'TransformerForecaster not callable';"
        "print('Transformer callable: OK')",
    ],
    capture_output=True,
    text=True,
    timeout=30,
)
print(_r.stdout[:200])
if _r.returncode == 0 and "OK" in _r.stdout:
    ok("TransformerForecaster is callable")
else:
    fail(
        f"Transformer callable test failed: "
        f"{(_r.stdout + _r.stderr)[:200]}"
    )

# Fix 2 — no st.components.v1.html
_found = []
for root, dirs, files in os.walk("."):
    dirs[:] = [
        d
        for d in dirs
        if d
        not in (
            "evolve_venv",
            ".git",
            "__pycache__",
            "_archive",
            ".venv",
            "venv",
            "scripts",
        )
    ]
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
            if "components.v1.html" in t:
                _found.append(p)
        except Exception:
            pass
if _found:
    fail(f"st.components.v1.html still in: {_found}")
else:
    ok("No st.components.v1.html remaining")

# Fix 3 — RL references in active code
_rl_found = []
_skip = (
    "test",
    "old_",
    "verify_",
    "scripts",
    ".venv",
    "evolve_venv",
    "_archive",
)
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if not any(s in d for s in _skip)]
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        if any(s in p for s in _skip):
            continue
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
            if re.search(
                r"gymnasium|"
                r"stable_baselines|"
                r"rl_trainer|RLTrainer",
                t,
            ):
                _rl_found.append(p)
        except Exception:
            pass
if _rl_found:
    fail(f"RL references remain in active code: {_rl_found}")
else:
    ok("RL trainer references removed from active code")

# Syntax spot-check key files
for fpath in [
    "trading/models/forecast_router.py",
    "trading/models/advanced/transformer/time_series_transformer.py",
]:
    try:
        ast.parse(
            open(fpath, encoding="utf-8", errors="replace").read()
        )
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
_out = result.stdout + result.stderr
_passes = _out.count("PASS:")
_fails = _out.count("FAIL:")
print(f"Smoke: {_passes} PASS, {_fails} FAIL")
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.6.2.")
    sys.exit(0)
