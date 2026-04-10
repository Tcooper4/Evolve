import ast
import os
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 54 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 — no legacy components.v1 iframe HTML API in app code
_html_needle = "st.components.v1." + "html"
_found = []
_EXCLUDE = {
    "evolve_venv", ".git", "__pycache__", "_archive",
    ".venv", "venv", "node_modules",
}
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in _EXCLUDE]
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        t = open(
            p, encoding="utf-8",
            errors="replace"
        ).read()
        if _html_needle in t:
            _found.append(p)
if _found:
    fail(f"legacy iframe HTML API still in: {_found}")
else:
    ok("No legacy components.v1.html in app code")

# Fix 2 — no deprecated container width kwarg
_ucw = []
_needle = "use" + "_container_width"
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in _EXCLUDE]
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(root, f)
        t = open(
            p, encoding="utf-8",
            errors="replace"
        ).read()
        if _needle in t:
            _ucw.append(p)
if _ucw:
    fail(
        f"deprecated width kwarg still in "
        f"{len(_ucw)} files: {_ucw[:3]}"
    )
else:
    ok("deprecated width kwarg fully removed")

# Fix 3 — trading/signals gone or clean
if os.path.exists("trading/signals"):
    _contents = [
        f for f in
        os.listdir("trading/signals")
        if f != "__pycache__"
        and not f.endswith(".pyc")
    ]
    if len(_contents) <= 1:
        ok(
            "trading/signals empty "
            "or __init__ only"
        )
    else:
        fail(
            f"trading/signals has "
            f"content: {_contents}"
        )
else:
    ok("trading/signals directory removed")

# Syntax spot-check key files
for fpath in [
    "app.py",
    "pages/1_Dashboard.py",
    "pages/2_Analyze.py",
    "components/theme.py",
    "trading/analysis/ai_score.py",
]:
    try:
        ast.parse(open(
            fpath, encoding="utf-8",
            errors="replace"
        ).read())
        ok(f"Syntax valid: {fpath}")
    except SyntaxError as e:
        fail(f"Syntax error {fpath}: {e}")

print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True, text=True
)
print((result.stdout + result.stderr)[-800:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(
    f"\n=== {len(PASS)} passed, "
    f"{len(FAIL)} failed ==="
)
if FAIL:
    sys.exit(1)
else:
    print(
        "All checks passed. "
        "Ready to commit v4.4.7."
    )
    sys.exit(0)
