import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 27 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Check Dashboard has fragment
text = open(
    "pages/1_Dashboard.py",
    encoding="utf-8",
    errors="replace",
).read()
if "@st.fragment" in text:
    ok("Dashboard: @st.fragment present")
else:
    fail("Dashboard: @st.fragment missing")
if "_render_briefing" in text:
    ok("Dashboard: _render_briefing function present")
else:
    fail("Dashboard: _render_briefing missing")

# Check Chat has fragment
text = open(
    "pages/6_Chat.py",
    encoding="utf-8",
    errors="replace",
).read()
if "@st.fragment" in text:
    ok("Chat: @st.fragment present")
else:
    fail("Chat: @st.fragment missing")
if "_render_chat_briefing" in text:
    ok("Chat: _render_chat_briefing present")
else:
    fail("Chat: _render_chat_briefing missing")

# Syntax check
for f in [
    "pages/1_Dashboard.py",
    "pages/6_Chat.py",
]:
    try:
        ast.parse(
            open(f, encoding="utf-8", errors="replace").read()
        )
        ok(f"Syntax valid: {f}")
    except SyntaxError as e:
        fail(f"Syntax error in {f}: {e}")

# Smoke test
print()
print("--- Smoke test ---")
result = subprocess.run(
    [python, "tests/model_smoke_test.py"],
    capture_output=True,
    text=True,
)
print((result.stdout + result.stderr)[-1500:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    for f in FAIL:
        print(f"  FAIL  {f}")
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.1.4.")
    sys.exit(0)
