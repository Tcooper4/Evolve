import ast
import subprocess
import sys

python = r".\evolve_venv\Scripts\python.exe"
print("=== Session 34 Verification ===\n")
PASS, FAIL = [], []


def ok(msg):
    PASS.append(msg)
    print(f"OK    {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"FAIL  {msg}")


# Fix 1 - page_link removed
t_app = open("app.py", encoding="utf-8", errors="replace").read()
if "st.page_link" not in t_app:
    ok("page_link removed from app.py")
else:
    fail("page_link still in app.py")

# Fix 2 - button-triggered briefing
t_dash = open(
    "pages/1_Dashboard.py",
    encoding="utf-8",
    errors="replace",
).read()
if "gen_briefing_btn" in t_dash:
    ok("Generate briefing button present")
else:
    fail("Generate briefing button missing")
if "refresh_briefing_btn" in t_dash:
    ok("Refresh briefing button present")
else:
    fail("Refresh briefing button missing")
if "load_user_preferences" in t_dash:
    ok("User prefs loaded in briefing")
else:
    fail("User prefs not loaded")

# Fix 3 - options cache
t_opt = open(
    "trading/data/options_flow.py",
    encoding="utf-8",
    errors="replace",
).read()
if "_get_options_cache" in t_opt:
    ok("Options SQLite cache present")
else:
    fail("Options SQLite cache missing")
if "timeout=8" in t_opt:
    ok("8s timeout present")
else:
    fail("8s timeout missing")
if "_neutral_options" in t_opt:
    ok("Neutral fallback present")
else:
    fail("Neutral fallback missing")

# Fix 4 - chat cache keys match
t_chat = open(
    "pages/6_Chat.py",
    encoding="utf-8",
    errors="replace",
).read()
if "home_briefing_report" in t_chat:
    ok("Chat uses correct cache key")
else:
    fail("Chat cache key mismatch")

# Syntax checks
for fpath in [
    "app.py",
    "pages/1_Dashboard.py",
    "trading/data/options_flow.py",
    "pages/6_Chat.py",
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
print((result.stdout + result.stderr)[-1500:])
if result.returncode == 0:
    ok("Smoke test passed")
else:
    fail("Smoke test FAILED")

print(f"\n=== {len(PASS)} passed, {len(FAIL)} failed ===")
if FAIL:
    sys.exit(1)
else:
    print("All checks passed. Ready to commit v4.2.2.")
    sys.exit(0)
